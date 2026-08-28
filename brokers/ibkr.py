import logging
import time
from datetime import date
from threading import Thread, Event

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import TickerId

import config
from brokers.base import Broker
from engine.portfolio import Position

logger = logging.getLogger(__name__)


class _IbkrApp(EWrapper, EClient):
    """Internal TWS API handler with callbacks for positions, orders, and prices."""

    def __init__(self):
        EClient.__init__(self, self)
        self.positions: dict[str, dict] = {}
        self.account_info: dict = {}
        self.next_order_id: int | None = None
        self.order_results: dict[int, dict] = {}  # orderId -> result
        self.price_data: dict[int, float] = {}     # reqId -> price
        self._connected_event = Event()
        self._order_events: dict[int, Event] = {}
        self._price_events: dict[int, Event] = {}
        self._positions_done = Event()
        self._account_done = Event()

    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        if errorCode in (2104, 2106, 2158):
            return  # Informational messages
        if reqId > -1:
            logger.error(f"IBKR Error Id:{reqId} Code:{errorCode} Msg:{errorString}")

    def nextValidId(self, orderId: int):
        self.next_order_id = orderId
        self._connected_event.set()

    def position(self, account, contract, pos, avgCost):
        if pos != 0:
            self.positions[contract.symbol] = {
                "symbol": contract.symbol,
                "quantity": int(pos),
                "avg_cost": avgCost,
                "sec_type": contract.secType,
                "currency": contract.currency,
                "account": account,
            }

    def positionEnd(self):
        self._positions_done.set()

    def accountSummary(self, reqId, account, tag, value, currency):
        self.account_info[tag] = {"value": float(value), "currency": currency}

    def accountSummaryEnd(self, reqId):
        self._account_done.set()

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice,
                    permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        self.order_results[orderId] = {
            "status": status,
            "filled": filled,
            "remaining": remaining,
            "avg_fill_price": avgFillPrice,
        }
        if status in ("Filled", "Cancelled", "Inactive"):
            event = self._order_events.get(orderId)
            if event:
                event.set()

    def tickPrice(self, reqId, tickType, price, attrib):
        # tickType: 1 = bid, 2 = ask, 4 = last
        if price <= 0:
            return
        if reqId not in self.price_data:
            self.price_data[reqId] = {}
        if tickType == 1:
            self.price_data[reqId]["bid"] = price
        elif tickType == 2:
            self.price_data[reqId]["ask"] = price
        elif tickType == 4:
            self.price_data[reqId]["last"] = price

        # Signal when we have bid+ask or at least last
        data = self.price_data[reqId]
        if ("bid" in data and "ask" in data) or "last" in data:
            event = self._price_events.get(reqId)
            if event:
                event.set()


def _make_stock_contract(symbol: str) -> Contract:
    """Create a US stock contract."""
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract


class IbkrBroker(Broker):
    """Interactive Brokers TWS API broker integration.

    Requires TWS or IB Gateway running on localhost.
    Configure via .env: IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID.
    """

    name = "ibkr"

    def __init__(self):
        self.app = _IbkrApp()
        self._thread: Thread | None = None
        self._connected = False
        self._next_req_id = 1000

    def _get_req_id(self) -> int:
        rid = self._next_req_id
        self._next_req_id += 1
        return rid

    def _get_order_id(self) -> int:
        oid = self.app.next_order_id
        self.app.next_order_id += 1
        return oid

    def connect(self) -> None:
        host = config.IBKR_HOST
        port = config.IBKR_PORT
        client_id = config.IBKR_CLIENT_ID

        logger.info(f"Connecting to IBKR TWS at {host}:{port} (clientId={client_id})...")

        self.app.connect(host, port, client_id)
        self._thread = Thread(target=self.app.run, daemon=True)
        self._thread.start()

        if not self.app._connected_event.wait(timeout=10):
            raise ConnectionError("Failed to connect to IBKR TWS — is TWS/Gateway running?")

        self._connected = True
        logger.info(f"Connected to IBKR TWS, next order ID: {self.app.next_order_id}")

    def disconnect(self) -> None:
        if self._connected:
            self.app.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR TWS")

    def get_account_info(self) -> dict:
        self.app._account_done.clear()
        self.app.account_info = {}
        self.app.reqAccountSummary(
            self._get_req_id(), "All", "NetLiquidation,AvailableFunds"
        )
        self.app._account_done.wait(timeout=10)

        nav = self.app.account_info.get("NetLiquidation", {}).get("value", 0.0)
        cash = self.app.account_info.get("AvailableFunds", {}).get("value", 0.0)
        currency = self.app.account_info.get("NetLiquidation", {}).get("currency", "USD")

        return {
            "cash": cash,
            "portfolio_value": nav,
            "currency": currency,
        }

    def get_positions(self) -> dict[str, Position]:
        self.app._positions_done.clear()
        self.app.positions = {}
        self.app.reqPositions()
        self.app._positions_done.wait(timeout=10)

        positions = {}
        for symbol, data in self.app.positions.items():
            positions[symbol] = Position(
                symbol=symbol,
                quantity=data["quantity"],
                avg_cost=data["avg_cost"],
                entry_date=date.today(),
            )

        return positions

    def buy(self, symbol: str, quantity: int, price: float) -> dict | None:
        return self._place_order(symbol, quantity, "BUY", price)

    def sell(self, symbol: str, quantity: int, price: float) -> dict | None:
        return self._place_order(symbol, quantity, "SELL", price)

    def _place_order(self, symbol: str, quantity: int, action: str, price: float) -> dict | None:
        contract = _make_stock_contract(symbol)

        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = "MKT"  # Market order for simplicity
        order.eTradeOnly = False
        order.firmQuoteOnly = False

        order_id = self._get_order_id()
        event = Event()
        self.app._order_events[order_id] = event

        logger.info(f"IBKR placing {action} {quantity} {symbol} (MKT order, id={order_id})")
        self.app.placeOrder(order_id, contract, order)

        # Wait for fill (up to 30 seconds)
        if not event.wait(timeout=30):
            logger.warning(f"IBKR order {order_id} timed out waiting for fill")

        result = self.app.order_results.get(order_id)
        if result and result.get("status") == "Filled":
            fill_price = result.get("avg_fill_price", price)
            logger.info(f"IBKR {action} {quantity} {symbol} filled @ {fill_price:.2f}")
            return {
                "broker": "ibkr",
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price": fill_price,
                "order_id": order_id,
                "raw": result,
            }

        status = result.get("status", "Unknown") if result else "No response"
        logger.error(f"IBKR {action} {symbol} order status: {status}")
        return None

    def get_current_price(self, symbol: str) -> float | None:
        contract = _make_stock_contract(symbol)
        req_id = self._get_req_id()

        event = Event()
        self.app._price_events[req_id] = event

        self.app.reqMktData(req_id, contract, "", True, False, [])

        if event.wait(timeout=10):
            data = self.app.price_data.get(req_id, {})
            # Prefer bid/ask mid price, fall back to last
            if "bid" in data and "ask" in data:
                return (data["bid"] + data["ask"]) / 2
            if "last" in data:
                return data["last"]

        logger.warning(f"IBKR: could not get price for {symbol}")
        return None
