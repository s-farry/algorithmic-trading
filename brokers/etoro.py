import json
import logging
import time
import uuid
from datetime import date, datetime

import requests

import config
from brokers.base import Broker
from engine.portfolio import Position

logger = logging.getLogger(__name__)

API_BASE = "https://public-api.etoro.com"
API_PREFIX = "/api/v1"

# Persistent cache for instrument ID <-> symbol mapping
_INSTRUMENTS_CACHE_FILE = config.RESULTS_DIR / "etoro_instruments_cache.json"


class EtoroBroker(Broker):
    """eToro REST API broker integration.

    Authentication requires two keys set in .env:
        ETORO_API_KEY      - API key (x-api-key header)
        ETORO_USER_KEY     - User-specific key (x-user-key header)

    Supports both real and demo trading via the `demo` parameter.
    """

    name = "etoro"

    def __init__(self, demo: bool = False):
        self.demo = demo
        self.session = requests.Session()
        self._instruments_cache: dict[str, int] = {}  # symbol -> instrumentId
        self._id_to_symbol: dict[int, str] = {}  # instrumentId -> symbol (reverse)
        self._connected = False
        self._load_disk_cache()

    def _load_disk_cache(self):
        """Load the persistent instrument ID <-> symbol cache from disk."""
        if _INSTRUMENTS_CACHE_FILE.exists():
            try:
                with open(_INSTRUMENTS_CACHE_FILE) as f:
                    data = json.load(f)
                # JSON keys are strings, convert IDs back to int
                for symbol, inst_id in data.items():
                    self._instruments_cache[symbol] = int(inst_id)
                    self._id_to_symbol[int(inst_id)] = symbol
                logger.info(f"Loaded {len(self._instruments_cache)} instruments from disk cache")
            except Exception as e:
                logger.warning(f"Failed to load instruments cache: {e}")

    def _save_disk_cache(self):
        """Persist the instrument ID <-> symbol cache to disk."""
        try:
            with open(_INSTRUMENTS_CACHE_FILE, "w") as f:
                json.dump(self._instruments_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save instruments cache: {e}")

    def _cache_instrument(self, symbol: str, inst_id):
        """Add a symbol <-> instrumentId mapping to both in-memory and disk cache."""
        symbol = symbol.upper()
        try:
            inst_id = int(inst_id)
        except (ValueError, TypeError):
            return
        if symbol not in self._instruments_cache or self._instruments_cache[symbol] != inst_id:
            self._instruments_cache[symbol] = inst_id
            self._id_to_symbol[inst_id] = symbol
            self._save_disk_cache()

    def _headers(self) -> dict:
        return {
            "x-api-key": config.ETORO_API_KEY,
            "x-user-key": config.ETORO_USER_KEY,
            "x-request-id": str(uuid.uuid4()),
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs) -> dict | None:
        """Make an authenticated API request with retry logic."""
        url = f"{API_BASE}{API_PREFIX}{path}"
        headers = self._headers()
        headers.update(kwargs.pop("headers", {}))

        for attempt in range(3):
            try:
                resp = self.session.request(method, url, headers=headers, **kwargs)

                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue

                if resp.status_code >= 400:
                    body = resp.text[:1000] if resp.text else "(empty)"
                    logger.error(
                        f"eToro API error ({method} {path}): "
                        f"{resp.status_code} {resp.reason}\n"
                        f"  Request body: {kwargs.get('json', '(none)')}\n"
                        f"  Response: {body}"
                    )
                    return None

                return resp.json() if resp.content else {}

            except requests.exceptions.RequestException as e:
                logger.error(f"eToro API error ({method} {path}): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return None

        return None

    def _trading_prefix(self) -> str:
        """Return path prefix for demo or real trading."""
        return "/trading/execution/demo" if self.demo else "/trading/execution"

    def _info_prefix(self) -> str:
        """Return path prefix for demo or real portfolio info."""
        return "/trading/info/demo" if self.demo else "/trading/info"

    def connect(self) -> None:
        if not config.ETORO_API_KEY or not config.ETORO_USER_KEY:
            raise ValueError(
                "eToro credentials not set. Add ETORO_API_KEY and ETORO_USER_KEY to .env"
            )

        # Test connection by fetching instruments
        logger.info("Connecting to eToro API...")
        self._load_instruments()
        self._connected = True
        mode = "DEMO" if self.demo else "REAL"
        logger.info(f"Connected to eToro ({mode} mode), {len(self._instruments_cache)} instruments loaded")

    def disconnect(self) -> None:
        self.session.close()
        self._connected = False
        logger.info("Disconnected from eToro")

    def _load_instruments(self):
        """Fetch instrument metadata and build symbol -> instrumentId mapping."""
        data = self._request("GET", "/market-data/instruments")
        if not data:
            logger.warning("Instruments endpoint returned no data, will resolve symbols on demand")
            return

        # Debug: log the response structure so we can see the actual field names
        if isinstance(data, dict):
            logger.debug(f"Instruments response keys: {list(data.keys())}")
            # Try to find the list inside the response
            instruments = None
            for key in data:
                if isinstance(data[key], list) and len(data[key]) > 0:
                    instruments = data[key]
                    logger.debug(f"Found instrument list under key '{key}', first item keys: {list(instruments[0].keys())}")
                    break
            if instruments is None:
                # Maybe the dict itself has instrument-like keys
                logger.debug(f"Response structure: {str(data)[:500]}")
                instruments = []
        elif isinstance(data, list):
            instruments = data
            if instruments:
                logger.debug(f"Instruments list, first item keys: {list(instruments[0].keys())}")
        else:
            instruments = []

        for inst in instruments:
            # Try various known field names for the symbol and ID
            symbol = (
                inst.get("symbolFull") or inst.get("SymbolFull")
                or inst.get("tickerSymbol") or inst.get("TickerSymbol")
                or inst.get("symbol") or inst.get("Symbol")
                or inst.get("ticker") or inst.get("Ticker")
                or inst.get("name") or inst.get("Name")
                or ""
            )
            inst_id = (
                inst.get("instrumentId") or inst.get("InstrumentId")
                or inst.get("instrumentID") or inst.get("InstrumentID")
                or inst.get("id") or inst.get("Id")
                or inst.get("instrumentDisplayId") or inst.get("InstrumentDisplayId")
            )
            if symbol and inst_id:
                self._cache_instrument(symbol, inst_id)

        if not self._instruments_cache and instruments:
            # Still nothing — log first item so we can fix the field names
            logger.warning(f"Could not parse instruments. Sample item: {instruments[0]}")

    def _resolve_instrument_id(self, symbol: str) -> int | None:
        """Resolve a ticker symbol to an eToro instrument ID."""
        inst_id = self._instruments_cache.get(symbol.upper())
        if inst_id:
            return inst_id

        # Try searching the API
        data = self._request("GET", "/market-data/search", params={"query": symbol})
        if data:
            results = data if isinstance(data, list) else data.get("instruments", data.get("data", []))
            for inst in results:
                s = inst.get("symbolFull") or inst.get("symbol") or inst.get("ticker", "")
                i = inst.get("instrumentID") or inst.get("instrumentId") or inst.get("id")
                if s and i:
                    self._cache_instrument(s, i)
                if s.upper() == symbol.upper() and i:
                    return i

        logger.warning(f"Could not resolve eToro instrument ID for {symbol}")
        return None

    def _resolve_symbol_from_id(self, instrument_id: int) -> str | None:
        """Resolve an eToro instrument ID back to a ticker symbol.

        Relies on the reverse cache (_id_to_symbol). To populate the cache,
        call warm_cache() with known symbol names before get_positions().
        """
        return self._id_to_symbol.get(instrument_id)

    def warm_cache(self, symbols: list[str] | set[str]):
        """Pre-populate the instrument cache by resolving a list of symbols.

        Call this with known holdings (e.g. from the transaction log) before
        get_positions() so the reverse ID->symbol lookup works.
        """
        missing = [s for s in symbols if s.upper() not in self._instruments_cache]
        if not missing:
            return

        logger.info(f"Warming instrument cache for {len(missing)} symbols...")
        resolved = 0
        for sym in missing:
            inst_id = self._resolve_instrument_id(sym)
            if inst_id:
                resolved += 1
        logger.info(f"Resolved {resolved}/{len(missing)} symbols to instrument IDs")

    @staticmethod
    def _parse_date(value) -> date | None:
        """Parse a date string from the eToro API into a date object."""
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except (ValueError, AttributeError):
            return None

    def _unwrap(self, data: dict) -> dict:
        """Unwrap eToro's nested response (data is under 'clientPortfolio')."""
        if isinstance(data, dict) and "clientPortfolio" in data:
            return data["clientPortfolio"]
        return data

    def get_account_info(self) -> dict:
        data = self._request("GET", f"{self._info_prefix()}/portfolio")
        if not data:
            logger.warning("Portfolio endpoint returned no data")
            return {"cash": 0.0, "portfolio_value": 0.0, "currency": "USD"}

        portfolio = self._unwrap(data)

        credit = portfolio.get("credit", 0.0)
        # Estimate portfolio value as credit + unrealised P&L from positions
        positions = portfolio.get("positions", [])
        unrealised = sum(p.get("netProfit", 0.0) for p in positions)

        return {
            "cash": credit,
            "portfolio_value": credit + unrealised,
            "currency": "USD",
        }

    def get_positions(self) -> dict[str, Position]:
        data = self._request("GET", f"{self._info_prefix()}/pnl")
        if not data:
            logger.warning("PnL endpoint returned no data")
            return {}

        portfolio = self._unwrap(data)
        positions_list = portfolio.get("positions", [])

        if not positions_list:
            # Log the response structure to help debug missing positions
            logger.info(
                f"PnL endpoint returned 0 positions. "
                f"Response keys: {list(portfolio.keys()) if isinstance(portfolio, dict) else type(portfolio)}"
            )
            if isinstance(portfolio, dict):
                for key, val in portfolio.items():
                    if isinstance(val, list) and val:
                        logger.info(f"  Key '{key}' has {len(val)} items, first: {str(val[0])[:200]}")

        positions: dict[str, Position] = {}
        unresolved = []
        for p in positions_list:
            # Resolve instrument ID back to symbol — always normalise to int
            raw_id = p.get("instrumentID") or p.get("instrumentId") or p.get("InstrumentId")
            try:
                inst_id = int(raw_id) if raw_id is not None else None
            except (ValueError, TypeError):
                inst_id = None

            symbol = self._id_to_symbol.get(inst_id) if inst_id else None

            if not symbol:
                symbol = (
                    p.get("symbolFull") or p.get("symbol")
                    or p.get("ticker") or str(inst_id or "UNKNOWN")
                )
                unresolved.append((inst_id, symbol))

            symbol = symbol.upper()
            qty = int(p.get("units", p.get("amount", p.get("Units", 0))))
            cost = float(p.get("openRate", p.get("OpenRate", p.get("averagePrice", 0.0))))
            entry = self._parse_date(p.get("openDateTime"))

            if symbol in positions:
                # eToro creates a separate position record per buy order.
                # Aggregate them so positions_value reflects all invested capital.
                existing = positions[symbol]
                total_qty = existing.quantity + qty
                avg_cost = (
                    (existing.quantity * existing.avg_cost + qty * cost) / total_qty
                    if total_qty > 0 else cost
                )
                earliest = (
                    min(existing.entry_date, entry)
                    if existing.entry_date and entry
                    else existing.entry_date or entry
                )
                positions[symbol] = Position(
                    symbol=symbol, quantity=total_qty, avg_cost=avg_cost, entry_date=earliest
                )
            else:
                positions[symbol] = Position(
                    symbol=symbol, quantity=qty, avg_cost=cost, entry_date=entry
                )

        if unresolved:
            logger.warning(
                f"{len(unresolved)} positions could not be resolved to symbols. "
                f"Instrument IDs: {[iid for iid, _ in unresolved[:10]]}"
            )
        logger.info(
            f"eToro: {len(positions_list)} position records → {len(positions)} unique symbols "
            f"({len(unresolved)} unresolved)"
        )

        return positions

    def buy(self, symbol: str, quantity: int, price: float) -> dict | None:
        inst_id = self._resolve_instrument_id(symbol)
        if inst_id is None:
            logger.error(f"Cannot buy {symbol}: instrument not found on eToro")
            return None

        payload = {
            "InstrumentID": inst_id,
            "AmountInUnits": float(quantity),
            "IsBuy": True,
            "Leverage": 1,
            "IsNoStopLoss": True,
            "IsNoTakeProfit": True,
        }

        prefix = self._trading_prefix()
        result = self._request("POST", f"{prefix}/market-open-orders/by-units", json=payload)

        if result:
            order_info = result.get("orderForOpen", result)
            logger.info(f"eToro BUY {quantity} {symbol} @ ~{price:.2f} -> order: {result}")
            return {
                "broker": "etoro",
                "symbol": symbol,
                "action": "BUY",
                "quantity": quantity,
                "price": price,
                "order_id": order_info.get("orderID") or order_info.get("orderId") or order_info.get("positionId"),
                "raw": result,
            }

        logger.error(f"eToro BUY {quantity} {symbol} failed")
        return None

    def sell(self, symbol: str, quantity: int, price: float) -> dict | None:
        # eToro creates one position record per buy order, so there may be
        # multiple position IDs for the same symbol. Close all of them.
        data = self._request("GET", f"{self._info_prefix()}/pnl")
        if not data:
            logger.error(f"Cannot sell {symbol}: failed to fetch positions")
            return None

        portfolio = self._unwrap(data)
        positions_list = portfolio.get("positions", [])

        # Collect ALL position IDs for this symbol
        position_ids = []
        for p in positions_list:
            raw_id = p.get("instrumentID") or p.get("instrumentId") or p.get("InstrumentId")
            try:
                inst_id = int(raw_id) if raw_id is not None else None
            except (ValueError, TypeError):
                inst_id = None

            matched_symbol = self._id_to_symbol.get(inst_id) if inst_id else None
            if not matched_symbol:
                matched_symbol = (p.get("symbolFull") or p.get("symbol") or p.get("ticker", "")).upper()

            if matched_symbol and matched_symbol.upper() == symbol.upper():
                pos_id = p.get("positionID") or p.get("positionId") or p.get("PositionId") or p.get("id")
                if pos_id:
                    position_ids.append(pos_id)

        if not position_ids:
            logger.error(f"Cannot sell {symbol}: no open position found")
            return None

        prefix = self._trading_prefix()
        closed = []
        for pos_id in position_ids:
            result = self._request(
                "POST",
                f"{prefix}/market-close-orders/positions/{pos_id}",
                json={},
            )
            if result is not None:
                closed.append(pos_id)
                logger.info(f"eToro SELL {symbol} position {pos_id} @ ~{price:.2f} ✓")
            else:
                logger.error(f"eToro SELL {symbol} position {pos_id} failed")

        if closed:
            logger.info(f"eToro SELL {symbol}: closed {len(closed)}/{len(position_ids)} positions")
            return {
                "broker": "etoro",
                "symbol": symbol,
                "action": "SELL",
                "quantity": quantity,
                "price": price,
                "position_ids": closed,
                "raw": {},
            }

        logger.error(f"eToro SELL {symbol}: all {len(position_ids)} close attempts failed")
        return None

    def get_current_price(self, symbol: str) -> float | None:
        inst_id = self._resolve_instrument_id(symbol)
        if inst_id is None:
            return None

        data = self._request(
            "GET",
            "/market-data/instruments/rates",
            params={"instrumentIds": inst_id},
        )
        if not data:
            return None

        rates = data if isinstance(data, list) else data.get("rates", data.get("data", []))
        for rate in rates:
            bid = rate.get("bid", rate.get("lastPrice", 0))
            ask = rate.get("ask", bid)
            return (bid + ask) / 2  # mid price

        return None
