from brokers.base import Broker
from brokers.etoro import EtoroBroker
from brokers.ibkr import IbkrBroker

BROKERS = {
    "etoro": EtoroBroker,
    "ibkr": IbkrBroker,
}
