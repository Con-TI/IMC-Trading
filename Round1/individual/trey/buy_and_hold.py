from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string
import jsonpickle
from statistics import median, mean, stdev
import math

class Product:
    KELP = "KELP"
    RESIN = "RAINFOREST_RESIN"
    SQUID = "SQUID_INK"

class Trader:
    def run(self, state: TradingState):
        result = {}
        conversions = 1 
        
        for product, depth in state.order_depths.items():
            sell_ords = depth.sell_orders
            volumes = [vol for price,vol in sell_ords.items()]
            biggest_volume = min(volumes)
            ask_p = [*sell_ords.keys()][volumes.index(biggest_volume)]
            result[product] = [Order(product, ask_p, -biggest_volume)]
        
        traderData = "SAMPLE"
        return result, conversions, traderData