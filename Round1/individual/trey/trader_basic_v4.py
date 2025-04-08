"""
This file focuses on maximizing resin profit. It only trades resin.

Strategy:
- RESIN: buy at 9998, sell at 10002
- SQUID: buy when price 

"""

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

POSITION_LIMIT = {
    Product.KELP: 50,
    Product.RESIN: 50,
    Product.SQUID: 50
} 

SQUID_PARAMETERS = {
    "history_length" : 5,
}
class Trader:
    def __init__(self):
        self.inventory : Dict[Product, int] = {Product.KELP : 0, Product.RESIN : 0, Product.SQUID : 0}
        self.max_orderable : Dict[Product, Dict[str, int]] = {}
        self.fair_prices : Dict[Product, float] = {Product.RESIN : 10000}
        
        self.squid_micros : List[float] = []
    
    def run(self, state: TradingState):
        squid_order_depth : OrderDepth = state.order_depths[Product.SQUID]
        self.__unpack_state(state)
        self.__update_lists(state.order_depths)
        
        squid_orders = self.squid_ords()
        
        result = {}
        result[Product.SQUID] = squid_orders
        traderData = jsonpickle.encode({
            "squid_micros" : self.squid_micros,
        })        
        conversions = 1 
        
        print(self.fair_prices[Product.SQUID])
        return result, conversions, traderData
            
    # UTILS________________________________________________________-
    def __update_lists(self, order_depths : Dict[Product,OrderDepth]):
        squid_order_depth : OrderDepth = order_depths[Product.SQUID]
        buy_side = squid_order_depth.buy_orders
        sell_side = squid_order_depth.sell_orders
        
        buy_vols = [abs(buy_side[price]) for price in buy_side]
        high_vol_bid = [*buy_side.keys()][buy_vols.index(max(buy_vols))]
        sell_vols = [abs(sell_side[price]) for price in sell_side]
        high_vol_ask = [*sell_side.keys()][sell_vols.index(max(sell_vols))]
        
        # self.squid_mids.append((high_vol_ask+high_vol_bid)/2)
        self.squid_micros.append((high_vol_ask*high_vol_bid+high_vol_bid*high_vol_ask)/(high_vol_bid+high_vol_ask))
        self.squid_bids.append(high_vol_bid)
        self.squid_asks.append(high_vol_ask)
        
        if len(self.squid_micros) > SQUID_PARAMETERS['history_length']:
            self.squid_micros.pop(0)
            self.squid_bids.pop(0)
            self.squid_asks.pop(0)
   
    def __unpack_state(self, state : TradingState):
        if state.position:
            self.inventory = state.position
        self.max_orderable = {product: {"buy":max(POSITION_LIMIT[product]-self.inventory[product],0),"sell":min(-POSITION_LIMIT[product]-self.inventory[product],0)} for product in self.inventory}
                
        if state.traderData:
            traderData = jsonpickle.decode(state.traderData)
            self.squid_micros = traderData['squid_micros']
            self.squid_bids = traderData['squid_bids']
            self.squid_asks = traderData['squid_asks']
        
"""
Sidenote:
bid, ask fixed at +-1 trades 80% of the time
bid, ask fixed at +-2 trades 76% of the time
bid, ask fixed at +-3 trades 30% of the time
bid, ask fixed at +-4 trades 17% of the time
bid, ask fixed at +-5 trades 0% of the time
"""