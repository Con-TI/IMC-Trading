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
    "history_length" : 1000,
    # "coeff": -0.01412783,
    "make_edge": 1,
    "take_edge": 0,
    "clear_edge": 3,
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
        
        self.squid_fair_val()
        
        # squid_kwargs : dict = {
        #     # "buy_vol" : buy_vol,
        #     # "sell_vol" : sell_vol,
        #     "order_depth" : squid_order_depth,
        #     "product" : Product.SQUID,
        #     "edge" : SQUID_PARAMETERS['make_edge'],
        # }
        # squid_make_orders = self.market_maker_orders(**squid_kwargs)
        squid_make_orders = []
        
        squid_kwargs : dict = {
            # "buy_vol" : buy_vol,
            # "sell_vol" : sell_vol,
            "order_depth" : squid_order_depth,
            "product" : Product.SQUID,
            "edge" : SQUID_PARAMETERS['make_edge'],
        }
        squid_take_orders = self.market_taker_orders(**squid_kwargs)
        
        squid_orders = squid_take_orders + squid_make_orders
        
        result = {}
        result[Product.SQUID] = squid_orders
        traderData = jsonpickle.encode({
            "squid_fair" : self.squid_mids,
        })        
        conversions = 1 
        
        print(self.fair_prices[Product.SQUID])
        return result, conversions, traderData
    
    def market_taker_orders(self, *, order_depth : OrderDepth, product : Product, edge : int, minimum_volume_signal : int = 0):
        orders : List[Order] = []
        return orders
             
                
    def market_maker_orders(self, *, buy_vol = 0, sell_vol = 0, order_depth : OrderDepth, product : Product, edge : int) -> List[Order]:
        orders = []

        inventory = self.inventory[product]
        position_after_take = inventory + buy_vol - sell_vol
        pos_lim = POSITION_LIMIT[product]
        # How much we can buy after our buy order
        buy_q = (pos_lim - (inventory + buy_vol))//2
        # How much we can sell after our buy order
        sell_q = (pos_lim + (inventory - sell_vol))//2
        
        predicted_fair = self.fair_prices[product]
        buy_order = Order(Product.SQUID,int(predicted_fair-edge),buy_q)
        orders.append(buy_order)
        sell_order = Order(Product.SQUID,int(predicted_fair+edge),-sell_q)
        orders.append(sell_order)
        
        return orders
    
    def squid_fair_val(self):
        if len(self.squid_mids)==SQUID_PARAMETERS['history_length']:
            rolling_mean = median(self.squid_mids)
            diff = self.squid_mids[-1] - rolling_mean
            if diff>0: trend = 1 
            elif diff<0: trend=-1
            else: trend =0
            
            changes = [(self.squid_mids[i+1]-self.squid_mids[i])/self.squid_mids[i] for i in range(len(self.squid_mids)-1)]
            last_return = changes[-1]
            volatility = abs(stdev(changes))
            
            coeff1 = -0.1
            coeff2 = 0
            self.fair_prices[Product.SQUID] = self.squid_mids[-1]*(1 - coeff1*last_return*trend*math.exp(coeff2*volatility))
        else:
            self.fair_prices[Product.SQUID] = self.squid_mids[-1]
        
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
        self.squid_mids.append((high_vol_ask*high_vol_bid+high_vol_bid*high_vol_ask)/(high_vol_bid+high_vol_ask))
        
        if len(self.squid_mids) > SQUID_PARAMETERS['history_length']:
            self.squid_mids.pop(0)
   
    def __unpack_state(self, state : TradingState):
        if state.position:
            self.inventory = state.position
        self.max_orderable = {product: {"buy":max(POSITION_LIMIT[product]-self.inventory[product],0),"sell":min(-POSITION_LIMIT[product]-self.inventory[product],0)} for product in self.inventory}
                
        if state.traderData:
            traderData = jsonpickle.decode(state.traderData)
            self.squid_mids = traderData['squid_fair']
        
"""
Sidenote:
bid, ask fixed at +-1 trades 80% of the time
bid, ask fixed at +-2 trades 76% of the time
bid, ask fixed at +-3 trades 30% of the time
bid, ask fixed at +-4 trades 17% of the time
bid, ask fixed at +-5 trades 0% of the time
"""