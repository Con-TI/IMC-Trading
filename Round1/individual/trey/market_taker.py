from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string
import jsonpickle
import numpy as np

UPLIM = 8
LOWLIM = 5

class Product:
    KELP = "KELP"
    RESIN = "RAINFOREST_RESIN"
    SQUID = "SQUID_INK"

class Trader():
    def __init__(self):
        self.price_memory = {"ask":[],"bid":[]}
        self.last_desired = 0
        
    def run(self, state : TradingState):
        result = {}
        orders = []
        
        if state.traderData:
            self.price_memory = jsonpickle.decode(state.traderData)['price_memory']
            self.last_desired = jsonpickle.decode(state.traderData)['last_desired']
        
        if state.position.get(Product.SQUID,0)!=0: position=state.position[Product.SQUID] 
        else: position=0
        
        order_depth = state.order_depths[Product.SQUID]
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        
        volumes = [vol for p,vol in buy_orders.items()]
        highest_vol_bid_idx = volumes.index(max(volumes))
        highest_vol_bid = [*buy_orders.keys()][highest_vol_bid_idx]
        
        volumes = [vol for p,vol in sell_orders.items()]
        highest_vol_ask_idx = volumes.index(min(volumes))
        highest_vol_ask = [*sell_orders.keys()][highest_vol_ask_idx]
        
        self.price_memory['ask'].append(highest_vol_ask)
        self.price_memory['bid'].append(highest_vol_bid)
        
        if len(self.price_memory['ask']) > 5:
            self.price_memory['ask'].pop(0)
            self.price_memory['bid'].pop(0)
        
        if len(self.price_memory['bid']) < 5:
            bid_change = 0
            ask_change = 0
        else:
            bid_change = self.price_memory['bid'][-1]-self.price_memory['bid'][-3]
            ask_change = self.price_memory['ask'][-1]-self.price_memory['ask'][-3]
        
        if (UPLIM> bid_change > LOWLIM) and (UPLIM> ask_change > LOWLIM):
            desired_position = 50
        elif (-UPLIM <bid_change < -LOWLIM) and (-UPLIM < ask_change < -LOWLIM):
            desired_position = -50
        else:
            desired_position = self.last_desired
        
        quantity = desired_position - position
        if quantity > 0:
            print(Order(Product.SQUID,highest_vol_ask-1,quantity))
            orders.append(Order(Product.SQUID,highest_vol_ask-1,quantity))
        elif quantity < 0: 
            print(Order(Product.SQUID,highest_vol_bid+1,quantity))
            orders.append(Order(Product.SQUID,highest_vol_bid+1,quantity))       
        result[Product.SQUID] = orders
        
        tradeData = jsonpickle.encode({
            "price_memory": self.price_memory,
            'last_desired': desired_position
        })
        conversions = 1
        
        return result, conversions, tradeData
        
        
        
        