from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import pandas as pd
import math

class Product:
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    
PARAMS = {
    Product.CROISSANTS : {
        "history_length" : 10
    },
    Product.JAMS : {
        
    },
    Product.DJEMBES : {
        "history_length" : 33,
        "fast" : 15,
        "slow" : 30,
        "macd" : 3,
    }
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        
        self.LIMIT = {
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
        }

    def get_best_ask_best_bid(self, state : TradingState, product : Product, first : bool = False):
        order_depth = state.order_depths[product]
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        
        if first:
            buy_p = [*buy_orders.keys()]
            best_bid = None
            if len(buy_p) > 1:
                best_bid = max(*buy_orders.keys())
            else:
                best_bid = buy_p[0]
            
            sell_p = [*sell_orders.keys()]
            best_ask = None
            if len(sell_p) > 1:
                best_ask = min(*sell_orders.keys())
            else:
                best_ask = sell_p[0]

            mid = (best_ask + best_bid)//2
            return mid, (best_bid, buy_orders[best_bid]), (best_ask, sell_orders[best_ask])    
        
        max_vol = 0
        bid = 0
        if len(buy_orders) > 1:
            for p,q in buy_orders.items():
                if abs(q)>max_vol:
                    max_vol = abs(q)
                    bid = p 
            bid_vol = max_vol
        else:
            bid = [*buy_orders.keys()][0]
            bid_vol = buy_orders[bid]
        
        max_vol = 0
        ask = 0
        if len(sell_orders) > 1:
            for p,q in sell_orders.items():
                if abs(q)>max_vol:
                    max_vol = abs(q)
                    ask = p
            ask_vol = max_vol
        else:
            ask = [*sell_orders.keys()][0]
            ask_vol = sell_orders[ask]
            
        mid = (ask + bid)//2
        
        return mid, (bid, bid_vol), (ask, ask_vol)

    def update_history(self, state : TradingState, product : Product, traderObject, synth = False):
        # if synth:
        #     if product == Product.SYNTHETIC_1:
        #         synth_od = self.get_synthetic_basket_depth(state, Product.PICNIC_1)
        #         bid = [*synth_od.buy_orders.keys()][0]
        #         ask = [*synth_od.sell_orders.keys()][0]
        #         synth_mid = (bid+ask)/2
        #         traderObject[product]['mid_price'].append(synth_mid)
        #         window_limit = self.params[product]['history_length']
        #         if len(traderObject[product]['mid_price']) > window_limit:
        #             traderObject[product]['mid_price'].pop(0)
        #         return
        #     if product == Product.SYNTHETIC_2:
        #         synth_od = self.get_synthetic_basket_depth(state, Product.PICNIC_2)
        #         bid = [*synth_od.buy_orders.keys()][0]
        #         ask = [*synth_od.sell_orders.keys()][0]
        #         synth_mid = (bid+ask)/2
        #         traderObject[product]['mid_price'].append(synth_mid)
        #         window_limit = self.params[product]['history_length']
        #         if len(traderObject[product]['mid_price']) > window_limit:
        #             traderObject[product]['mid_price'].pop(0)
        #         return

        _, bid_tup, ask_tup = self.get_best_ask_best_bid(state, product)
        mid = (bid_tup[0] + ask_tup[0])/2
        traderObject[product]['mid_price'].append(mid)
        window_limit = self.params[product]['history_length']
        if len(traderObject[product]['mid_price']) > window_limit:
            traderObject[product]['mid_price'].pop(0)
        return
           
    def djembes_orders(self, state, traderObject):
        _, bid_tup, ask_tup = self.get_best_ask_best_bid(state, Product.DJEMBES)
        bid, ask = bid_tup[0], ask_tup[0]
        
        orders = []
        
        if len(traderObject[Product.DJEMBES]['mid_price']) == self.params[Product.DJEMBES]['history_length']:
            macd_sig = 0
            short_mean = np.mean(traderObject[Product.DJEMBES]['mid_price'][-self.params[Product.DJEMBES]['fast']:])
            long_mean = np.mean(traderObject[Product.DJEMBES]['mid_price'][-self.params[Product.DJEMBES]['slow']:])
            difference1 = short_mean-long_mean
            short_mean = np.mean(traderObject[Product.DJEMBES]['mid_price'][-self.params[Product.DJEMBES]['fast']-1:-1])
            long_mean = np.mean(traderObject[Product.DJEMBES]['mid_price'][-self.params[Product.DJEMBES]['slow']-1:-1])
            difference2 = short_mean-long_mean
            short_mean = np.mean(traderObject[Product.DJEMBES]['mid_price'][-self.params[Product.DJEMBES]['fast']-2:-2])
            long_mean = np.mean(traderObject[Product.DJEMBES]['mid_price'][-self.params[Product.DJEMBES]['slow']-2:-2])
            difference3 = short_mean-long_mean
            macd_sig = (difference1 + difference2 + difference3)/3
            sig = difference1-macd_sig
            sign_cur_sig = np.sign(sig)
            if (sign_cur_sig == -1):
                traderObject[Product.DJEMBES]['desired_position'] = -self.LIMIT[Product.DJEMBES]
            elif (sign_cur_sig == 1):
                traderObject[Product.DJEMBES]['desired_position'] = self.LIMIT[Product.DJEMBES]
        position = state.position.get(Product.DJEMBES,0)
        desired_pos = traderObject[Product.DJEMBES]['desired_position']
        q = desired_pos-position
        if q < 0:
            orders.append(Order(Product.DJEMBES, bid, -q))
        elif q > 0:
            orders.append(Order(Product.DJEMBES, ask, q))
        
        return orders
           
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        if Product.DJEMBES not in traderObject:
            traderObject[Product.DJEMBES] = {
                 "mid_price" : [],
                 "desired_position" : 0,
            } 
        
        self.update_history(state, Product.DJEMBES, traderObject)
        
        
        result = {}
        
        if Product.DJEMBES in state.order_depths:
            ink_orders = self.djembes_orders(
                state, traderObject    
            )
            result[Product.DJEMBES] = ink_orders
            
        traderData = jsonpickle.encode(traderObject)

        # Conversions (purpose not clear from context, possibly a trading feature)
        conversions = 1

        return result, conversions, traderData
        
        
            
         

