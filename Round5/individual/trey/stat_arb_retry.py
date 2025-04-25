from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import pandas as pd
import math
from statistics import NormalDist

class Product:
    SQUID = "SQUID_INK"
    KELP = "KELP"
    RESIN = "RAINFOREST_RESIN"
    PICNIC_1 = "PICNIC_BASKET1"
    PICNIC_2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC_1 = "SYNTHETIC_1"
    SYNTHETIC_2 = "SYNTHETIC_2"
    ROCK = "VOLCANIC_ROCK"
    VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOUCHER_9750 = 'VOLCANIC_ROCK_VOUCHER_9750'
    VOUCHER_10000 = 'VOLCANIC_ROCK_VOUCHER_10000'
    VOUCHER_10250 = 'VOLCANIC_ROCK_VOUCHER_10250'
    VOUCHER_10500 = 'VOLCANIC_ROCK_VOUCHER_10500'
    

PARAMS = {
    Product.PICNIC_2 : {
        'history_length' :30,
        'threshold' :0.5
    }    
}

BASKET_WEIGHTS = {
    Product.PICNIC_1: {
        Product.CROISSANTS: 6,
        Product.DJEMBES: 1,
        Product.JAMS: 3,
        },
    Product.PICNIC_2: {
        Product.CROISSANTS: 4,
        Product.JAMS: 2,
    }
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        
        self.LIMIT = {
            Product.PICNIC_1: 60,
            Product.PICNIC_2: 100,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.SQUID : 50,
            Product.KELP : 50,
            Product.RESIN : 50,
            Product.ROCK: 400,
            Product.VOUCHER_9500: 200,
            Product.VOUCHER_9750: 200, #200
            Product.VOUCHER_10000: 200, #200
            Product.VOUCHER_10250: 200, #200
            Product.VOUCHER_10500: 200, #200
        }

    
    def get_synthetic_basket_depth(self, state : TradingState, basket : Product):
        best_asks = {}
        best_bids = {}
        
        basket_bid = 0
        basket_ask = 0
        
        for product in BASKET_WEIGHTS[basket].keys():
            order_depth = state.order_depths[product]
            
            buy_orders = order_depth.buy_orders
            sell_orders = order_depth.sell_orders
            
            bid_price = None
            max_vol = 0
            for price, quantity in buy_orders.items():
                if abs(quantity) > max_vol:
                    max_vol = abs(quantity)
                    bid_price = price
            best_bids[product] = (bid_price, max_vol)
            basket_bid += bid_price * BASKET_WEIGHTS[basket][product]

            ask_price = None
            max_vol = 0
            for price, quantity in sell_orders.items():
                if abs(quantity) > max_vol:
                    max_vol = abs(quantity)
                    ask_price = price
            best_asks[product] = (ask_price, max_vol)
            basket_ask += ask_price * BASKET_WEIGHTS[basket][product]                
        
        basket_bid_vol = min([p_q[1]//BASKET_WEIGHTS[basket][product] for product, p_q  in best_bids.items()])
        basket_ask_vol = min([p_q[1]//BASKET_WEIGHTS[basket][product] for product, p_q  in best_asks.items()])
        
        synthetic_od = OrderDepth()
        synthetic_od.buy_orders = {basket_bid : basket_bid_vol}
        synthetic_od.sell_orders = {basket_ask : basket_ask_vol}
        
        return synthetic_od

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
        if synth:
            if product == Product.SYNTHETIC_1:
                synth_od = self.get_synthetic_basket_depth(state, Product.PICNIC_1)
                bid = [*synth_od.buy_orders.keys()][0]
                ask = [*synth_od.sell_orders.keys()][0]
                synth_mid = (bid+ask)/2
                traderObject[product]['mid_price'].append(synth_mid)
                window_limit = self.params[product]['history_length']
                if len(traderObject[product]['mid_price']) > window_limit:
                    traderObject[product]['mid_price'].pop(0)
                return
            if product == Product.SYNTHETIC_2:
                synth_od = self.get_synthetic_basket_depth(state, Product.PICNIC_2)
                bid = [*synth_od.buy_orders.keys()][0]
                ask = [*synth_od.sell_orders.keys()][0]
                synth_mid = (bid+ask)/2
                traderObject[product]['mid_price'].append(synth_mid)
                window_limit = self.params[product]['history_length']
                if len(traderObject[product]['mid_price']) > window_limit:
                    traderObject[product]['mid_price'].pop(0)
                return

        _, bid_tup, ask_tup = self.get_best_ask_best_bid(state, product)
        mid = (bid_tup[0] + ask_tup[0])/2
        traderObject[product]['mid_price'].append(mid)
        window_limit = self.params[product]['history_length']
        if len(traderObject[product]['mid_price']) > window_limit:
            traderObject[product]['mid_price'].pop(0)

        return
                
    def get_buy_sell_limits(self, state : TradingState, product : Product):
        position = state.position.get(product,0)
        lim = self.LIMIT[product]
        return (lim-position,-lim-position)

    def determine_regime(self, state : TradingState, traderObject, basket : Product):
        if len(traderObject[basket]['spread_history']) > 5:
            regime_indicator = np.max(traderObject[basket]['spread_history'][-5:]) - np.min(traderObject[basket]['spread_history'][-5:])
        else:
            regime_indicator = 0
        if regime_indicator >= 6:            
            traderObject[basket]['market_state'] = "trending"
        else:
            traderObject[basket]['market_state'] = "mean_reverting"
        return
        
    def basket_2_orders(self, state : TradingState, traderObject):
        synthetic_od = self.get_synthetic_basket_depth(state, Product.PICNIC_2)
        mid = ([*synthetic_od.buy_orders.keys()][0] + [*synthetic_od.sell_orders.keys()][0])/2
        picnic_mid = traderObject[Product.PICNIC_2]['mid_price'][-1]
        spread = picnic_mid - mid
        window_limit = self.params[Product.PICNIC_2]['history_length']

        if len(traderObject[Product.PICNIC_2]['spread_history']) < 50:
            return [], [], []
        
        self.determine_regime(state, traderObject, Product.PICNIC_2)        

        picnic_orders = []
        croissant_orders = []
        jams_orders =[]

        if traderObject[Product.PICNIC_2]['market_state'] == "mean_reverting":
            average_spread = np.mean(traderObject[Product.PICNIC_2]['spread_history'])
            spread_std = np.std(traderObject[Product.PICNIC_2]['spread_history'])
            z_score = (spread - average_spread)/spread_std
            threshold = self.params[Product.PICNIC_2]['threshold']
            if z_score > threshold:
                _, bid_tup, ask_tup = self.get_best_ask_best_bid(state, Product.PICNIC_2)
                picnic_orders.append(Order(Product.PICNIC_2, bid_tup[0], -1))
                _, bid_tup, ask_tup = self.get_best_ask_best_bid(state, Product.CROISSANTS)
                croissant_orders.append(Order(Product.CROISSANTS, ask_tup[0],4))
                _, bid_tup, ask_tup = self.get_best_ask_best_bid(state, Product.JAMS)
                jams_orders.append(Order(Product.JAMS, ask_tup[0], 2))
            elif z_score < -threshold:
                _, bid_tup, ask_tup = self.get_best_ask_best_bid(state, Product.PICNIC_2)
                picnic_orders.append(Order(Product.PICNIC_2, ask_tup[0], 1))
                _, bid_tup, ask_tup = self.get_best_ask_best_bid(state, Product.CROISSANTS)
                croissant_orders.append(Order(Product.CROISSANTS, bid_tup[0],-4))
                _, bid_tup, ask_tup = self.get_best_ask_best_bid(state, Product.JAMS)
                jams_orders.append(Order(Product.JAMS, bid_tup[0], -2))
        elif traderObject[Product.PICNIC_2]['market_state'] == "trending":
            trend = traderObject[Product.PICNIC_2]['spread_history'][-1] - traderObject[Product.PICNIC_2]['spread_history'][-10]
            if trend > 0:
                desired_position = 50
                position = state.position.get(Product.PICNIC_2,0)
                q = desired_position - position
                _, bid_tup, ask_tup = self.get_best_ask_best_bid(state, Product.PICNIC_2)
                if q>0:
                    picnic_orders.append(Order(Product.PICNIC_2, ask_tup[0], q))
                    
                desired_position = -200
                position = state.position.get(Product.CROISSANTS,0)
                q = desired_position - position
                _, bid_tup, ask_tup = self.get_best_ask_best_bid(state, Product.CROISSANTS)
                if q<0:
                    croissant_orders.append(Order(Product.CROISSANTS, bid_tup[0], q))
                    
                desired_position = -100
                position = state.position.get(Product.JAMS,0)
                q = desired_position - position
                _, bid_tup, ask_tup = self.get_best_ask_best_bid(state, Product.JAMS)
                if q<0:
                    jams_orders.append(Order(Product.JAMS, bid_tup[0], q))
            elif trend < 0:
                desired_position = -50
                position = state.position.get(Product.PICNIC_2,0)
                q = desired_position - position
                _, bid_tup, ask_tup = self.get_best_ask_best_bid(state, Product.PICNIC_2)
                if q<0:
                    picnic_orders.append(Order(Product.PICNIC_2, bid_tup[0], q))
                    
                desired_position = 200
                position = state.position.get(Product.CROISSANTS,0)
                q = desired_position - position
                _, bid_tup, ask_tup = self.get_best_ask_best_bid(state, Product.CROISSANTS)
                if q>0:
                    croissant_orders.append(Order(Product.CROISSANTS, ask_tup[0], q))
                    
                desired_position = 100
                position = state.position.get(Product.JAMS,0)
                q = desired_position - position
                _, bid_tup, ask_tup = self.get_best_ask_best_bid(state, Product.JAMS)
                if q>0:
                    jams_orders.append(Order(Product.JAMS, ask_tup[0], q))

        return picnic_orders, croissant_orders, jams_orders
        
    def update_spread_history(self, state : TradingState, basket : Product, traderObject):
        synthetic_od = self.get_synthetic_basket_depth(state, Product.PICNIC_2)
        mid = ([*synthetic_od.buy_orders.keys()][0] + [*synthetic_od.sell_orders.keys()][0])/2
        picnic_mid = traderObject[Product.PICNIC_2]['mid_price'][-1]
        spread = picnic_mid - mid

        window_limit = self.params[Product.PICNIC_2]['history_length']
        traderObject[Product.PICNIC_2]['spread_history'].append(spread)
        if len(traderObject[Product.PICNIC_2]['spread_history']) > window_limit:
            traderObject[Product.PICNIC_2]['spread_history'].pop(0)

        return
        
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
            
        if Product.PICNIC_1 not in traderObject:
            traderObject[Product.PICNIC_1] = {
                "mid_price" : [],
                "last_spread_mean" : None
            }
        if Product.PICNIC_2 not in traderObject:
            traderObject[Product.PICNIC_2] = {
                "mid_price" : [],
                "spread_history" : [],
                "market_state" : "mean_reverting",
            }

        
        self.update_history(state, Product.PICNIC_2, traderObject)
        self.update_spread_history(state, Product.PICNIC_2, traderObject)
        
        result = {}
        
        print(traderObject)
        traderData = jsonpickle.encode(traderObject)
        
        picnic_2_orders, croissant_orders, jam_orders =  self.basket_2_orders(state, traderObject)
        result[Product.PICNIC_2] = picnic_2_orders
        result[Product.CROISSANTS] = croissant_orders
        result[Product.JAMS] = jam_orders
        # Conversions (purpose not clear from context, possibly a trading feature)
        conversions = 1

        return result, conversions, traderData