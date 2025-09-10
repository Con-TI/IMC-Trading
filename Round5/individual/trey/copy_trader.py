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
    ROCK = "VOLCANIC_ROCK"
    VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOUCHER_9750 = 'VOLCANIC_ROCK_VOUCHER_9750'
    VOUCHER_10000 = 'VOLCANIC_ROCK_VOUCHER_10000'
    VOUCHER_10250 = 'VOLCANIC_ROCK_VOUCHER_10250'
    VOUCHER_10500 = 'VOLCANIC_ROCK_VOUCHER_10500'
    MACARONS = "MAGNIFICENT_MACARONS"
    

PARAMS = {
    Product.CROISSANTS : {
        'name':'Camilla',
        'opposite':False
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
            Product.MACARONS: 75
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


    def copy_trade(self, state : TradingState, traderObject, product):
        orders = []
        market_trades = state.market_trades.get(product,[])
        copier = self.params[product]['name']
        opposite = self.params[product]['opposite']
        
        desired_position = traderObject[product]['desired_position']
        
        if market_trades:
            for trade in market_trades:
                if (trade.buyer == copier):
                    if opposite:
                        desired_position += -trade.quantity
                    else:
                        desired_position += trade.quantity
                    break
                elif (trade.seller == copier):
                    if opposite:
                        desired_position += trade.quantity
                    else:
                        desired_position += -trade.quantity
                    break
        
        traderObject[product]['desired_position'] = desired_position
        
        mid, bid_tup, ask_tup = self.get_best_ask_best_bid(state, product, first=True)
        
        position = state.position.get(product,0)
        diff = desired_position-position
        if diff > 0:
            orders.append(Order(product,ask_tup[0],diff))
        elif diff < 0:            
            orders.append(Order(product,bid_tup[0],diff))
        
        return orders

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        if Product.CROISSANTS not in state.traderData:
            traderObject[Product.CROISSANTS] = {
                'desired_position' : 0,
            }
        
        results = {}
            
        results[Product.CROISSANTS] = self.copy_trade(state, traderObject, Product.CROISSANTS)
        
        
        traderData = jsonpickle.encode(traderObject)

        # Conversions (purpose not clear from context, possibly a trading feature)
        conversions = 0

        return results, conversions, traderData
        