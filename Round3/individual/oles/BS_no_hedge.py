from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string
import jsonpickle
import sys
import numpy as np
import pandas as pd
from statistics import NormalDist
from math import log, sqrt, exp



class Product:
    V_9500 = 'VOLCANIC_ROCK_VOUCHER_9500'
    V_9750 = 'VOLCANIC_ROCK_VOUCHER_9750'
    V_10000 = 'VOLCANIC_ROCK_VOUCHER_10000'
    V_10250 = 'VOLCANIC_ROCK_VOUCHER_10250'
    V_10500 = 'VOLCANIC_ROCK_VOUCHER_10500'


class Trader:
    def __init__(self):
        self.LIMIT = {
            Product.V_9500: 200,
            Product.V_9750: 200,
            Product.V_10000: 200,
            Product.V_10250: 200,
            Product.V_10500: 200
        }

    def run(self, state: TradingState):
        traderObject = {}
        result = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        
        timestamp = state.timestamp   

        TTE = 4 - timestamp/100000  

        strikes = [Product.V_9500, Product.V_9750, Product.V_10000, Product.V_10250, Product.V_10500]
        strike_price = {Product.V_9500: 9500, 
                        Product.V_9750: 9750, 
                        Product.V_10000: 100000, 
                        Product.V_10250: 10250, 
                        Product.V_10500: 10500}


        order_depth: OrderDepth = state.order_depths['VOLCANIC_ROCK']

        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        if best_bid == None or best_ask == None:
            return [], 1, jsonpickle.encode(traderObject)
        rock_mid_price = (best_ask + best_bid) / 2


        for strike in strikes:
            orders = []

            order_depth: OrderDepth = state.order_depths[strike]
            position = state.position.get(strike, 0)
            position_limit = self.LIMIT[strike]
            max_buys = position_limit - position
            max_sells = - position_limit - position

            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            if best_ask == None or best_bid == None:
                continue
            voucher_mid_price = (best_ask + best_bid) / 2

            action, BS_price = self.call_buy_sell(voucher_mid_price, rock_mid_price, strike_price[strike], TTE)

            if action == 'sell':
                if BS_price < best_bid:
                    orders.append(Order(strike, best_bid, max_sells))
                else:
                    orders.append(Order(strike, best_ask, max_sells))
            elif action == 'buy':
                if BS_price > best_ask:
                    orders.append(Order(strike, best_ask, max_buys))
                else:
                    orders.append(Order(strike, best_bid, max_buys))
            
            result[strike] = orders

            
        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
    
    
    def black_scholes_call(self, spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price


    def implied_volatility(
            self, call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
        ):
            low_vol = 0.01
            high_vol = 1.0
            volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
            for _ in range(max_iterations):
                estimated_price = self.black_scholes_call(
                    spot, strike, time_to_expiry, volatility
                )
                diff = estimated_price - call_price
                if abs(diff) < tolerance:
                    break
                elif diff > 0:
                    high_vol = volatility
                else:
                    low_vol = volatility
                volatility = (low_vol + high_vol) / 2.0
            return volatility
    
    def call_buy_sell(self, call_price, spot, strike, time_to_expiry):
      tolerance = 1
      IV = self.implied_volatility(call_price, spot, strike, time_to_expiry)
      BS_price = self.black_scholes_call(spot, strike, time_to_expiry, IV)

      if call_price - BS_price>tolerance:
            return 'sell', BS_price
      elif BS_price - call_price>tolerance:
            return 'buy', BS_price
      
      return 'hold', BS_price