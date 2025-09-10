from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any, Tuple
import string
import jsonpickle
import numpy as np
import math
import pandas as pd

class Product:
    PICNIC_1 = "PICNIC_BASKET1"
    PICNIC_2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC_1 = "SYNTHETIC_1"
    SYNTHETIC_2 = "SYNTHETIC_2"

PARAMS = {
    Product.PICNIC_1 : {
        "history_length" : 10,
        "gamma" : 1,
        "k" : 0.5,
        "reversion_coefficient": 1,
        "inventory_coeff" : 12.5,
        "theta_coeff" : 1e4/3,
        "trend_coeff" : 50000,
        "linear_coeff" : 6e7,
        "predictive_shift_coeff" : 2,
        "default_delt_shift" : 0,
        "regime_delt_shift" : 1,
        "spread_mean" : 0
    },
    Product.SYNTHETIC_1 : {
        "history_length" : 10,
    },
    Product.PICNIC_2 : {
        "history_length" : 10,
        "gamma" : 1,
        "k" : 0.5,
        "reversion_coefficient": 2,
        "inventory_coeff" : 100,
        "theta_coeff" : 4e2,
        "trend_coeff" : 50000,
        "linear_coeff" : 3.3e7,
        "predictive_shift_coeff" : 0,
        "default_delt_shift" : 0,
        "regime_delt_shift" : 0,
        "spread_mean" : 0
    },
    Product.SYNTHETIC_2 : {
        "history_length" : 10,
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

    def retrieve_reservation_quotes(self, state : TradingState, traderObject, product : Product):
        gamma, k = self.params[product]['gamma'], self.params[product]['k']
        volatility = pd.Series(traderObject[product]['mid_price'])
        volatility = volatility.pct_change().std()**2
        mid, _, _ = self.get_best_ask_best_bid(state, product)
        linear = max(1-volatility*self.params[product]['linear_coeff'],0)
        theta = self.retrieve_signal_skew(state, traderObject, product)
        trend = self.retrieve_trend_sig(state, traderObject, Product.PICNIC_1)
        mid += (theta*linear + trend*(1-linear))*self.params[product]['predictive_shift_coeff']
        q = state.position.get(product,0)
        time_hor = 1000000-state.timestamp
        default_delt = 1/gamma*np.log(1+gamma/k) + self.params[product]['default_delt_shift']
        if abs(trend) > 1:
            default_delt += self.params[product]['regime_delt_shift']
        bid_off = default_delt + self.params[product]['inventory_coeff']*gamma*volatility*q*(time_hor)
        ask_off = default_delt - self.params[product]['inventory_coeff']*gamma*volatility*q*(time_hor)
        bid_price = mid-bid_off
        ask_price = mid+ask_off
        return bid_price, ask_price

    def retrieve_signal_skew(self, state : TradingState, traderObject, product:Product):
        time_hor = 1000000-state.timestamp
        mid, _, _ = self.get_best_ask_best_bid(state, product)
        
        # Reversion based on spread
        synthetic_od = self.get_synthetic_basket_depth(state, product)
        synthetic_bid = [*synthetic_od.buy_orders.keys()][0]
        synthetic_ask = [*synthetic_od.sell_orders.keys()][0]
        synthetic_mid = (synthetic_bid+synthetic_ask)/2
        
        spread = np.log(mid) - np.log(synthetic_mid)
        mean = self.params[product]['spread_mean']
        theta = self.params[product]['theta_coeff']*(mean-spread)*(1-np.exp(-time_hor*self.params[product]['reversion_coefficient']))
        return theta

    def retrieve_trend_sig(self, state : TradingState, traderObject, product : Product):
        trend = np.log(traderObject[product]['mid_price'][-1])-np.log(traderObject[product]['mid_price'][0])
        volatility = pd.Series(traderObject[product]['mid_price'])
        volatility = volatility.pct_change().std()**2*self.params[product]['trend_coeff']
        trend /= volatility
        return trend

    def arbitrage_orders(self, state : TradingState, traderObject, product:Product):
        mid, b, a = self.get_best_ask_best_bid(state, product)
        theta = self.retrieve_signal_skew(state, traderObject, product)
        mid += theta

        od = state.order_depths[product]
        buys = od.buy_orders
        sells = od.sell_orders
        to_sell = [price for price, quantity in buys.items() if price > mid]
        to_buy = [price for price, quantity in sells.items() if price < mid]
        orders = []
        if to_buy:
            for price in to_buy:
                q = min(abs(sells[price]),5)
                orders.append(Order(product, price, q))
                orders.append(Order(product, a[0]-1, -q))
        if to_sell:
            for price in to_sell:            
                q = min(abs(buys[price]),5)
                orders.append(Order(product, price, -q))
                orders.append(Order(product, b[0]+1, q))
        return orders

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        if Product.PICNIC_1 not in traderObject:
            traderObject[Product.PICNIC_1] = {
                "mid_price" : [],
            }
        if Product.SYNTHETIC_1 not in traderObject:
            traderObject[Product.SYNTHETIC_1] = {
                "mid_price" : [],
            }    
        if Product.PICNIC_2 not in traderObject:
            traderObject[Product.PICNIC_2] = {
                "mid_price" : [],
            }
        if Product.SYNTHETIC_2 not in traderObject:
            traderObject[Product.SYNTHETIC_2] = {
                "mid_price" : [],
            }

        self.update_history(state, Product.PICNIC_1, traderObject)
        self.update_history(state, Product.SYNTHETIC_1, traderObject, synth = True)    
        self.update_history(state, Product.PICNIC_2, traderObject)
        self.update_history(state, Product.SYNTHETIC_2, traderObject, synth = True)   

        result = {}

        orders = []        
        if state.timestamp>500:
            bid, ask = self.retrieve_reservation_quotes(state, traderObject, Product.PICNIC_1)

            # MM Orders
            buy_lim, sell_lim = self.get_buy_sell_limits(state, Product.PICNIC_1)
            fixed_q = 15
            orders.append(Order(Product.PICNIC_1,int(math.floor(bid)),min(fixed_q,buy_lim)))
            orders.append(Order(Product.PICNIC_1,int(math.ceil(ask)),max(-fixed_q,sell_lim)))

            # Arbitrage Orders
            # orders += self.arbitrage_orders(state, traderObject, Product.PICNIC_1)

        result[Product.PICNIC_1] = orders
                        
        # orders = []        
        # if state.timestamp>500:
        #     bid, ask = self.retrieve_reservation_quotes(state, traderObject, Product.PICNIC_2)

        #     # MM Orders
        #     buy_lim, sell_lim = self.get_buy_sell_limits(state, Product.PICNIC_2)
        #     fixed_q = 1
        #     orders.append(Order(Product.PICNIC_2,int(math.floor(bid)),min(fixed_q,buy_lim)))
        #     orders.append(Order(Product.PICNIC_2,int(math.ceil(ask)),max(-fixed_q,sell_lim)))

        #     # Arbitrage Orders
        #     # orders += self.arbitrage_orders(state, traderObject, Product.PICNIC_1)

        # result[Product.PICNIC_2] = orders
        
        conversions = 1
        
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
    
