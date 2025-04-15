from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any, Tuple
import string
import jsonpickle
import numpy as np
import math


class Product:
    PICNIC_1 = "PICNIC_BASKET1"
    PICNIC_2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"
    SPREAD_1 = "SPREAD_1"
    SPREAD_2 = "SPREAD_2"


PARAMS = {
    Product.SPREAD_1: {
        "default_spread_mean": 48.762433333333334,
        "default_spread_std": 85.11945080948948944,
        "difference_std" : 0.5,
        "spread_std_window": 45,
    },
    Product.SPREAD_2: {
        "default_spread_mean": 30.23596666666666,
        "default_spread_std": 59.849200222652364,
        "spread_std_window": 20,
        "zscore_threshold": 1.5,
        "target_position": 95
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

    def fair_price_calc(self, state : TradingState, basket : Product, spread_data : Dict[str, Any], adjust_inventory : bool = False):
        """
        ver 1
        Fair price = midprice of basket - z_score*(std of spread differences)
        ver 2
        Fair price = midprice of basket - z_score*(std of spread differences) - position*volatility*risk_aversion_param
        """
        
        synthetic_od : OrderDepth = self.get_synthetic_basket_depth(state, basket)
        synthetic_basket_bid = [*synthetic_od.buy_orders.keys()][0]
        synthetic_basket_ask = [*synthetic_od.sell_orders.keys()][0]
        synthetic_mid = (synthetic_basket_ask+synthetic_basket_bid)/2
        
        actual_depth = state.order_depths[basket]
        
        best_bid = None
        if actual_depth.buy_orders:
            max_vol = 0
            for price, quantity in actual_depth.buy_orders.items():
                if abs(quantity)>max_vol:
                    max_vol = abs(quantity)
                    best_bid = price
        else:
            best_bid = spread_data['prev_basket_bid']
        spread_data['prev_basket_bid'] = best_bid
        
        best_ask = None
        if actual_depth.sell_orders:
            max_vol = 0
            for price, quantity in actual_depth.sell_orders.items():
                if abs(quantity)>max_vol:
                    max_vol = abs(quantity)
                    best_ask = price
        else:
            best_ask = spread_data['prev_basket_ask']
        spread_data['prev_basket_ask'] = best_ask

        actual_mid = (best_bid+best_ask)/2
        
        spread = synthetic_mid-actual_mid     
        spread_product = None
        if basket == Product.PICNIC_1:
            spread_product = Product.SPREAD_1
        elif basket == Product.PICNIC_2:
            spread_product = Product.SPREAD_2    

        window_size = self.params[spread_product]['spread_std_window']
        spread_data['spread_history'].append(spread)
        spread_hist = spread_data['spread_history']
        if len(spread_hist) > window_size:
            spread_hist.pop(0)

        mu = self.params[spread_product]['default_spread_mean']
        std = None
        if len(spread_hist) < window_size:
            std = self.params[spread_product]['default_spread_std']
        else:
            std = np.std(spread_hist)
        
        z_score = (spread - mu)/std
        shift = max(abs(z_score)*self.params[spread_product]['difference_std'],3)

        current_basket_position = state.position.get(basket,0)

        if adjust_inventory:
            basket_fair_price = actual_mid + (abs(z_score)/z_score)*shift + current_basket_position//3
        else:
            basket_fair_price = actual_mid - (abs(z_score)/z_score)*shift
            
        return basket_fair_price

    def basket_orders(self, state : TradingState, basket : Product, spread_data : Dict[str, Any], adjust_inventory : bool = False):
        orders = []
        
        fair_value = self.fair_price_calc(state, basket, spread_data, adjust_inventory=adjust_inventory)
        print(fair_value)
        
        order_depth = state.order_depths[basket]
        if len(order_depth.buy_orders) > 1:
            best_bid = max(*order_depth.buy_orders.keys())
        else:
            best_bid = order_depth.buy_orders.keys()
        if len(order_depth.sell_orders) > 1:
            best_ask = min(*order_depth.sell_orders.keys())
        else:
            best_ask = order_depth.sell_orders.keys()

        position = state.position.get(basket, 0)
        position_limit = self.LIMIT[basket]
        take_width = 1
        
        max_buys = position_limit - position
        max_sells = - position_limit - position
        
        fair_bid = np.floor(fair_value)-take_width
        fair_ask = np.ceil(fair_value)+take_width
        
        if fair_bid < best_bid:
            fair_bid = best_bid
        if fair_ask > fair_ask:
            fair_ask = fair_ask

        if best_bid < fair_value < best_ask:        
            orders.append(Order(basket, round(fair_bid), max_buys))
            orders.append(Order(basket, round(fair_ask), max_sells))
        elif fair_value < best_bid:
            orders.append(Order(basket, best_bid, max_sells))
        elif fair_value > best_ask:
            orders.append(Order(basket, best_ask, max_buys))
        
        return orders

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        
        if Product.SPREAD_1 not in traderObject:
            traderObject[Product.SPREAD_1] = {
                "spread_history" : [],
                "prev_basket_bid" : None,
                "prev_basket_ask" : None,
            }
        if Product.SPREAD_2 not in traderObject:
            traderObject[Product.SPREAD_2] = {
                "spread_history": [],
                "prev_basket_bid" : None,
                "prev_basket_ask" : None,
            }            

        result = {}
        conversions = 1
        result[Product.PICNIC_1] = self.basket_orders(state, Product.PICNIC_1, traderObject[Product.SPREAD_1])

        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
    
