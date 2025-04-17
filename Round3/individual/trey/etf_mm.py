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

PARAMS = {
    Product.PICNIC_1 : {
        "trade_impulse_adj" : 5.0,
        "tick_size" : 1.0,
        "history_length" : 6,
        "default_volatility" : 4.86,
        "default_spread_mean": 48.762433333333334,
        # "de"
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

    def update_history(self, state : TradingState, product : Product, traderObject):
        _, bid_tup, ask_tup = self.get_best_ask_best_bid(state, product)
        mid = (bid_tup[0] + ask_tup[0])/2
        exec_trades = state.market_trades.get(product, [])
        _ = []
        for trade in exec_trades:
            if trade.price < mid:
                _.append((trade.price, -trade.quantity))
            else:
                _.append((trade.price, trade.quantity))
        exec_trades = _
        if exec_trades:
            traderObject[product]['time_since_last_trade'] = 0 
        else:
            traderObject[product]['time_since_last_trade'] += 1

        traderObject[product]['executed_orders'].append(exec_trades)
        traderObject[product]['mid_price'].append(mid)
        window_limit = self.params[product]['history_length']

        if len(traderObject[product]['executed_orders']) > window_limit:
            traderObject[product]['executed_orders'].pop(0)
        if len(traderObject[product]['mid_price']) > window_limit:
            traderObject[product]['mid_price'].pop(0)
        
    def get_book_pressure_trade_impulse(self, state: TradingState, product: Product, traderObject):
        productObject =  traderObject[product] 
        exec_orders = productObject['executed_orders']
        mid_prices = productObject['mid_price']

        exists_trades = any([len(orders)>0 for orders in exec_orders])

        # Book pressure : Volume + Time WAP derived from executed trades
        # Trade impulse : volume ratio metric of executed trades, time weighted
        book_pressure = 0
        total_quantity = 0
        trade_impulse = 0        
        if exists_trades:
            for i, trades in enumerate(exec_orders):
                for trade in trades:
                    p,q = trade
                    book_pressure += abs(q)*p*np.exp(i)
                    total_quantity += abs(q)*np.exp(i)    
            book_pressure /= total_quantity  
            last_trade_set = exec_orders[-1] 
            
            for trade in last_trade_set:
                _,q = trade
                trade_impulse += q
            trade_impulse *= np.exp(i)   
            trade_impulse /= total_quantity
                 
        else:
            book_pressure = mid_prices[-1]
            total_quantity = 0
            trade_impulse = 0

        trade_impulse_adj = self.params[product]['trade_impulse_adj']
        tick_size = self.params[product]['tick_size']

        fair_value = book_pressure + trade_impulse_adj*trade_impulse*tick_size
        
        return fair_value, book_pressure, trade_impulse

    def time_weighted_fair_price_trend_vol_adjusted(self, state:TradingState, product : Product, traderObject):
        fair, book, impulse = self.get_book_pressure_trade_impulse(state, product, traderObject)
        mid, _, _ = self.get_best_ask_best_bid(state, product)
        exp_coeff = np.exp(traderObject[product]['time_since_last_trade'])
        trend_coeff = self.get_trend_coefficient(state, product, traderObject)
        volatility = self.get_volatility(state, product , traderObject)
        twap = (mid*exp_coeff + fair/(volatility/8)) / (1/(volatility/8) + exp_coeff)        
        return twap + trend_coeff
        
    def get_buy_sell_limits(self, state : TradingState, product : Product):
        position = state.position.get(product,0)
        lim = self.LIMIT[product]
        return (lim-position,-lim-position)

    def get_volatility(self, state : TradingState, product : Product, traderObject):
        mids = traderObject[product]['mid_price']
        if len(mids) < self.params[product]['history_length']:
            return self.params[product]['default_volatility']
        else:
            diffs = []
            for i in range(len(mids)-1):
                diffs.append(mids[i+1]-mids[i])
            return np.std(diffs)

    def get_trend_coefficient(self, state : TradingState, product : Product, traderObject):
        mids = traderObject[product]['mid_price']
        if len(mids) < 2:
            return 0
        else:
            diffs = []
            for i in range(len(mids)-1):
                diffs.append(mids[i+1]-mids[i])
            return np.mean(diffs)

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        
        if Product.PICNIC_1 not in traderObject:
            traderObject[Product.PICNIC_1] = {
                "executed_orders" : [],
                "mid_price" : [],
                "time_since_last_trade" : 0
            }
        
        result = {}
        
        self.update_history(state, Product.PICNIC_1, traderObject)
        mid, bid, ask = self.get_best_ask_best_bid(state, Product.PICNIC_1)
        # fair, pressure, impulse = self.get_book_pressure_trade_impulse(state,Product.PICNIC_1,traderObject)
        fair = self.time_weighted_fair_price_trend_vol_adjusted(state, Product.PICNIC_1, traderObject)
        
        buy_lim, sell_lim = self.get_buy_sell_limits(state, Product.PICNIC_1)
        
        orders = []
        volatility = self.get_volatility(state, Product.PICNIC_1, traderObject)
        buffer = min(2*volatility,0)
                
        if fair >  ask[0] + buffer:
            order = Order(Product.PICNIC_1, ask[0], buy_lim)
            orders.append(order)
        elif fair < bid[0] - buffer:
            order = Order(Product.PICNIC_1, bid[0], sell_lim)        
            orders.append(order)
        else:
            order = Order(Product.PICNIC_1, int(math.floor(fair - 3)), buy_lim)
            orders.append(order)
            order = Order(Product.PICNIC_1, int(math.ceil(fair + 3)), sell_lim)
            orders.append(order)

        print((fair, buffer))        
        result[Product.PICNIC_1] = orders

               
        conversions = 1
        
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
    
