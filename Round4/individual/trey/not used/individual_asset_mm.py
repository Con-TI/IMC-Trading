from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any, Tuple
import string
import jsonpickle
import numpy as np
import math
import pandas as pd

class Product:
    # PICNIC_1 = "PICNIC_BASKET1"
    # PICNIC_2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    # SYNTHETIC_1 = "SYNTHETIC_1"
    # SYNTHETIC_2 = "SYNTHETIC_2"
    SQUID = "SQUID_INK"
    KELP = "KELP"
    RESIN = "RAINFOREST_RESIN"
    
PARAMS = {
    Product.RESIN : {
        },
    Product.KELP : {
        "history_length" : 5,
        "gamma" : 1,
        "k" : 2,
        "reversion_coefficient": 1,
        "mean_rev_coeff" : 1000,
        "inventory_coeff" : 0.5,
        "trend_coeff" : 50000,
        "predictive_shift_coeff" : 10,
        "default_delt_shift" : 1,
        "regime_delt_shift" : 1,
        'average_delta' : 1.5,
        'store_executed_trades' : True,
        'use_AS_offset' : False
        },
    Product.SQUID : {
        "history_length" : 5,
        "gamma" : 1,
        "k" : 2,
        "reversion_coefficient": 1,
        "mean_rev_coeff" : 1000,
        "inventory_coeff" : 0.5,
        "trend_coeff" : 50000,
        "predictive_shift_coeff" : 10,
        "default_delt_shift" : 1,
        "regime_delt_shift" : 1,
        'average_delta' : 1.5,
        'store_executed_trades' : True,
        'use_AS_offset' : False
        },
    Product.CROISSANTS : {
        },
    Product.JAMS : {
        },
    Product.DJEMBES : {
        
    }
}
    
class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            # Product.PICNIC_1: 60,
            # Product.PICNIC_2: 100,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.SQUID : 50,
            Product.KELP : 50,
            Product.RESIN : 100,
        }
    
    def get_best_ask_best_bid(self, state : TradingState, product : Product, traderObject, first : bool = False):
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
        if buy_orders:
            if len(buy_orders) > 1:
                for p,q in buy_orders.items():
                    if abs(q)>max_vol:
                        max_vol = abs(q)
                        bid = p 
                bid_vol = max_vol
            else:
                bid = [*buy_orders.keys()][0]
                bid_vol = buy_orders[bid]
        else:
            bid = traderObject['mid_price'][-1]-self.params[product]['average_delta']
        
        max_vol = 0
        ask = 0
        if sell_orders:
            if len(sell_orders) > 1:
                for p,q in sell_orders.items():
                    if abs(q)>max_vol:
                        max_vol = abs(q)
                        ask = p
                ask_vol = max_vol
            else:
                ask = [*sell_orders.keys()][0]
                ask_vol = sell_orders[ask]
        else:
            ask = traderObject['mid_price'][-1]+self.params[product]['average_delta']
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

        _, bid_tup, ask_tup = self.get_best_ask_best_bid(state, product, traderObject)
        mid = (bid_tup[0] + ask_tup[0])/2
        traderObject[product]['mid_price'].append(mid)
        window_limit = self.params[product]['history_length']
        if len(traderObject[product]['mid_price']) > window_limit:
            traderObject[product]['mid_price'].pop(0)
            
        if self.params[product]['store_executed_trades']:
            exec_trades = []
            if state.market_trades.get(product,False):
                for trade in state.market_trades[product]:
                    if trade.price > mid:
                        exec_trades.append((trade.price,trade.quantity))
                    else:
                        exec_trades.append((trade.price,-trade.quantity))            
            traderObject[product]['executed_trades'].append(exec_trades)
            if len(traderObject[product]['executed_trades']) > window_limit:
                traderObject[product]['executed_trades'].pop(0)

        return
    
    def get_buy_sell_limits(self, state : TradingState, product : Product):
        position = state.position.get(product,0)
        lim = self.LIMIT[product]
        return (lim-position,-lim-position)

    def retrieve_reservation_quotes(self, state : TradingState, traderObject, product : Product):
        gamma, k = self.params[product]['gamma'], self.params[product]['k']
        volatility = pd.Series(traderObject[product]['mid_price'])
        volatility = volatility.pct_change().std()**2
        mid, _, _ = self.get_best_ask_best_bid(state, product, traderObject)
        dp_offset = self.dynamic_position_skew(state, traderObject, product)
        mid += dp_offset
        q = state.position.get(product,0)
        time_hor = 1000000-state.timestamp
        if self.params[product]['use_AS_offset']:
            default_delt = 1/gamma*np.log(1+gamma/k) + self.params[product]['default_delt_shift']
        else:
            default_delt = self.params[product]['default_delt_shift']
        bid_off = default_delt + self.params[product]['inventory_coeff']*gamma*volatility*q*(time_hor)
        ask_off = default_delt - self.params[product]['inventory_coeff']*gamma*volatility*q*(time_hor)
        print(mid, bid_off, ask_off,state.position.get(product,0))
        bid_price = mid-bid_off
        ask_price = mid+ask_off
        return bid_price, ask_price
    
    def dynamic_position_skew(self, state : TradingState, traderObject, product:Product):
        previous_pos = traderObject[product]['prev_position']
        current_pos = state.position.get(product,0)
        offset = traderObject[product]['dp_offset']
        if current_pos != 0:
            if current_pos > previous_pos:
                offset -= 0.5
            elif current_pos < previous_pos:
                offset += 0.5
            traderObject[product]['dp_offset'] = offset
        return offset

    def retrieve_ofi_skew(self, state : TradingState, traderObject, product : Product):
        exec_trades = traderObject[product]['executed_trades']
        total_q = 0+1e-6
        for trades in exec_trades:
            for price, quantity in trades:
                total_q += abs(quantity)
                
        signed_quantity = 0
        if exec_trades[-1]:
            for price, quantity in exec_trades[-1]:
                signed_quantity += quantity
        
        return signed_quantity/total_q

    def retrieve_trend_sig(self, state : TradingState, traderObject, product : Product):
        trend = np.log(traderObject[product]['mid_price'][-1])-np.log(traderObject[product]['mid_price'][0])
        return trend

    def retrieve_mean_revert_skew(self, state : TradingState, traderObject, product : Product):
        weights = np.exp(np.arange(1,len(traderObject[product]['mid_price'])+1)*0.5)
        mean = np.dot(np.array(traderObject[product]['mid_price']),weights)/weights.sum()
        diff =  (np.log(mean)-np.log(traderObject[product]['mid_price'][-1]))*self.params[product]['mean_rev_coeff']
        time_hor = 1000000-state.timestamp
        return diff*(1-np.exp(-self.params[product]['reversion_coefficient']*time_hor))
    
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        if Product.KELP not in traderObject:
            traderObject[Product.KELP] = {
                "mid_price" : [],
                "executed_trades" : [],
                "prev_position" : None,
                "dp_offset" : 0
            }    
        if Product.SQUID not in traderObject:
            traderObject[Product.SQUID] = {
                "mid_price" : [],
                "executed_trades" : [],
                "prev_position" : None,
                "dp_offset" : 0
            }    


        self.update_history(state, Product.KELP, traderObject)

        result = {}

        # orders = []        
        # if state.timestamp>500:
        #     bid, ask = self.retrieve_reservation_quotes(state, traderObject, Product.KELP)
            
        #     # MM Orders
        #     buy_lim, sell_lim = self.get_buy_sell_limits(state, Product.KELP)
        #     fixed_q = 2
        #     orders.append(Order(Product.KELP, int(math.floor(bid)), min(fixed_q,buy_lim)))
        #     orders.append(Order(Product.KELP, int(math.ceil(ask)), max(fixed_q,sell_lim)))

        #     traderObject[Product.KELP]['prev_position'] = state.position.get(Product.KELP,0)

        # result[Product.KELP] = orders

        orders = []        
        if state.timestamp>500:
            bid, ask = self.retrieve_reservation_quotes(state, traderObject, Product.SQUID)
            
            # MM Orders
            buy_lim, sell_lim = self.get_buy_sell_limits(state, Product.SQUID)
            fixed_q = 2
            orders.append(Order(Product.SQUID, int(math.floor(bid)), min(fixed_q,buy_lim)))
            orders.append(Order(Product.SQUID, int(math.ceil(ask)), max(fixed_q,sell_lim)))

            traderObject[Product.SQUID]['prev_position'] = state.position.get(Product.KELP,0)

        result[Product.SQUID] = orders

        conversions = 1
        
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
    
