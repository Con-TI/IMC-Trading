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
from statistics import median, mean


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
    "history_length" : 5,
    # "coeff": -0.01412783,
    "std" : 0.001182054165445773,
    "take_edge": 1,
    'default_edge':1,
    "clear_edge": 1,
    "volatility coeff":1,
    "minimum_volume_signal":5,
    "n_params":2,
    "Q" : 1e-5,
    "R" : 0.01
}

class Trader:
    def __init__(self):
        self.inventory : Dict[Product, int] = {Product.KELP : 0, Product.RESIN : 0, Product.SQUID : 0}
        self.max_orderable : Dict[Product, Dict[str, int]] = {}
        self.fair_prices : Dict[Product, float] = {Product.RESIN : 10000}
    
        self.kalman_params = {"n_params":SQUID_PARAMETERS['n_params'],
                              "beta":[0.0 for i in range(SQUID_PARAMETERS['n_params'])],
                              "P":[[1.0 if i == j else 0.0 for i in range(SQUID_PARAMETERS['n_params'])] for j in range(SQUID_PARAMETERS['n_params'])],
                              "Q":[[SQUID_PARAMETERS['Q'] if i == j else 0.0 for i in range(SQUID_PARAMETERS['n_params'])] for j in range(SQUID_PARAMETERS['n_params'])],
                              "R": SQUID_PARAMETERS['R']}
        self.squid_fair : List[float] = []
    
    def run(self, state: TradingState):
        squid_order_depth : OrderDepth = state.order_depths[Product.SQUID]
        self.__unpack_state(state)
        self.__update_lists(state.order_depths)
        
        # squid_kwargs : dict = {
        #     "order_depth" : squid_order_depth,
        #     "product" : Product.SQUID,
        #     "edge" : SQUID_PARAMETERS['take_edge'],
        #     "minimum_volume_signal" : SQUID_PARAMETERS['minimum_volume_signal']
        # }
        self.squid_fair_val()
        # squid_take_orders, buy_vol, sell_vol = self.market_taker_orders(**squid_kwargs)
        squid_take_orders = []
        
        # squid_kwargs : dict = {
        #     "order_depth" : squid_order_depth,
        #     "product" : Product.SQUID,
        #     "edge" : SQUID_PARAMETERS['clear_edge'],
        #     "minimum_volume_signal" : SQUID_PARAMETERS['minimum_volume_signal']
        # }
        # squid_clear_orders, buy_vol, sell_vol = self.clear_positions_orders(buy_vol,sell_vol, **squid_kwargs)
        squid_clear_orders = []
        
        squid_market_make = self.squid_market_maker_orders(order_depth = squid_order_depth)
        
        squid_orders = squid_take_orders + squid_clear_orders + squid_market_make
        
        result = {}
        result[Product.SQUID] = squid_orders
        traderData = jsonpickle.encode({
            "squid_fair" : self.squid_fair,
            "kalman_params" : self.kalman_params,
        })        
        conversions = 1 
        return result, conversions, traderData
    
    def market_taker_orders(self, *, order_depth : OrderDepth, product : Product, edge : int, minimum_volume_signal : int = 0):
        orders : List[Order] = []
        buy_order_volume : int = 0
        sell_order_volume : int = 0
        
        predicted_fair = self.fair_prices[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if best_ask <= predicted_fair - edge:
                quantity = min(best_ask_amount, self.max_orderable[product]['buy'])
                if quantity > minimum_volume_signal:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]
            
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= predicted_fair - edge:
                quantity = min(best_bid_amount, abs(self.max_orderable[product]['sell']))
                if quantity > minimum_volume_signal:
                    orders.append(Order(product, best_bid, -quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
                    
        return orders, buy_order_volume, sell_order_volume
    
    def clear_positions_orders(self, buy_vol, sell_vol, *, order_depth : OrderDepth, product : Product, edge : int):
        orders : List[Order] = []
        
        inventory = self.inventory[product]
        position_after_take = inventory + buy_vol - sell_vol
        pos_lim = POSITION_LIMIT[product]
        # How much we can buy after our buy order
        buy_q = pos_lim - (inventory + buy_vol)
        # How much we can sell after our buy order
        sell_q = pos_lim + (inventory - sell_vol)
        
        # Market take results in positive position
        if position_after_take > 0:
            fair_ask = int(self.fair_prices[product] + edge)
            # Total volume of remaining arbable prices
            clear_quantity = sum(vol for price, vol in order_depth.buy_orders.items() if price >= fair_ask)
            
            # Choose the minimum between how much we can sell based on existing orders, 
            # what our position needs to go back to 0
            # and the volume of arbable prices
            clear_quantity = min(clear_quantity, position_after_take)
            clear_quantity = min(clear_quantity,sell_q)
            if clear_quantity > 0:
                orders.append(Order(product, fair_ask, -clear_quantity))
                sell_vol += clear_quantity
        
        # Market take results in negative position
        elif position_after_take < 0:
            fair_bid = int(self.fair_prices[product] + edge)
            # Total volume of remaining arbable prices
            clear_quantity = sum(vol for price, vol in order_depth.buy_orders.items() if price <= fair_bid)
            
            # Choose the minimum between how much we can buy based on existing orders, 
            # what our position needs to go back to 0
            # and the volume of arbable prices
            clear_quantity = min(clear_quantity, abs(position_after_take))
            clear_quantity = min(clear_quantity,buy_q)
            if clear_quantity > 0:
                orders.append(Order(product, fair_bid, clear_quantity))
                buy_vol += clear_quantity
        
        return orders, buy_vol, sell_vol
        
                
    def squid_market_maker_orders(self, *, order_depth : OrderDepth = None) -> List[Order]:
        orders = []
        mid = max(order_depth.sell_orders.keys())+max(order_depth.buy_orders.keys())/2
        predicted_fair = self.fair_prices[Product.SQUID]
        edge = SQUID_PARAMETERS['default_edge']
        buy_order = Order(Product.SQUID,int(predicted_fair-edge),self.max_orderable[Product.SQUID]['buy'])
        orders.append(buy_order)
        sell_order = Order(Product.SQUID,int(predicted_fair+edge),self.max_orderable[Product.SQUID]['sell'])
        orders.append(sell_order)
        
        return orders
    
    def squid_fair_val(self):
        if len(self.squid_fair)==SQUID_PARAMETERS['history_length']:
            x_t = [self.squid_fair[-4],(self.squid_fair[-1]-self.squid_fair[-4])/(1e-6+abs(self.squid_fair[-1]-self.squid_fair[-4]))]
            y_t = self.squid_fair[-1]
        
            beta_prior = self.kalman_params['beta']
            P_prior = [row[:] for row in self.kalman_params['P']]  # Copy the covariance matrix
            for i in range(self.kalman_params['n_params']):
                P_prior[i][i] += self.kalman_params['Q'][i][i]  # Add process noise
            
            # Kalman gain
            S = sum(x_t[i] * sum(P_prior[i][j] * x_t[j] for j in range(self.kalman_params['n_params'])) for i in range(self.kalman_params['n_params'])) + self.kalman_params['R']
            K = [sum(P_prior[i][j] * x_t[j] for j in range(self.kalman_params['n_params'])) / S for i in range(self.kalman_params['n_params'])]
            
            # Update the estimate of beta
            y_pred = sum(x_t[i] * beta_prior[i] for i in range(self.kalman_params['n_params']))
            self.kalman_params['beta'] = [beta_prior[i] + K[i] * (y_t - y_pred) for i in range(self.kalman_params['n_params'])]
            
            # Update the covariance matrix
            self.kalman_params['P'] = [[P_prior[i][j] - K[i] * x_t[j] * P_prior[i][j] for j in range(self.kalman_params['n_params'])] for i in range(self.kalman_params['n_params'])]
            
            predicted_fair = sum(x_t[i] * self.kalman_params['beta'][i] for i in range(self.kalman_params['n_params']))
            self.fair_prices[Product.SQUID] = predicted_fair
        else:
            self.fair_prices[Product.SQUID] = self.squid_fair[-1]
        
    # UTILS________________________________________________________-
    def __update_lists(self, order_depths : Dict[Product,OrderDepth]):
        squid_order_depth : OrderDepth = order_depths[Product.SQUID]
        buy_side = squid_order_depth.buy_orders
        sell_side = squid_order_depth.sell_orders
        
        buy_vols = [abs(buy_side[price]) for price in buy_side]
        high_vol_bid = [*buy_side.keys()][buy_vols.index(max(buy_vols))]
        sell_vols = [abs(sell_side[price]) for price in sell_side]
        high_vol_ask = [*sell_side.keys()][sell_vols.index(max(sell_vols))]
        
        self.squid_fair.append((high_vol_ask+high_vol_bid)/2)
        
        if len(self.squid_fair) > SQUID_PARAMETERS['history_length']:
            self.squid_fair.pop(0)
   
    def __unpack_state(self, state : TradingState):
        if state.position:
            self.inventory = state.position
        self.max_orderable = {product: {"buy":max(POSITION_LIMIT[product]-self.inventory[product],0),"sell":min(-POSITION_LIMIT[product]-self.inventory[product],0)} for product in self.inventory}
                
        if state.traderData:
            traderData = jsonpickle.decode(state.traderData)
            self.squid_fair = traderData['squid_fair']
            self.kalman_params = traderData['kalman_params']
        
"""
Sidenote:
bid, ask fixed at +-1 trades 80% of the time
bid, ask fixed at +-2 trades 76% of the time
bid, ask fixed at +-3 trades 30% of the time
bid, ask fixed at +-4 trades 17% of the time
bid, ask fixed at +-5 trades 0% of the time
"""