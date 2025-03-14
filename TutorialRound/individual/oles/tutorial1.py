from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    '''
    Attempting to copy whale from our traing site
    '''

    def __init__(self):
        self.kelp_position = None
        self.resin_position = None
        self.kelp_pending_orders = None
        self.resin_pending_orders = None
        self.kelp_limit = self.resin_limit = 50
        self.memory = []
        self.poi_params = None

        #dict of total bids and asks desired
        self.order_ratio = None

        #gradients for leaky relu used to determine poisson parameter
        self.m1 = 0.01
        self.m2 = 0.05

        self.kelp_market_conditions = {'spread':None,'midprice':None, 'microprice':None, 'sigma_norm':None, 'best_ask':None, 'best_bid':None}
        self.resin_market_conditions = {'spread':None,'midprice':None, 'microprice':None, 'sigma_norm':None, 'best_ask':None, 'best_bid':None}


    def _poi_param(self):
        # Calculates the poisson parameter given orderbook imbalance
        order_ratio = self.order_ratio
        leaky_relu = lambda x : self.m1*(50-x)+1 if x>=50 else self.m2*(10-x)+3
        self.poi_params = {'kelp_bid':leaky_relu(order_ratio['kelp_bids']),'kelp_ask':leaky_relu(order_ratio['kelp_asks']), 'resin_bid':leaky_relu(order_ratio['resin_bids']),'resin_ask':leaky_relu(order_ratio['resin_asks'])}

    def _derive_order_ratio(self):
        # Calculates order ratio based off of microprice and spread
        kelp_micro_minus_bid = self.kelp_market_conditions['microprice']-self.kelp_market_conditions['best_bid']
        kelp_numeric_spread = self.kelp_market_conditions['spread']
        if kelp_numeric_spread == 0:
            kelp_numeric_spread = 1
        kelp_bid_order_num = min(max(int(kelp_micro_minus_bid/kelp_numeric_spread*100),30),70)
        kelp_ask_order_num = 100 - kelp_bid_order_num

        resin_micro_minus_bid = self.resin_market_conditions['microprice']-self.resin_market_conditions['best_bid']
        resin_numeric_spread = self.resin_market_conditions['spread']
        if resin_numeric_spread == 0:
            resin_numeric_spread = 1
        resin_bid_order_num = min(max(int(resin_micro_minus_bid/resin_numeric_spread*100),30),70)
        resin_ask_order_num = 100 - resin_bid_order_num

        self.order_ratio = {"kelp_bids": kelp_bid_order_num, "kelp_asks": kelp_ask_order_num, "resin_bids": resin_bid_order_num, "resin_asks": resin_ask_order_num}

        def _fetch_inventory_and_pending_orders(self):
            # Fetches trading state
            state = TradingState()
            state = state.toJSON()

            #getting your position (just a single number as buys and sells added)
            positions = state['position']
            self.kelp_position = positions['KELP']
            self.resin_position = positions['RAINFORES_RESIN']

            self.kelp_pending_orders = [{'price':order.price,'quantity':order.quantity} for order in pending_orders]
 
