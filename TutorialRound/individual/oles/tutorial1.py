from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np

class Trader:
    '''
    Attempting to copy whale from our traing site
    '''

    def __init__(self):
        
        self.kelp_position = None
        self.resin_position = None
        self.kelp_limit = self.resin_limit = 50
        self.poi_params = None

        #dict of total bids and asks desired
        self.order_ratio = None

        #gradients for leaky relu used to determine poisson parameter
        self.m1 = 0.01
        self.m2 = 0.05

        self.kelp_market_conditions = {'spread':None,'midprice':None, 'microprice':None, 'best_ask':None, 'best_bid':None}
        self.resin_market_conditions = {'spread':None,'midprice':None, 'microprice':None, 'best_ask':None, 'best_bid':None}


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


    def _fetch_inventory(self):
        # Fetches trading state
        state = TradingState()
        state = state.toJSON()

        #getting your position (just a single number as buys and sells added)
        positions = state['position']
        self.kelp_position = positions['KELP']
        self.resin_position = positions['RAINFORES_RESIN']


    def _order_distribution_shift(self):
        #just matching the old spread - dont have past data to calculate the rel vol
        return self.kelp_market_conditions['spread'], self.resin_market_conditions['spread']


    def _bid_generator(self):
        # Generates orders based on order ratio and poisson distribution
        order_ratio = self.order_ratio
        steps_from_mid_kelp = np.random.poisson(self.poi_params['kelp_bid'],order_ratio['kelp_bids']) + self._order_distribution_shift()[0]
        bid_array_kelp = self.kelp_market_conditions['midprice']-steps_from_mid_kelp
        bid_array_kelp = bid_array_kelp[bid_array_kelp>0]
        bid_prices_kelp = np.unique(bid_array_kelp)

        bid_orders_kelp = [{'kelp price':price,'quantity':len(bid_array_kelp[bid_array_kelp==price])} for price in bid_prices_kelp]

        steps_from_mid_resin = np.random.poisson(self.poi_params['resin_bid'],order_ratio['resin_bids']) + self._order_distribution_shift()[1]
        bid_array_resin = self.resin_market_conditions['midprice']-steps_from_mid_resin
        bid_array_resin = bid_array_resin[bid_array_resin>0]
        bid_prices_resin = np.unique(bid_array_resin)

        bid_orders_resin = [{'resin price':price,'quantity':len(bid_array_resin[bid_array_resin==price])} for price in bid_prices_resin]

        return {'kelp bids': bid_orders_kelp, 'resin bids': bid_orders_resin}

    def _ask_generator(self):
        # Generates orders based on order ratio and poisson distribution
        # Generates orders based on order ratio and poisson distribution
        order_ratio = self.order_ratio
        steps_from_mid_kelp = np.random.poisson(self.poi_params['kelp_ask'],order_ratio['kelp_asks']) + self._order_distribution_shift()[0]
        ask_array_kelp = self.kelp_market_conditions['midprice']+steps_from_mid_kelp
        ask_array_kelp = ask_array_kelp[ask_array_kelp>0]
        ask_prices_kelp = np.unique(ask_array_kelp)

        ask_orders_kelp = [{'kelp price':price,'quantity':-len(ask_array_kelp[ask_array_kelp==price])} for price in ask_prices_kelp]

        steps_from_mid_resin = np.random.poisson(self.poi_params['resin_ask'],order_ratio['resin_asks']) + self._order_distribution_shift()[1]
        ask_array_resin = self.resin_market_conditions['midprice']+steps_from_mid_resin
        ask_array_resin = ask_array_resin[ask_array_resin>0]
        ask_prices_resin = np.unique(ask_array_resin)

        ask_orders_resin = [{'resin price':price,'quantity':-len(ask_array_resin[ask_array_resin==price])} for price in ask_prices_resin]

        return {'kelp asks': ask_orders_kelp, 'resin asks': ask_orders_resin}
    

    def update_all(self):
        self._update_market_conditions()
        self._fetch_inventory()
        self._derive_order_ratio()
        self._poi_param()
        self._order_distribution_shift()

    def _update_market_conditions(self):
        # Fetches current market conditions
        # Fetches trading state
        state = TradingState()
        state = state.toJSON()

        kelp_order_depths = state['order_depths']['KELP']
        resin_order_depths = state['order_depths']['RAINFOREST_RESIN']

        kelp_max_buy = None
        kelp_max_buy_vol = None
        kelp_min_ask = None
        kelp_min_ask_vol = None
        for order_price, order_vol in getattr(kelp_order_depths, 'buy_orders'):
            if kelp_max_buy is None or order_price>kelp_max_buy:
                kelp_max_buy = order_price
                kelp_max_buy_vol = order_vol

        for order_price, order_vol in getattr(kelp_order_depths, 'sell_orders'):
            if kelp_min_ask is None or order_price>kelp_min_ask:
                kelp_min_ask = order_price
                kelp_min_ask_vol = order_vol

        resin_max_buy = None
        resin_max_buy_vol = None
        resin_min_ask = None
        resin_min_ask_vol = None
        for order_price, order_vol in getattr(resin_order_depths, 'buy_orders'):
            if resin_max_buy is None or order_price>resin_max_buy:
                resin_max_buy = order_price
                resin_max_buy_vol = order_vol

        for order_price, order_vol in getattr(resin_order_depths, 'sell_orders'):
            if resin_min_ask is None or order_price>resin_min_ask:
                resin_min_ask = order_price
                resin_min_ask_vol = order_vol

        self.kelp_market_conditions['midprice'] =  (kelp_min_ask+kelp_max_buy)/2
        self.kelp_market_conditions['microprice'] = (kelp_min_ask*kelp_max_buy_vol+kelp_max_buy*kelp_min_ask_vol)/(kelp_max_buy_vol+kelp_min_ask_vol)
        self.kelp_market_conditions['spread'] = kelp_max_buy-kelp_min_ask
        self.kelp_market_conditions['best_ask'] = kelp_min_ask
        self.kelp_market_conditions['best_bid'] =  kelp_max_buy

        self.resin_market_conditions['midprice'] =  (resin_min_ask+resin_max_buy)/2
        self.resin_market_conditions['microprice'] = (resin_min_ask*resin_max_buy_vol+resin_max_buy*resin_min_ask_vol)/(resin_max_buy_vol+resin_min_ask_vol)
        self.resin_market_conditions['spread'] = resin_max_buy-resin_min_ask
        self.resin_market_conditions['best_ask'] = resin_min_ask
        self.resin_market_conditions['best_bid'] =  resin_max_buy
        


    #need to adapt to my formats above
    def run(self, state: TradingState):
        pass
