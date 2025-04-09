"""
Trying MACD on squid
"""

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string
import jsonpickle
import numpy as np

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
    "history_length" : 35, #number of datapoints needed for MACD
}

class Trader:
    def __init__(self):
        self.inventory : Dict[Product, int] = {Product.KELP : 0, Product.RESIN : 0, Product.SQUID : 0}
        self.max_orderable : Dict[Product, Dict[str, int]] = {}
        self.fair_prices : Dict[Product, int] = {Product.RESIN : 10000}
        
        self.squid_prices : List[float] = []
        self.squid_best_bids : List[float] = []
        self.squid_best_asks : List[float] = []
        #self.squid_mid_history : List[float] = []

    def run(self, state: TradingState):
        # ____________________________ STEP 1 : Set up values __________________________
        # Step 1: Setting up values for reference:  
        # # Order depths
        kelp_order_depth : OrderDepth = state.order_depths[Product.KELP]
        resin_order_depth : OrderDepth = state.order_depths[Product.RESIN]
        squid_order_depth : OrderDepth = state.order_depths[Product.SQUID]
        
        # # Positions/Inventory      
        if state.position:
            self.inventory = state.position
        self.max_orderable = {product: {"buy":max(POSITION_LIMIT[product]-self.inventory[product],0),"sell":min(-POSITION_LIMIT[product]-self.inventory[product],0)} for product in self.inventory}
                
        if state.traderData:
            traderData = jsonpickle.decode(state.traderData)
            self.squid_prices = traderData['squid_prices']
            self.squid_best_bids = traderData['squid_best_bids']
            self.squid_best_asks = traderData['squid_best_asks']
            #self.squid_mid_history.append((self.squid_best_asks[0] + self.squid_best_bids[0])/2)

         #____________________________ STEP 2 : Generate trades __________________________
        # Resin
        resin_kwargs : dict = {
            "order_depth" : resin_order_depth
        }
        generated_resin_orders = self.resin_orders(**resin_kwargs)
        
        # Kelp
        kelp_kwargs : dict = {
            "order_depth" : kelp_order_depth
        }
        generated_kelp_orders = self.kelp_orders(**kelp_kwargs)
        
        # Squid
        squid_kwargs : dict = {
            "order_depth" : squid_order_depth
        }
        generated_squid_orders = self.squid_orders(**squid_kwargs)

        #____________________________ STEP 3 : Submit __________________________
        result = {}
        # result[Product.KELP] = generated_kelp_orders
        #result[Product.RESIN] = generated_resin_orders
        result[Product.SQUID] = generated_squid_orders

        best_bid = max([*squid_order_depth.buy_orders.keys()])
        best_ask = min([*squid_order_depth.sell_orders.keys()])
        
        self.squid_best_bids.append(best_bid)
        self.squid_best_asks.append(best_ask)
        self.squid_prices.append((best_bid+best_ask)/2)
        if len(self.squid_prices) > SQUID_PARAMETERS["history_length"]:
            self.squid_prices.pop(0)
                
        traderData = jsonpickle.encode({
            "squid_prices" : self.squid_prices,
            "squid_best_bids" : self.squid_best_bids,
            "squid_best_asks" : self.squid_best_asks
        })        
        conversions = 1 
        return result, conversions, traderData

    def squid_orders(self, *, order_depth : OrderDepth) -> List[Order]:
        orders : List[Order] = []
        
        prices = np.array(self.squid_prices)

        if len(prices) < 35: 
            print(1)
            return []

        ema_12 = self.calculate_ema(prices, 12)
        ema_26 = self.calculate_ema(prices, 26)

        macd_line = ema_12 - ema_26
        macd_line[:26] = 0  

        # Signal line = 10 EMA of MACD
        valid_macd = macd_line[macd_line != 0]
        signal_line = self.calculate_ema(valid_macd, 10)

        if len(signal_line) == 0:
            print(2)
            return []

        # Get the most recent values
        macd = valid_macd[-1]
        signal = signal_line[-1]
        hist = macd - signal

        if macd>signal:
            order = Order(Product.SQUID, self.squid_best_asks[0], int(self.max_orderable[Product.SQUID]['buy']))
            orders.append(order)
            print(3)
            return orders
        elif macd<signal:
            order = Order(Product.SQUID, self.squid_best_bids[0], int(self.max_orderable[Product.SQUID]['sell']))
            orders.append(order)
            print(4)
            return orders
        print(5)
        return []

    def calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
            if len(prices) < period:
                return np.array([])

            ema = np.zeros_like(prices)
            k = 2 / (period + 1)
            ema[period - 1] = np.mean(prices[:period])

            for i in range(period, len(prices)):
                ema[i] = prices[i] * k + ema[i - 1] * (1 - k)

            return ema

    
    def kelp_orders(self, *, order_depth : OrderDepth) -> List[Order]:
        return []        
    
    def resin_orders(self, *, order_depth : OrderDepth) -> List[Order]:
        return []
    





