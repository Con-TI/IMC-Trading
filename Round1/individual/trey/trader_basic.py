"""
This file focuses on maximizing resin profit. It only trades resin.

Strategy:
- Bid and ask quotes stay within the range 9996 and 10000
- Arbitrage:
    - We have a fair bid and fair ask
    - If the ask is below fair bid, buy at ask. Sell back at a higher price.
    - If the bid is above the fair ask, sell at bid. Buy back at a lower price.
    - Trade the maximal amount upon arbing.
- Market Make:
    - If no arb, place market making orders at fair bid and fair ask
"""

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string
import jsonpickle


class Product:
    KELP = "KELP"
    RESIN = "RAINFOREST_RESIN"
    SQUID = "SQUID_INK"

POSITION_LIMIT = {
    Product.KELP: 50,
    Product.RESIN: 50,
    Product.SQUID: 50
} 

RESIN_PARAMETERS = {
    "history_length" : 3,
}

class Trader:
    def __init__(self):
        self.inventory : Dict[Product, int] = {Product.KELP : 0, Product.RESIN : 0, Product.SQUID : 0}
        self.max_orderable : Dict[Product, Dict[str, int]] = {}
        self.fair_prices : Dict[Product, int] = {Product.RESIN : 10000}
        
        self.resin_prices : List[float] = []
        self.resin_best_bids : List[float] = []
        self.resin_best_asks : List[float] = []
    
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
            self.resin_prices = traderData['resin_prices']
            self.resin_best_bids = traderData['resin_best_bids']
            self.resin_best_asks = traderData['resin_best_asks']
        
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
        result[Product.RESIN] = generated_resin_orders
        # result[Product.SQUID] = generated_squid_orders

        best_bid = max([*resin_order_depth.buy_orders.keys()])
        best_ask = min([*resin_order_depth.sell_orders.keys()])
        
        self.resin_best_bids.append(best_bid)
        self.resin_best_asks.append(best_ask)
        self.resin_prices.append((best_bid+best_ask)/2)
        if len(self.resin_prices) > RESIN_PARAMETERS["history_length"]:
            self.resin_prices.pop(0)
                
        traderData = jsonpickle.encode({
            "resin_prices" : self.resin_prices,
            "resin_best_bids" : self.resin_best_bids,
            "resin_best_asks" : self.resin_best_asks
        })        
        conversions = 1 
        return result, conversions, traderData

    def resin_orders(self, *, order_depth : OrderDepth) -> List[Order]:
        orders : List[Order] = [] 
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        
        if (not buy_orders) or (not sell_orders) or (not self.resin_prices):
            quantity = 1
            order = Order(Product.RESIN, 10002, -int(quantity))
            orders.append(order)
            order = Order(Product.RESIN, 9998, int(quantity))
            orders.append(order)
            return orders
        else:
            fair_value = 10000
            arbbids = {price:buy_orders[price] for price in buy_orders if price > fair_value-1}
            arbasks = {price:sell_orders[price] for price in sell_orders if price < fair_value+1}
            if arbbids:
                allowable_vol = abs(self.max_orderable[Product.RESIN]['sell'])
                for price in arbbids:
                    if allowable_vol == 0:
                        break
                    quantity = 1
                    order = Order(Product.RESIN, int(price), -int(quantity))
                    orders.append(order)
                    order = Order(Product.RESIN, 9998, int(quantity))
                    orders.append(order)
                    allowable_vol -= quantity
            if arbasks:
                allowable_vol = abs(self.max_orderable[Product.RESIN]['buy'])
                for price in arbbids:
                    if allowable_vol == 0:
                        break
                    quantity = 1
                    order = Order(Product.RESIN,int(price),int(quantity))
                    orders.append(order)
                    order = Order(Product.RESIN,10002,-int(quantity))
                    orders.append(order)
                    allowable_vol -= quantity
            if (not arbasks) and (not arbbids):
                quantity = 1
                order = Order(Product.RESIN, 10002, -int(quantity))
                orders.append(order)
                order = Order(Product.RESIN, 9998, int(quantity))
                orders.append(order)
                return orders
        return orders
    
    def kelp_orders(self, *, order_depth : OrderDepth) -> List[Order]:
        return []        
    
    def squid_orders(self, *, order_depth : OrderDepth) -> List[Order]:
        return []
    
"""
Sidenote:
bid, ask fixed at +-1 trades 80% of the time
bid, ask fixed at +-2 trades 76% of the time
bid, ask fixed at +-3 trades 30% of the time
bid, ask fixed at +-4 trades 17% of the time
bid, ask fixed at +-5 trades 0% of the time
"""