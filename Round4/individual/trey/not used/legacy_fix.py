from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import pandas as pd
import math

class Product:
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SQUID = "SQUID_INK"
    KELP = "KELP"
    RESIN = "RAINFOREST_RESIN"
    
PARAMS = {
    # Fixed parameters for Rainforest Resin trading
    # [x] Optimized
    Product.RESIN : {
        "rainforest_resin_fair_value" : 10000,
        "rainforest_resin_width" : 1,
    },
    
    # Parameters for Kelp trading
    # [x] Optimized
    Product.KELP : {        
        "make_width" : 3.5,
        "take_width" : 1,
    },
    
    # Parameters for ink trading
    # [] Optimized
    Product.SQUID : {
        "timespan" : 4
    }
    
}
    

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        
        self.LIMIT = {
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.SQUID : 50,
            Product.KELP : 50,
            Product.RESIN : 100,
        }
        
    def rainforest_resin_orders(self, order_depth: OrderDepth, fair_value: int, width: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        # Track buy and sell order volumes
        buy_order_volume = 0
        sell_order_volume = 0
        
        # Find the best ask above fair value and best bid below fair value
        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])

        # Buy logic: look for sell orders below fair value
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            if best_ask < fair_value:

                # Calculate buy quantity within position limits
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(Product.RESIN, best_ask, quantity)) 
                    buy_order_volume += quantity

        # Sell logic: look for buy orders above fair value
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:

                # Calculate sell quantity within position limits
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(Product.RESIN, best_bid, -1 * quantity))
                    sell_order_volume += quantity
        
        # Clear any excess position and adjust orders
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, Product.RESIN, 
            buy_order_volume, sell_order_volume, fair_value, 1
        )

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.RESIN, bbbf + width, buy_quantity))

        # Place additional sell orders to approach position limit
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.RESIN, baaf - width, -sell_quantity))

        return orders
    
    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int, product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float, width: int) -> List[Order]:
        
        # Calculate position after current trades
        position_after_take = position + buy_order_volume - sell_order_volume
        fair = round(fair_value)
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        # Calculate remaining buy and sell quantities
        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, int(fair_for_ask), -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        # If position is negative, try to buy at fair ask price
        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, int(fair_for_bid), abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
    
        return buy_order_volume, sell_order_volume

    def kelp_orders(self, order_depth: OrderDepth, width: float, kelp_take_width: float, position: int, position_limit: int) -> List[Order]:
        
        # Generate orders for Kelp trading
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0

        # Ensure both buy and sell orders exist
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:    

            # Find best ask and bid prices
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            
            # Filter orders with significant volume (>= 15)
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]
            
            # Use filtered prices or fallback to best prices
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
            
            # Calculate mid-price
            mmmid_price = (mm_ask + mm_bid) / 2    
            
            fair_value = mmmid_price

            # Take liquidity when price is significantly away from fair value
            if best_ask <= fair_value - kelp_take_width:
                ask_amount = -1 * order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(Product.KELP, int(math.ceil(best_ask)), quantity))
                        buy_order_volume += quantity
            
            if best_bid >= fair_value + kelp_take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(Product.KELP, int(math.floor(best_bid)), -1 * quantity))
                        sell_order_volume += quantity

            # Clear any excess position
            buy_order_volume, sell_order_volume = self.clear_position_order(
                orders, order_depth, position, position_limit, Product.KELP, 
                buy_order_volume, sell_order_volume, fair_value, 2
            )
            
            # Find prices for additional orders
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + width]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - width]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + width*2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - width*2
           
            # Place additional buy orders to approach position limit
            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order(Product.KELP, int(math.floor(bbbf + width)), buy_quantity))

            # Place additional sell orders to approach position limit
            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order(Product.KELP, int(math.ceil(baaf - width)), -sell_quantity))

        return orders
    
    def ink_orders(self, order_depth: OrderDepth, timespan:int, position: int, position_limit: int, traderObject, state) -> List[Order]:
        
        # Generate orders for Squid trading
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0

        # Ensure both buy and sell orders exist
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:    

            # Find best ask and bid prices
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            
            # Filter orders with significant volume (>= 15)
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]
            
            # Use filtered prices or fallback to best prices
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
            
            # Calculate mid-price
            mmmid_price = (mm_ask + mm_bid) / 2    
            traderObject[Product.SQUID]['mid_price'].append(mmmid_price)
            
            if len(traderObject[Product.SQUID]['mid_price']) > timespan:
                traderObject[Product.SQUID]['mid_price'].pop(0)
        
            fair_value = mmmid_price
            predictive_shift = self.floor_func(self.retrieve_mean_revert_skew(state, traderObject, Product.SQUID),4)
            print(fair_value, predictive_shift, best_bid, best_ask)
            direction = np.sign(predictive_shift) if predictive_shift != 0 else 0

            if traderObject[Product.SQUID]['trade_filled']:
                if traderObject[Product.SQUID]["stop_loss_flag"]:
                    own_trades = state.own_trades.get(Product.SQUID,None)
                    if own_trades is None:
                        if (traderObject[Product.SQUID]['side'] == 'buy'):
                            orders.append(Order(Product.SQUID, int(math.ceil((best_bid+mm_bid)/2)), -1))
                        elif (traderObject[Product.SQUID]['side'] == 'sell'):
                            orders.append(Order(Product.SQUID, int(math.floor(((best_ask+mm_ask)/2))), 1))
                        return orders
                    else:
                        traderObject[Product.SQUID]['trade_sent'] = False
                        traderObject[Product.SQUID]['side'] = None
                        traderObject[Product.SQUID]['traded_price'] = None
                        traderObject[Product.SQUID]['trade_filled'] = False
                        traderObject[Product.SQUID]['stop_loss_flag'] = False
                if (traderObject[Product.SQUID]['side'] == 'buy') and (mmmid_price < traderObject[Product.SQUID]['traded_price']-1):
                    traderObject[Product.SQUID]["stop_loss_flag"] = True
                    orders.append(Order(Product.SQUID, int(math.ceil((best_bid+mm_bid)/2)), -1))
                    return orders
                elif (traderObject[Product.SQUID]['side'] == 'sell') and (mmmid_price > traderObject[Product.SQUID]['traded_price']+1):
                    traderObject[Product.SQUID]["stop_loss_flag"] = True
                    orders.append(Order(Product.SQUID, int(math.floor(((best_ask+mm_ask)/2))), 1))
                    return orders
            if not traderObject[Product.SQUID]['trade_sent']:
                if direction == 1.0:
                    orders.append(Order(Product.SQUID, int(math.floor(((best_ask+mm_ask)/2))), 1))
                    traderObject[Product.SQUID]['trade_sent'] = True
                    traderObject[Product.SQUID]['side'] = "buy"
                    traderObject[Product.SQUID]['traded_price'] = best_ask
                elif direction == -1.0:
                    orders.append(Order(Product.SQUID, int(math.ceil((best_bid+mm_bid)/2)), -1))
                    traderObject[Product.SQUID]['trade_sent'] = True
                    traderObject[Product.SQUID]['side'] = "sell"
                    traderObject[Product.SQUID]['traded_price'] = best_bid
            else:
                if not traderObject[Product.SQUID]['trade_filled']:
                    own_trades = state.own_trades.get(Product.SQUID,None)
                    if own_trades is None:
                        if direction == 1.0:
                            orders.append(Order(Product.SQUID, best_ask, 1))
                            traderObject[Product.SQUID]['trade_sent'] = True
                            traderObject[Product.SQUID]['side'] = "buy"
                            traderObject[Product.SQUID]['traded_price'] = best_ask
                        elif direction == -1.0:
                            orders.append(Order(Product.SQUID, best_bid, -1))
                            traderObject[Product.SQUID]['trade_sent'] = True
                            traderObject[Product.SQUID]['side'] = "sell"
                            traderObject[Product.SQUID]['traded_price'] = best_bid
                        else:
                            traderObject[Product.SQUID]['trade_sent'] = False
                            traderObject[Product.SQUID]['side'] = None
                            traderObject[Product.SQUID]['traded_price'] = None
                    else:    
                        traderObject[Product.SQUID]['trade_filled'] = True
                        price = traderObject[Product.SQUID]['traded_price']
                        if traderObject[Product.SQUID]['side'] == "sell":
                            if price > best_ask:
                                orders.append(Order(Product.SQUID,best_ask,1))
                        if traderObject[Product.SQUID]['side'] == "buy":
                            if price < best_bid:
                                orders.append(Order(Product.SQUID,best_bid,-1))
                else:
                    own_trades = state.own_trades.get(Product.SQUID,None)
                    if own_trades is None:
                        price = traderObject[Product.SQUID]['traded_price']
                        if traderObject[Product.SQUID]['side'] == "sell":
                            if price > best_ask:
                                orders.append(Order(Product.SQUID,best_ask,1))
                        if traderObject[Product.SQUID]['side'] == "buy":
                            if price < best_bid:
                                orders.append(Order(Product.SQUID,best_bid,-1))
                    else:
                        traderObject[Product.SQUID]['trade_sent'] = False
                        traderObject[Product.SQUID]['side'] = None
                        traderObject[Product.SQUID]['traded_price'] = None
                        traderObject[Product.SQUID]['trade_filled'] = False
        return orders


    
    def retrieve_trend_sig(self, traderObject, product : Product):
        if len(traderObject[product]['mid_price']) == self.params[product]['timespan']:
            trend = np.log(traderObject[product]['mid_price'][-1])-np.log(traderObject[product]['mid_price'][0])
            # volatility = pd.Series(traderObject[product]['mid_price']).pct_change().std()**2
            # if volatility != 0:
            #     trend /= volatility*50000
            return trend*100
            return 0
        return 0

    def retrieve_mean_revert_skew(self, state : TradingState, traderObject, product : Product):
        if len(traderObject[product]['mid_price']) == self.params[product]['timespan']:
            weights = np.arange(1,len(traderObject[product]['mid_price'][:-1])+1)
            ave = np.dot(np.array(traderObject[product]['mid_price'][:-1]),weights)/weights.sum()
            diff1 = ave-traderObject[product]['mid_price'][-1]
            diff2 = traderObject[product]['mid_price'][-2]-traderObject[product]['mid_price'][-1]
            diff3 = traderObject[product]['mid_price'][-3]-traderObject[product]['mid_price'][-1]
            # Short timeframe mean reversion
            return (diff1+diff2+diff3)/3

            # Longer timeframe mean reversion
            time_hor = 1000000-state.timestamp
            return diff*(1-np.exp(-1*time_hor))
        return 0
    
    def floor_func(self, num, bounds):
        if abs(num)<bounds:
            return 0
        return num

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        if Product.SQUID not in traderObject:
            traderObject[Product.SQUID] = {
                 "mid_price" : [],
                 "trade_sent" : False,
                 "trade_filled" : False,
                 "traded_price" : None,
                 "side" : None,
                 "stop_loss_flag" : False
            }    
        
        # Main trading method called for each trading iteration
        result = {}
        
        # # Generate orders for Rainforest Resin if market exists
        # if Product.RESIN in state.order_depths:
        #     rainforest_resin_orders = self.rainforest_resin_orders(
        #         state.order_depths[Product.RESIN], 
        #         self.params[Product.RESIN]["rainforest_resin_fair_value"], 
        #         self.params[Product.RESIN]["rainforest_resin_width"], 
        #         state.position.get(Product.RESIN,0), 
        #         self.LIMIT[Product.RESIN]
        #     )
        #     result[Product.RESIN] = rainforest_resin_orders

        # Generate orders for Kelp if market exists
        if Product.KELP in state.order_depths:
            kelp_orders = self.kelp_orders(
                state.order_depths[Product.KELP], 
                self.params[Product.KELP]['make_width'], 
                self.params[Product.KELP]['take_width'], 
                state.position.get(Product.KELP,0), 
                self.LIMIT[Product.KELP]
            )
            result[Product.KELP] = kelp_orders

        # Generate orders for SQUID if market exists
        # if Product.SQUID in state.order_depths:
        #     ink_orders = self.ink_orders(
        #         state.order_depths[Product.SQUID], 
        #         self.params[Product.SQUID]['timespan'], 
        #         state.position.get(Product.SQUID, 0), 
        #         self.LIMIT[Product.SQUID],
        #         traderObject,
        #         state
        #     )
        #     result[Product.SQUID] = ink_orders

        traderData = jsonpickle.encode(traderObject)

        # Conversions (purpose not clear from context, possibly a trading feature)
        conversions = 1

        return result, conversions, traderData
    

    