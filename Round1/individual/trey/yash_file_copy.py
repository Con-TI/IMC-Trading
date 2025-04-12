from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math

class Trader:
    def __init__(self):
        # Initialize lists to store historical price and volume-weighted average price data

        self.kelp_prices = []  # Stores mid-prices for Kelp
        self.kelp_vwap = []    # Stores volume-weighted average price information
        self.ink_prices = []
        self.ink_vwap = []

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
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity)) 
                    buy_order_volume += quantity

        # Sell logic: look for buy orders above fair value
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:

                # Calculate sell quantity within position limits
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
                    sell_order_volume += quantity
        
        # Clear any excess position and adjust orders
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "RAINFOREST_RESIN", 
            buy_order_volume, sell_order_volume, fair_value, 1
        )

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_quantity))

        # Place additional sell orders to approach position limit
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_quantity))

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
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        # If position is negative, try to buy at fair ask price
        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
    
        return buy_order_volume, sell_order_volume
    
    def kelp_fair_value(self, order_depth: OrderDepth, method = "mid_price", min_vol = 0) -> float:
        
        # Calculate fair value for Kelp with different methods
        if method == "mid_price":
            # Simple mid-price calculation
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            return mid_price
        
        elif method == "mid_price_with_vol_filter":

            # Mid-price calculation with volume filtering
            if len([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]) ==0 or \
               len([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]) ==0:
                
                # Fallback to simple mid-price if volume filtering fails
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                mid_price = (best_ask + best_bid) / 2
                return mid_price
            else:   

                # Calculate mid-price using only orders with sufficient volume
                best_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol])
                best_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol])
                mid_price = (best_ask + best_bid) / 2
            return mid_price
        
    def ink_fair_value(self, order_depth: OrderDepth, method = "mid_price", min_vol = 0) -> float:
        
        # Calculate fair value for Kelp with different methods
        if method == "mid_price":
            # Simple mid-price calculation
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            return mid_price
        
        elif method == "mid_price_with_vol_filter":

            # Mid-price calculation with volume filtering
            if len([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]) ==0 or \
               len([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]) ==0:
                
                # Fallback to simple mid-price if volume filtering fails
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                mid_price = (best_ask + best_bid) / 2
                return mid_price
            else:   

                # Calculate mid-price using only orders with sufficient volume
                best_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol])
                best_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol])
                mid_price = (best_ask + best_bid) / 2
            return mid_price

    def kelp_orders(self, order_depth: OrderDepth, timespan:int, width: float, kelp_take_width: float, position: int, position_limit: int) -> List[Order]:
        
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
            self.kelp_prices.append(mmmid_price)

            # Calculate volume-weighted average price (VWAP)
            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.kelp_vwap.append({"vol": volume, "vwap": vwap})
            
            # Maintain historical data for specified timespan
            if len(self.kelp_vwap) > timespan:
                self.kelp_vwap.pop(0)
            
            if len(self.kelp_prices) > timespan:
                self.kelp_prices.pop(0)
        
            # Calculate fair value (commented out in favor of mid-price)
            # fair_value = sum([x["vwap"]*x['vol'] for x in self.kelp_vwap]) / sum([x['vol'] for x in self.kelp_vwap])
            
            fair_value = mmmid_price

            # Take liquidity when price is significantly away from fair value
            if best_ask <= fair_value - kelp_take_width:
                ask_amount = -1 * order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order("KELP", best_ask, quantity))
                        buy_order_volume += quantity
            
            if best_bid >= fair_value + kelp_take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order("KELP", best_bid, -1 * quantity))
                        sell_order_volume += quantity

            # Clear any excess position
            buy_order_volume, sell_order_volume = self.clear_position_order(
                orders, order_depth, position, position_limit, "KELP", 
                buy_order_volume, sell_order_volume, fair_value, 2
            )
            
            # Find prices for additional orders
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2
           
            # Place additional buy orders to approach position limit
            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order("KELP", bbbf + 1, buy_quantity))

            # Place additional sell orders to approach position limit
            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order("KELP", baaf - 1, -sell_quantity))

        return orders
    
    def ink_orders(self, order_depth: OrderDepth, timespan:int, width: float, ink_take_width: float, position: int, position_limit: int) -> List[Order]:
        
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
            self.ink_prices.append(mmmid_price)

            # Calculate volume-weighted average price (VWAP)
            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.ink_vwap.append({"vol": volume, "vwap": vwap})
            
            # Maintain historical data for specified timespan
            if len(self.ink_vwap) > timespan:
                self.ink_vwap.pop(0)
            
            if len(self.ink_prices) > timespan:
                self.ink_prices.pop(0)
        
            # Calculate fair value (commented out in favor of mid-price)
            # fair_value = sum([x["vwap"]*x['vol'] for x in self.kelp_vwap]) / sum([x['vol'] for x in self.kelp_vwap])
            if len(self.ink_prices)>2:
                last_return = (mmmid_price-self.ink_prices[-2])/self.ink_prices[-2]
                c = 0
                factor = 1 + last_return*c
            else:
                last_return = 0
                factor = 1
            fair_value = mmmid_price*factor

            # Take liquidity when price is significantly away from fair value
            if best_ask <= fair_value - ink_take_width:
                ask_amount = -1 * order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order("SQUID_INK", best_ask, quantity))
                        buy_order_volume += quantity
            
            if best_bid >= fair_value + ink_take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order("SQUID_INK", best_bid, -1 * quantity))
                        sell_order_volume += quantity

            # Clear any excess position
            buy_order_volume, sell_order_volume = self.clear_position_order(
                orders, order_depth, position, position_limit, "SQUID_INK", 
                buy_order_volume, sell_order_volume, fair_value, 2
            )
            
            # Find prices for additional orders
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2
           
            # Place additional buy orders to approach position limit
            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order("SQUID_INK", bbbf + 1, buy_quantity))

            # Place additional sell orders to approach position limit
            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order("SQUID_INK", baaf - 1, -sell_quantity))

        return orders

    def run(self, state: TradingState):
        # Main trading method called for each trading iteration
        result = {}

        # Fixed parameters for Rainforest Resin trading
        rainforest_resin_fair_value = 10000
        rainforest_resin_width = 2
        rainforest_resin_position_limit = 50

        # Parameters for Kelp trading
        kelp_make_width = 3.5
        kelp_take_width = 1
        kelp_position_limit = 50
        kelp_timespan = 10

        ink_make_width = 1
        ink_take_width = 0.5
        ink_position_limit = 50
        ink_timespan = 10
        
        # Commented out data restoration (potentially for persistent state)
        if state.traderData:
            traderData = jsonpickle.decode(state.traderData)
            self.kelp_prices = traderData["kelp_prices"]
            self.kelp_vwap = traderData["kelp_vwap"]
            self.ink_prices = traderData['ink_prices']
            self.ink_vwap = traderData['ink_vwap']
        # Generate orders for Rainforest Resin if market exists
        if "RAINFOREST_RESIN" in state.order_depths:
            rainforest_resin_position = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
            rainforest_resin_orders = self.rainforest_resin_orders(
                state.order_depths["RAINFOREST_RESIN"], 
                rainforest_resin_fair_value, 
                rainforest_resin_width, 
                rainforest_resin_position, 
                rainforest_resin_position_limit
            )
            result["RAINFOREST_RESIN"] = rainforest_resin_orders

        # Generate orders for Kelp if market exists
        if "KELP" in state.order_depths:
            kelp_position = state.position["KELP"] if "KELP" in state.position else 0
            kelp_orders = self.kelp_orders(
                state.order_depths["KELP"], 
                kelp_timespan, 
                kelp_make_width, 
                kelp_take_width, 
                kelp_position, 
                kelp_position_limit
            )
            result["KELP"] = kelp_orders

        # Generate orders for Kelp if market exists
        if "SQUID_INK" in state.order_depths:
            ink_position = state.position["SQUID_INK"] if "SQUID_INK" in state.position else 0
            ink_orders = self.ink_orders(
                state.order_depths["SQUID_INK"], 
                ink_timespan, 
                ink_make_width, 
                ink_take_width, 
                ink_position, 
                ink_position_limit
            )
            result["SQUID_INK"] = ink_orders
        

        # Encode trader data for potential state preservation
        traderData = jsonpickle.encode({"kelp_prices": self.kelp_prices, 
                                        "kelp_vwap": self.kelp_vwap,
                                        "ink_prices": self.ink_prices, 
                                        "ink_vwap": self.ink_vwap})

        # Conversions (purpose not clear from context, possibly a trading feature)
        conversions = 1

        return result, conversions, traderData