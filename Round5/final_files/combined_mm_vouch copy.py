from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import pandas as pd
import math
from statistics import NormalDist

class Product:
    SQUID = "SQUID_INK"
    KELP = "KELP"
    RESIN = "RAINFOREST_RESIN"
    MACARONS = "MAGNIFICENT_MACARONS"
    PICNIC_1 = "PICNIC_BASKET1"
    PICNIC_2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"
    SPREAD_1 = "SPREAD_1"
    SPREAD_2 = "SPREAD_2"
    ROCK = "VOLCANIC_ROCK"
    VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOUCHER_9750 = 'VOLCANIC_ROCK_VOUCHER_9750'
    VOUCHER_10000 = 'VOLCANIC_ROCK_VOUCHER_10000'
    VOUCHER_10250 = 'VOLCANIC_ROCK_VOUCHER_10250'
    VOUCHER_10500 = 'VOLCANIC_ROCK_VOUCHER_10500'
    

PARAMS = {
    Product.RESIN : {
        "rainforest_resin_fair_value" : 10000,
        "rainforest_resin_width" : 2,
    },
    
    Product.KELP : {        
        "make_width" : 1,
        "take_width" : 1,
    },
    
    Product.SQUID : {
        "timespan" : 300,
        "timespan_bound" : 200,
        "trend_timespan_bound" : 25,
        "make_width" : 2,
        "take_width" : .5,
        "trend_thresh" : 6,
    },
    
    Product.MACARONS : {
        # Critical Sunlight Index threshold
        "CSI" : 49, #original 49
        # Trading parameters
        "DEFAULT_FAIR_VALUE" : 650,  # Default fair value
        "LOW_SUNLIGHT_PREMIUM_BASE" : 300,  # Base premium to add when sunlight is below CSI
        "MIN_EDGE" : 2,  # Minimum edge for market making
        "MAX_EDGE" : 10,  # Maximum edge for market making
        "TAKE_EDGE" : 5,  # Edge for taking orders
    },
    Product.SPREAD_1: {
        "default_spread_mean": 48.762433333333334,
        "default_spread_std": 85.11945080948948944,
        "spread_std_window": 49.5,
        "zscore_threshold": 3,
        "target_position": 60,
    },
    Product.SPREAD_2: {
        "default_spread_mean": 30.23596666666666,
        "default_spread_std": 59.849200222652364,
        "spread_std_window": 24,
        "zscore_threshold": 1.5,
        "target_position": 100
    },
    
    # 10k vouch
    Product.VOUCHER_10000: {
        "strike": 10000, 
        "time_to_expiry": 3,
        "VOUCHER_SPREAD_FACTOR": 0.0001,
        "MIN_VOUCHER_SPREAD": .1,
        "RISK_AVERSION": 0.05,
        "WINDOW_SIZE": 40,
        "BASE_ORDER_SIZE": 100
    },
    
    # 9.5k vouch
    Product.VOUCHER_9500: {
        "strike": 9500, 
        "time_to_expiry": 3,
        "VOUCHER_SPREAD_FACTOR": 0.0001, #.2
        "MIN_VOUCHER_SPREAD": 0.01, #.1
        "RISK_AVERSION": 0.05, #0.05
        "WINDOW_SIZE": 5, #10
        "BASE_ORDER_SIZE": 10 #10
    },
    
    # 9.75k vouch
    Product.VOUCHER_9750: {
        "strike": 9750, 
        "time_to_expiry": 3,
        "VOUCHER_SPREAD_FACTOR": 0.0001,
        "MIN_VOUCHER_SPREAD": 0.1,
        "RISK_AVERSION": 0.05,
        "WINDOW_SIZE": 40,
        "BASE_ORDER_SIZE": 100
    },
    
    # Product.VOUCHER_10250: {
    #     "strike": 10250, 
    #     "time_to_expiry": 4,
    #     "VOUCHER_SPREAD_FACTOR": .1,  
    #     "MIN_VOUCHER_SPREAD": 0.1,
    #     "RISK_AVERSION": 0.05,
    #     "WINDOW_SIZE": 40,
    #     "BASE_ORDER_SIZE": 100           
    # },
    
    # 10.5k vouch
    Product.VOUCHER_10500: {
        "strike": 10500, 
        "time_to_expiry": 5,
        "VOUCHER_SPREAD_FACTOR": 0.0001,
        "MIN_VOUCHER_SPREAD": 0.1,
        "RISK_AVERSION": 0.05,
        "WINDOW_SIZE": 40,
        "BASE_ORDER_SIZE": 100
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
        Product.DJEMBES: 0,
    }
}

class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
            math.log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * math.sqrt(time_to_expiry))
        d2 = d1 - volatility * math.sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (math.log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * math.sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * math.sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            math.log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * math.sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.001
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility

class Trader:
    def __init__(self, params = None):
        if params is None:
            params = PARAMS
        self.params = params
    
        self.LIMIT = {
            Product.SQUID : 50,
            Product.KELP : 50,
            Product.RESIN : 50,
            
            Product.MACARONS : 75,
            
            Product.PICNIC_1: 60,
            Product.PICNIC_2: 100,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            
            Product.ROCK: 400,
            Product.VOUCHER_9500: 200,
            Product.VOUCHER_9750: 200, #200
            Product.VOUCHER_10000: 200, #200
            Product.VOUCHER_10250: 200, #200
            Product.VOUCHER_10500: 200, #200
        }
        
        self.CONVERSION_LIMIT = {Product.MACARONS: 10}
        
        self.DEFAULT_VOUCHER_PARAMS = {
            "VOUCHER_SPREAD_FACTOR": 0.0001,  
            "MIN_VOUCHER_SPREAD": 0.1,        
            "RISK_AVERSION": 0.05,            
            "WINDOW_SIZE": 40,               
            "BASE_ORDER_SIZE": 100,
            "ROCK_SPREAD" : 0.001,
            "ROCK_ORDER_SIZE" : 200 ,
            "MAX_POSITION_FACTOR" : 1          
        }
    
    # _____________________________________ RESIN SQUID KELP ___________________________________________________________________________________
    
    def rainforest_resin_orders(self, order_depth: OrderDepth, fair_value: int, width: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        # Track buy and sell order volumes
        buy_order_volume = 0
        sell_order_volume = 0
        
        # Find the best ask above fair value and best bid below fair value
        baaf = 10000
        bbbf = 10000
        if [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]:
            baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        if [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]:
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
            orders.append(Order(Product.RESIN, bbbf + 1, buy_quantity))

        # Place additional sell orders to approach position limit
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.RESIN, baaf - 1, -sell_quantity))

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
    
    def ink_orders(self, order_depth: OrderDepth, timespan:int, take_width : float, width : float, position: int, position_limit: int, traderObject, state) -> List[Order]:
        
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
            predictive_shift = self.retrieve_squid_ink_signal(traderObject)
            direction = np.sign(predictive_shift) if predictive_shift != 0 else 0
            desired_position = 0
            if direction != 0: 
                desired_position = (position_limit-2) if direction == 1.0 else - (position_limit-2)
            else:
                desired_position = traderObject[Product.SQUID]['desired_position']
            traderObject[Product.SQUID]['desired_position'] = desired_position

            if direction != 0:
                if position != desired_position:
                    if position > desired_position:
                        orders.append(Order(Product.SQUID, best_bid, desired_position-position))
                    elif position < desired_position:
                        orders.append(Order(Product.SQUID, best_ask, desired_position-position))
            else:
                # Take liquidity when price is significantly away from fair value
                position_limit = position_limit//2
                if best_ask <= fair_value - take_width:
                    ask_amount = -1 * order_depth.sell_orders[best_ask]
                    if ask_amount <= 20:
                        quantity = min(ask_amount, position_limit - position)
                        if quantity > 0:
                            orders.append(Order(Product.SQUID, int(math.ceil(best_ask)), quantity))
                            buy_order_volume += quantity
                
                if best_bid >= fair_value + take_width:
                    bid_amount = order_depth.buy_orders[best_bid]
                    if bid_amount <= 20:
                        quantity = min(bid_amount, position_limit + position)
                        if quantity > 0:
                            orders.append(Order(Product.SQUID, int(math.floor(best_bid)), -1 * quantity))
                            sell_order_volume += quantity

                # Clear any excess position
                buy_order_volume, sell_order_volume = self.clear_position_order(
                    orders, order_depth, position, position_limit, Product.SQUID, 
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
                    orders.append(Order(Product.SQUID, int(math.floor(bbbf + width)), buy_quantity))

                # Place additional sell orders to approach position limit
                sell_quantity = position_limit + (position - sell_order_volume)
                if sell_quantity > 0:
                    orders.append(Order(Product.SQUID, int(math.ceil(baaf - width)), -sell_quantity))
        return orders

    def retrieve_squid_ink_signal(self, traderObject, get_trend_only = False):
        """
            # Short term mean reversion
            signal1 = df.mid_price.rolling(50).mean()-df.mid_price
            # Longer term mean reversion
            signal2 = (df.mid_price.rolling(150).quantile(0.25)+df.mid_price.rolling(150).quantile(0.75))/2-df.mid_price
            # Short term trend
            signal3 = ((df.mid_price.rolling(12).quantile(0.75)-df.mid_price.rolling(12).quantile(0.25))/12)**2*np.sign(df.mid_price.diff(12))

            bound = df.mid_price.diff().rolling(30).quantile(.85)-df.mid_price.diff().rolling(30).quantile(.15)*4
        """
        if get_trend_only:
            if len(traderObject[Product.SQUID]['mid_price']) >= self.params[Product.SQUID]['trend_timespan_bound']:    
                trend = np.quantile(traderObject[Product.SQUID]['mid_price'][-25:],0.75)-np.quantile(traderObject[Product.SQUID]['mid_price'][-25:],0.25)
                trend *= np.sign(traderObject[Product.SQUID]['mid_price'][-1]-traderObject[Product.SQUID]['mid_price'][-25])
                trend *= 1.5
                return trend
            elif len(traderObject[Product.SQUID]['mid_price']) > 5:
                trend = np.quantile(traderObject[Product.SQUID]['mid_price'],0.75)-np.quantile(traderObject[Product.SQUID]['mid_price'],0.25)
                trend *= np.sign(traderObject[Product.SQUID]['mid_price'][-1]-traderObject[Product.SQUID]['mid_price'][0])
                trend *= 25/len(traderObject[Product.SQUID]['mid_price'])
                trend *= 1.5
                return trend

        if len(traderObject[Product.SQUID]['mid_price']) >= self.params[Product.SQUID]['timespan_bound']:
            mean = np.mean(traderObject[Product.SQUID]['mid_price'][-150:]) - traderObject[Product.SQUID]['mid_price'][-1]
            quantile_mean = (np.quantile(traderObject[Product.SQUID]['mid_price'],0.75)+np.quantile(traderObject[Product.SQUID]['mid_price'],0.25))/2 - traderObject[Product.SQUID]['mid_price'][-1]
            trend = np.quantile(traderObject[Product.SQUID]['mid_price'][-25:],0.75)-np.quantile(traderObject[Product.SQUID]['mid_price'][-25:],0.25)
            trend /= 12
            trend = trend**2
            trend *= np.sign(traderObject[Product.SQUID]['mid_price'][-1]-traderObject[Product.SQUID]['mid_price'][-25])
            difference = [traderObject[Product.SQUID]['mid_price'][-i]-traderObject[Product.SQUID]['mid_price'][-1-i] for i in range(1,31)]
            bound = max((np.quantile(difference,0.85) - np.quantile(difference,0.15))*4,50)
            signal = mean + quantile_mean + trend
            signal /= 10
            signal = signal**2
            if abs(signal) > bound:
                return mean + quantile_mean + trend
            return 0
        return 0
    
    def floor_func(self, num, bounds):
        if abs(num)<bounds:
            return 0
        return num

    # _________________________________________ MACARONS __________________________________________________________-
    def update_macaron_history(self, state : TradingState, traderObject):
        _, bid_tup, ask_tup = self.get_best_ask_best_bid(state, Product.MACARONS)
        mid = (bid_tup[0] + ask_tup[0])/2
        traderObject[Product.MACARONS]['mid_price'].append(mid)
        window_limit = 10
        if len(traderObject[Product.MACARONS]['mid_price']) > window_limit:
            traderObject[Product.MACARONS]['mid_price'].pop(0)
            traderObject['macaron_rel'] = [t/traderObject[Product.MACARONS]['mid_price'][0] for t in traderObject[Product.MACARONS]['mid_price']]
        
        current_sunlight = state.observations.conversionObservations[Product.MACARONS].sunlightIndex
        
        # Record sunlight value with timestamp
        traderObject["sunlight_history"].append({
            "timestamp": state.timestamp,
            "value": current_sunlight
        })

        current_sugar = state.observations.conversionObservations[Product.MACARONS].sugarPrice
        
        # Record sugar value

        traderObject["sugar_history"].append({
            "timestamp": state.timestamp,
            "value": current_sugar
        })
        
        # Keep only recent entries (last 10 timestamps)
        if len(traderObject["sunlight_history"]) > 10:
            traderObject["sunlight_history"] = traderObject["sunlight_history"][-10:]
        if len(traderObject['sugar_history'])>10:
            traderObject['sugar_history'] = traderObject['sugar_history'][-10:]
            traderObject['sugar_rel'] = [t['value']/traderObject['sugar_history'][0]['value'] for t in traderObject['sugar_history']]
        return

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

    def calculate_recent_trend(self, sunlight_history):
        """Calculate if sunlight is trending downward."""
        if len(sunlight_history) < 10:
            return 0
        
        # Get the most recent values
        recent_values = [entry["value"] for entry in sunlight_history[-10:]]
        
        # Simple slope calculation
        x = np.arange(len(recent_values))
        slope, _ = np.polyfit(x, recent_values, 1)
        
        return slope


    def execute_below_csi_strategy(self, state : TradingState, position, current_sunlight, trend, trader_data):
        """Strategy when sunlight is below CSI."""
        orders = []

        # Adjusting fair value based off difference between sugar and macaroon price
        price_adj = 0

        macaroon_prices = trader_data['macaron_rel']
        sugar_prices = trader_data["sugar_rel"]

        x = np.arange(len(macaroon_prices))

        sugar_grad, _ = np.polyfit(x, sugar_prices, 1)
        macaroon_grad, _ = np.polyfit(x, macaroon_prices, 1)

        
        grad_threshold = 0.0
        diff_threshold = 1000
        diff_scaler = 10
        grad_scaler = 1000

        current_sugar = sugar_prices[-1]
        current_macaroon = macaroon_prices[-1]
        price_diff = current_sugar-current_macaroon

        if abs(price_diff)>diff_threshold:
            price_adj+=price_diff*diff_scaler

        grad_diff = sugar_grad-macaroon_grad

        if abs(grad_diff)>grad_threshold:
            price_adj+=grad_diff*grad_scaler

        # Get order depth for macarons
        order_depth = state.order_depths[Product.MACARONS]
        
        # Calculate how far below CSI we are (normalized from 0 to 1)
        # The lower the sunlight compared to CSI, the higher this value
        severity = min(1.0, max(0.0, (self.params[Product.MACARONS]['CSI'] - current_sunlight) / self.params[Product.MACARONS]['CSI']))
        
        # Calculate premium based on severity and trend
        # Higher severity and negative trend means higher premium
        premium_multiplier = 1.0 + severity * 3.0
        if trend < 0:
            # Negative trend means sunlight is decreasing - amplify premium
            premium_multiplier *= (1.0 + min(1.5, abs(trend) * 2))
        
        premium = int(self.params[Product.MACARONS]["LOW_SUNLIGHT_PREMIUM_BASE"] * premium_multiplier)
        
        # Calculate implied prices
        observation = state.observations.conversionObservations[Product.MACARONS]
        implied_bid = observation.bidPrice - observation.exportTariff - observation.transportFees
        implied_ask = observation.askPrice + observation.importTariff + observation.transportFees
        
        # Base fair value from market
        base_fair_value = (implied_bid + implied_ask) / 2
        
        # Add premium to fair value
        fair_value = base_fair_value + premium + price_adj
        
        print(f"Below CSI Strategy - Severity: {severity:.2f}, Premium: {premium}")
        print(f"Base Fair Value: {base_fair_value}, Adjusted Fair Value: {fair_value}")
        
        # Calculate available buy/sell capacity
        buy_capacity = self.LIMIT[Product.MACARONS] - position
        sell_capacity = self.LIMIT[Product.MACARONS] + position
        
        # Take undervalued sell orders
        for price in sorted(order_depth.sell_orders.keys()):
            if price > fair_value - self.params[Product.MACARONS]["TAKE_EDGE"]:
                break
                
            quantity = min(abs(order_depth.sell_orders[price]), buy_capacity)
            if quantity > 0:
                orders.append(Order(Product.MACARONS, price, quantity))
                buy_capacity -= quantity
        
        # If trend is negative, be more aggressive in buying
        bid_edge = self.params[Product.MACARONS]["MIN_EDGE"]
        ask_edge = self.params[Product.MACARONS]["MAX_EDGE"]
        
        if trend < -0.5:  # Strong downward trend in sunlight
            # Be more aggressive with buying, less aggressive with selling
            bid_edge = max(1, self.params[Product.MACARONS]["MIN_EDGE"] - int(abs(trend) * 2))
            ask_edge = self.params[Product.MACARONS]["MAX_EDGE"] + int(premium / 2)
        
        # Place bid at adjusted fair value
        if buy_capacity > 0:
            bid_price = int(fair_value - bid_edge)
            orders.append(Order(Product.MACARONS, bid_price, buy_capacity))
        
        # Place ask at adjusted fair value plus premium
        if sell_capacity > 0:
            ask_price = int(fair_value + ask_edge)
            orders.append(Order(Product.MACARONS, ask_price, -sell_capacity))
        
        # Determine conversions based on position and strategy
        # When below CSI, we want to limit selling and favor buying
        if position > 8:  # If we have a large position
            conversions = max(-self.CONVERSION_LIMIT[Product.MACARONS], -position + 4)
        else:
            # Don't convert if position is reasonable or negative
            conversions = 0
            
        return orders, conversions

    def execute_normal_strategy(self, state : TradingState, position):
        """Strategy when sunlight is above CSI."""
        orders = []
        
        # Get order depth for macarons
        order_depth = state.order_depths[Product.MACARONS]
        
        # Calculate implied prices
        observation = state.observations.conversionObservations[Product.MACARONS]
        implied_bid = observation.bidPrice - observation.exportTariff - observation.transportFees
        implied_ask = observation.askPrice + observation.importTariff + observation.transportFees
        
        # Fair value is the midpoint of implied bid and ask
        fair_value = (implied_bid + implied_ask) / 2
        
        print(f"Normal Strategy - Fair Value: {fair_value}")
        print(f"Implied bid: {implied_bid}, Implied ask: {implied_ask}")
        
        # Calculate available buy/sell capacity
        buy_capacity = self.LIMIT[Product.MACARONS] - position
        sell_capacity = self.LIMIT[Product.MACARONS] + position
        
        # Take arbitrage opportunities
        # Buy if price < implied_bid - edge
        for price in sorted(order_depth.sell_orders.keys()):
            if price >= implied_bid - self.params[Product.MACARONS]["MIN_EDGE"]:
                break
                
            quantity = min(abs(order_depth.sell_orders[price]), buy_capacity)
            if quantity > 0:
                orders.append(Order(Product.MACARONS, price, quantity))
                buy_capacity -= quantity
        
        # Sell if price > implied_ask + edge
        for price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if price <= implied_ask + self.params[Product.MACARONS]["MIN_EDGE"]:
                break
                
            quantity = min(order_depth.buy_orders[price], sell_capacity)
            if quantity > 0:
                orders.append(Order(Product.MACARONS, price, -quantity))
                sell_capacity -= quantity
        
        # Place market making orders
        if buy_capacity > 0:
            bid_price = int(implied_bid - self.params[Product.MACARONS]["MIN_EDGE"])
            orders.append(Order(Product.MACARONS, bid_price, buy_capacity))
        
        if sell_capacity > 0:
            ask_price = int(implied_ask + self.params[Product.MACARONS]["MIN_EDGE"])
            orders.append(Order(Product.MACARONS, ask_price, -sell_capacity))
        
        # In normal market conditions, try to keep position balanced
        # If position is large in either direction, convert to reduce it
        if abs(position) > 8:
            conversions = max(min(-position, self.CONVERSION_LIMIT[Product.MACARONS]), 
                             -self.CONVERSION_LIMIT[Product.MACARONS])
        else:
            conversions = 0
            
        return orders, conversions

    # _______________________________________________ BASKET ORDERS ___________________________________________________________________________________
        # VWAP
    def get_swmid(self, order_depth : OrderDepth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def get_synthetic_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth], basket : Product
    ) -> OrderDepth:
        # Constants
        CROISSANTS_PER_BASKET = BASKET_WEIGHTS[basket][Product.CROISSANTS]
        JAMS_PER_BASKET = BASKET_WEIGHTS[basket][Product.JAMS]
        DJEMBES_PER_BASKET = BASKET_WEIGHTS[basket][Product.DJEMBES]

        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        croissants_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        croissants_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        jams_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        jams_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )
        djembes_best_bid = (
            max(order_depths[Product.DJEMBES].buy_orders.keys())
            if order_depths[Product.DJEMBES].buy_orders
            else 0
        )
        djembes_best_ask = (
            min(order_depths[Product.DJEMBES].sell_orders.keys())
            if order_depths[Product.DJEMBES].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
            croissants_best_bid * CROISSANTS_PER_BASKET
            + jams_best_bid * JAMS_PER_BASKET
            + djembes_best_bid * DJEMBES_PER_BASKET
        )
        implied_ask = (
            croissants_best_ask * CROISSANTS_PER_BASKET
            + jams_best_ask * JAMS_PER_BASKET
            + djembes_best_ask * DJEMBES_PER_BASKET
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            croissants_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[croissants_best_bid]
                // CROISSANTS_PER_BASKET
            )
            jams_bid_volume = (
                order_depths[Product.JAMS].buy_orders[jams_best_bid]
                // JAMS_PER_BASKET
            )
            if DJEMBES_PER_BASKET != 0:
                djembes_bid_volume = (
                    order_depths[Product.DJEMBES].buy_orders[djembes_best_bid]
                    // DJEMBES_PER_BASKET
                )
            else:
                djembes_bid_volume = 10000
            implied_bid_volume = min(
                croissants_bid_volume, jams_bid_volume, djembes_bid_volume
            )
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            croissants_ask_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[croissants_best_ask]
                // CROISSANTS_PER_BASKET
            )
            jams_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[jams_best_ask]
                // JAMS_PER_BASKET
            )
            if DJEMBES_PER_BASKET != 0:
                djembes_ask_volume = (
                    -order_depths[Product.DJEMBES].sell_orders[djembes_best_ask]
                    // DJEMBES_PER_BASKET
                )
            else:
                djembes_ask_volume = 10000
            implied_ask_volume = min(
                croissants_ask_volume, jams_ask_volume, djembes_ask_volume
            )
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def convert_synthetic_basket_orders(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth], basket : Product
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
            Product.DJEMBES: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(
            order_depths, basket
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                croissants_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                jams_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
                djembes_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            elif quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                croissants_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                jams_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
                djembes_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            croissants_order = Order(
                Product.CROISSANTS,
                croissants_price,
                -quantity * BASKET_WEIGHTS[basket][Product.CROISSANTS],
            )
            jams_order = Order(
                Product.JAMS,
                jams_price,
                -quantity * BASKET_WEIGHTS[basket][Product.JAMS],
            )
            djembes_order = Order(
                Product.DJEMBES, 
                djembes_price, 
                -quantity * BASKET_WEIGHTS[basket][Product.DJEMBES]
            )

            # Add the component orders to the respective lists
            component_orders[Product.CROISSANTS].append(croissants_order)
            component_orders[Product.JAMS].append(jams_order)
            component_orders[Product.DJEMBES].append(djembes_order)

        return component_orders

    def execute_spread_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
        basket: Product
    ):

        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[basket]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths, basket)

        if target_position < basket_position:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(basket, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths, basket
            )
            aggregate_orders[basket] = basket_orders
            return aggregate_orders

        else:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(basket, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths, basket
            )
            aggregate_orders[basket] = basket_orders
            return aggregate_orders

    def spread_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position: int,
        spread_data: Dict[str, Any],
    ):
        if product not in order_depths.keys():
            return None

        basket_order_depth = order_depths[product]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths, product)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        
        # Invert spread calculation logic for better profitability
        if product == Product.PICNIC_1:
            spread = basket_swmid - synthetic_swmid  # INVERTED
        elif product == Product.PICNIC_2:
            spread = synthetic_swmid - basket_swmid  # INVERTED
            
        spread_data["spread_history"].append(spread)

        if product == Product.PICNIC_1:
            spread_product = Product.SPREAD_1
        elif product == Product.PICNIC_2:
            spread_product = Product.SPREAD_2
            
        if (
            len(spread_data["spread_history"])
            < self.params[spread_product]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[spread_product]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])

        zscore = (
            spread - self.params[spread_product]["default_spread_mean"]
        ) / spread_std

        thresh = self.params[spread_product]["zscore_threshold"]

        # Invert the trading signals based on the zscore
        if zscore <= -thresh:
            if basket_position != self.params[spread_product]["target_position"]:  # INVERTED
                return self.execute_spread_orders(
                    self.params[spread_product]["target_position"],  # INVERTED
                    basket_position,
                    order_depths,
                    product
                )

        if zscore >= thresh:
            if basket_position != -self.params[spread_product]["target_position"]:  # INVERTED
                return self.execute_spread_orders(
                    -self.params[spread_product]["target_position"],  # INVERTED
                    basket_position,
                    order_depths,
                    product
                )

        spread_data["prev_zscore"] = zscore
        return None
    
    # _____________________________________________________ VOUCHERS __________________________________________________________________________________
    # Single coupon trades.py + volatility_smiles.py + 10500.py
    def get_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate the mid price from an order book"""
        if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        elif len(order_depth.buy_orders) > 0:
            return max(order_depth.buy_orders.keys())
        elif len(order_depth.sell_orders) > 0:
            return min(order_depth.sell_orders.keys())
        return None

    def calculate_fair_voucher_price(self, rock_price, strike, time_to_expiry, volatility):
        """Calculate fair price for voucher using Black-Scholes"""
        return BlackScholes.black_scholes_call(rock_price, strike, time_to_expiry, volatility)

    def calculate_position_adjustment(self, position, product_limit, risk_aversion):
        """Calculate price adjustment based on current position"""
        # Adjust prices based on inventory risk
        position_ratio = position / product_limit
        return position_ratio * risk_aversion * 10

    def make_market_rock(self, order_depth: OrderDepth, position: int) -> List[Order]:
        """Create market making orders for volcanic rock"""
        orders = []
        mid_price = self.get_mid_price(order_depth)
        
        if mid_price is None:
            return orders
            
        # Adjust spread based on position
        position_adjustment = self.calculate_position_adjustment(
            position, 
            self.LIMIT[Product.ROCK], 
            self.DEFAULT_VOUCHER_PARAMS["RISK_AVERSION"]
        )
        
        bid_price = math.floor(mid_price - self.DEFAULT_VOUCHER_PARAMS["ROCK_SPREAD"]/2 - position_adjustment)
        ask_price = math.ceil(mid_price + self.DEFAULT_VOUCHER_PARAMS["ROCK_SPREAD"]/2 - position_adjustment)
        
        remaining_buy_capacity = self.LIMIT[Product.ROCK] - position
        remaining_sell_capacity = self.LIMIT[Product.ROCK] + position
        
        buy_size = min(self.DEFAULT_VOUCHER_PARAMS['ROCK_ORDER_SIZE'], round(remaining_buy_capacity * self.DEFAULT_VOUCHER_PARAMS["MAX_POSITION_FACTOR"]))
        sell_size = min(self.DEFAULT_VOUCHER_PARAMS['ROCK_ORDER_SIZE'], round(remaining_sell_capacity * self.DEFAULT_VOUCHER_PARAMS["MAX_POSITION_FACTOR"]))
        
        if buy_size > 0 and bid_price not in order_depth.buy_orders:
            orders.append(Order(Product.ROCK, bid_price, buy_size))
            
        if sell_size > 0 and ask_price not in order_depth.sell_orders:
            orders.append(Order(Product.ROCK, ask_price, -sell_size))
            
        return orders

    def make_market_voucher(self, product: str, order_depth: OrderDepth, rock_price: float, position: int, volatility: float, tte : float) -> List[Order]:
        """Create market making orders for vouchers"""
        orders = []
        
        if product not in [Product.VOUCHER_9500, Product.VOUCHER_9750, Product.VOUCHER_10000, Product.VOUCHER_10250, Product.VOUCHER_10500]:
            return orders
            
        params = self.params[product]
        
        spread_factor = self.params.get("VOUCHER_SPREAD_FACTOR", self.DEFAULT_VOUCHER_PARAMS["VOUCHER_SPREAD_FACTOR"])
        min_spread = params.get("MIN_VOUCHER_SPREAD", self.DEFAULT_VOUCHER_PARAMS["MIN_VOUCHER_SPREAD"])
        risk_aversion = params.get("RISK_AVERSION", self.DEFAULT_VOUCHER_PARAMS["RISK_AVERSION"])
        order_size = params.get("BASE_ORDER_SIZE", self.DEFAULT_VOUCHER_PARAMS["BASE_ORDER_SIZE"])
        
        fair_price = self.calculate_fair_voucher_price(
            rock_price, 
            params["strike"], 
            tte, 
            volatility
        )
        
        position_adjustment = self.calculate_position_adjustment(position, self.LIMIT[product], risk_aversion)
        vol_adjustment = volatility * 100
        
        spread = max(min_spread, fair_price * spread_factor * (1 + vol_adjustment))
        
        bid_price = math.floor(fair_price - spread/2 - position_adjustment)
        ask_price = math.ceil(fair_price + spread/2 - position_adjustment)
        
        remaining_buy_capacity = self.LIMIT[product] - position
        remaining_sell_capacity = self.LIMIT[product] + position
        
        buy_size = min(order_size, round(remaining_buy_capacity * self.DEFAULT_VOUCHER_PARAMS["MAX_POSITION_FACTOR"]))
        sell_size = min(order_size, round(remaining_sell_capacity * self.DEFAULT_VOUCHER_PARAMS["MAX_POSITION_FACTOR"]))
        
        # Only place orders if we have capacity and avoid placing at existing levels
        if buy_size > 0 and bid_price not in order_depth.buy_orders:
            orders.append(Order(product, bid_price, buy_size))
            
        if sell_size > 0 and ask_price not in order_depth.sell_orders:
            orders.append(Order(product, ask_price, -sell_size))
            
        return orders

    def opportunistic_orders(self, product: str, order_depth: OrderDepth, fair_price: float, position: int) -> List[Order]:
        """Take advantage of mispriced orders in the market"""
        orders = []
        
        remaining_buy_capacity = self.LIMIT[product] - position
        remaining_sell_capacity = self.LIMIT[product] + position
        
        if len(order_depth.buy_orders) > 0 and remaining_sell_capacity > 0:
            best_bid = max(order_depth.buy_orders.keys())
            if best_bid > fair_price * 1.01:  # 1% margin
                quantity = min(order_depth.buy_orders[best_bid], remaining_sell_capacity)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
        
        if len(order_depth.sell_orders) > 0 and remaining_buy_capacity > 0:
            best_ask = min(order_depth.sell_orders.keys())
            if best_ask < fair_price * 0.99:  # 1% margin
                quantity = min(-order_depth.sell_orders[best_ask], remaining_buy_capacity)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
        
        return orders

    def calculate_volatility(self, traderObject: Dict[str, Any], rock_price: float, voucher_price: float, product: str, tte : float) -> float:
        """Calculate implied volatility and update history"""
        if product not in [Product.VOUCHER_9500, Product.VOUCHER_9750, Product.VOUCHER_10000, Product.VOUCHER_10250, Product.VOUCHER_10500]:
            return 0.0001  # Default volatility
        
        # Get product-specific window size
        window_size = self.params[product].get("WINDOW_SIZE", self.DEFAULT_VOUCHER_PARAMS["WINDOW_SIZE"])
            
        if product not in traderObject or "vol_history" not in traderObject[product]:
            traderObject[product] = {"vol_history": []}
            
        params = self.params[product]
        
        try:
            volatility = BlackScholes.implied_volatility(
                voucher_price,
                rock_price,
                params["strike"],
                tte
            )
        except:
            volatility = 0.0001  # Default if calculation fails
            
        # Update volatility history
        traderObject[product]["vol_history"].append(volatility)
        if len(traderObject[product]["vol_history"]) > window_size:
            traderObject[product]["vol_history"].pop(0)
            
        # Use average volatility for stability
        return sum(traderObject[product]["vol_history"]) / len(traderObject[product]["vol_history"])


    # _______________________________________________ RUN FUNCTION _____________________________________________________________________________________
    def run(self, state: TradingState):
        #____________________________________________________ Updating/Initilizing traderObject ________________________________________________________
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        if Product.SQUID not in traderObject:
            traderObject[Product.SQUID] = {
                "mid_price" : [],
                "desired_position" : 0
            }    
        
        # Initialize sunlight history if not exists
        if "sunlight_history" not in traderObject:
            traderObject["sunlight_history"] = []

        if "sugar_history" not in traderObject:
            traderObject["sugar_history"] = []

        if "sugar_rel" not in traderObject:
            traderObject["sugar_rel"] = []

        if "macaron_rel" not in traderObject:
            traderObject["macaron_rel"] = []

        if Product.MACARONS not in traderObject:
            traderObject[Product.MACARONS] = {
                "mid_price" : [],
            }
            
        if Product.SPREAD_1 not in traderObject:
            traderObject[Product.SPREAD_1] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }
        
        if Product.SPREAD_2 not in traderObject:
            traderObject[Product.SPREAD_2] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }


        # Store current sunlight index
        if Product.MACARONS not in state.observations.conversionObservations:
            # No macaron data available
            return {}, 0, jsonpickle.encode(traderObject)
        
        self.update_macaron_history(state, traderObject)
        
        #____________________________________________________ Orders ________________________________________________________
        
        result = {}
        conversions = 0

        # Generate orders for Rainforest Resin if market exists
        if Product.RESIN in state.order_depths:
            rainforest_resin_orders = self.rainforest_resin_orders(
                state.order_depths[Product.RESIN], 
                self.params[Product.RESIN]["rainforest_resin_fair_value"], 
                self.params[Product.RESIN]["rainforest_resin_width"], 
                state.position.get(Product.RESIN,0), 
                self.LIMIT[Product.RESIN]
            )
            result[Product.RESIN] = rainforest_resin_orders

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
        if Product.SQUID in state.order_depths:
            ink_orders = self.ink_orders(
                state.order_depths[Product.SQUID], 
                self.params[Product.SQUID]['timespan'], 
                self.params[Product.SQUID]['take_width'],
                self.params[Product.SQUID]['make_width'],
                state.position.get(Product.SQUID, 0), 
                self.LIMIT[Product.SQUID],
                traderObject,
                state
            )
            result[Product.SQUID] = ink_orders
        
        # _____________________________________________________________MACARONS_________________________________________________________________________
        
        current_sunlight = state.observations.conversionObservations[Product.MACARONS].sunlightIndex
        current_sugar = state.observations.conversionObservations[Product.MACARONS].sugarPrice
        
        # Get current position
        position = state.position.get(Product.MACARONS, 0)
        
        # Check if we're below CSI
        below_csi = current_sunlight < self.params[Product.MACARONS]['CSI']
        
        # Calculate the trend to determine if sunlight is expected to stay low
        recent_trend = self.calculate_recent_trend(traderObject["sunlight_history"])
        
        if Product.MACARONS in state.order_depths:
            # Log critical information to verify strategy is working
            print(f"CSI: {self.params[Product.MACARONS]["CSI"]}, Current Sunlight: {current_sunlight}")
            print(f"Below CSI: {below_csi}, Trend: {recent_trend}")
            
            if below_csi:
                orders, conversions = self.execute_below_csi_strategy(
                    state,
                    position,
                    current_sunlight,
                    recent_trend
                )
            else:
                orders, conversions = self.execute_normal_strategy(
                    state,
                    position
                )
            
            result[Product.MACARONS] = orders
                
        # _____________________________________________________________ BASKETS _____________________________________________________________
                
        picnic_1_position = (
            state.position[Product.PICNIC_1]
            if Product.PICNIC_1 in state.position
            else 0
        )
        
        picnic_2_position = (
            state.position[Product.PICNIC_2]
            if Product.PICNIC_2 in state.position
            else 0
        )
        
        spread_orders = self.spread_orders(
            state.order_depths,
            Product.PICNIC_1,
            picnic_1_position,
            traderObject[Product.SPREAD_1],
        )
        
        spread_orders_2 = self.spread_orders(
            state.order_depths,
            Product.PICNIC_2,
            picnic_2_position,
            traderObject[Product.SPREAD_2]
        )
        
        if (spread_orders != None) and (spread_orders_2 != None):
            croissant_final_q = spread_orders[Product.CROISSANTS][0].quantity + spread_orders_2[Product.CROISSANTS][0].quantity
            croissant_position = (state.position[Product.CROISSANTS] if Product.CROISSANTS in state.position else 0)
            croissant_final_q = max(min(croissant_final_q,self.LIMIT[Product.CROISSANTS]-croissant_position),-self.LIMIT[Product.CROISSANTS]-croissant_position)
            
            
            jams_final_q = spread_orders[Product.JAMS][0].quantity + spread_orders_2[Product.JAMS][0].quantity
            jams_position = (state.position[Product.JAMS] if Product.JAMS in state.position else 0)
            jams_final_q = max(min(jams_final_q,self.LIMIT[Product.JAMS])-jams_position,-self.LIMIT[Product.JAMS]-jams_position)

            if croissant_final_q < 0:
                # Switched buy/sell logic
                croissants_price = max(state.order_depths[Product.CROISSANTS].buy_orders.keys())
                result[Product.CROISSANTS] = [Order(Product.CROISSANTS, croissants_price, croissant_final_q)]
            else:
                # Switched buy/sell logic
                croissants_price = min(state.order_depths[Product.CROISSANTS].sell_orders.keys())
                result[Product.CROISSANTS] = [Order(Product.CROISSANTS, croissants_price, croissant_final_q)]

            if jams_final_q < 0:
                # Switched buy/sell logic
                jams_price = max(state.order_depths[Product.JAMS].buy_orders.keys())
                result[Product.JAMS] = [Order(Product.JAMS, jams_price, jams_final_q)]
            else:
                # Switched buy/sell logic
                jams_price = min(state.order_depths[Product.JAMS].sell_orders.keys())
                result[Product.JAMS] = [Order(Product.JAMS, jams_price, jams_final_q)]

            result[Product.DJEMBES] = spread_orders[Product.DJEMBES]
            result[Product.PICNIC_1] = spread_orders[Product.PICNIC_1]
            result[Product.PICNIC_2] = spread_orders_2[Product.PICNIC_2]
        elif spread_orders != None:
            result[Product.CROISSANTS] = spread_orders[Product.CROISSANTS]
            result[Product.JAMS] = spread_orders[Product.JAMS]
            result[Product.DJEMBES] = spread_orders[Product.DJEMBES]
            result[Product.PICNIC_1] = spread_orders[Product.PICNIC_1]
        elif spread_orders_2 != None:
            result[Product.CROISSANTS] = spread_orders_2[Product.CROISSANTS]
            result[Product.JAMS] = spread_orders_2[Product.JAMS]
            result[Product.PICNIC_2] = spread_orders_2[Product.PICNIC_2]
        
        # _____________________________________________________________ VOUCHERS _____________________________________________________________
        
        # Get volcanic rock price
        rock_price = None
        if Product.ROCK in state.order_depths:
            rock_order_depth = state.order_depths[Product.ROCK]
            rock_price = self.get_mid_price(rock_order_depth)
        
        if rock_price is None:
            # Can't trade without a rock price
            return result, 0, jsonpickle.encode(traderObject)
        
        # Time to expiry
        tte = 3 - ((state.timestamp) / 1000000)
        
        # Calculate volatilities and deltas for each voucher
        volatilities = {}
        deltas = {}
        for product in [Product.VOUCHER_9500, Product.VOUCHER_9750, Product.VOUCHER_10000, Product.VOUCHER_10500]:
            if product in state.order_depths:
                voucher_order_depth = state.order_depths[product]
                voucher_price = self.get_mid_price(voucher_order_depth)
                
                if voucher_price:
                    volatility = self.calculate_volatility(traderObject, rock_price, voucher_price, product, tte)
                    volatilities[product] = volatility
                    
                    # Calculate delta for hedging
                    delta = BlackScholes.delta(
                        rock_price,
                        self.params[product]["strike"],
                        tte,
                        volatility
                    )
                    deltas[product] = delta
                    
        # Get positions
        rock_position = state.position.get(Product.ROCK, 0)
        
        # Voucher 10k
        voucher_orders = {}

        for product in [Product.VOUCHER_9500, Product.VOUCHER_9750, Product.VOUCHER_10000, Product.VOUCHER_10500]:
            if product in state.order_depths and product in volatilities:
                # Get current position and volatility
                position = state.position.get(product, 0)
                volatility = volatilities[product]
                        
                # First take any opportunistic trades
                fair_price = self.calculate_fair_voucher_price(
                    rock_price,
                    self.params[product]["strike"],
                    tte,
                    volatility
                )
                        
                orders = self.opportunistic_orders(
                    product, 
                    state.order_depths[product], 
                    fair_price, 
                    position,
                )
                        
                # Then add market making orders
                orders.extend(self.make_market_voucher(
                    product, 
                    state.order_depths[product], 
                    rock_price, 
                    position, 
                    volatility,
                    tte
                ))
                        
                if orders:
                    voucher_orders[product] = orders
            result[product] = orders        

        #____________________________________________________ Submit ________________________________________________________
        
        traderData = jsonpickle.encode(traderObject)

        # Conversions (purpose not clear from context, possibly a trading feature)

        return result, conversions, traderData