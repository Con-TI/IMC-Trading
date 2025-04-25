from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
from statistics import NormalDist


class Product:
    ROCK = "VOLCANIC_ROCK"
    VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOUCHER_9750 = 'VOLCANIC_ROCK_VOUCHER_9750'
    VOUCHER_10000 = 'VOLCANIC_ROCK_VOUCHER_10000'
    VOUCHER_10250 = 'VOLCANIC_ROCK_VOUCHER_10250'
    VOUCHER_10500 = 'VOLCANIC_ROCK_VOUCHER_10500'


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
    def __init__(self):

        self.LIMIT = {
            Product.ROCK: 1,
            Product.VOUCHER_9500: 1,
            Product.VOUCHER_9750: 1,
            Product.VOUCHER_10000: 1,
            Product.VOUCHER_10250: 1,
            Product.VOUCHER_10500: 200,
        }
        
        '''
        # Market making parameters
        self.ROCK_SPREAD = 2  # Spread in rock prices
        self.VOUCHER_SPREAD_FACTOR = 1.5  # Voucher spread as multiple of fair value
        self.MIN_VOUCHER_SPREAD = 5  # Minimum spread for vouchers
        
        # Risk management parameters
        self.RISK_AVERSION = 0.05  # Increases spread as position increases
        self.WINDOW_SIZE = 10  # Window size for volatility estimation
        
        '''

        # Market making parameters
        self.ROCK_SPREAD = .001  # Spread in rock prices # prev 1: 5295, 5314 with .5. 5744 with .1, 5925 with .01, 5976 with 0.001.
        self.VOUCHER_SPREAD_FACTOR = .000 # 7359, with 0.01, 9906 with 0.001, 10.2k with 0.0001
        self.MIN_VOUCHER_SPREAD = .0 # 5878, with 1, 5976 with .1. 
        # voucher spread factor, min spread - profit
        #9500 on 0.0001, 0.1 - 2.64k
        #9500 on 0.001, 0.1 - 1.6k
        #9500 on 0.0001, 0.01 - 2.7k

        #9750 on 0.0001, 0.01 - 7.7k
        #9750 on 0.0001, 0.1 - 8.1k
        #9750 on 0.0001, 0.15 - 7.5k

        #10250 on 0.0001, 0.1 - -800
        #10250 on 0.001, 0.1 - -600
        #10250 on 0.001, 0.5 - -300

        #10500 on 0.0001, 0.1 - 0?
        #10500 on 0.001, 0.1 - 0?

        
        self.RISK_AVERSION = 0.05  # Increases spread as position increases
        self.WINDOW_SIZE = 40  # Window size for volatility estimation
        

        self.VOUCHER_PARAMS = {
            Product.VOUCHER_9500: {"strike": 9500, "time_to_expiry": 3},
            Product.VOUCHER_9750: {"strike": 9750, "time_to_expiry": 3},
            Product.VOUCHER_10000: {"strike": 10000, "time_to_expiry": 3},
            Product.VOUCHER_10250: {"strike": 10250, "time_to_expiry": 3},
            Product.VOUCHER_10500: {"strike": 10500, "time_to_expiry": 3},
        }
        
        self.BASE_ORDER_SIZE = 100
        self.ROCK_ORDER_SIZE = 200
        self.MAX_POSITION_FACTOR = 1

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

    def calculate_position_adjustment(self, position, product_limit):
        """Calculate price adjustment based on current position"""
        # Adjust prices based on inventory risk
        position_ratio = position / product_limit
        return position_ratio * self.RISK_AVERSION * 10

    def make_market_rock(self, order_depth: OrderDepth, position: int) -> List[Order]:
        """Create market making orders for volcanic rock"""
        orders = []
        mid_price = self.get_mid_price(order_depth)
        
        if mid_price is None:
            return orders
            
        # Adjust spread based on position
        position_adjustment = self.calculate_position_adjustment(position, self.LIMIT[Product.ROCK])
        
        # Calculate bid and ask prices
        bid_price = math.floor(mid_price - self.ROCK_SPREAD/2 - position_adjustment)
        ask_price = math.ceil(mid_price + self.ROCK_SPREAD/2 - position_adjustment)
        
        # Calculate order sizes - reduce size as we approach position limits
        remaining_buy_capacity = self.LIMIT[Product.ROCK] - position
        remaining_sell_capacity = self.LIMIT[Product.ROCK] + position
        
        buy_size = min(self.ROCK_ORDER_SIZE, round(remaining_buy_capacity * self.MAX_POSITION_FACTOR))
        sell_size = min(self.ROCK_ORDER_SIZE, round(remaining_sell_capacity * self.MAX_POSITION_FACTOR))
        
        # Only place orders if we have capacity and avoid placing at existing levels
        if buy_size > 0 and bid_price not in order_depth.buy_orders:
            orders.append(Order(Product.ROCK, bid_price, buy_size))
            
        if sell_size > 0 and ask_price not in order_depth.sell_orders:
            orders.append(Order(Product.ROCK, ask_price, -sell_size))
            
        return orders

    def make_market_voucher(self, product: str, order_depth: OrderDepth, rock_price: float, position: int, volatility: float) -> List[Order]:
        """Create market making orders for vouchers"""
        orders = []
        
        if product not in self.VOUCHER_PARAMS:
            return orders
            
        params = self.VOUCHER_PARAMS[product]
        
        # Calculate theoretical price
        fair_price = self.calculate_fair_voucher_price(
            rock_price, 
            params["strike"], 
            params["time_to_expiry"], 
            volatility
        )
        print(fair_price)
        # Adjust spread based on position and volatility
        position_adjustment = self.calculate_position_adjustment(position, self.LIMIT[product])
        vol_adjustment = volatility * 100  # Higher volatility = wider spread
        
        # Calculate spread
        spread = max(self.MIN_VOUCHER_SPREAD, fair_price * self.VOUCHER_SPREAD_FACTOR * (1 + vol_adjustment))
        print(spread)
        # Calculate bid and ask prices
        bid_price = math.floor(fair_price - spread/2 - position_adjustment)
        ask_price = math.ceil(fair_price + spread/2 - position_adjustment)
        
        # Calculate order sizes - reduce size as we approach position limits
        remaining_buy_capacity = self.LIMIT[product] - position
        remaining_sell_capacity = self.LIMIT[product] + position
        
        order_size = self.BASE_ORDER_SIZE
        buy_size = min(order_size, round(remaining_buy_capacity * self.MAX_POSITION_FACTOR))
        sell_size = min(order_size, round(remaining_sell_capacity * self.MAX_POSITION_FACTOR))
        
        # Only place orders if we have capacity and avoid placing at existing levels
        if buy_size > 0 and bid_price not in order_depth.buy_orders:
            orders.append(Order(product, bid_price, buy_size))
            
        if sell_size > 0 and ask_price not in order_depth.sell_orders:
            orders.append(Order(product, ask_price, -sell_size))
            
        return orders

    def opportunistic_orders(self, product: str, order_depth: OrderDepth, fair_price: float, position: int) -> List[Order]:
        """Take advantage of mispriced orders in the market"""
        orders = []
        
        # Only trade if we have capacity
        remaining_buy_capacity = self.LIMIT[product] - position
        remaining_sell_capacity = self.LIMIT[product] + position
        
        # Look for selling opportunities (if market bid is above our fair price)
        if len(order_depth.buy_orders) > 0 and remaining_sell_capacity > 0:
            best_bid = max(order_depth.buy_orders.keys())
            if best_bid > fair_price * 1.01:  # 1% margin
                quantity = min(order_depth.buy_orders[best_bid], remaining_sell_capacity)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
        
        # Look for buying opportunities (if market ask is below our fair price)
        if len(order_depth.sell_orders) > 0 and remaining_buy_capacity > 0:
            best_ask = min(order_depth.sell_orders.keys())
            if best_ask < fair_price * 0.99:  # 1% margin
                quantity = min(-order_depth.sell_orders[best_ask], remaining_buy_capacity)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
        print(orders)
        return orders

    def delta_hedge_position(self, rock_orders: List[Order], voucher_positions: Dict[str, int], rock_position: int, deltas: Dict[str, float]) -> List[Order]:
        """Hedge the overall voucher position by adjusting rock position"""
        # Calculate target rock position based on voucher positions and deltas
        target_rock_position = 0
        for product, position in voucher_positions.items():
            if product in deltas:
                target_rock_position -= position * deltas[product]
        
        # Calculate how many more rocks we need to buy/sell
        current_rock_orders = sum(order.quantity for order in rock_orders) if rock_orders else 0
        remaining_adjustment = int(target_rock_position - (rock_position + current_rock_orders))
        
        if abs(remaining_adjustment) < 5:  # Don't hedge small positions
            return []
            
        return [] # [Order(Product.ROCK, 0, remaining_adjustment)]  # Price will be filled in with best available

    def calculate_volatility(self, trader_data: Dict[str, Any], rock_price: float, voucher_price: float, product: str) -> float:
        """Calculate implied volatility and update history"""
        if product not in self.VOUCHER_PARAMS:
            return 0.0001  # Default volatility
            
        if product not in trader_data or "vol_history" not in trader_data[product]:
            trader_data[product] = {"vol_history": []}
            
        params = self.VOUCHER_PARAMS[product]
        
        try:
            volatility = BlackScholes.implied_volatility(
                voucher_price,
                rock_price,
                params["strike"],
                params["time_to_expiry"]
            )
        except:
            volatility = 0.0001  # Default if calculation fails
            
        # Update volatility history
        trader_data[product]["vol_history"].append(volatility)
        if len(trader_data[product]["vol_history"]) > self.WINDOW_SIZE:
            trader_data[product]["vol_history"].pop(0)
            
        # Use average volatility for stability
        return sum(trader_data[product]["vol_history"]) / len(trader_data[product]["vol_history"])

    def run(self, state: TradingState):
        # Initialize the result dict and trader data
        result = {}
        trader_data = {}

        tte = (3 - (state.timestamp / 1000000)) #need to change to 1000000 when submitting 
        
        # Load trader data from previous iteration
        if state.traderData and state.traderData != "":
            try:
                trader_data = jsonpickle.decode(state.traderData)
            except:
                trader_data = {}
        
        # Get volcanic rock price
        rock_price = None
        if Product.ROCK in state.order_depths:
            rock_order_depth = state.order_depths[Product.ROCK]
            rock_price = self.get_mid_price(rock_order_depth)
        
        if rock_price is None:
            # Can't trade without a rock price
            return result, 0, jsonpickle.encode(trader_data)
        
        # Get positions
        rock_position = state.position.get(Product.ROCK, 0)
        voucher_positions = {}
        for product in self.VOUCHER_PARAMS:
            voucher_positions[product] = state.position.get(product, 0)
        
        # Calculate volatilities and deltas for each voucher
        volatilities = {}
        deltas = {}
        for product in self.VOUCHER_PARAMS:
            if product in state.order_depths:
                voucher_order_depth = state.order_depths[product]
                voucher_price = self.get_mid_price(voucher_order_depth)
                
                if voucher_price:
                    volatility = self.calculate_volatility(trader_data, rock_price, voucher_price, product)
                    volatilities[product] = volatility
                    
                    # Calculate delta for hedging
                    delta = BlackScholes.delta(
                        rock_price,
                        self.VOUCHER_PARAMS[product]["strike"],
                        tte,
                        volatility
                    )
                    deltas[product] = delta
        
        # Start with market making for volcanic rock
        rock_orders = []
        # if Product.ROCK in state.order_depths:
        #     # First take any opportunistic trades
        #     fair_rock_price = rock_price  # In a real market, might have some external fair value
        #     rock_orders.extend(self.opportunistic_orders(
        #         Product.ROCK, 
        #         state.order_depths[Product.ROCK], 
        #         fair_rock_price, 
        #         rock_position
        #     ))
            
        #     # Then add market making orders
        #     rock_orders.extend(self.make_market_rock(
        #         state.order_depths[Product.ROCK], 
        #         rock_position
        #     ))
        
        # Now handle voucher products
        voucher_orders = {}
        for product in self.VOUCHER_PARAMS:
            if product == Product.VOUCHER_10500:
                if product in state.order_depths and product in volatilities:
                    # Get current position and volatility
                    position = voucher_positions.get(product, 0)
                    volatility = volatilities[product]
                    
                    # First take any opportunistic trades
                    fair_price = self.calculate_fair_voucher_price(
                        rock_price,
                        self.VOUCHER_PARAMS[product]["strike"],
                        tte,
                        volatility
                    )

                    print(fair_price)
                    
                    orders = self.opportunistic_orders(
                        product, 
                        state.order_depths[product], 
                        fair_price, 
                        position
                    )
                    
                    # Then add market making orders
                    orders.extend(self.make_market_voucher(
                        product, 
                        state.order_depths[product], 
                        rock_price, 
                        position, 
                        volatility
                    ))
                    
                    if orders:
                        voucher_orders[product] = orders
        
        # Delta hedge rock position based on voucher positions
        hedge_orders = self.delta_hedge_position(
            rock_orders, 
            voucher_positions, 
            rock_position, 
            deltas
        )
        
        # Integrate hedge orders into rock orders with proper pricing
        if hedge_orders:
            for order in hedge_orders:
                if order.quantity > 0 and Product.ROCK in state.order_depths:
                    # Buying rock - use best ask price
                    order_depth = state.order_depths[Product.ROCK]
                    if len(order_depth.sell_orders) > 0:
                        best_ask = min(order_depth.sell_orders.keys())
                        order.price = best_ask
                        rock_orders.append(order)
                elif order.quantity < 0 and Product.ROCK in state.order_depths:
                    # Selling rock - use best bid price
                    order_depth = state.order_depths[Product.ROCK]
                    if len(order_depth.buy_orders) > 0:
                        best_bid = max(order_depth.buy_orders.keys())
                        order.price = best_bid
                        rock_orders.append(order)
        
        # Add all orders to the result
        if rock_orders:
            result[Product.ROCK] = rock_orders
            
        for product, orders in voucher_orders.items():
            result[product] = orders
        
        return result, 0, jsonpickle.encode(trader_data)