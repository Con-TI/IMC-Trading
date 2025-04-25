from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Tuple, Dict
import jsonpickle
import numpy as np


class Product:
    MACARONS = "MAGNIFICENT_MACARONS"


class Trader:
    def __init__(self):
        # Position limits
        self.POSITION_LIMIT = {Product.MACARONS: 75}
        self.CONVERSION_LIMIT = {Product.MACARONS: 10}
        
        # Critical Sunlight Index threshold
        self.CSI = 49
        
        # Trading parameters
        self.DEFAULT_FAIR_VALUE = 10000  # Default fair value
        self.LOW_SUNLIGHT_PREMIUM_BASE = 300  # Base premium to add when sunlight is below CSI
        self.MIN_EDGE = 2  # Minimum edge for market making
        self.MAX_EDGE = 10  # Maximum edge for market making
        self.TAKE_EDGE = 5  # Edge for taking orders

    def run(self, state: TradingState):
        # Initialize trader data
        trader_data = {}
        if state.traderData and state.traderData != "":
            trader_data = jsonpickle.decode(state.traderData)
        
        # Initialize sunlight history if not exists
        if "sunlight_history" not in trader_data:
            trader_data["sunlight_history"] = []
        
        # Store current sunlight index
        if Product.MACARONS not in state.observations.conversionObservations:
            # No macaron data available
            return {}, 0, jsonpickle.encode(trader_data)
            
        current_sunlight = state.observations.conversionObservations[Product.MACARONS].sunlightIndex
        
        # Record sunlight value with timestamp
        trader_data["sunlight_history"].append({
            "timestamp": state.timestamp,
            "value": current_sunlight
        })
        
        # Keep only recent entries (last 10 timestamps)
        if len(trader_data["sunlight_history"]) > 10:
            trader_data["sunlight_history"] = trader_data["sunlight_history"][-10:]
        
        # Get current position
        position = state.position.get(Product.MACARONS, 0)
        
        # Check if we're below CSI
        below_csi = current_sunlight < self.CSI
        
        # Calculate the trend to determine if sunlight is expected to stay low
        recent_trend = self.calculate_recent_trend(trader_data["sunlight_history"])
        
        # Make trading decisions based on sunlight conditions
        result = {}
        conversions = 0
        
        if Product.MACARONS in state.order_depths:
            # Log critical information to verify strategy is working
            print(f"CSI: {self.CSI}, Current Sunlight: {current_sunlight}")
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
        
        # Save trader data
        trader_data_encoded = jsonpickle.encode(trader_data)
        
        return result, conversions, trader_data_encoded

    def calculate_recent_trend(self, sunlight_history):
        """Calculate if sunlight is trending downward."""
        if len(sunlight_history) < 3:
            return 0
        
        # Get the most recent values
        recent_values = [entry["value"] for entry in sunlight_history[-3:]]
        
        # Simple slope calculation
        x = np.arange(len(recent_values))
        slope, _ = np.polyfit(x, recent_values, 1)
        
        return slope

    def execute_below_csi_strategy(self, state, position, current_sunlight, trend):
        """Strategy when sunlight is below CSI."""
        orders = []
        
        # Get order depth for macarons
        order_depth = state.order_depths[Product.MACARONS]
        
        # Calculate how far below CSI we are (normalized from 0 to 1)
        # The lower the sunlight compared to CSI, the higher this value
        severity = min(1.0, max(0.0, (self.CSI - current_sunlight) / self.CSI))
        
        # Calculate premium based on severity and trend
        # Higher severity and negative trend means higher premium
        premium_multiplier = 1.0 + severity * 3.0
        if trend < 0:
            # Negative trend means sunlight is decreasing - amplify premium
            premium_multiplier *= (1.0 + min(1.5, abs(trend) * 2))
        
        premium = int(self.LOW_SUNLIGHT_PREMIUM_BASE * premium_multiplier)
        
        # Calculate implied prices
        observation = state.observations.conversionObservations[Product.MACARONS]
        implied_bid = observation.bidPrice - observation.exportTariff - observation.transportFees
        implied_ask = observation.askPrice + observation.importTariff + observation.transportFees
        
        # Base fair value from market
        base_fair_value = (implied_bid + implied_ask) / 2
        
        # Add premium to fair value
        fair_value = base_fair_value + premium
        
        print(f"Below CSI Strategy - Severity: {severity:.2f}, Premium: {premium}")
        print(f"Base Fair Value: {base_fair_value}, Adjusted Fair Value: {fair_value}")
        
        # Calculate available buy/sell capacity
        buy_capacity = self.POSITION_LIMIT[Product.MACARONS] - position
        sell_capacity = self.POSITION_LIMIT[Product.MACARONS] + position
        
        # Take undervalued sell orders
        for price in sorted(order_depth.sell_orders.keys()):
            if price > fair_value - self.TAKE_EDGE:
                break
                
            quantity = min(abs(order_depth.sell_orders[price]), buy_capacity)
            if quantity > 0:
                orders.append(Order(Product.MACARONS, price, quantity))
                buy_capacity -= quantity
        
        # If trend is negative, be more aggressive in buying
        bid_edge = self.MIN_EDGE
        ask_edge = self.MAX_EDGE
        
        if trend < -0.5:  # Strong downward trend in sunlight
            # Be more aggressive with buying, less aggressive with selling
            bid_edge = max(1, self.MIN_EDGE - int(abs(trend) * 2))
            ask_edge = self.MAX_EDGE + int(premium / 2)
        
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
        if position > 50:  # If we have a large position
            conversions = max(-self.CONVERSION_LIMIT[Product.MACARONS], -position + 40)
        else:
            # Don't convert if position is reasonable or negative
            conversions = 0
            
        return orders, conversions

    def execute_normal_strategy(self, state, position):
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
        buy_capacity = self.POSITION_LIMIT[Product.MACARONS] - position
        sell_capacity = self.POSITION_LIMIT[Product.MACARONS] + position
        
        # Take arbitrage opportunities
        # Buy if price < implied_bid - edge
        for price in sorted(order_depth.sell_orders.keys()):
            if price >= implied_bid - self.MIN_EDGE:
                break
                
            quantity = min(abs(order_depth.sell_orders[price]), buy_capacity)
            if quantity > 0:
                orders.append(Order(Product.MACARONS, price, quantity))
                buy_capacity -= quantity
        
        # Sell if price > implied_ask + edge
        for price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if price <= implied_ask + self.MIN_EDGE:
                break
                
            quantity = min(order_depth.buy_orders[price], sell_capacity)
            if quantity > 0:
                orders.append(Order(Product.MACARONS, price, -quantity))
                sell_capacity -= quantity
        
        # Place market making orders
        if buy_capacity > 0:
            bid_price = int(implied_bid - self.MIN_EDGE)
            orders.append(Order(Product.MACARONS, bid_price, buy_capacity))
        
        if sell_capacity > 0:
            ask_price = int(implied_ask + self.MIN_EDGE)
            orders.append(Order(Product.MACARONS, ask_price, -sell_capacity))
        
        # In normal market conditions, try to keep position balanced
        # If position is large in either direction, convert to reduce it
        if abs(position) > 40:
            conversions = max(min(-position, self.CONVERSION_LIMIT[Product.MACARONS]), 
                             -self.CONVERSION_LIMIT[Product.MACARONS])
        else:
            conversions = 0
            
        return orders, conversions