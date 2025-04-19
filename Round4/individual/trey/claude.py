from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any, Tuple
import jsonpickle
import numpy as np
import math
import pandas as pd
from collections import deque

class Product:
    PICNIC_1 = "PICNIC_BASKET1"
    PICNIC_2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC_1 = "SYNTHETIC_1"
    SYNTHETIC_2 = "SYNTHETIC_2"

class SignalType:
    MEAN_REVERSION = "MEAN_REVERSION"
    TREND_FOLLOWING = "TREND_FOLLOWING"
    HYBRID = "HYBRID"

# Enhanced parameters with more sophisticated coefficients
PARAMS = {
    Product.PICNIC_1: {
        "history_length": 50,           # Increased history for better statistical significance
        "gamma": 0.85,                  # Risk aversion parameter
        "k": 0.45,                      # Market impact parameter
        "reversion_coefficient": 1.25,  # Mean reversion strength
        "inventory_coeff": 15.5,        # Higher inventory penalty
        "theta_coeff": 1.2e4,           # Signal coefficient
        "trend_coeff": 65000,           # Trend strength
        "volatility_coeff": 7e7,        # Volatility impact on spreads
        "predictive_shift_coeff": 2.5,  # Forward-looking adjustment
        "default_delt_shift": 0.2,      # Base spread adjustment
        "regime_delt_shift": 1.5,       # Regime-specific spread adjustment
        "spread_mean": 0,               # Expected equilibrium spread
        "signal_type": SignalType.HYBRID, # Dynamic signal selection
        "crossover_threshold": 0.35,    # Threshold for signal switching
        "jump_detection_z": 3.2,        # Z-score for detecting price jumps
        "kalman_process_noise": 1e-5,   # Kalman filter process noise
        "kalman_measurement_noise": 1e-3, # Kalman measurement noise
        "arb_threshold": 0.15           # Threshold for arbitrage opportunities
    },
    Product.SYNTHETIC_1: {
        "history_length": 50,
        "arb_threshold": 0.15          
    },
    Product.PICNIC_2: {
        "history_length": 50,
        "gamma": 0.9,
        "k": 0.4,
        "reversion_coefficient": 2.2,
        "inventory_coeff": 120,
        "theta_coeff": 5e2,
        "trend_coeff": 60000,
        "volatility_coeff": 4e7,
        "predictive_shift_coeff": 0.5,
        "default_delt_shift": 0.25,
        "regime_delt_shift": 0.5,
        "spread_mean": 0,
        "signal_type": SignalType.HYBRID,
        "crossover_threshold": 0.4,
        "jump_detection_z": 3.0,
        "kalman_process_noise": 2e-5,
        "kalman_measurement_noise": 2e-3,
        "arb_threshold": 0.18
    },
    Product.SYNTHETIC_2: {
        "history_length": 50,
        "arb_threshold": 0.18
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
    }
}

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_value=0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_value
        self.estimation_variance = 1.0
        
    def update(self, measurement):
        # Prediction update
        prediction = self.estimate
        prediction_variance = self.estimation_variance + self.process_variance
        
        # Measurement update
        kalman_gain = prediction_variance / (prediction_variance + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimation_variance = (1 - kalman_gain) * prediction_variance
        
        return self.estimate


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        # Position limits with incremental scaling
        self.LIMIT = {
            Product.PICNIC_1: 60,
            Product.PICNIC_2: 100,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
        }
        
        # Initialize Kalman filters for each product
        self.kalman_filters = {}
        for product in [Product.PICNIC_1, Product.PICNIC_2]:
            self.kalman_filters[product] = KalmanFilter(
                process_variance=self.params[product]["kalman_process_noise"],
                measurement_variance=self.params[product]["kalman_measurement_noise"]
            )
            
    def detect_regime(self, prices, window=20):
        """Detect market regime based on price patterns"""
        if len(prices) < window:
            return SignalType.HYBRID
        
        # Calculate short and long moving averages
        short_ma = np.mean(prices[-10:])
        long_ma = np.mean(prices[-window:])
        
        # Calculate volatility
        volatility = np.std(prices[-window:])
        
        # Calculate momentum
        momentum = prices[-1] - prices[-5]
        
        # Calculate mean reversion score (negative autocorrelation)
        returns = np.diff(prices[-window:])
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        else:
            autocorr = 0
            
        # Determine regime
        if abs(autocorr) < 0.15:  # Low autocorrelation suggests random walk
            return SignalType.HYBRID
        elif autocorr < -0.2:     # Negative autocorrelation suggests mean reversion
            return SignalType.MEAN_REVERSION
        elif autocorr > 0.2:      # Positive autocorrelation suggests trend following
            return SignalType.TREND_FOLLOWING
        else:
            return SignalType.HYBRID
            
    def get_synthetic_basket_depth(self, state: TradingState, basket: Product):
        """Calculate synthetic ETF order book from component assets"""
        best_asks = {}
        best_bids = {}
        
        basket_bid = 0
        basket_ask = float('inf')
        
        # Calculate quantities at different price levels
        all_bid_quantities = {}
        all_ask_quantities = {}
        
        # Get all components for this basket
        for product in BASKET_WEIGHTS[basket].keys():
            if product not in state.order_depths:
                return None  # Cannot calculate if missing component
                
            order_depth = state.order_depths[product]
            buy_orders = order_depth.buy_orders
            sell_orders = order_depth.sell_orders
            
            # Calculate weighted component values 
            weight = BASKET_WEIGHTS[basket][product]
            
            # Find best bid and ask with volumes
            if buy_orders:
                bid_price = max(buy_orders.keys())
                bid_vol = buy_orders[bid_price]
                best_bids[product] = (bid_price, abs(bid_vol))
                basket_bid += bid_price * weight
            else:
                return None  # No valid bid price
                
            if sell_orders:
                ask_price = min(sell_orders.keys())
                ask_vol = sell_orders[ask_price]
                best_asks[product] = (ask_price, abs(ask_vol))
                basket_ask = min(basket_ask, ask_price * weight)
            else:
                return None  # No valid ask price
        
        # Calculate maximum executable quantities
        basket_bid_vol = min([p_q[1]//BASKET_WEIGHTS[basket][product] for product, p_q in best_bids.items()])
        basket_ask_vol = min([p_q[1]//BASKET_WEIGHTS[basket][product] for product, p_q in best_asks.items()])
        
        synthetic_od = OrderDepth()
        synthetic_od.buy_orders = {basket_bid: basket_bid_vol}
        synthetic_od.sell_orders = {basket_ask: -basket_ask_vol}  # Negative for sell orders
        
        return synthetic_od

    def get_order_book_imbalance(self, order_depth, price_levels=3):
        """Calculate order book imbalance ratio to predict short-term price movements"""
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        
        if not buy_orders or not sell_orders:
            return 0.0
            
        # Get sorted price levels
        bid_prices = sorted(buy_orders.keys(), reverse=True)[:price_levels]
        ask_prices = sorted(sell_orders.keys())[:price_levels]
        
        # Calculate total volume at top levels
        bid_volume = sum(abs(buy_orders[price]) for price in bid_prices if price in buy_orders)
        ask_volume = sum(abs(sell_orders[price]) for price in ask_prices if price in sell_orders)
        
        # Calculate imbalance ratio (-1 to 1)
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
            
        return (bid_volume - ask_volume) / total_volume

    def get_best_ask_best_bid(self, state: TradingState, product: Product, weighted=True):
        """Get best bid and ask prices with improved volume weighting"""
        if product not in state.order_depths:
            return None, None, None
            
        order_depth = state.order_depths[product]
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        
        if not buy_orders or not sell_orders:
            return None, None, None
            
        bid_prices = sorted(buy_orders.keys(), reverse=True)
        ask_prices = sorted(sell_orders.keys())
        
        if not bid_prices or not ask_prices:
            return None, None, None
            
        best_bid = bid_prices[0]
        best_ask = ask_prices[0]
        
        # Find highest volume bid and ask if weighted is True
        if weighted:
            best_bid_vol = abs(buy_orders[best_bid])
            best_ask_vol = abs(sell_orders[best_ask])
            
            for price in bid_prices:
                if abs(buy_orders[price]) > best_bid_vol:
                    best_bid = price
                    best_bid_vol = abs(buy_orders[price])
                    
            for price in ask_prices:
                if abs(sell_orders[price]) > best_ask_vol:
                    best_ask = price
                    best_ask_vol = abs(sell_orders[price])
        else:
            best_bid_vol = abs(buy_orders[best_bid])
            best_ask_vol = abs(sell_orders[best_ask])
        
        mid = (best_ask + best_bid) / 2
        
        return mid, (best_bid, best_bid_vol), (best_ask, best_ask_vol)

    def update_history(self, state: TradingState, product: Product, trader_object, synth=False):
        """Update price history with improved filtering"""
        if synth:
            if product == Product.SYNTHETIC_1:
                synth_od = self.get_synthetic_basket_depth(state, Product.PICNIC_1)
                if synth_od is None:
                    return
                    
                bid_price = list(synth_od.buy_orders.keys())[0]
                ask_price = list(synth_od.sell_orders.keys())[0]
                synth_mid = (bid_price + ask_price) / 2
                
                # Apply Kalman filtering to synthetic price
                filtered_mid = self.kalman_filters[Product.PICNIC_1].update(synth_mid)
                
                trader_object[product]['mid_price'].append(filtered_mid)
                if 'raw_price' not in trader_object[product]:
                    trader_object[product]['raw_price'] = []
                trader_object[product]['raw_price'].append(synth_mid)
                
            elif product == Product.SYNTHETIC_2:
                synth_od = self.get_synthetic_basket_depth(state, Product.PICNIC_2)
                if synth_od is None:
                    return
                    
                bid_price = list(synth_od.buy_orders.keys())[0]
                ask_price = list(synth_od.sell_orders.keys())[0]
                synth_mid = (bid_price + ask_price) / 2
                
                # Apply Kalman filtering
                filtered_mid = self.kalman_filters[Product.PICNIC_2].update(synth_mid)
                
                trader_object[product]['mid_price'].append(filtered_mid)
                if 'raw_price' not in trader_object[product]:
                    trader_object[product]['raw_price'] = []
                trader_object[product]['raw_price'].append(synth_mid)
                
        else:
            mid, bid_tup, ask_tup = self.get_best_ask_best_bid(state, product)
            if mid is not None:
                # Apply Kalman filtering to actual price
                if product in self.kalman_filters:
                    filtered_mid = self.kalman_filters[product].update(mid)
                else:
                    filtered_mid = mid
                
                trader_object[product]['mid_price'].append(filtered_mid)
                if 'raw_price' not in trader_object[product]:
                    trader_object[product]['raw_price'] = []
                trader_object[product]['raw_price'].append(mid)
                
                # Calculate and store order book imbalance
                if 'imbalance' not in trader_object[product]:
                    trader_object[product]['imbalance'] = []
                imbalance = self.get_order_book_imbalance(state.order_depths[product])
                trader_object[product]['imbalance'].append(imbalance)
                
        # Maintain history length
        window_limit = self.params[product]['history_length']
        for key in ['mid_price', 'raw_price', 'imbalance']:
            if key in trader_object[product] and len(trader_object[product][key]) > window_limit:
                trader_object[product][key] = trader_object[product][key][-window_limit:]
                
    def get_buy_sell_limits(self, state: TradingState, product: Product):
        """Calculate dynamic position limits based on market conditions"""
        position = state.position.get(product, 0)
        base_limit = self.LIMIT[product]
        
        # Dynamic limit adjustment based on volatility if we have enough history
        if product in state.traderData and len(state.traderData[product].get('mid_price', [])) > 10:
            prices = state.traderData[product]['mid_price']
            volatility = np.std(prices[-10:]) / np.mean(prices[-10:])
            
            # Reduce limits during high volatility
            vol_factor = max(0.5, 1.0 - volatility * 5)
            adjusted_limit = int(base_limit * vol_factor)
            
            # Never go below half the base limit
            limit = max(adjusted_limit, base_limit // 2)
        else:
            limit = base_limit
            
        return (limit - position, -limit - position)

    def calculate_volatility(self, prices, method='garch'):
        """Enhanced volatility estimation using GARCH-like approach"""
        if len(prices) < 4:
            return 0.0001  # Default low volatility when insufficient data
            
        returns = np.diff(np.log(prices))
        
        if method == 'simple':
            return np.std(returns) ** 2
            
        elif method == 'ewma':
            # Exponentially weighted moving average
            lambda_param = 0.94
            weights = np.array([(1 - lambda_param) * lambda_param ** i for i in range(len(returns))])
            weights = weights / np.sum(weights)
            return np.sum(weights * returns ** 2)
            
        elif method == 'garch':
            # Simple GARCH(1,1)-like implementation
            omega = 0.000001
            alpha = 0.05
            beta = 0.9
            
            sigma2 = np.var(returns)
            for r in returns:
                sigma2 = omega + alpha * (r ** 2) + beta * sigma2
                
            return sigma2
            
        return np.std(returns) ** 2  # Fallback

    def retrieve_reservation_quotes(self, state: TradingState, trader_object, product: Product):
        """Calculate optimal bid/ask quotes using enhanced models"""
        params = self.params[product]
        gamma, k = params['gamma'], params['k']
        
        # Get mid price and current market conditions
        mid, bid_tup, ask_tup = self.get_best_ask_best_bid(state, product)
        if mid is None:
            return None, None
            
        # Calculate volatility with improved estimator
        if len(trader_object[product]['mid_price']) > 3:
            volatility = self.calculate_volatility(trader_object[product]['mid_price'], method='garch')
        else:
            volatility = 0.0001  # Default when insufficient data
            
        # Detect current market regime
        current_regime = self.detect_regime(trader_object[product]['mid_price'])
        trader_object[product]['current_regime'] = current_regime
        
        # Calculate signals based on current regime
        theta = self.retrieve_signal_skew(state, trader_object, product)
        trend = self.retrieve_trend_sig(state, trader_object, product)
        
        # Dynamic weight allocation between mean reversion and trend signals
        if current_regime == SignalType.MEAN_REVERSION:
            alpha = 0.8  # 80% weight on mean reversion
        elif current_regime == SignalType.TREND_FOLLOWING:
            alpha = 0.2  # 20% weight on mean reversion, 80% on trend
        else:  # Hybrid
            alpha = 0.5  # Equal weighting
            
        # Include order book imbalance for short-term prediction
        imbalance = 0
        if 'imbalance' in trader_object[product] and trader_object[product]['imbalance']:
            imbalance = trader_object[product]['imbalance'][-1] * 0.5  # Scale factor
            
        # Combined directional signal
        combined_signal = alpha * theta + (1 - alpha) * trend + imbalance
        
        # Apply signal to mid price based on volatility conditions
        linear_factor = max(1 - volatility * params['volatility_coeff'], 0)
        
        # More aggressive in low volatility, more conservative in high volatility
        adjusted_mid = mid + combined_signal * params['predictive_shift_coeff'] * linear_factor
        
        # Inventory management with time decay
        q = state.position.get(product, 0)
        time_remaining = max(1000000 - state.timestamp, 1)  # Avoid division by zero
        time_weight = math.sqrt(1000000 / time_remaining)  # Square root scaling
        
        # Optimal spread calculation based on Avellaneda-Stoikov with inventory and time effects
        inventory_risk = params['inventory_coeff'] * gamma * volatility * q * time_weight
        
        # Base half-spread calculation
        default_delt = 1 / gamma * math.log(1 + gamma / k) + params['default_delt_shift']
        
        # Adjust spread in high volatility or strong directional markets
        if abs(trend) > params['crossover_threshold'] or volatility > 0.0005:
            default_delt += params['regime_delt_shift']
            
        # Calculate asymmetric bid/ask spreads based on inventory
        bid_spread = default_delt + inventory_risk
        ask_spread = default_delt - inventory_risk
        
        # Final quote prices
        bid_price = adjusted_mid - bid_spread
        ask_price = adjusted_mid + ask_spread
        
        return bid_price, ask_price

    def retrieve_signal_skew(self, state: TradingState, trader_object, product: Product):
        """Calculate cointegration-based mean reversion signal"""
        if product not in [Product.PICNIC_1, Product.PICNIC_2]:
            return 0
            
        synthetic_product = Product.SYNTHETIC_1 if product == Product.PICNIC_1 else Product.SYNTHETIC_2
        
        # Need history for both actual and synthetic
        if (len(trader_object[product].get('mid_price', [])) < 5 or 
            len(trader_object[synthetic_product].get('mid_price', [])) < 5):
            return 0
            
        # Get latest prices
        actual_price = trader_object[product]['mid_price'][-1]
        synthetic_price = trader_object[synthetic_product]['mid_price'][-1]
        
        # Calculate log spread
        spread = np.log(actual_price) - np.log(synthetic_price)
        
        # Calculate z-score of spread using recent history
        if len(trader_object[product]['mid_price']) > 10:
            actual_hist = trader_object[product]['mid_price'][-20:]
            synth_hist = trader_object[synthetic_product]['mid_price'][-20:]
            
            hist_spreads = [np.log(a) - np.log(s) for a, s in zip(actual_hist, synth_hist)]
            mean_spread = np.mean(hist_spreads)
            std_spread = np.std(hist_spreads) or 0.001  # Avoid division by zero
            
            z_score = (spread - mean_spread) / std_spread
        else:
            # Not enough history for z-score
            z_score = spread
            
        # Calculate mean reversion signal with time decay
        time_remaining = max(1000000 - state.timestamp, 1)
        mean_reversion_strength = self.params[product]['reversion_coefficient']
        
        # Non-linear signal transformation to handle outliers
        if abs(z_score) > self.params[product]['jump_detection_z']:
            # Strong mean reversion for extreme deviations
            signal = -np.sign(z_score) * self.params[product]['theta_coeff'] * math.sqrt(abs(z_score))
        else:
            # Linear mean reversion for normal deviations
            signal = -z_score * self.params[product]['theta_coeff']
            
        # Apply time decay - stronger signals near end of trading
        time_factor = 1 - math.exp(-time_remaining * mean_reversion_strength / 1000000)
        return signal * time_factor

    def retrieve_trend_sig(self, state: TradingState, trader_object, product: Product):
        """Enhanced trend detection with multiple timeframes"""
        if len(trader_object[product].get('mid_price', [])) < 10:
            return 0
            
        prices = trader_object[product]['mid_price']
        
        # Multi-timeframe momentum
        short_trend = np.log(prices[-1]) - np.log(prices[-5]) if len(prices) >= 5 else 0
        medium_trend = np.log(prices[-1]) - np.log(prices[-10]) if len(prices) >= 10 else 0
        long_trend = np.log(prices[-1]) - np.log(prices[0]) if len(prices) >= 20 else 0
        
        # Calculate volatility scaled by trend strength coefficient
        volatility = self.calculate_volatility(prices)
        vol_scaling = self.params[product]['trend_coeff'] * volatility
        
        # Weighted combination of multiple timeframes
        combined_trend = (0.5 * short_trend + 0.3 * medium_trend + 0.2 * long_trend) / vol_scaling
        
        return combined_trend

    def arbitrage_orders(self, state: TradingState, trader_object, product: Product):
        """Execute statistical arbitrage between ETF and components"""
        if product not in [Product.PICNIC_1, Product.PICNIC_2]:
            return []
            
        synthetic_product = Product.SYNTHETIC_1 if product == Product.PICNIC_1 else Product.SYNTHETIC_2
        
        # Check if we have both markets available
        mid, bid_tup, ask_tup = self.get_best_ask_best_bid(state, product)
        if mid is None:
            return []
            
        synthetic_od = self.get_synthetic_basket_depth(state, product)
        if synthetic_od is None:
            return []
            
        synthetic_bid = list(synthetic_od.buy_orders.keys())[0]
        synthetic_ask = list(synthetic_od.sell_orders.keys())[0]
        synthetic_bid_vol = synthetic_od.buy_orders[synthetic_bid]
        synthetic_ask_vol = abs(synthetic_od.sell_orders[synthetic_ask])
        
        orders = []
        arb_threshold = self.params[product]['arb_threshold']
        
        # ETF cheaper than components - buy ETF, sell components
        if bid_tup[0] < synthetic_ask * (1 - arb_threshold):
            # Buy ETF at bid price
            etf_buy_qty = min(abs(bid_tup[1]), 5)
            max_position = self.LIMIT[product] - state.position.get(product, 0)
            etf_buy_qty = min(etf_buy_qty, max_position)
            
            if etf_buy_qty > 0:
                orders.append(Order(product, bid_tup[0], etf_buy_qty))
                
                # Sell components
                for component, weight in BASKET_WEIGHTS[product].items():
                    comp_mid, comp_bid, comp_ask = self.get_best_ask_best_bid(state, component)
                    if comp_mid is None:
                        continue
                        
                    # Sell component at bid price
                    comp_sell_qty = -etf_buy_qty * weight
                    comp_position_limit = -self.LIMIT.get(component, 100) - state.position.get(component, 0)
                    comp_sell_qty = max(comp_sell_qty, comp_position_limit)
                    
                    if comp_sell_qty < 0:
                        orders.append(Order(component, comp_bid[0], comp_sell_qty))
                        
        # ETF more expensive than components - sell ETF, buy components
        elif ask_tup[0] > synthetic_bid * (1 + arb_threshold):
            # Sell ETF at ask price
            etf_sell_qty = -min(abs(ask_tup[1]), 5)
            min_position = -self.LIMIT[product] - state.position.get(product, 0)
            etf_sell_qty = max(etf_sell_qty, min_position)
            
            if etf_sell_qty < 0:
                orders.append(Order(product, ask_tup[0], etf_sell_qty))
                
                # Buy components
                for component, weight in BASKET_WEIGHTS[product].items():
                    comp_mid, comp_bid, comp_ask = self.get_best_ask_best_bid(state, component)
                    if comp_mid is None:
                        continue
                        
                    # Buy component at ask price
                    comp_buy_qty = -etf_sell_qty * weight
                    comp_position_limit = self.LIMIT.get(component, 100) - state.position.get(component, 0)
                    comp_buy_qty = min(comp_buy_qty, comp_position_limit)
                    
                    if comp_buy_qty > 0:
                        orders.append(Order(component, comp_ask[0], comp_buy_qty))
        
        return orders

    def intra_market_arb(self, state: TradingState, trader_object, product: Product):
        """Exploit inefficiencies within the same market"""
        if product not in state.order_depths:
            return []
            
        od = state.order_depths[product]
        buys = od.buy_orders
        sells = od.sell_orders
        
        if not buys or not sells:
            return []
            
        # Get current fair price estimate
        if len(trader_object[product].get('mid_price', [])) > 0:
            fair_price = trader_object[product]['mid_price'][-1]
        else:
            bid_prices = sorted(buys.keys(), reverse=True)
            ask_prices = sorted(sells.keys())
            fair_price = (bid_prices[0] + ask_prices[0]) / 2
            
        orders = []
        
        # Find mispriced bids (too high)
        to_sell = [(price, abs(quantity)) for price, quantity in buys.items() if price > fair_price * 1.01]
        to_sell.sort(key=lambda x: x[0], reverse=True)  # Sort by highest price first
        
        # Find mispriced asks (too low)
        to_buy = [(price, abs(quantity)) for price, quantity in sells.items() if price < fair_price * 0.99]
        to_buy.sort(key=lambda x: x[0])  # Sort by lowest price first
        
        # Execute arbitrage with position limits
        position = state.position.get(product, 0)
        buy_limit, sell_limit = self.get_buy_sell_limits(state, product)
        
        # Execute buy orders
        remaining_buy = buy_limit
        for price, quantity in to_buy:
            if remaining_buy <= 0:
                break
                
            qty = min(quantity, remaining_buy, 10)  # Cap individual orders at 10
            if qty > 0:
                orders.append(Order(product, price, qty))
                remaining_buy -= qty
                
        # Execute sell orders
        remaining_sell = abs(sell_limit)
        for price, quantity in to_sell:
            if remaining_sell <= 0:
                break
                
            qty = min(quantity, remaining_sell, 10)  # Cap individual orders at 10
            if qty > 0:
                orders.append(Order(product, price, -qty))
                remaining_sell -= qty
                
        return orders

    def run(self, state: TradingState):
        """Main trading logic with improved execution"""
        # Initialize or load trader object
        trader_object = {}
        if state.traderData and state.traderData != "":
            trader_object = jsonpickle.decode(state.traderData)
        
        # Initialize products in trader_object
        for product in [Product.PICNIC_1, Product.PICNIC_2, Product.SYNTHETIC_1, Product.SYNTHETIC_2, 
                    Product.CROISSANTS, Product.DJEMBES, Product.JAMS]:
            if product not in trader_object:
                trader_object[product] = {'mid_price': [], 'current_regime': SignalType.HYBRID}
        
        # Update price history for all available products
        for product in [Product.PICNIC_1, Product.PICNIC_2, Product.CROISSANTS, Product.DJEMBES, Product.JAMS]:
            if product in state.order_depths:
                self.update_history(state, product, trader_object)
        
        # Update synthetic basket prices
        for synth_product in [Product.SYNTHETIC_1, Product.SYNTHETIC_2]:
            self.update_history(state, synth_product, trader_object, synth=True)
        
        # Initialize result container for orders
        result = {}
        conversions = 0  # No conversions in this implementation
        
        # Process each product
        for product in [Product.PICNIC_1, Product.PICNIC_2, Product.CROISSANTS, Product.DJEMBES, Product.JAMS]:
            if product not in state.order_depths:
                continue
            
            orders = []
            
            # Apply different strategies based on product type
            if product in [Product.PICNIC_1, Product.PICNIC_2]:
                # ETF trading with market making and arbitrage
                
                # First try arbitrage between ETF and components
                arb_orders = self.arbitrage_orders(state, trader_object, product)
                orders.extend(arb_orders)
                
                # Then apply market making with optimal quotes
                bid_price, ask_price = self.retrieve_reservation_quotes(state, trader_object, product)
                
                if bid_price is not None and ask_price is not None:
                    # Apply position limits
                    buy_limit, sell_limit = self.get_buy_sell_limits(state, product)
                    
                    # Execute market making orders with size limits
                    max_order_size = 15  # Limit individual order size
                    
                    # Place bid if we can buy more
                    if buy_limit > 0:
                        bid_size = min(buy_limit, max_order_size)
                        orders.append(Order(product, bid_price, bid_size))
                    
                    # Place ask if we can sell more
                    if sell_limit < 0:
                        ask_size = max(sell_limit, -max_order_size)
                        orders.append(Order(product, ask_price, ask_size))
            else:
                # Component asset trading (simpler strategies)
                # Look for intra-market arbitrage opportunities
                arb_orders = self.intra_market_arb(state, trader_object, product)
                orders.extend(arb_orders)
                
                # Simple market making for components with wider spreads
                mid, bid_tup, ask_tup = self.get_best_ask_best_bid(state, product)
                if mid is not None:
                    # Apply position limits
                    buy_limit, sell_limit = self.get_buy_sell_limits(state, product)
                    
                    # Use wider spreads for less liquid components
                    spread_factor = 0.01  # 1% spread
                    
                    # Place bid if we can buy more
                    if buy_limit > 0:
                        bid_price = mid * (1 - spread_factor)
                        bid_size = min(buy_limit, 10)
                        orders.append(Order(product, bid_price, bid_size))
                    
                    # Place ask if we can sell more
                    if sell_limit < 0:
                        ask_price = mid * (1 + spread_factor)
                        ask_size = max(sell_limit, -10)
                        orders.append(Order(product, ask_price, ask_size))
            
            # Add orders to result if we have any
            if orders:
                result[product] = orders
        
        # Encode updated trader object
        trader_data = jsonpickle.encode(trader_object)
        
        return result, conversions, trader_data