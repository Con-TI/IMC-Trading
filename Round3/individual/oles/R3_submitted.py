from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math


class Product:
    ROCK = "VOLCANIC_ROCK"
    VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOUCHER_9750 = 'VOLCANIC_ROCK_VOUCHER_9750'
    VOUCHER_10000 = 'VOLCANIC_ROCK_VOUCHER_10000'
    VOUCHER_10250 = 'VOLCANIC_ROCK_VOUCHER_10250'
    VOUCHER_10500 = 'VOLCANIC_ROCK_VOUCHER_10500'


# Updated parameters for all vouchers with more aggressive thresholds
PARAMS = {
    Product.VOUCHER_9500: {
        "mean_volatility": 0.009323628758797987,
        "threshold": 0.007365006856641405,
        "strike": 9500,
        "starting_time_to_expiry": 5,
        "std_window": 20,  # Increased window size for better mean reversion signals
        "zscore_threshold": 1,  # More aggressive threshold
        "position_limit": 200,
    },
    Product.VOUCHER_9750: {
        "mean_volatility": 0.01116066088348751,
        "threshold": 0.003556422560280647,
        "strike": 9750,
        "starting_time_to_expiry": 5,
        "std_window": 20,
        "zscore_threshold": 1,
        "position_limit": 200,
    },
    Product.VOUCHER_10000: {
        "mean_volatility": 0.01048148171413482,
        "threshold": 0.0008021323602629242,
        "strike": 10000,
        "starting_time_to_expiry": 5,
        "std_window": 20,
        "zscore_threshold": 1,
        "position_limit": 200,
    },
    Product.VOUCHER_10250: {
        "mean_volatility": 0.00959143413175075,
        "threshold": 0.0005408212015501674,
        "strike": 10250,
        "starting_time_to_expiry": 5,
        "std_window": 20,
        "zscore_threshold": 1,
        "position_limit": 200,
    },
    Product.VOUCHER_10500: {
        "mean_volatility": 0.00947690768351717,
        "threshold": 0.000421320253747233,
        "strike": 10500,
        "starting_time_to_expiry": 5,
        "std_window": 20,
        "zscore_threshold": 1,
        "position_limit": 200,
    },
}


from math import log, sqrt, exp
from statistics import NormalDist


class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        if time_to_expiry <= 0:
            return max(0, spot - strike)
        
        d1 = (
            log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        if time_to_expiry <= 0:
            return max(0, strike - spot)
            
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        if time_to_expiry <= 0:
            return 1.0 if spot > strike else 0.0
            
        d1 = (
            log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        if time_to_expiry <= 0:
            return 0.0
            
        d1 = (
            log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        if time_to_expiry <= 0:
            return 0.0
            
        d1 = (
            log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return spot * sqrt(time_to_expiry) * NormalDist().pdf(d1) / 100

    @staticmethod
    def theta(spot, strike, time_to_expiry, volatility):
        """Calculate time decay (theta) of an option"""
        if time_to_expiry <= 0:
            return 0.0
            
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        
        theta = -spot * NormalDist().pdf(d1) * volatility / (2 * sqrt(time_to_expiry))
        return theta / 365.0  # Daily theta

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=100, tolerance=1e-4
    ):
        if time_to_expiry <= 0:
            return 0.0
            
        # Intrinsic value check
        intrinsic = max(0, spot - strike)
        if call_price <= intrinsic:
            return 0.00001  # Return minimum volatility
            
        # Initialize boundaries
        low_vol = 0.00001
        high_vol = 5.0  # Increased upper bound for extreme scenarios
        
        # Initial guess using midpoint
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
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        # Position limits for each product
        self.LIMIT = {
            Product.ROCK: 400,
            Product.VOUCHER_9500: 200,
            Product.VOUCHER_9750: 200,
            Product.VOUCHER_10000: 200,
            Product.VOUCHER_10250: 200,
            Product.VOUCHER_10500: 200,
        }
        
        # All voucher products
        self.VOUCHERS = [
            Product.VOUCHER_9500,
            Product.VOUCHER_9750,
            Product.VOUCHER_10000,
            Product.VOUCHER_10250,
            Product.VOUCHER_10500,
        ]
        
        # Market making parameters
        self.SPREAD_FACTOR = 0.4  # Spread as percentage of volatility
        self.VOL_EDGE_FACTOR = 2.0  # Percentage of vol edge to capture

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate the mid price from an order book"""
        if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        return None
        
    def get_best_bid(self, order_depth: OrderDepth) -> tuple:
        """Get best bid price and volume"""
        if len(order_depth.buy_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            return best_bid, order_depth.buy_orders[best_bid]
        return None, 0
        
    def get_best_ask(self, order_depth: OrderDepth) -> tuple:
        """Get best ask price and volume"""
        if len(order_depth.sell_orders) > 0:
            best_ask = min(order_depth.sell_orders.keys())
            return best_ask, order_depth.sell_orders[best_ask]
        return None, 0

    def get_voucher_mid_price(self, voucher_order_depth: OrderDepth, voucher_data: Dict[str, Any]) -> float:
        """Get mid price for voucher, fallback to previous price if necessary"""
        if len(voucher_order_depth.buy_orders) > 0 and len(voucher_order_depth.sell_orders) > 0:
            best_bid = max(voucher_order_depth.buy_orders.keys())
            best_ask = min(voucher_order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            voucher_data["prev_price"] = mid_price
            return mid_price
        else:
            return voucher_data.get("prev_price", 0)

    def delta_hedge_rock_position(
        self,
        rock_order_depth: OrderDepth,
        total_delta_exposure: float,
        rock_position: int,
        aggressive=False
    ) -> List[Order]:
        """
        Delta hedge the overall position in all vouchers by creating orders in ROCK.
        If aggressive=True, use market orders to immediately rebalance.
        """
        target_rock_position = -int(total_delta_exposure)
        hedge_quantity = target_rock_position - rock_position
        
        orders: List[Order] = []
        
        if hedge_quantity == 0:
            return orders
            
        if hedge_quantity > 0:
            # Buy ROCK
            best_ask, ask_vol = self.get_best_ask(rock_order_depth)
            if best_ask is not None:
                quantity = min(
                    abs(hedge_quantity), 
                    abs(ask_vol),
                    self.LIMIT[Product.ROCK] - rock_position
                )
                if quantity > 0:
                    # Add a small increment to price if aggressive to ensure execution
                    # Convert price to integer to avoid format errors
                    price = int(best_ask) if not aggressive else int(best_ask * 1.001)
                    orders.append(Order(Product.ROCK, price, -quantity))
                    
        elif hedge_quantity < 0:
            # Sell ROCK
            best_bid, bid_vol = self.get_best_bid(rock_order_depth)
            if best_bid is not None:
                quantity = min(
                    abs(hedge_quantity), 
                    abs(bid_vol),
                    self.LIMIT[Product.ROCK] + rock_position
                )
                if quantity > 0:
                    # Reduce price slightly if aggressive to ensure execution
                    # Convert price to integer to avoid format errors
                    price = int(best_bid) if not aggressive else int(best_bid * 0.999)
                    orders.append(Order(Product.ROCK, price, quantity))

        return orders

    def calculate_historical_volatility(self, prices: List[float]) -> float:
        """Calculate historical volatility from a list of prices"""
        if len(prices) < 2:
            return 0.01  # Default volatility
            
        # Calculate log returns
        log_returns = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]
        
        # Calculate volatility (standard deviation of log returns)
        vol = np.std(log_returns) if len(log_returns) > 0 else 0.01
        
        # Annualize the volatility (assuming daily data)
        annual_vol = vol * math.sqrt(252)  # Standard trading days in a year
        
        return annual_vol

    def find_volatility_arbitrage_opportunities(
        self,
        rock_mid_price: float,
        rock_prices: List[float],  # Historical prices for HV calculation
        voucher_order_depths: Dict[str, OrderDepth],
        time_to_expiry: float,
        trader_data: Dict[str, Any]
    ) -> List[Dict]:
        """
        Find volatility arbitrage opportunities across all vouchers.
        """
        opportunities = []
        
        # Calculate historical volatility if we have enough data
        historical_vol = self.calculate_historical_volatility(rock_prices) if len(rock_prices) > 5 else None
        
        # For each voucher
        for voucher in self.VOUCHERS:
            if voucher not in voucher_order_depths:
                continue
                
            voucher_depth = voucher_order_depths[voucher]
            strike = self.params[voucher]["strike"]
            
            # Check if we have bids and asks
            best_bid, bid_vol = self.get_best_bid(voucher_depth)
            best_ask, ask_vol = self.get_best_ask(voucher_depth)
            
            if best_bid is None or best_ask is None:
                continue
            
            # Initialize voucher data if not exists
            if voucher not in trader_data:
                trader_data[voucher] = {
                    "prev_price": 0,
                    "past_vol": [],
                    "historical_prices": []
                }
                
            # Calculate implied volatilities
            bid_iv = BlackScholes.implied_volatility(
                best_bid, rock_mid_price, strike, time_to_expiry
            )
            ask_iv = BlackScholes.implied_volatility(
                best_ask, rock_mid_price, strike, time_to_expiry
            )
            
            # Use the mean volatility from parameters as our baseline
            baseline_vol = self.params[voucher]["mean_volatility"]
            
            # If we have historical volatility, use a blend of historical and mean vol
            if historical_vol:
                baseline_vol = (baseline_vol + historical_vol) / 2
            
            # Calculate fair price using our baseline volatility
            fair_price = BlackScholes.black_scholes_call(
                rock_mid_price, strike, time_to_expiry, baseline_vol
            )
            
            # Store the latest implied volatility for mean reversion
            mid_iv = (bid_iv + ask_iv) / 2
            trader_data[voucher]["past_vol"].append(mid_iv)
            
            # Keep historical window at proper size
            window_size = self.params[voucher]["std_window"]
            if len(trader_data[voucher]["past_vol"]) > window_size:
                trader_data[voucher]["past_vol"] = trader_data[voucher]["past_vol"][-window_size:]
            
            # Calculate z-score if we have enough data
            if len(trader_data[voucher]["past_vol"]) >= 3:  # Need at least 3 data points for meaningful std
                vol_mean = np.mean(trader_data[voucher]["past_vol"])
                vol_std = np.std(trader_data[voucher]["past_vol"])
                
                # Prevent division by zero
                if vol_std > 0:
                    vol_zscore = (mid_iv - vol_mean) / vol_std
                else:
                    vol_zscore = (mid_iv - baseline_vol) / (baseline_vol * 0.1)  # Fallback
            else:
                # If not enough data, compare to baseline
                vol_zscore = (mid_iv - baseline_vol) / (baseline_vol * 0.1)
            
            # Calculate option greeks
            delta = BlackScholes.delta(rock_mid_price, strike, time_to_expiry, mid_iv)
            gamma = BlackScholes.gamma(rock_mid_price, strike, time_to_expiry, mid_iv)
            vega = BlackScholes.vega(rock_mid_price, strike, time_to_expiry, mid_iv)
            theta = BlackScholes.theta(rock_mid_price, strike, time_to_expiry, mid_iv)
            
            # Calculate mispricing edges
            bid_edge = best_bid - fair_price
            ask_edge = fair_price - best_ask
            
            # Calculate vol edge - difference between implied and baseline vol
            bid_vol_edge = bid_iv - baseline_vol
            ask_vol_edge = baseline_vol - ask_iv
            
            opportunity = {
                "voucher": voucher,
                "fair_price": fair_price,
                "best_bid": best_bid,
                "best_ask": best_ask, 
                "bid_vol": bid_vol,
                "ask_vol": ask_vol,
                "bid_price_edge": bid_edge,
                "ask_price_edge": ask_edge,
                "bid_vol_edge": bid_vol_edge,
                "ask_vol_edge": ask_vol_edge,
                "delta": delta,
                "gamma": gamma,
                "vega": vega,
                "theta": theta,
                "vol_zscore": vol_zscore,
                "baseline_vol": baseline_vol,
                "time_to_expiry": time_to_expiry
            }
            
            opportunities.append(opportunity)
            
        return opportunities

    def generate_volatility_arbitrage_orders(
        self,
        opportunities: List[Dict],
        positions: Dict[str, int],
    ) -> Dict[str, List[Order]]:
        """
        Generate orders for vouchers based on inverted volatility arbitrage opportunities
        We sell options when IV is low (negative z-score) and buy when IV is high (positive z-score)
        This is the opposite of traditional vol trading (buying low IV, selling high IV)
        """
        orders = {}
        
        # Process each opportunity to find tradable volatility edges
        for opp in opportunities:
            voucher = opp["voucher"]
            position = positions.get(voucher, 0)
            zscore_threshold = self.params[voucher]["zscore_threshold"]
            
            # Initialize orders list for this voucher if needed
            if voucher not in orders:
                orders[voucher] = []
            
            spread = self.SPREAD_FACTOR * opp["baseline_vol"] * self.params[voucher]["strike"]

            # INVERTED: Sell when volatility is low (IV < historical/mean vol)
            if opp["vol_zscore"] <= -zscore_threshold and opp["ask_vol_edge"] > 0:
                # Check if we're above negative position limit
                if position > -self.LIMIT[voucher]:
                    # Size based on edge magnitude and z-score
                    edge_factor = min(1.0, abs(opp["vol_zscore"]) / (2 * zscore_threshold))*self.VOL_EDGE_FACTOR/100
                    desired_quantity = int(self.LIMIT[voucher] * edge_factor)
                    
                    # Calculate how many we can sell
                    quantity = min(
                        desired_quantity,
                        self.LIMIT[voucher] + position,
                        abs(opp["bid_vol"])  # Use bid volume since we're selling
                    )
                    
                    if quantity > 0:
                        price = int(opp["fair_price"] + spread)  # Adjusted to sell above fair price
                        orders[voucher].append(Order(voucher, price, quantity))  # Positive quantity for sell
            
            # INVERTED: Buy when volatility is high (IV > historical/mean vol)
            elif opp["vol_zscore"] >= zscore_threshold and opp["bid_vol_edge"] > 0:
                # Check if we're below position limit
                if position < self.LIMIT[voucher]:
                    # Size based on edge magnitude and z-score
                    edge_factor = min(1.0, abs(opp["vol_zscore"]) / (2 * zscore_threshold))*self.VOL_EDGE_FACTOR/100
                    desired_quantity = int(self.LIMIT[voucher] * edge_factor)
                    
                    # Calculate how many we can buy
                    quantity = min(
                        desired_quantity,
                        self.LIMIT[voucher] - position,
                        abs(opp["ask_vol"])  # Use ask volume since we're buying
                    )
                    
                    if quantity > 0:
                        price = int(opp["fair_price"] - spread)  # Adjusted to buy below fair price
                        orders[voucher].append(Order(voucher, price, -quantity))  # Negative quantity for buy
        
        # Remove empty order lists
        orders = {k: v for k, v in orders.items() if v}
        
        return orders

    def calculate_greeks_exposure(
        self,
        positions: Dict[str, int],
        voucher_orders: Dict[str, List[Order]],
        opportunities: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate the total exposure to different option greeks from positions and new orders
        """
        # Create lookup dictionary of opportunities by voucher
        opp_by_voucher = {opp["voucher"]: opp for opp in opportunities}
        
        # Initialize exposures
        exposures = {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0
        }
        
        # Add current positions exposure
        for voucher, position in positions.items():
            if voucher in opp_by_voucher:
                opp = opp_by_voucher[voucher]
                exposures["delta"] += position * opp["delta"]
                exposures["gamma"] += position * opp["gamma"]
                exposures["vega"] += position * opp["vega"]
                exposures["theta"] += position * opp["theta"]
        
        # Add new orders exposure
        for voucher, orders in voucher_orders.items():
            if voucher in opp_by_voucher:
                opp = opp_by_voucher[voucher]
                for order in orders:
                    # Negative because we're executing the order
                    exposures["delta"] += -order.quantity * opp["delta"]
                    exposures["gamma"] += -order.quantity * opp["gamma"]
                    exposures["vega"] += -order.quantity * opp["vega"]
                    exposures["theta"] += -order.quantity * opp["theta"]
        
        return exposures

    def optimize_volatility_exposure(
        self,
        voucher_orders: Dict[str, List[Order]],
        positions: Dict[str, int],
        opportunities: List[Dict],
        voucher_order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        """
        Adjust orders to optimize vega exposure while keeping delta near zero
        """
        # Calculate initial exposures
        exposures = self.calculate_greeks_exposure(positions, voucher_orders, opportunities)
        
        # No optimization needed if we don't have any vega exposure
        if abs(exposures["vega"]) < 0.1:
            return voucher_orders
            
        # Create lookup dictionary of opportunities by voucher
        opp_by_voucher = {opp["voucher"]: opp for opp in opportunities}
        
        # Prioritize options with high vega to delta ratio for vol trading
        vega_efficient_opps = sorted(
            opportunities, 
            key=lambda x: abs(x["vega"] / (x["delta"] + 0.0001)) if x["delta"] != 0 else float('inf'),
            reverse=True
        )
        
        # If we have large vega exposure, adjust positions in most vega-efficient options
        if abs(exposures["vega"]) > 10:
            for opp in vega_efficient_opps:
                voucher = opp["voucher"]
                position = positions.get(voucher, 0)
                
                # Skip if no opportunity or already at position limits
                if voucher not in opp_by_voucher or voucher not in voucher_order_depths:
                    continue
                
                voucher_depth = voucher_order_depths[voucher]
                    
                # If positive vega and we need to reduce it
                if exposures["vega"] > 10 and opp["vega"] > 0:
                    # Can we sell more of this option?
                    if position > -self.LIMIT[voucher]:
                        # Try to sell at bid
                        best_bid, bid_vol = self.get_best_bid(voucher_depth)
                        if best_bid:
                            quantity = min(
                                1 + position + self.LIMIT[voucher],  # How many we can sell
                                abs(bid_vol),  # Available volume
                                int(exposures["vega"] / (opp["vega"] + 0.0001))  # How many needed to reduce vega
                            )
                            
                            if quantity > 0:
                                if voucher not in voucher_orders:
                                    voucher_orders[voucher] = []
                                voucher_orders[voucher].append(Order(voucher, best_bid, quantity))
                                
                                # Update exposure
                                exposures["vega"] -= quantity * opp["vega"]
                                exposures["delta"] -= quantity * opp["delta"]
                                
                # If negative vega and we need to increase it
                elif exposures["vega"] < -10 and opp["vega"] > 0:
                    # Can we buy more of this option?
                    if position < self.LIMIT[voucher]:
                        # Try to buy at ask
                        best_ask, ask_vol = self.get_best_ask(voucher_depth)
                        if best_ask:
                            quantity = min(
                                self.LIMIT[voucher] - position,  # How many we can buy
                                abs(ask_vol),  # Available volume
                                int(abs(exposures["vega"]) / (opp["vega"] + 0.0001))  # How many needed to increase vega
                            )
                            
                            if quantity > 0:
                                if voucher not in voucher_orders:
                                    voucher_orders[voucher] = []
                                voucher_orders[voucher].append(Order(voucher, best_ask, -quantity))
                                
                                # Update exposure
                                exposures["vega"] += quantity * opp["vega"]
                                exposures["delta"] += quantity * opp["delta"]
        
        return voucher_orders

    def run(self, state: TradingState):
        # Initialization
        result = {}
        
        # Load trader data from previous iteration
        trader_data = {}
        if state.traderData and state.traderData != "":
            trader_data = jsonpickle.decode(state.traderData)
        
        # Get positions
        positions = state.position
        rock_position = positions.get(Product.ROCK, 0)
        
        # Skip if rock isn't available
        if Product.ROCK not in state.order_depths:
            return result, 0, jsonpickle.encode(trader_data)
        
        # Get rock order depth and mid price
        rock_order_depth = state.order_depths[Product.ROCK]
        rock_mid_price = self.get_mid_price(rock_order_depth)
        
        # Skip if rock mid price is not available
        if rock_mid_price is None:
            return result, 0, jsonpickle.encode(trader_data)
        
        # Track rock prices for historical volatility calculation
        if "rock_prices" not in trader_data:
            trader_data["rock_prices"] = []
        trader_data["rock_prices"].append(rock_mid_price)
        
        # Keep only the most recent 30 prices
        if len(trader_data["rock_prices"]) > 30:
            trader_data["rock_prices"] = trader_data["rock_prices"][-30:]
        
        # Collect available voucher order depths
        voucher_order_depths = {}
        for voucher in self.VOUCHERS:
            if voucher in state.order_depths:
                voucher_order_depths[voucher] = state.order_depths[voucher]
                
                # Initialize voucher data if not exists
                if voucher not in trader_data:
                    trader_data[voucher] = {
                        "prev_price": 0,
                        "past_vol": [],
                    }
        
        # Skip if no vouchers are available
        if not voucher_order_depths:
            return result, 0, jsonpickle.encode(trader_data)
        
        # Calculate time to expiry (days)
        time_to_expiry = max(0, PARAMS[Product.VOUCHER_9500]["starting_time_to_expiry"] - (state.timestamp / 1000000))
        
        # Find volatility arbitrage opportunities
        opportunities = self.find_volatility_arbitrage_opportunities(
            rock_mid_price,
            trader_data.get("rock_prices", [rock_mid_price]),
            voucher_order_depths,
            time_to_expiry,
            trader_data
        )
        
        # Skip if no opportunities
        if not opportunities:
            return result, 0, jsonpickle.encode(trader_data)
        
        # Generate volatility arbitrage orders
        voucher_orders = self.generate_volatility_arbitrage_orders(
            opportunities,
            positions
        )
        
        # Optimize vega exposure while keeping delta neutral
        voucher_orders = self.optimize_volatility_exposure(
            voucher_orders,
            positions,
            opportunities,
            voucher_order_depths
        )
        
        # Calculate total delta exposure from voucher positions and orders
        exposures = self.calculate_greeks_exposure(positions, voucher_orders, opportunities)
        total_delta_exposure = exposures["delta"]
        
        # Generate rock hedge orders to maintain delta neutrality
        rock_orders = self.delta_hedge_rock_position(
            rock_order_depth,
            total_delta_exposure,
            rock_position,
            aggressive=True  # Use aggressive orders to ensure delta neutrality
        )
        
        # Add orders to result
        for voucher, orders in voucher_orders.items():
            if orders:
                result[voucher] = orders
                
        if rock_orders:
            result[Product.ROCK] = rock_orders
        
        # Store delta exposure data for debugging
        trader_data["last_exposures"] = exposures
        
        # Return the result with updated trader data
        return result, 0, jsonpickle.encode(trader_data)