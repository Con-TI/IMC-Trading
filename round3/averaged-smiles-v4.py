from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any, Tuple
import numpy as np
import math
import jsonpickle
from statistics import NormalDist


class Product:
    ROCK = "VOLCANIC_ROCK"
    VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOUCHER_9750 = 'VOLCANIC_ROCK_VOUCHER_9750'
    VOUCHER_10000 = 'VOLCANIC_ROCK_VOUCHER_10000'
    VOUCHER_10250 = 'VOLCANIC_ROCK_VOUCHER_10250'
    VOUCHER_10500 = 'VOLCANIC_ROCK_VOUCHER_10500'


PARAMS = {
    Product.VOUCHER_9500: {
        "strike": 9500,
        "position_sizing": 0.1,
        "vol_edge_threshold": 0.0001,
    },
    Product.VOUCHER_9750: {
        "strike": 9750,
        "position_sizing": 0.1,
        "vol_edge_threshold": 0.0001,
    },
    Product.VOUCHER_10000: {
        "strike": 10000,
        "position_sizing": 0.1,
        "vol_edge_threshold": 0.0001,
    },
    Product.VOUCHER_10250: {
        "strike": 10250,
        "position_sizing": 0.1,
        "vol_edge_threshold": 0.0001,
    },
    Product.VOUCHER_10500: {
        "strike": 10500,
        "position_sizing": 0.1,
        "vol_edge_threshold": 0.0001,
    },
}


class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        if time_to_expiry <= 0:
            return max(0, spot - strike)
        
        d1 = (
            math.log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * math.sqrt(time_to_expiry))
        d2 = d1 - volatility * math.sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        if time_to_expiry <= 0:
            return 1.0 if spot > strike else 0.0
            
        d1 = (
            math.log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * math.sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=100, tolerance=1e-5
    ):
        if time_to_expiry <= 0:
            return 0.0
            
        intrinsic = max(0, spot - strike)
        if call_price <= intrinsic:
            return 0.000001  # Return minimum volatility
            
        low_vol = 0.000001
        high_vol = 2.0
        
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

        self.LIMIT = {
            Product.ROCK: 400,
            Product.VOUCHER_9500: 200,
            Product.VOUCHER_9750: 200,
            Product.VOUCHER_10000: 200,
            Product.VOUCHER_10250: 200,
            Product.VOUCHER_10500: 200,
        }
        
        self.VOUCHERS = [
            Product.VOUCHER_9500,
            Product.VOUCHER_9750,
            Product.VOUCHER_10000,
            Product.VOUCHER_10250,
            Product.VOUCHER_10500,
        ]
    
    def parabola_function(self, x, a, b, c):
        return a * (x ** 2) + b * x + c
    
    def fit_parabola(self, x_data, y_data) -> Tuple[Dict, float]:
        if len(x_data) < 3:
            return None, float('inf')
            
        x = np.array(x_data)
        y = np.array(y_data)        
        X = np.column_stack([x**2, x, np.ones(len(x))])
        
        try:
            XTX = X.T @ X
            XTy = X.T @ y
            
            params = np.linalg.solve(XTX, XTy)
            
            a, b, c = params            
            predicted = self.parabola_function(x, a, b, c)
            residuals = y - predicted
            mse = np.mean(residuals**2)
            
            print(mse)

            if a < 0:
                print('Negative coeff, failed')
                return None, float('inf')

                # return {"a": 3.385, "b": 0.0035136, "c": 0.013, the averaged smile
            
            return {"a": a, "b": b, "c": c}, mse
        except:
            print('Parabola failed')
            return None, float('inf')

    def get_mid_price(self, order_depth: OrderDepth) -> float:

        if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        elif len(order_depth.buy_orders) > 0:
            return max(order_depth.buy_orders.keys())
        elif len(order_depth.sell_orders) > 0:
            return min(order_depth.sell_orders.keys())
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
        
    def get_weighted_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate the weighted mid price from an order book"""
        if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            bid_vol = abs(order_depth.buy_orders[best_bid])
            ask_vol = abs(order_depth.sell_orders[best_ask])
            total_vol = bid_vol + ask_vol
            
            if total_vol > 0:
                return (best_bid * ask_vol + best_ask * bid_vol) / total_vol
            return (best_bid + best_ask) / 2
        return self.get_mid_price(order_depth)

    def compute_volatility_smile(
        self,
        rock_mid_price: float,
        voucher_order_depths: Dict[str, OrderDepth],
        time_to_expiry: float,
    ) -> Dict:
 
        moneyness_values = []
        implied_volatilities = []
        voucher_data = {}
        
        for voucher in self.VOUCHERS:
            if voucher not in voucher_order_depths:
                continue
                
            voucher_depth = voucher_order_depths[voucher]
            strike = self.params[voucher]["strike"]
            
            best_bid, bid_vol = self.get_best_bid(voucher_depth)
            best_ask, ask_vol = self.get_best_ask(voucher_depth)
            
            if best_bid is None or best_ask is None:
                continue
            
            mid_price = (best_bid + best_ask) / 2
            
            # Moneyness (m_t = ln(S_t / K) / sqrt(τ))
            moneyness = math.log(rock_mid_price / strike) / math.sqrt(time_to_expiry)
            
            try:
                implied_vol = BlackScholes.implied_volatility(
                    mid_price, rock_mid_price, strike, time_to_expiry
                )
                
                moneyness_values.append(moneyness)
                implied_volatilities.append(implied_vol)
                
                voucher_data[voucher] = {
                    "moneyness": moneyness,
                    "implied_vol": implied_vol,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "bid_vol": bid_vol,
                    "ask_vol": ask_vol,
                    "strike": strike,
                }
            except:
                print('Volatility calculation failed')
                continue
        
        smile_fit = None
        if len(moneyness_values) >= 3:
            try:
                smile_fit, mse = self.fit_parabola(moneyness_values, implied_volatilities)
            except:

                print('Fitting failed')
                pass
        
        return {
            "voucher_data": voucher_data,
            "smile_fit": smile_fit
        }

    def analyze_vouchers_with_smile(
        self,
        rock_mid_price: float,
        time_to_expiry: float,
        volatility_smile: Dict,
    ) -> List[Dict]:
    
        voucher_analysis = []
        
        if volatility_smile["smile_fit"] is None:
            return voucher_analysis
            
        smile_fit = volatility_smile["smile_fit"]
        voucher_data = volatility_smile["voucher_data"]
        
        for voucher, data in voucher_data.items():
            moneyness = data["moneyness"]
            implied_vol = data["implied_vol"]
            
            # Calculate theoretical vol from parabola fit
            theoretical_vol = self.parabola_function(
                moneyness, smile_fit["a"], smile_fit["b"], smile_fit["c"]
            )
            
            theoretical_vol = max(0.000001, theoretical_vol)
            
            vol_spread = implied_vol - theoretical_vol

            print(f'This is the {voucher} spread {vol_spread}')

            fair_price = BlackScholes.black_scholes_call(
                rock_mid_price, data["strike"], time_to_expiry, theoretical_vol
            )
            
            bid_price_edge = data["best_bid"] - fair_price  # Positive means overpriced, we sell
            ask_price_edge = fair_price - data["best_ask"]  # Positive means underpriced, we buy
            
            delta = BlackScholes.delta(
                rock_mid_price, data["strike"], time_to_expiry, implied_vol
            )
            
            analysis = {
                "voucher": voucher,
                "moneyness": moneyness,
                "theoretical_vol": theoretical_vol,
                "implied_vol": implied_vol,
                "vol_spread": vol_spread,
                "fair_price": fair_price,
                "best_bid": data["best_bid"],
                "best_ask": data["best_ask"], 
                "bid_vol": data["bid_vol"],
                "ask_vol": data["ask_vol"],
                "bid_price_edge": bid_price_edge,
                "ask_price_edge": ask_price_edge,
                "delta": delta
            }
            
            voucher_analysis.append(analysis)
            
        return voucher_analysis

    def generate_voucher_orders(
        self,
        voucher_analysis: List[Dict],
        positions: Dict[str, int],
    ) -> Dict[str, List[Order]]:
 
        orders = {}
        
        # Buy when IV < theoretical vol (negative vol_spread)
        buy_opps = sorted([op for op in voucher_analysis 
                           if op["vol_spread"] < -self.params[op["voucher"]]["vol_edge_threshold"]],
                          key=lambda x: x["vol_spread"])  # Sort by most negative vol_spread
        
        # Sell when IV > theoretical vol (positive vol_spread)
        sell_opps = sorted([op for op in voucher_analysis 
                            if op["vol_spread"] > self.params[op["voucher"]]["vol_edge_threshold"]],
                           key=lambda x: x["vol_spread"], reverse=True)
        
        for opp in buy_opps:
            voucher = opp["voucher"]
            position = positions.get(voucher, 0)
            
            if position < self.LIMIT[voucher]:
                max_quantity = self.LIMIT[voucher] - position
                vol_edge_ratio = min(5.0, abs(opp["vol_spread"]) / self.params[voucher]["vol_edge_threshold"])
                
                target_quantity = min(
                    max_quantity,
                    int(max_quantity * self.params[voucher]["position_sizing"] * vol_edge_ratio),
                    abs(opp["ask_vol"])
                )
                
                if target_quantity > 0:
                    if voucher not in orders:
                        orders[voucher] = []
                    orders[voucher].append(Order(voucher, opp["best_ask"], target_quantity))
        
        for opp in sell_opps:
            voucher = opp["voucher"]
            position = positions.get(voucher, 0)
            
            if position > -self.LIMIT[voucher]:
        
                # Determine quantity based on strength of signal and position sizing parameter
                max_quantity = self.LIMIT[voucher] + position
                vol_edge_ratio = min(5.0, abs(opp["vol_spread"]) / self.params[voucher]["vol_edge_threshold"])
                
                target_quantity = min(
                    max_quantity,
                    int(max_quantity * self.params[voucher]["position_sizing"] * vol_edge_ratio),
                    abs(opp["bid_vol"])
                )
                
                if target_quantity > 0:
                    if voucher not in orders:
                        orders[voucher] = []
                    orders[voucher].append(Order(voucher, opp["best_bid"], -target_quantity))
        
        return orders

    def run(self, state: TradingState):
        result = {}
        
        trader_data = {}
        if state.traderData and state.traderData != "":
            try:
                trader_data = jsonpickle.decode(state.traderData)
            except:
                trader_data = {}
        
        positions = state.position if state.position else {}
        
        # Skip if rock isn't available
        if Product.ROCK not in state.order_depths:
            return result, 0, jsonpickle.encode(trader_data)
        
        rock_order_depth = state.order_depths[Product.ROCK]
        rock_mid_price = self.get_weighted_mid_price(rock_order_depth)
        
        if rock_mid_price is None:
            return result, 0, jsonpickle.encode(trader_data)
        
        voucher_order_depths = {}
        for voucher in self.VOUCHERS:
            if voucher in state.order_depths:
                voucher_order_depths[voucher] = state.order_depths[voucher]
        
        if not voucher_order_depths:
            return result, 0, jsonpickle.encode(trader_data)
        
        # Calculate time to expiry (days)
        time_to_expiry = max(0.001, 4 - (state.timestamp / 1000000))
        
        volatility_smile = self.compute_volatility_smile(
            rock_mid_price,
            voucher_order_depths,
            time_to_expiry
        )
        
        if not volatility_smile or volatility_smile["smile_fit"] is None:
            return result, 0, jsonpickle.encode(trader_data)
        
        if "smile_history" not in trader_data:
            trader_data["smile_history"] = []
            
        trader_data["smile_history"].append({
            "timestamp": state.timestamp,
            "fit": volatility_smile["smile_fit"],
            "time_to_expiry": time_to_expiry
        })
        
        # Keep history to a reasonable size
        if len(trader_data["smile_history"]) > 100:
            trader_data["smile_history"].pop(0)
        
        voucher_analysis = self.analyze_vouchers_with_smile(
            rock_mid_price,
            time_to_expiry,
            volatility_smile
        )
        
        if not voucher_analysis:
            return result, 0, jsonpickle.encode(trader_data)
        
        voucher_orders = self.generate_voucher_orders(
            voucher_analysis,
            positions
        )
        
        for voucher, orders in voucher_orders.items():
            if orders:
                result[voucher] = orders
        
        return result, 0, jsonpickle.encode(trader_data)