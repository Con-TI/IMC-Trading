from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import jsonpickle
from statistics import NormalDist
from math import log, sqrt, exp


class Product:
    V_9500 = 'VOLCANIC_ROCK_VOUCHER_9500'
    V_9750 = 'VOLCANIC_ROCK_VOUCHER_9750'
    V_10000 = 'VOLCANIC_ROCK_VOUCHER_10000'
    V_10250 = 'VOLCANIC_ROCK_VOUCHER_10250'
    V_10500 = 'VOLCANIC_ROCK_VOUCHER_10500'
    ROCK = 'VOLCANIC_ROCK'


class Trader:
    def __init__(self):
        self.LIMIT = {
            Product.V_9500: 200,
            Product.V_9750: 200,
            Product.V_10000: 200,
            Product.V_10250: 200,
            Product.V_10500: 200,
            Product.ROCK: 400
        }

    def run(self, state: TradingState):
        traderObject = {}
        result = {}
        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)
        
        hedge_threshold = 10

        timestamp = state.timestamp
        TTE = 4 - timestamp / 100_000  # Time to expiry (in years)

        strikes = [Product.V_9500, Product.V_9750, Product.V_10000, Product.V_10250, Product.V_10500]
        strike_price = {
            Product.V_9500: 9500,
            Product.V_9750: 9750,
            Product.V_10000: 10000,
            Product.V_10250: 10250,
            Product.V_10500: 10500
        }

        rock_order_depth: OrderDepth = state.order_depths[Product.ROCK]
        if not rock_order_depth.sell_orders or not rock_order_depth.buy_orders:
            return {}, 1, jsonpickle.encode(traderObject)

        best_ask = min(rock_order_depth.sell_orders.keys())
        best_bid = max(rock_order_depth.buy_orders.keys())
        rock_mid_price = (best_ask + best_bid) / 2

        net_delta = 0  

        for strike in strikes:
            orders = []

            order_depth: OrderDepth = state.order_depths.get(strike)
            if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
                continue  

            position = state.position.get(strike, 0)
            position_limit = self.LIMIT[strike]
            max_buys = position_limit - position
            max_sells = -position_limit - position

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            voucher_mid_price = (best_ask + best_bid) / 2

            IV = self.implied_volatility(voucher_mid_price, rock_mid_price, strike_price[strike], TTE)
            delta = self.black_scholes_delta(rock_mid_price, strike_price[strike], TTE, IV)
            net_delta += position * delta

            action, BS_price = self.call_buy_sell(voucher_mid_price, rock_mid_price, strike_price[strike], TTE, best_ask-best_bid)

            if action == 'sell':
                price = best_bid if BS_price < best_bid else best_ask
                orders.append(Order(strike, price, max_sells))
            elif action == 'buy':
                price = best_ask if BS_price > best_ask else best_bid
                orders.append(Order(strike, price, max_buys))

            result[strike] = orders

        rock_position = state.position.get(Product.ROCK, 0)
        rock_limit = self.LIMIT[Product.ROCK]

        hedge_units = round(-net_delta - rock_position)
        hedge_units = max(min(hedge_units, rock_limit - rock_position), -rock_limit - rock_position)

        hedge_orders = []
        best_ask = min(rock_order_depth.sell_orders.keys()) if rock_order_depth.sell_orders else None
        best_bid = max(rock_order_depth.buy_orders.keys()) if rock_order_depth.buy_orders else None

        if abs(hedge_units)>hedge_threshold:
            if hedge_units > 0 and best_ask is not None:
                hedge_orders.append(Order(Product.ROCK, best_ask, hedge_units))
            elif hedge_units < 0 and best_bid is not None:
                hedge_orders.append(Order(Product.ROCK, best_bid, hedge_units))

            result[Product.ROCK] = hedge_orders

        traderData = jsonpickle.encode(traderObject)
        conversions = 1
        return result, conversions, traderData

    def black_scholes_call(self, spot, strike, time_to_expiry, volatility):
        if time_to_expiry == 0:
            return max(spot - strike, 0)
        d1 = (log(spot / strike) + 0.5 * volatility**2 * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        return spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)

    def black_scholes_delta(self, spot, strike, time_to_expiry, volatility):
        if time_to_expiry == 0:
            return 0
        d1 = (log(spot / strike) + 0.5 * volatility**2 * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    def implied_volatility(self, call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0
        for _ in range(max_iterations):
            estimated_price = self.black_scholes_call(spot, strike, time_to_expiry, volatility)
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility

    def call_buy_sell(self, call_price, spot, strike, time_to_expiry, spread):
        tolerance = 1.0
        IV = self.implied_volatility(call_price, spot, strike, time_to_expiry)
        BS_price = self.black_scholes_call(spot, strike, time_to_expiry, IV)

        if call_price - BS_price > tolerance + spread/2:
            return 'sell', BS_price
        elif BS_price - call_price > tolerance:
            return 'buy', BS_price

        return 'hold', BS_price
