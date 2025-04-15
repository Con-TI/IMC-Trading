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


PARAMS = {
    Product.VOUCHER_9500: {
        "mean_volatility": 0.01106732809843326,
        "threshold": .0036609522447843975/2,
        "strike": 9750,
        "starting_time_to_expiry": 5,
        "std_window": 50,
        "zscore_threshold": 20,
    },
}

from math import log, sqrt, exp
from statistics import NormalDist


class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        # print(f"d1: {d1}")
        # print(f"vol: {volatility}")
        # print(f"spot: {spot}")
        # print(f"strike: {strike}")
        # print(f"time: {time_to_expiry}")
        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
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
        }

    # Returns buy_order_volume, sell_order_volume
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if best_ask <= fair_value - take_width:
                quantity = min(
                    best_ask_amount, position_limit - position
                )  # max amt to buy
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(
                    best_bid_amount, position_limit + position
                )  # should be the max we can sell
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def take_best_orders_with_adverse(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        adverse_volume: int,
    ) -> (int, int):

        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if prevent_adverse:
            buy_order_volume, sell_order_volume = self.take_best_orders_with_adverse(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
                adverse_volume,
            )
        else:
            buy_order_volume, sell_order_volume = self.take_best_orders(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
            )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def get_VOUCHER_9500_mid_price(
            self, VOUCHER_9500_order_depth: OrderDepth, traderData: Dict[str, Any]
        ):
            if (
                len(VOUCHER_9500_order_depth.buy_orders) > 0
                and len(VOUCHER_9500_order_depth.sell_orders) > 0
            ):
                best_bid = max(VOUCHER_9500_order_depth.buy_orders.keys())
                best_ask = min(VOUCHER_9500_order_depth.sell_orders.keys())
                traderData["prev_coupon_price"] = (best_bid + best_ask) / 2
                return (best_bid + best_ask) / 2
            else:
                return traderData["prev_coupon_price"]

    def delta_hedge_rock_position(
            self,
            rock_order_depth: OrderDepth,
            VOUCHER_9500_position: int,
            rock_position: int,
            rock_buy_orders: int,
            rock_sell_orders: int,
            delta: float,
        ) -> List[Order]:
            """
            Delta hedge the overall position in VOUCHER_9500 by creating orders in ROCK.

            Args:
                rock_order_depth (OrderDepth): The order depth for the ROCK product.
                VOUCHER_9500_position (int): The current position in VOUCHER_9500.
                rock_position (int): The current position in ROCK.
                rock_buy_orders (int): The total quantity of buy orders for ROCK in the current iteration.
                rock_sell_orders (int): The total quantity of sell orders for ROCK in the current iteration.
                delta (float): The current value of delta for the VOUCHER_9500 product.
                traderData (Dict[str, Any]): The trader data for the VOUCHER_9500 product.

            Returns:
                List[Order]: A list of orders to delta hedge the VOUCHER_9500 position.
            """

            target_rock_position = -int(delta * VOUCHER_9500_position)
            hedge_quantity = target_rock_position - (
                rock_position + rock_buy_orders - rock_sell_orders
            )

            orders: List[Order] = []
            if hedge_quantity > 0:
                # Buy ROCK
                best_ask = min(rock_order_depth.sell_orders.keys())
                quantity = min(
                    abs(hedge_quantity), -rock_order_depth.sell_orders[best_ask]
                )
                quantity = min(
                    quantity,
                    self.LIMIT[Product.ROCK] - (rock_position + rock_buy_orders),
                )
                if quantity > 0:
                    orders.append(Order(Product.ROCK, best_ask, quantity))
            elif hedge_quantity < 0:
                # Sell ROCK
                best_bid = max(rock_order_depth.buy_orders.keys())
                quantity = min(
                    abs(hedge_quantity), rock_order_depth.buy_orders[best_bid]
                )
                quantity = min(
                    quantity,
                    self.LIMIT[Product.ROCK] + (rock_position - rock_sell_orders),
                )
                if quantity > 0:
                    orders.append(Order(Product.ROCK, best_bid, -quantity))

            return orders

    def delta_hedge_VOUCHER_9500_orders(
            self,
            rock_order_depth: OrderDepth,
            VOUCHER_9500_orders: List[Order],
            rock_position: int,
            rock_buy_orders: int,
            rock_sell_orders: int,
            delta: float,
        ) -> List[Order]:
            """
            Delta hedge the new orders for VOUCHER_9500 by creating orders in ROCK.

            Args:
                rock_order_depth (OrderDepth): The order depth for the ROCK product.
                VOUCHER_9500_orders (List[Order]): The new orders for VOUCHER_9500.
                rock_position (int): The current position in ROCK.
                rock_buy_orders (int): The total quantity of buy orders for ROCK in the current iteration.
                rock_sell_orders (int): The total quantity of sell orders for ROCK in the current iteration.
                delta (float): The current value of delta for the VOUCHER_9500 product.

            Returns:
                List[Order]: A list of orders to delta hedge the new VOUCHER_9500 orders.
            """
            if len(VOUCHER_9500_orders) == 0:
                return None

            net_VOUCHER_9500_quantity = sum(
                order.quantity for order in VOUCHER_9500_orders
            )
            target_rock_quantity = -int(delta * net_VOUCHER_9500_quantity)

            orders: List[Order] = []
            if target_rock_quantity > 0:
                # Buy ROCK
                best_ask = min(rock_order_depth.sell_orders.keys())
                quantity = min(
                    abs(target_rock_quantity), -rock_order_depth.sell_orders[best_ask]
                )
                quantity = min(
                    quantity,
                    self.LIMIT[Product.ROCK] - (rock_position + rock_buy_orders),
                )
                if quantity > 0:
                    orders.append(Order(Product.ROCK, best_ask, quantity))
            elif target_rock_quantity < 0:
                # Sell ROCK
                best_bid = max(rock_order_depth.buy_orders.keys())
                quantity = min(
                    abs(target_rock_quantity), rock_order_depth.buy_orders[best_bid]
                )
                quantity = min(
                    quantity,
                    self.LIMIT[Product.ROCK] + (rock_position - rock_sell_orders),
                )
                if quantity > 0:
                    orders.append(Order(Product.ROCK, best_bid, -quantity))

            return orders

    def rock_hedge_orders(
            self,
            rock_order_depth: OrderDepth,
            VOUCHER_9500_order_depth: OrderDepth,
            VOUCHER_9500_orders: List[Order],
            rock_position: int,
            VOUCHER_9500_position: int,
            delta: float,
        ) -> List[Order]:
            if VOUCHER_9500_orders == None or len(VOUCHER_9500_orders) == 0:
                VOUCHER_9500_position_after_trade = VOUCHER_9500_position
            else:
                VOUCHER_9500_position_after_trade = VOUCHER_9500_position + sum(
                    order.quantity for order in VOUCHER_9500_orders
                )

            target_rock_position = -delta * VOUCHER_9500_position_after_trade

            if target_rock_position == rock_position:
                return None

            target_rock_quantity = target_rock_position - rock_position

            orders: List[Order] = []
            if target_rock_quantity > 0:
                # Buy ROCK
                best_ask = min(rock_order_depth.sell_orders.keys())
                quantity = min(
                    abs(target_rock_quantity),
                    self.LIMIT[Product.ROCK] - rock_position,
                )
                if quantity > 0:
                    orders.append(Order(Product.ROCK, best_ask, round(quantity)))

            elif target_rock_quantity < 0:
                # Sell ROCK
                best_bid = max(rock_order_depth.buy_orders.keys())
                quantity = min(
                    abs(target_rock_quantity),
                    self.LIMIT[Product.ROCK] + rock_position,
                )
                if quantity > 0:
                    orders.append(Order(Product.ROCK, best_bid, -round(quantity)))

            return orders

    def VOUCHER_9500_orders(
            self,
            VOUCHER_9500_order_depth: OrderDepth,
            VOUCHER_9500_position: int,
            traderData: Dict[str, Any],
            volatility: float,
        ) -> List[Order]:
            traderData["past_coupon_vol"].append(volatility)
            if (
                len(traderData["past_coupon_vol"])
                < self.params[Product.VOUCHER_9500]["std_window"]
            ):
                return None, None

            if (
                len(traderData["past_coupon_vol"])
                > self.params[Product.VOUCHER_9500]["std_window"]
            ):
                traderData["past_coupon_vol"].pop(0)

            vol_z_score = (
                volatility - self.params[Product.VOUCHER_9500]["mean_volatility"]
            ) / np.std(traderData["past_coupon_vol"])
            # print(f"vol_z_score: {vol_z_score}")
            # print(f"zscore_threshold: {self.params[Product.VOUCHER_9500]['zscore_threshold']}")
            if vol_z_score >= self.params[Product.VOUCHER_9500]["zscore_threshold"]:
                if VOUCHER_9500_position != -self.LIMIT[Product.VOUCHER_9500]:
                    target_VOUCHER_9500_position = -self.LIMIT[Product.VOUCHER_9500]
                    if len(VOUCHER_9500_order_depth.buy_orders) > 0:
                        best_bid = max(VOUCHER_9500_order_depth.buy_orders.keys())
                        target_quantity = abs(
                            target_VOUCHER_9500_position - VOUCHER_9500_position
                        )
                        quantity = min(
                            target_quantity,
                            abs(VOUCHER_9500_order_depth.buy_orders[best_bid]),
                        )
                        quote_quantity = target_quantity - quantity
                        if quote_quantity == 0:
                            return [Order(Product.VOUCHER_9500, best_bid, -quantity)], []
                        else:
                            return [Order(Product.VOUCHER_9500, best_bid, -quantity)], [
                                Order(Product.VOUCHER_9500, best_bid, -quote_quantity)
                            ]

            elif vol_z_score <= -self.params[Product.VOUCHER_9500]["zscore_threshold"]:
                if VOUCHER_9500_position != self.LIMIT[Product.VOUCHER_9500]:
                    target_VOUCHER_9500_position = self.LIMIT[Product.VOUCHER_9500]
                    if len(VOUCHER_9500_order_depth.sell_orders) > 0:
                        best_ask = min(VOUCHER_9500_order_depth.sell_orders.keys())
                        target_quantity = abs(
                            target_VOUCHER_9500_position - VOUCHER_9500_position
                        )
                        quantity = min(
                            target_quantity,
                            abs(VOUCHER_9500_order_depth.sell_orders[best_ask]),
                        )
                        quote_quantity = target_quantity - quantity
                        if quote_quantity == 0:
                            return [Order(Product.VOUCHER_9500, best_ask, quantity)], []
                        else:
                            return [Order(Product.VOUCHER_9500, best_ask, quantity)], [
                                Order(Product.VOUCHER_9500, best_ask, quote_quantity)
                            ]

            return None, None

    def get_past_returns(
        self,
        traderObject: Dict[str, Any],
        order_depths: Dict[str, OrderDepth],
        timeframes: Dict[str, int],
    ):
        returns_dict = {}

        for symbol, timeframe in timeframes.items():
            traderObject_key = f"{symbol}_price_history"
            if traderObject_key not in traderObject:
                traderObject[traderObject_key] = []

            price_history = traderObject[traderObject_key]

            if symbol in order_depths:
                order_depth = order_depths[symbol]
                if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                    current_price = (
                        max(order_depth.buy_orders.keys())
                        + min(order_depth.sell_orders.keys())
                    ) / 2
                else:
                    if len(price_history) > 0:
                        current_price = float(price_history[-1])
                    else:
                        returns_dict[symbol] = None
                        continue
            else:
                if len(price_history) > 0:
                    current_price = float(price_history[-1])
                else:
                    returns_dict[symbol] = None
                    continue

            price_history.append(
                f"{current_price:.1f}"
            )  # Convert float to string with 1 decimal place

            if len(price_history) > timeframe:
                price_history.pop(0)

            if len(price_history) == timeframe:
                past_price = float(price_history[0])  # Convert string back to float
                returns = (current_price - past_price) / past_price
                returns_dict[symbol] = returns
            else:
                returns_dict[symbol] = None

        return returns_dict

    def run(self, state: TradingState):
            traderObject = {}
            if state.traderData != None and state.traderData != "":
                traderObject = jsonpickle.decode(state.traderData)


            result = {}
            conversions = 1

            if Product.VOUCHER_9500 not in traderObject:
                traderObject[Product.VOUCHER_9500] = {
                    "prev_coupon_price": 0,
                    "past_coupon_vol": [],
                }

            if (
                Product.VOUCHER_9500 in self.params
                and Product.VOUCHER_9500 in state.order_depths
            ):
                VOUCHER_9500_position = (
                    state.position[Product.VOUCHER_9500]
                    if Product.VOUCHER_9500 in state.position
                    else 0
                )

                rock_position = (
                    state.position[Product.ROCK]
                    if Product.ROCK in state.position
                    else 0
                )
                # print(f"VOUCHER_9500_position: {VOUCHER_9500_position}")
                # print(f"rock_position: {rock_position}")
                rock_order_depth = state.order_depths[Product.ROCK]
                VOUCHER_9500_order_depth = state.order_depths[Product.VOUCHER_9500]
                rock_mid_price = (
                    min(rock_order_depth.buy_orders.keys())
                    + max(rock_order_depth.sell_orders.keys())
                ) / 2
                VOUCHER_9500_mid_price = self.get_VOUCHER_9500_mid_price(
                    VOUCHER_9500_order_depth, traderObject[Product.VOUCHER_9500]
                )
                tte = (
                    self.params[Product.VOUCHER_9500]["starting_time_to_expiry"]
                    - (state.timestamp) / 1000000 / 250
                )
                volatility = BlackScholes.implied_volatility(
                    VOUCHER_9500_mid_price,
                    rock_mid_price,
                    self.params[Product.VOUCHER_9500]["strike"],
                    tte,
                )
                delta = BlackScholes.delta(
                    rock_mid_price,
                    self.params[Product.VOUCHER_9500]["strike"],
                    tte,
                    volatility,
                )

                VOUCHER_9500_take_orders, VOUCHER_9500_make_orders = (
                    self.VOUCHER_9500_orders(
                        state.order_depths[Product.VOUCHER_9500],
                        VOUCHER_9500_position,
                        traderObject[Product.VOUCHER_9500],
                        volatility,
                    )
                )

                rock_orders = self.rock_hedge_orders(
                    state.order_depths[Product.ROCK],
                    state.order_depths[Product.VOUCHER_9500],
                    VOUCHER_9500_take_orders,
                    rock_position,
                    VOUCHER_9500_position,
                    delta,
                )

                if VOUCHER_9500_take_orders != None or VOUCHER_9500_make_orders != None:
                    result[Product.VOUCHER_9500] = (
                        VOUCHER_9500_take_orders + VOUCHER_9500_make_orders
                    )
                    # print(f"VOUCHER_9500: {result[Product.VOUCHER_9500]}")

                if rock_orders != None:
                    result[Product.ROCK] = rock_orders
                    # print(f"ROCK: {result[Product.ROCK]}")

            traderData = jsonpickle.encode(traderObject)

            return result, conversions, traderData
