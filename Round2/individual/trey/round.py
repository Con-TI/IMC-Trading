from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any, Tuple
import string
import jsonpickle
import numpy as np
import math


class Product:
    PICNIC_1 = "PICNIC_BASKET1"
    PICNIC_2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"
    SPREAD_1 = "SPREAD_1"
    SPREAD_2 = "SPREAD_2"


PARAMS = {
    Product.SPREAD_1: {
        "default_spread_mean": 48.762433333333334,
        "default_spread_std": 85.11945080948948944,
        "spread_std_window": 45,
        "zscore_threshold": 3,
        "target_position": 58,
    },
    Product.SPREAD_2: {
        "default_spread_mean": 30.23596666666666,
        "default_spread_std": 59.849200222652364,
        "spread_std_window": 20,
        "zscore_threshold": 1.5,
        "target_position": 95
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


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.PICNIC_1: 60,
            Product.PICNIC_2: 100,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
        }

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
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                croissants_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                jams_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
                djembes_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                croissants_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                jams_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
                djembes_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            croissants_order = Order(
                Product.CROISSANTS,
                croissants_price,
                quantity * BASKET_WEIGHTS[basket][Product.CROISSANTS],
            )
            jams_order = Order(
                Product.JAMS,
                jams_price,
                quantity * BASKET_WEIGHTS[basket][Product.JAMS],
            )
            djembes_order = Order(
                Product.DJEMBES, 
                djembes_price, 
                quantity * BASKET_WEIGHTS[basket][Product.DJEMBES]
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

        if target_position > basket_position:
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

        else:
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
        if product == Product.PICNIC_1:
            spread = basket_swmid - synthetic_swmid
        elif product == Product.PICNIC_2:
            spread = synthetic_swmid - basket_swmid
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

        if zscore >= thresh:
            if basket_position != -self.params[spread_product]["target_position"]:
                return self.execute_spread_orders(
                    -self.params[spread_product]["target_position"],
                    basket_position,
                    order_depths,
                    product
                )

        if zscore <= -thresh:
            if basket_position != self.params[spread_product]["target_position"]:
                return self.execute_spread_orders(
                    self.params[spread_product]["target_position"],
                    basket_position,
                    order_depths,
                    product
                )

        spread_data["prev_zscore"] = zscore
        return None

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

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
        spread_orders=  None
        
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

            if croissant_final_q > 0:
                croissants_price = min(state.order_depths[Product.CROISSANTS].sell_orders.keys())
            else:
                croissants_price = max(state.order_depths[Product.CROISSANTS].buy_orders.keys())

            if jams_final_q > 0:
                jams_final_q = min(state.order_depths[Product.JAMS].sell_orders.keys())
            else:
                jams_price = max(state.order_depths[Product.JAMS].buy_orders.keys())

            result[Product.CROISSANTS] = [Order(Product.CROISSANTS, croissants_price, croissant_final_q)]
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

        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
    
