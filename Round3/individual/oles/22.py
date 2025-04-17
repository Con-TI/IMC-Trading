from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Tuple, Any
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
    def __init__(self, params = None):
        # Initialize lists to store historical price and volume-weighted average price data

        self.kelp_prices = []  # Stores mid-prices for Kelp
        self.kelp_vwap = []    # Stores volume-weighted average price information
        self.ink_prices = []
        self.ink_vwap = []
        
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

    def rainforest_resin_orders(self, order_depth: OrderDepth, fair_value: int, width: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        # Track buy and sell order volumes
        buy_order_volume = 0
        sell_order_volume = 0
        
        # Find the best ask above fair value and best bid below fair value
        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])

        # INVERTED: Sell logic now looks for buy orders below fair value
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid < fair_value:
                # Calculate sell quantity within position limits
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
                    sell_order_volume += quantity

        # INVERTED: Buy logic now looks for sell orders above fair value
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            if best_ask > fair_value:
                # Calculate buy quantity within position limits
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity)) 
                    buy_order_volume += quantity
        
        # Clear any excess position and adjust orders
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "RAINFOREST_RESIN", 
            buy_order_volume, sell_order_volume, fair_value, 1
        )

        # INVERTED: Now selling at lower price and buying at higher price
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", baaf - 1, buy_quantity))

        # Place additional sell orders to approach position limit
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", bbbf + 1, -sell_quantity))

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

            # INVERTED LOGIC: Take liquidity when price is significantly away from fair value
            if best_ask <= fair_value - kelp_take_width:
                # Instead of buying when price is far below fair value, sell at best bid
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order("KELP", best_bid, -1 * quantity))
                        sell_order_volume += quantity
            
            if best_bid >= fair_value + kelp_take_width:
                # Instead of selling when price is far above fair value, buy at best ask
                ask_amount = -1 * order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order("KELP", best_ask, quantity))
                        buy_order_volume += quantity

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
        
        # Generate orders for Squid Ink trading
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
        
            # Use mid-price as fair value
            fair_value = mmmid_price

            # INVERTED LOGIC: Take liquidity when price is significantly away from fair value
            if best_ask <= fair_value - ink_take_width:
                # Instead of buying when price is far below fair value, sell at best bid
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order("SQUID_INK", best_bid, -1 * quantity))
                        sell_order_volume += quantity
            
            if best_bid >= fair_value + ink_take_width:
                # Instead of selling when price is far above fair value, buy at best ask
                ask_amount = -1 * order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order("SQUID_INK", best_ask, quantity))
                        buy_order_volume += quantity

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
        
        # Invert spread calculation logic for better profitability
        if product == Product.PICNIC_1:
            spread = synthetic_swmid - basket_swmid  # INVERTED
        elif product == Product.PICNIC_2:
            spread = basket_swmid - synthetic_swmid  # INVERTED
            
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
        if zscore >= thresh:
            if basket_position != self.params[spread_product]["target_position"]:  # INVERTED
                return self.execute_spread_orders(
                    self.params[spread_product]["target_position"],  # INVERTED
                    basket_position,
                    order_depths,
                    product
                )

        if zscore <= -thresh:
            if basket_position != -self.params[spread_product]["target_position"]:  # INVERTED
                return self.execute_spread_orders(
                    -self.params[spread_product]["target_position"],  # INVERTED
                    basket_position,
                    order_depths,
                    product
                )

        spread_data["prev_zscore"] = zscore
        return None
    
    def run(self, state: TradingState):
        # Main trading method called for each trading iteration
        result = {}

        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        conversions = 0

        # Fixed parameters for Rainforest Resin trading
        rainforest_resin_fair_value = 10000
        rainforest_resin_width = 2
        rainforest_resin_position_limit = 50

        # Parameters for Kelp trading
        kelp_make_width = 3.5
        kelp_take_width = 1
        kelp_position_limit = 50
        kelp_timespan = 10

        ink_make_width = 2
        ink_take_width = .5
        ink_position_limit = 50
        ink_timespan = 1000
        
        
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
                result[Product.CROISSANTS] = [Order(Product.CROISSANTS, croissants_price, croissant_final_q)]

            else:
                croissants_price = max(state.order_depths[Product.CROISSANTS].buy_orders.keys())
                result[Product.CROISSANTS] = [Order(Product.CROISSANTS, croissants_price, croissant_final_q)]

            if jams_final_q > 0:
                jams_price = min(state.order_depths[Product.JAMS].sell_orders.keys())
                result[Product.JAMS] = [Order(Product.JAMS, jams_price, jams_final_q)]
            else:
                jams_price = max(state.order_depths[Product.JAMS].buy_orders.keys())
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

        # Commented out data restoration (potentially for persistent state)
        # traderData = jsonpickle.decode(state.traderData)
        # self.kelp_prices = traderData["kelp_prices"]
        # self.kelp_vwap = traderData["kelp_vwap"]

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
        traderData = jsonpickle.encode({"kelp_prices": self.kelp_prices, "kelp_vwap": self.kelp_vwap})
        traderData = jsonpickle.encode({"ink_prices": self.ink_prices, "ink_vwap": self.ink_vwap})
        traderData = jsonpickle.encode(traderObject)

        # Conversions (purpose not clear from context, possibly a trading feature)
        conversions = 1

        return result, conversions, traderData