from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Tuple

class Trader:
    def __init__(self):
        # Position limits for each product
        self.position_limits = {
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBE": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100
        }
        
        # Basket compositions
        self.basket_components = {
            "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBE": 1},
            "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2}
        }
    
    def get_best_price(self, order_depth: OrderDepth, is_buy: bool) -> Tuple[int, int]:
        """Get the best price and quantity from the order depth."""
        if is_buy and order_depth.sell_orders:
            best_price = min(order_depth.sell_orders.keys())
            return best_price, order_depth.sell_orders[best_price]
        elif not is_buy and order_depth.buy_orders:
            best_price = max(order_depth.buy_orders.keys())
            return best_price, order_depth.buy_orders[best_price]
        return None, 0
    
    def calculate_fair_value(self, state: TradingState, basket: str) -> int:
        """Calculate the fair value of a basket based on current component prices."""
        fair_value = 0
        components = self.basket_components[basket]
        
        for product, quantity in components.items():
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                if order_depth.buy_orders and order_depth.sell_orders:
                    # Use mid price for fair value calculation
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    mid_price = (best_bid + best_ask) // 2
                    fair_value += mid_price * quantity
                elif order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    fair_value += best_bid * quantity
                elif order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    fair_value += best_ask * quantity
        
        return fair_value
    
    def max_trade_size(self, state: TradingState, product: str, is_buy: bool) -> int:
        """Calculate maximum trade size based on position limits."""
        current_position = state.position.get(product, 0)
        
        if is_buy:
            return self.position_limits[product] - current_position
        else:
            return current_position + self.position_limits[product]
    
    def run(self, state: TradingState):
        """Main trading logic."""
        result = {}
        
        # Calculate fair values for baskets
        basket1_fair_value = self.calculate_fair_value(state, "PICNIC_BASKET1")
        basket2_fair_value = self.calculate_fair_value(state, "PICNIC_BASKET2")
        
        # Process all products
        for product in state.order_depths:
            orders = []
            order_depth = state.order_depths[product]
            
            # Handle basket arbitrage
            if product == "PICNIC_BASKET1" and basket1_fair_value > 0:
                best_ask, ask_amount = self.get_best_price(order_depth, True)
                best_bid, bid_amount = self.get_best_price(order_depth, False)
                
                # Buy basket if cheaper than components
                if best_ask and best_ask < basket1_fair_value:
                    # Calculate how many we can buy
                    max_buy = min(
                        abs(ask_amount),
                        self.max_trade_size(state, "PICNIC_BASKET1", True)
                    )
                    
                    # Check if we can also take the component positions
                    for comp, qty in self.basket_components["PICNIC_BASKET1"].items():
                        component_max = self.max_trade_size(state, comp, True) // qty
                        max_buy = min(max_buy, component_max)
                    
                    if max_buy > 0:
                        orders.append(Order(product, best_ask, max_buy))
                        
                        # Also create sell orders for the components
                        for comp, qty in self.basket_components["PICNIC_BASKET1"].items():
                            if comp in state.order_depths:
                                comp_best_bid, _ = self.get_best_price(state.order_depths[comp], False)
                                if comp_best_bid:
                                    if comp not in result:
                                        result[comp] = []
                                    result[comp].append(Order(comp, comp_best_bid, -max_buy * qty))
                
                # Sell basket if more expensive than components
                if best_bid and best_bid > basket1_fair_value:
                    max_sell = min(
                        abs(bid_amount),
                        self.max_trade_size(state, "PICNIC_BASKET1", False)
                    )
                    
                    if max_sell > 0:
                        orders.append(Order(product, best_bid, -max_sell))
                        
                        # Also create buy orders for the components
                        for comp, qty in self.basket_components["PICNIC_BASKET1"].items():
                            if comp in state.order_depths:
                                comp_best_ask, _ = self.get_best_price(state.order_depths[comp], True)
                                if comp_best_ask:
                                    if comp not in result:
                                        result[comp] = []
                                    result[comp].append(Order(comp, comp_best_ask, max_sell * qty))
            
            # Similar logic for PICNIC_BASKET2
            elif product == "PICNIC_BASKET2" and basket2_fair_value > 0:
                best_ask, ask_amount = self.get_best_price(order_depth, True)
                best_bid, bid_amount = self.get_best_price(order_depth, False)
                
                # Buy basket if cheaper than components
                if best_ask and best_ask < basket2_fair_value:
                    max_buy = min(
                        abs(ask_amount),
                        self.max_trade_size(state, "PICNIC_BASKET2", True)
                    )
                    
                    for comp, qty in self.basket_components["PICNIC_BASKET2"].items():
                        component_max = self.max_trade_size(state, comp, True) // qty
                        max_buy = min(max_buy, component_max)
                    
                    if max_buy > 0:
                        orders.append(Order(product, best_ask, max_buy))
                        
                        for comp, qty in self.basket_components["PICNIC_BASKET2"].items():
                            if comp in state.order_depths:
                                comp_best_bid, _ = self.get_best_price(state.order_depths[comp], False)
                                if comp_best_bid:
                                    if comp not in result:
                                        result[comp] = []
                                    result[comp].append(Order(comp, comp_best_bid, -max_buy * qty))
                
                # Sell basket if more expensive than components
                if best_bid and best_bid > basket2_fair_value:
                    max_sell = min(
                        abs(bid_amount),
                        self.max_trade_size(state, "PICNIC_BASKET2", False)
                    )
                    
                    if max_sell > 0:
                        orders.append(Order(product, best_bid, -max_sell))
                        
                        for comp, qty in self.basket_components["PICNIC_BASKET2"].items():
                            if comp in state.order_depths:
                                comp_best_ask, _ = self.get_best_price(state.order_depths[comp], True)
                                if comp_best_ask:
                                    if comp not in result:
                                        result[comp] = []
                                    result[comp].append(Order(comp, comp_best_ask, max_sell * qty))
            
            # Handle individual products (market making strategy for components)
            elif product in ["CROISSANTS", "JAMS", "DJEMBE"]:
                # Simple market making strategy for components
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    
                    # Only trade if there's a reasonable spread
                    if best_ask - best_bid > 2:
                        # Buy at bid, sell at ask (if we don't have pending orders from basket arbitrage)
                        if product not in result or not result[product]:
                            max_buy = min(
                                abs(order_depth.buy_orders[best_bid]),
                                self.max_trade_size(state, product, True)
                            )
                            
                            max_sell = min(
                                abs(order_depth.sell_orders[best_ask]),
                                self.max_trade_size(state, product, False)
                            )
                            
                            if max_buy > 0:
                                orders.append(Order(product, best_bid, max_buy))
                            
                            if max_sell > 0:
                                orders.append(Order(product, best_ask, -max_sell))
            
            if orders:
                result[product] = orders
        
        # Use trader data to persist information between rounds if needed
        trader_data = ""
        conversions = 1
        
        return result, conversions, trader_data