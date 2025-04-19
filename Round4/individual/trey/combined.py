from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import pandas as pd
import math

class Product:
    SQUID = "SQUID_INK"
    KELP = "KELP"
    RESIN = "RAINFOREST_RESIN"
    PICNIC_1 = "PICNIC_BASKET1"
    PICNIC_2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC_1 = "SYNTHETIC_1"
    SYNTHETIC_2 = "SYNTHETIC_2"
    
PARAMS = {
    Product.RESIN : {
        "rainforest_resin_fair_value" : 10000,
        "rainforest_resin_width" : 2,
    },
    
    Product.KELP : {        
        "make_width" : 1,
        "take_width" : 1,
    },
    
    Product.SQUID : {
        "timespan" : 300,
        "timespan_bound" : 200,
        "trend_timespan_bound" : 25,
        "make_width" : 2,
        "take_width" : .5,
        "trend_thresh" : 6,
    },
    
    Product.PICNIC_1 : {
        "history_length" : 10,
        "gamma" : 1,
        "k" : 0.5,
        "reversion_coefficient": 1,
        "inventory_coeff" : 12.5,
        "theta_coeff" : 1e4/3,
        "trend_coeff" : 50000,
        "linear_coeff" : 6e7,
        "predictive_shift_coeff" : 2,
        "default_delt_shift" : -0.2,
        "regime_delt_shift" : 0,
        "spread_mean" : 0
    },
    Product.SYNTHETIC_1 : {
        "history_length" : 10,
    },
    Product.PICNIC_2 : {
        "history_length" : 10,
        "gamma" : 1,
        "k" : 0.5,
        "reversion_coefficient": 2,
        "inventory_coeff" : 100,
        "theta_coeff" : 4e2,
        "trend_coeff" : 50000,
        "linear_coeff" : 3.3e7,
        "predictive_shift_coeff" : 0,
        "default_delt_shift" : 0,
        "regime_delt_shift" : 0,
        "spread_mean" : 0
    },
    Product.SYNTHETIC_2 : {
        "history_length" : 10,
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
            Product.SQUID : 50,
            Product.KELP : 50,
            Product.RESIN : 50,
        }
        
    def rainforest_resin_orders(self, order_depth: OrderDepth, fair_value: int, width: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        # Track buy and sell order volumes
        buy_order_volume = 0
        sell_order_volume = 0
        
        # Find the best ask above fair value and best bid below fair value
        baaf = 10000
        bbbf = 10000
        if [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]:
            baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        if [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]:
            bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])

        # Buy logic: look for sell orders below fair value
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            if best_ask < fair_value:

                # Calculate buy quantity within position limits
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(Product.RESIN, best_ask, quantity)) 
                    buy_order_volume += quantity

        # Sell logic: look for buy orders above fair value
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:

                # Calculate sell quantity within position limits
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(Product.RESIN, best_bid, -1 * quantity))
                    sell_order_volume += quantity
        
        # Clear any excess position and adjust orders
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, Product.RESIN, 
            buy_order_volume, sell_order_volume, fair_value, 1
        )

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.RESIN, bbbf + 1, buy_quantity))

        # Place additional sell orders to approach position limit
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.RESIN, baaf - 1, -sell_quantity))

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
                orders.append(Order(product, int(fair_for_ask), -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        # If position is negative, try to buy at fair ask price
        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, int(fair_for_bid), abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
    
        return buy_order_volume, sell_order_volume

    def kelp_orders(self, order_depth: OrderDepth, width: float, kelp_take_width: float, position: int, position_limit: int) -> List[Order]:
        
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
            
            fair_value = mmmid_price

            # Take liquidity when price is significantly away from fair value
            if best_ask <= fair_value - kelp_take_width:
                ask_amount = -1 * order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(Product.KELP, int(math.ceil(best_ask)), quantity))
                        buy_order_volume += quantity
            
            if best_bid >= fair_value + kelp_take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(Product.KELP, int(math.floor(best_bid)), -1 * quantity))
                        sell_order_volume += quantity

            # Clear any excess position
            buy_order_volume, sell_order_volume = self.clear_position_order(
                orders, order_depth, position, position_limit, Product.KELP, 
                buy_order_volume, sell_order_volume, fair_value, 2
            )
            
            # Find prices for additional orders
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + width]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - width]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + width*2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - width*2
           
            # Place additional buy orders to approach position limit
            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order(Product.KELP, int(math.floor(bbbf + width)), buy_quantity))

            # Place additional sell orders to approach position limit
            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order(Product.KELP, int(math.ceil(baaf - width)), -sell_quantity))

        return orders
    
    def ink_orders(self, order_depth: OrderDepth, timespan:int, take_width : float, width : float, position: int, position_limit: int, traderObject, state) -> List[Order]:
        
        # Generate orders for Squid trading
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
            traderObject[Product.SQUID]['mid_price'].append(mmmid_price)
            
            if len(traderObject[Product.SQUID]['mid_price']) > timespan:
                traderObject[Product.SQUID]['mid_price'].pop(0)
        
            fair_value = mmmid_price
            predictive_shift = self.retrieve_squid_ink_signal(traderObject)
            direction = np.sign(predictive_shift) if predictive_shift != 0 else 0
            desired_position = 0
            if direction != 0: 
                desired_position = (position_limit-2) if direction == 1.0 else - (position_limit-2)
            else:
                desired_position = traderObject[Product.SQUID]['desired_position']
            traderObject[Product.SQUID]['desired_position'] = desired_position

            if direction != 0:
                if position != desired_position:
                    if position > desired_position:
                        orders.append(Order(Product.SQUID, best_bid, desired_position-position))
                    elif position < desired_position:
                        orders.append(Order(Product.SQUID, best_ask, desired_position-position))
            else:
                # Take liquidity when price is significantly away from fair value
                position_limit = position_limit//2
                if best_ask <= fair_value - take_width:
                    ask_amount = -1 * order_depth.sell_orders[best_ask]
                    if ask_amount <= 20:
                        quantity = min(ask_amount, position_limit - position)
                        if quantity > 0:
                            orders.append(Order(Product.SQUID, int(math.ceil(best_ask)), quantity))
                            buy_order_volume += quantity
                
                if best_bid >= fair_value + take_width:
                    bid_amount = order_depth.buy_orders[best_bid]
                    if bid_amount <= 20:
                        quantity = min(bid_amount, position_limit + position)
                        if quantity > 0:
                            orders.append(Order(Product.SQUID, int(math.floor(best_bid)), -1 * quantity))
                            sell_order_volume += quantity

                # Clear any excess position
                buy_order_volume, sell_order_volume = self.clear_position_order(
                    orders, order_depth, position, position_limit, Product.SQUID, 
                    buy_order_volume, sell_order_volume, fair_value, 2
                )
                
                # Find prices for additional orders
                aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + width]
                bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - width]
                baaf = min(aaf) if len(aaf) > 0 else fair_value + width*2
                bbbf = max(bbf) if len(bbf) > 0 else fair_value - width*2
            
                # Place additional buy orders to approach position limit
                buy_quantity = position_limit - (position + buy_order_volume)
                if buy_quantity > 0:
                    orders.append(Order(Product.SQUID, int(math.floor(bbbf + width)), buy_quantity))

                # Place additional sell orders to approach position limit
                sell_quantity = position_limit + (position - sell_order_volume)
                if sell_quantity > 0:
                    orders.append(Order(Product.SQUID, int(math.ceil(baaf - width)), -sell_quantity))
        return orders

    def retrieve_squid_ink_signal(self, traderObject, get_trend_only = False):
        """
            # Short term mean reversion
            signal1 = df.mid_price.rolling(50).mean()-df.mid_price
            # Longer term mean reversion
            signal2 = (df.mid_price.rolling(150).quantile(0.25)+df.mid_price.rolling(150).quantile(0.75))/2-df.mid_price
            # Short term trend
            signal3 = ((df.mid_price.rolling(12).quantile(0.75)-df.mid_price.rolling(12).quantile(0.25))/12)**2*np.sign(df.mid_price.diff(12))

            bound = df.mid_price.diff().rolling(30).quantile(.85)-df.mid_price.diff().rolling(30).quantile(.15)*4
        """
        if get_trend_only:
            if len(traderObject[Product.SQUID]['mid_price']) >= self.params[Product.SQUID]['trend_timespan_bound']:    
                trend = np.quantile(traderObject[Product.SQUID]['mid_price'][-25:],0.75)-np.quantile(traderObject[Product.SQUID]['mid_price'][-25:],0.25)
                trend *= np.sign(traderObject[Product.SQUID]['mid_price'][-1]-traderObject[Product.SQUID]['mid_price'][-25])
                trend *= 1.5
                return trend
            elif len(traderObject[Product.SQUID]['mid_price']) > 5:
                trend = np.quantile(traderObject[Product.SQUID]['mid_price'],0.75)-np.quantile(traderObject[Product.SQUID]['mid_price'],0.25)
                trend *= np.sign(traderObject[Product.SQUID]['mid_price'][-1]-traderObject[Product.SQUID]['mid_price'][0])
                trend *= 25/len(traderObject[Product.SQUID]['mid_price'])
                trend *= 1.5
                return trend

        if len(traderObject[Product.SQUID]['mid_price']) >= self.params[Product.SQUID]['timespan_bound']:
            mean = np.mean(traderObject[Product.SQUID]['mid_price'][-150:]) - traderObject[Product.SQUID]['mid_price'][-1]
            quantile_mean = (np.quantile(traderObject[Product.SQUID]['mid_price'],0.75)+np.quantile(traderObject[Product.SQUID]['mid_price'],0.25))/2 - traderObject[Product.SQUID]['mid_price'][-1]
            trend = np.quantile(traderObject[Product.SQUID]['mid_price'][-25:],0.75)-np.quantile(traderObject[Product.SQUID]['mid_price'][-25:],0.25)
            trend /= 12
            trend = trend**2
            trend *= np.sign(traderObject[Product.SQUID]['mid_price'][-1]-traderObject[Product.SQUID]['mid_price'][-25])
            difference = [traderObject[Product.SQUID]['mid_price'][-i]-traderObject[Product.SQUID]['mid_price'][-1-i] for i in range(1,31)]
            bound = max((np.quantile(difference,0.85) - np.quantile(difference,0.15))*4,50)
            signal = mean + quantile_mean + trend
            signal /= 10
            signal = signal**2
            if abs(signal) > bound:
                return mean + quantile_mean + trend
            return 0
        return 0
    
    def floor_func(self, num, bounds):
        if abs(num)<bounds:
            return 0
        return num

    def get_synthetic_basket_depth(self, state : TradingState, basket : Product):
        best_asks = {}
        best_bids = {}
        
        basket_bid = 0
        basket_ask = 0
        
        for product in BASKET_WEIGHTS[basket].keys():
            order_depth = state.order_depths[product]
            
            buy_orders = order_depth.buy_orders
            sell_orders = order_depth.sell_orders
            
            bid_price = None
            max_vol = 0
            for price, quantity in buy_orders.items():
                if abs(quantity) > max_vol:
                    max_vol = abs(quantity)
                    bid_price = price
            best_bids[product] = (bid_price, max_vol)
            basket_bid += bid_price * BASKET_WEIGHTS[basket][product]

            ask_price = None
            max_vol = 0
            for price, quantity in sell_orders.items():
                if abs(quantity) > max_vol:
                    max_vol = abs(quantity)
                    ask_price = price
            best_asks[product] = (ask_price, max_vol)
            basket_ask += ask_price * BASKET_WEIGHTS[basket][product]                
        
        basket_bid_vol = min([p_q[1]//BASKET_WEIGHTS[basket][product] for product, p_q  in best_bids.items()])
        basket_ask_vol = min([p_q[1]//BASKET_WEIGHTS[basket][product] for product, p_q  in best_asks.items()])
        
        synthetic_od = OrderDepth()
        synthetic_od.buy_orders = {basket_bid : basket_bid_vol}
        synthetic_od.sell_orders = {basket_ask : basket_ask_vol}
        
        return synthetic_od

    def get_best_ask_best_bid(self, state : TradingState, product : Product, first : bool = False):
        order_depth = state.order_depths[product]
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        
        if first:
            buy_p = [*buy_orders.keys()]
            best_bid = None
            if len(buy_p) > 1:
                best_bid = max(*buy_orders.keys())
            else:
                best_bid = buy_p[0]
            
            sell_p = [*sell_orders.keys()]
            best_ask = None
            if len(sell_p) > 1:
                best_ask = min(*sell_orders.keys())
            else:
                best_ask = sell_p[0]

            mid = (best_ask + best_bid)//2
            return mid, (best_bid, buy_orders[best_bid]), (best_ask, sell_orders[best_ask])    
        
        max_vol = 0
        bid = 0
        if len(buy_orders) > 1:
            for p,q in buy_orders.items():
                if abs(q)>max_vol:
                    max_vol = abs(q)
                    bid = p 
            bid_vol = max_vol
        else:
            bid = [*buy_orders.keys()][0]
            bid_vol = buy_orders[bid]
        
        max_vol = 0
        ask = 0
        if len(sell_orders) > 1:
            for p,q in sell_orders.items():
                if abs(q)>max_vol:
                    max_vol = abs(q)
                    ask = p
            ask_vol = max_vol
        else:
            ask = [*sell_orders.keys()][0]
            ask_vol = sell_orders[ask]
            
        mid = (ask + bid)//2
        
        return mid, (bid, bid_vol), (ask, ask_vol)

    def update_history(self, state : TradingState, product : Product, traderObject, synth = False):
        if synth:
            if product == Product.SYNTHETIC_1:
                synth_od = self.get_synthetic_basket_depth(state, Product.PICNIC_1)
                bid = [*synth_od.buy_orders.keys()][0]
                ask = [*synth_od.sell_orders.keys()][0]
                synth_mid = (bid+ask)/2
                traderObject[product]['mid_price'].append(synth_mid)
                window_limit = self.params[product]['history_length']
                if len(traderObject[product]['mid_price']) > window_limit:
                    traderObject[product]['mid_price'].pop(0)
                return
            if product == Product.SYNTHETIC_2:
                synth_od = self.get_synthetic_basket_depth(state, Product.PICNIC_2)
                bid = [*synth_od.buy_orders.keys()][0]
                ask = [*synth_od.sell_orders.keys()][0]
                synth_mid = (bid+ask)/2
                traderObject[product]['mid_price'].append(synth_mid)
                window_limit = self.params[product]['history_length']
                if len(traderObject[product]['mid_price']) > window_limit:
                    traderObject[product]['mid_price'].pop(0)
                return

        _, bid_tup, ask_tup = self.get_best_ask_best_bid(state, product)
        mid = (bid_tup[0] + ask_tup[0])/2
        traderObject[product]['mid_price'].append(mid)
        window_limit = self.params[product]['history_length']
        if len(traderObject[product]['mid_price']) > window_limit:
            traderObject[product]['mid_price'].pop(0)
        return
                
    def get_buy_sell_limits(self, state : TradingState, product : Product):
        position = state.position.get(product,0)
        lim = self.LIMIT[product]
        return (lim-position,-lim-position)

    def retrieve_reservation_quotes(self, state : TradingState, traderObject, product : Product):
        gamma, k = self.params[product]['gamma'], self.params[product]['k']
        volatility = pd.Series(traderObject[product]['mid_price'])
        volatility = volatility.pct_change().std()**2
        mid, _, _ = self.get_best_ask_best_bid(state, product)
        linear = max(1-volatility*self.params[product]['linear_coeff'],0)
        theta = self.retrieve_signal_skew(state, traderObject, product)
        trend = self.retrieve_trend_sig(state, traderObject, Product.PICNIC_1)
        mid += (theta*linear + trend*(1-linear))*self.params[product]['predictive_shift_coeff']
        q = state.position.get(product,0)
        time_hor = 1000000-state.timestamp
        default_delt = 1/gamma*np.log(1+gamma/k) + self.params[product]['default_delt_shift']
        if abs(trend) > 1:
            default_delt += self.params[product]['regime_delt_shift']
        bid_off = default_delt + self.params[product]['inventory_coeff']*gamma*volatility*q*(time_hor)
        ask_off = default_delt - self.params[product]['inventory_coeff']*gamma*volatility*q*(time_hor)
        bid_price = mid-bid_off
        ask_price = mid+ask_off
        return bid_price, ask_price

    def retrieve_signal_skew(self, state : TradingState, traderObject, product:Product):
        time_hor = 1000000-state.timestamp
        mid, _, _ = self.get_best_ask_best_bid(state, product)
        
        # Reversion based on spread
        synthetic_od = self.get_synthetic_basket_depth(state, product)
        synthetic_bid = [*synthetic_od.buy_orders.keys()][0]
        synthetic_ask = [*synthetic_od.sell_orders.keys()][0]
        synthetic_mid = (synthetic_bid+synthetic_ask)/2
        
        spread = np.log(mid) - np.log(synthetic_mid)
        mean = self.params[product]['spread_mean']
        theta = self.params[product]['theta_coeff']*(mean-spread)*(1-np.exp(-time_hor*self.params[product]['reversion_coefficient']))
        return theta

    def retrieve_trend_sig(self, state : TradingState, traderObject, product : Product):
        trend = np.log(traderObject[product]['mid_price'][-1])-np.log(traderObject[product]['mid_price'][0])
        volatility = pd.Series(traderObject[product]['mid_price'])
        volatility = volatility.pct_change().std()**2*self.params[product]['trend_coeff']
        trend /= volatility
        return trend


    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        if Product.SQUID not in traderObject:
            traderObject[Product.SQUID] = {
                 "mid_price" : [],
                 "desired_position" : 0
            }    
        if Product.PICNIC_1 not in traderObject:
            traderObject[Product.PICNIC_1] = {
                "mid_price" : [],
            }
        if Product.SYNTHETIC_1 not in traderObject:
            traderObject[Product.SYNTHETIC_1] = {
                "mid_price" : [],
            }    
        if Product.PICNIC_2 not in traderObject:
            traderObject[Product.PICNIC_2] = {
                "mid_price" : [],
            }
        if Product.SYNTHETIC_2 not in traderObject:
            traderObject[Product.SYNTHETIC_2] = {
                "mid_price" : [],
            }
            
        self.update_history(state, Product.PICNIC_1, traderObject)
        self.update_history(state, Product.SYNTHETIC_1, traderObject, synth = True)    
        self.update_history(state, Product.PICNIC_2, traderObject)
        self.update_history(state, Product.SYNTHETIC_2, traderObject, synth = True)   
        
        # Main trading method called for each trading iteration
        result = {}
        
        # Generate orders for Rainforest Resin if market exists
        if Product.RESIN in state.order_depths:
            rainforest_resin_orders = self.rainforest_resin_orders(
                state.order_depths[Product.RESIN], 
                self.params[Product.RESIN]["rainforest_resin_fair_value"], 
                self.params[Product.RESIN]["rainforest_resin_width"], 
                state.position.get(Product.RESIN,0), 
                self.LIMIT[Product.RESIN]
            )
            result[Product.RESIN] = rainforest_resin_orders

        # Generate orders for Kelp if market exists
        if Product.KELP in state.order_depths:
            kelp_orders = self.kelp_orders(
                state.order_depths[Product.KELP], 
                self.params[Product.KELP]['make_width'], 
                self.params[Product.KELP]['take_width'], 
                state.position.get(Product.KELP,0), 
                self.LIMIT[Product.KELP]
            )
            result[Product.KELP] = kelp_orders

        # Generate orders for SQUID if market exists
        if Product.SQUID in state.order_depths:
            ink_orders = self.ink_orders(
                state.order_depths[Product.SQUID], 
                self.params[Product.SQUID]['timespan'], 
                self.params[Product.SQUID]['take_width'],
                self.params[Product.SQUID]['make_width'],
                state.position.get(Product.SQUID, 0), 
                self.LIMIT[Product.SQUID],
                traderObject,
                state
            )
            result[Product.SQUID] = ink_orders

        orders = []        
        if state.timestamp>500:
            bid, ask = self.retrieve_reservation_quotes(state, traderObject, Product.PICNIC_1)

            # MM Orders
            buy_lim, sell_lim = self.get_buy_sell_limits(state, Product.PICNIC_1)
            fixed_q = 15
            orders.append(Order(Product.PICNIC_1,int(math.floor(bid)),min(fixed_q,buy_lim)))
            orders.append(Order(Product.PICNIC_1,int(math.ceil(ask)),max(-fixed_q,sell_lim)))

        result[Product.PICNIC_1] = orders

        traderData = jsonpickle.encode(traderObject)



        # Conversions (purpose not clear from context, possibly a trading feature)
        conversions = 1

        return result, conversions, traderData
    

    