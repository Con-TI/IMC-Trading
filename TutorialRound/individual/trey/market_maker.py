from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Dict, List, Tuple
import jsonpickle
import string

""" 
Strategy:

Since RESIN is stable, just keep placing bids and asks at fixed levels with max position.
(Spread varies between 2 and 10 ticks)

For KELP, we take the following strategy:
(Spread varies between 1 and 4 ticks)
- Make the market by applying a 2 tick spread on our orders at all times.
- Principles for bid and ask placing (Only bids are described where asks could also be described)
    Measures of trajectory:
    - Suppose the highest bid volume is not at price 1 but at price 2/3. Then prices are going to go down.
    - Suppose the "width" of the range of bids is far larger than that of asks, then this indicates bots are expecting prices to go down.
    - Suppose the total volume on the bid side is higher than the ask, (does not influence based on data)
    - Suppose the midprice has been trending, then we expect either a continued trend or mean reversion based on the above.
    - If the microprice exceeds the midprice, we expect prices to go up.
    Could also check if the amount of autocorrelation in trends
    
    Arbitrages:
    - Suppose the spread goes to 1 (i.e. sudden volume moves bid and ask right next to each other), then this represent arbitrage.
    I.e., we can check the past mid-price or other reference to see if this was a misplacement on the bid side or ask side.
    If the bid is higher than our reference, we sell, then buy back at a lower price at the next timestep.
    
    Measures of fair price:
    - Microprice, previous midprice, previous microprice
- Trading principle:
    - Arbitrage if there's an opportunity 

Sidenote:
- In order to evaluate the effectiveness of the KELP strat, we turn off trades on RESIN.
"""

# Some global variables
class Product:
    RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"

POSITION_LIMIT = {
    Product.RESIN : 50,
    Product.KELP : 50
}

# ____________________________________________ Trader ________________________________________________

class Trader:
    def __init__(self):
        # Positions related
        self.inventory : Dict[Product, int] = {Product.RESIN:0, Product.KELP:0}
        self.position_limit : Dict[Product, int] = {}
        self.max_orderable : Dict[Product, Dict[str, int]] = {}

        # Prices related
        self.record_length : int = 10
        self.kelp_prices : List[float] = []
        self.kelp_vwap : List[float] = []
        self.kelp_bbid : List[float] = []
        self.kelp_bask : List[float] = []
        self.resin_bbid : List[float] = []
        self.resin_bask : List[float] = []

    def run(self, state: TradingState):
        
        # Step 1: Setting up values for reference:  
        # # Positions/Inventory      
        self.inventory = state.position
        self.position_limit = POSITION_LIMIT
        self.max_orderable = self.__calc_max_orderable()
        
        # # Order depths
        kelp_order_depth : OrderDepth = state.order_depths[Product.KELP]
        resin_order_depth : OrderDepth = state.order_depths[Product.RESIN]
        
        # # Unpack traderData
        # Start trading when we have data
        if state.traderData:
            traderData = jsonpickle.decode(state.traderData)
            self.inventory = state.traderData
            self.kelp_prices = traderData["kelp_prices"]
            self.kelp_vwap = traderData["kelp_vwap"]
            self.kelp_bask = traderData["kelp_bask"]
            self.kelp_bbid = traderData["kelp_bbid"]
            self.resin_bask = traderData["resin_bask"]
            self.resin_bbid = traderData["resin_bbid"]
            
            # Step 2: Generate the orders for resin
            resin_kwargs : dict = {
                "order_depth":resin_order_depth
            }
            generated_resin_orders = self.resin_orders(**resin_kwargs)
            
            # Step 3: Generate the orders for kelp
            kelp_kwargs : dict = {
                "order_depth":kelp_order_depth
            }
            generated_kelp_orders = self.kelp_orders(**kelp_kwargs)
        # Don't trade at the start
        else:
            traderData = {}
            generated_resin_orders = []
            generated_kelp_orders = []
        
        # Step 4: Record what's necessary for next iteration and submit:
        # # Setup the orders
        result = {}
        result[Product.KELP] = generated_kelp_orders
        result[Product.RESIN] = generated_resin_orders

        # Update the historical data we keep track of
        self.__update_lists(kelp_order_depth,resin_order_depth)
        
        # # To keep track of history
        traderData = jsonpickle.encode({"kelp_prices": self.kelp_prices,
                                        "kelp_vwap": self.kelp_vwap,
                                        "kelp_bask": self.kelp_bask,
                                        "kelp_bbid": self.kelp_bbid,
                                        "resin_bask": self.resin_bask,
                                        "resin_bbid": self.resin_bbid})
        
        # # Conversions to be used in later rounds.
        conversions = 1 
        
        return result, conversions, traderData
    
    def kelp_orders(self, *, order_depth : OrderDepth):
        orders : List[Order] = []
        unformatted_orders : List[Tuple[Product, int, int]] = []
        
        # Check for arbitrage
        last_mid = self.kelp_prices[-1]
        
        # Make market if no arbitrage

        orders = self.__convert_unformatted_to_formatted(unformatted_orders)    
        return orders
        
    def resin_orders(self, *, order_depth : OrderDepth, mean_val : int = 10000):
        orders : List[Order] = []
        unformatted_orders : List[Tuple[Product, int, int]] = []
        
        prod = Product.RESIN
        
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        spread = self.__spread(order_depth)
        
        # Check for arbitrage
        last_mid = self.kelp_prices[-1]
        
        if spread == 1:
            arbitrage_opportunity = [price>last_mid for price in buy_orders].any()
            # Sell above fair
            if arbitrage_opportunity:
                # Sell max_quantity at all bids above last mid.
                arb_bids = {price:buy_orders[price] for price in buy_orders if price>last_mid}
                allowable_sell_volume = abs(self.max_orderable[prod]['sell'])
                total_sell_volume = 0
                for bid in arb_bids:
                    if allowable_sell_volume == 0:
                        break
                    quantity = abs(arb_bids[bid])
                    q = min(quantity,allowable_sell_volume)
                    unformatted_orders.append((prod, bid, -q))
                    allowable_sell_volume -= q
                    total_sell_volume += q
                
                # Buy equivalent quantity at a price below last mid (by 1 unit, determined by data).
                inventory_after_arb = self.inventory[prod] - total_sell_volume
                allowable_buy_vol = self.position_limit[prod] - inventory_after_arb
                quantity_buy = min(total_sell_volume,allowable_buy_vol)
                price_buy = last_mid - 1
                unformatted_orders.append((prod, price_buy, quantity_buy))
            else:
                arbitrage_opportunity = [price<last_mid for price in sell_orders].any() 
                # Buy below fair
                if arbitrage_opportunity:
                    # Buy max_quantity at all asks below last mid.
                    arb_asks = {price:sell_orders[price] for price in sell_orders if price<last_mid}
                    allowable_buy_volume = abs(self.max_orderable[prod]['buy'])
                    total_buy_volume = 0
                    for ask in arb_asks:
                        if allowable_buy_volume == 0:
                            break
                        quantity = abs(arb_asks[ask])
                        q = min(quantity,allowable_buy_volume)
                        unformatted_orders.append((prod, ask, q))
                        allowable_buy_volume -= q
                        total_buy_volume += q
                    
                    # Buy equivalent quantity at a price below last mid (by 1 unit, determined by data).
                    inventory_after_arb = self.inventory[prod] - total_buy_volume
                    allowable_sell_vol = self.position_limit[prod] - inventory_after_arb
                    quantity_sell = min(total_buy_volume,allowable_sell_vol)
                    price_sell = last_mid + 1
                    unformatted_orders.append((prod, price_sell, quantity_sell)) 
                else:
                    # Make market if no arbitrage                  
                    ask_price = last_mid + 1
                    bid_price = last_mid - 1
                    q = min(abs(self.max_orderable[prod]['sell']),self.max_orderable[prod]['buy'])
                    unformatted_orders.append((prod, ask_price, -q))
                    unformatted_orders.append((prod, bid_price, q))
        else:
            # Make market if no arbitrage                  
            ask_price = last_mid + 1
            bid_price = last_mid - 1
            q = min(abs(self.max_orderable[prod]['sell']),self.max_orderable[prod]['buy'])
            unformatted_orders.append((prod, ask_price, -q))
            unformatted_orders.append((prod, bid_price, q))

        orders = self.__convert_unformatted_to_formatted(unformatted_orders)    
        return orders
    
    # ________________Utilities (Simple Calculations, Simple functions/loops)_______________
    def __update_lists(self, kelp_order_depth : OrderDepth, resin_order_depth : OrderDepth):
        self.kelp_prices.append(self.__mid(kelp_order_depth))
        self.kelp_vwap.append(self.__micro(kelp_order_depth))
        self.kelp_bask.append(self.__bask(kelp_order_depth))
        self.kelp_bbid.append(self.__bbid(kelp_order_depth))
        self.resin_bask.append(self.__bask(resin_order_depth))
        self.resin_bbid.append(self.__bbid(resin_order_depth))
        
        if len(self.kelp_prices) > self.record_length:
            self.kelp_prices.pop(0)
        if len(self.kelp_vwap) > self.record_length:
            self.kelp_vwap.pop(0)
        if len(self.kelp_bask) > self.record_length:
            self.kelp_bask.pop(0)
        if len(self.kelp_bbid) > self.record_length:
            self.kelp_bbid.pop(0)
        if len(self.resin_bask) > self.record_length:
            self.resin_bask.pop(0)
        if len(self.resin_bbid) > self.record_length:
            self.resin_bbid.pop(0)
    
    def __create_buy_order(self, product : Product, bid_p : int, quantity : int) -> Order:
        return Order(product, bid_p, quantity)
    
    def __create_sell_order(self, product : Product, ask_p : int, quantity : int) -> Order:
        return Order(product, ask_p, quantity)
    
    def __convert_unformatted_to_formatted(self, unformatted_orders : List[Tuple[Product, int, int]]) -> List[Order]:
        # Function to convert list of tuples to list of Order objects.
        orders : List[Order] = []
        for order in unformatted_orders:
            product, price, quantity = order
            # quantity < 0 implies sell, quantity > 0 implies buy
            if quantity<0:
                order : Order = self.__create_sell_order(product, round(price), quantity)
                orders.append(order)
            elif quantity>0:
                order : Order = self.__create_buy_order(product, round(price), quantity)
                orders.append(order)
        return orders
    
    def __calc_max_orderable(self):
        for product in self.inventory:
            inventory = self.inventory[product]
            pos_lim = self.position_limit[product]
            buy_max = max(pos_lim - inventory,0)
            sell_max = min(-pos_lim - inventory,0)
            self.max_orderable[product] = {"buy": buy_max, "sell": sell_max}
    
    def __bbid(self, order_depth : OrderDepth):
        buy_orders = order_depth.buy_orders
        return max([*buy_orders.keys()])

    def __bask(self, order_depth : OrderDepth):
        sell_orders = order_depth.sell_orders
        return max([*sell_orders.keys()])
    
    def __mid(self, order_depth : OrderDepth):
        sell_orders = order_depth.sell_orders
        buy_orders = order_depth.buy_orders
        
        sell_prices = [*sell_orders.keys()]
        buy_prices = [*buy_orders.keys()]
        
        return (min(sell_prices)+max(buy_prices))/2

    def __spread(self, order_depth : OrderDepth):
        sell_orders = order_depth.sell_orders
        buy_orders = order_depth.buy_orders
        
        sell_prices = [*sell_orders.keys()]
        buy_prices = [*buy_orders.keys()]
        
        return min(sell_prices)-max(buy_prices)

    def __micro(self, order_depth : OrderDepth):
        sell_orders = order_depth.sell_orders
        buy_orders = order_depth.buy_orders
        
        sell_prices = [*sell_orders.keys()]
        buy_prices = [*buy_orders.keys()]
        
        bbid = max(buy_prices)
        bask = min(sell_prices)
        
        bidvol = abs(buy_orders[bbid])
        askvol = abs(sell_orders[bask])
        
        return (bbid*askvol+bask*bidvol)/(bidvol+askvol)