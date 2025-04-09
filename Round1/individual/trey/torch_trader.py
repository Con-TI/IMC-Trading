from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string
import jsonpickle
import numpy as np

weight_matrix = np.array([[ 7.8487e-01,  3.0751e-02,  2.5513e-02, -6.9503e-02, -9.4861e-02,
          9.5216e-02, -4.9669e-03,  2.0373e-02,  4.0509e-03,  1.7254e-02,
          2.2188e-01, -3.7380e-02,  5.3508e-02, -1.7075e-01, -1.0644e-01],
        [ 2.5841e-02,  8.6753e-01,  4.2633e-02,  6.6994e-02, -2.2017e-01,
          3.7897e-04,  2.1455e-01, -2.2192e-03,  3.9439e-02,  4.2349e-02,
         -3.0342e-02,  1.3684e-01, -1.1450e-01, -1.6257e-01, -6.5367e-02]])
bias_matrix = np.array([-0.0551,  0.0386])

features = ["highest_vol_bid_change",
            "highest_vol_ask_change",
            "spread",
            "adjusted_spread",
            "total_volume_pressure",
            "adjusted_total_volume_pressure",
            "adjusted_microprice_change",
            'adjusted_micro_price_spread_ratio',
            'bidrange',
            'askrange',
            "overall_weighted_bid_change",
            "overall_weighted_ask_change",
            "overall_weighted_spread",
            "overall_weighted microprice change",
            "overall_weighted microprice spread_ratio"
            ]

class Product:
    KELP = "KELP"
    RESIN = "RAINFOREST_RESIN"
    SQUID = "SQUID_INK"

class Trader():
    def __init__(self):
        self.depth_memory : Dict[Product, OrderDepth] = {}
        
    def run(self, state : TradingState):
        result = {}
        
        if state.traderData:
            data = jsonpickle.decode(state.traderData)
            if data.get('depth_memory',0) != 0:
                self.depth_memory = data['depth_memory']
        
                order_depths = state.order_depths
                squid_order_depth = order_depths[Product.SQUID]
                input_vec, bid, ask = self.squid_input_vec(squid_order_depth)
                output_vec = input_vec @ weight_matrix + bias_matrix
                bid_change = output_vec[0]
                ask_change = output_vec[1]
                
                pred_bid = bid+bid_change
                pred_ask = ask+ask_change
                
                print(pred_bid, pred_ask)
        
        traderData = jsonpickle.encode({
            'depth_memory': state.order_depths
        })
        conversions = 1
        return result, conversions, traderData

    def squid_input_vec(self, current_depth : OrderDepth):
        previous_depth = self.depth_memory[Product.SQUID]
        
        prev_spread, prev_highest_vol_bid, prev_highest_vol_ask, prev_total_volume_pressure, prev_adjusted_total_volume_pressure, prev_adjusted_spread, prev_adjusted_microprice, prev_bidrange, prev_askrange, prev_weighted_bid, prev_weighted_ask, prev_weighted_microprice, prev_weighted_spread = self.calculate_important_values(previous_depth)
        spread, highest_vol_bid, highest_vol_ask, total_volume_pressure, adjusted_total_volume_pressure, adjusted_spread, adjusted_microprice, bidrange,askrange, weighted_bid, weighted_ask, weighted_microprice, weighted_spread =self.calculate_important_values(current_depth)
    
        highest_vol_bid_change = highest_vol_bid - prev_highest_vol_bid
        highest_vol_ask_change = highest_vol_ask - prev_highest_vol_ask
        adjusted_microprice_change = adjusted_microprice - prev_adjusted_microprice
        adjusted_micro_price_spread_ratio = (adjusted_microprice - highest_vol_bid)/spread
        weighted_bid_change = weighted_bid - prev_weighted_bid
        weighted_ask_change = weighted_ask - prev_weighted_ask
        weighted_microprice_change = weighted_microprice - prev_weighted_microprice
        weighted_micro_price_spread_ratio = (weighted_microprice- weighted_bid)/weighted_spread
    
        squid_input = np.array([highest_vol_bid_change, 
                       highest_vol_ask_change, 
                       spread, adjusted_spread, 
                       total_volume_pressure, 
                       adjusted_total_volume_pressure,
                       adjusted_microprice_change,
                       adjusted_micro_price_spread_ratio,
                       bidrange,
                       askrange,
                       weighted_bid_change,
                       weighted_ask_change,
                       weighted_spread,
                       weighted_microprice_change,
                       weighted_micro_price_spread_ratio])
        squid_input = np.expand_dims(squid_input,0).T
        
        return squid_input, highest_vol_bid, highest_vol_ask
    
    def calculate_important_values(self, depth : OrderDepth):
        buy_volumes = [int(v) for p,v in depth.buy_orders.items()]
        buy_prices = [int(p) for p,v in depth.buy_orders.items()]
        highest_vol_bid = buy_prices[np.argmax(buy_volumes)]
        
        sell_volumes = [int(v) for p,v in depth.sell_orders.items()]
        sell_prices = [int(p) for p,v in depth.sell_orders.items()]
        highest_vol_ask = buy_prices[np.argmin(sell_volumes)]
        
        spread = min(sell_prices) - max(buy_prices)
                
        total_volume_pressure = sum(buy_volumes)/(sum(buy_volumes)+abs(sum(sell_volumes)))
        adjusted_total_volume_pressure = max(buy_volumes)/(max(buy_volumes)+abs(min(sell_volumes)))
        
        adjusted_spread = highest_vol_ask-highest_vol_bid
        adjusted_microprice = (highest_vol_ask*max(buy_volumes) + highest_vol_bid*abs(min(sell_volumes)))/(max(buy_volumes)+abs(min(sell_volumes)))
        
        bidrange = max(buy_prices)-min(buy_prices)
        askrange = max(sell_prices)-min(sell_prices)
        
        weighted_mean_bids = sum([int(p)*int(v) for p,v in depth.buy_orders.items()])/sum(buy_volumes)
        weighted_mean_asks = sum([int(p)*int(v) for p,v in depth.buy_orders.items()])/sum(sell_volumes)
        weighted_spread = weighted_mean_asks-weighted_mean_bids
        weighted_microprice = (weighted_mean_bids*abs(sum(sell_volumes)) + weighted_mean_asks*sum(buy_volumes))/(sum(buy_volumes)+abs(sum(sell_volumes)))
        
        return (spread,
                highest_vol_bid, 
                highest_vol_ask, 
                total_volume_pressure, 
                adjusted_total_volume_pressure, 
                adjusted_spread, 
                adjusted_microprice, 
                bidrange,
                askrange,
                weighted_mean_bids,
                weighted_mean_asks,
                weighted_microprice,
                weighted_spread)
        