from datamodel import OrderDepth, UserId, TradingState, Order, Listing
from typing import List
import string

"""
Strategy:

Market-maker:
    - Sets constant boundaries on rainforest resin since it just fluctuates
    - Implements micro-price midprice strat on kelp

"""



class Trader:
    def run(self, state: TradingState):
        pass