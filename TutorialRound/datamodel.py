'''
Don't edit this file, this is a reference 
of the classes they want us to use
'''
import json
from typing import Dict, List
from json import JSONEncoder
import jsonpickle

Time = int
Symbol = str
Product = str
Position = int
UserId = str
ObservationValue = int


class Listing:
    '''
    A class to represent a product listed in the market.
    Each will have a symbol (to use in our orders and to reference)
    a product (the name of the good, different exchanges can sell the same good)
    and a denomination (currency used)
    '''
    def __init__(self, symbol: Symbol, product: Product, denomination: Product):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination
        
                 
class ConversionObservation:

    def __init__(self, bidPrice: float, askPrice: float, transportFees: float, exportTariff: float, importTariff: float, sugarPrice: float, sunlightIndex: float):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        self.sugarPrice = sugarPrice
        self.sunlightIndex = sunlightIndex
        

class Observation:

    def __init__(self, plainValueObservations: Dict[Product, ObservationValue], conversionObservations: Dict[Product, ConversionObservation]) -> None:
        self.plainValueObservations = plainValueObservations
        self.conversionObservations = conversionObservations
        
    def __str__(self) -> str:
        return "(plainValueObservations: " + jsonpickle.encode(self.plainValueObservations) + ", conversionObservations: " + jsonpickle.encode(self.conversionObservations) + ")"
     

class Order:
    '''
    What we submit (a list of Order objects, one per order) during each run call.
    
    Each order object will contain: 
    - the symbol of the product, 
    - the price to buy/sell (where if we buy its the max buy price,
    and if we sell its the min sell price (so the simulation would 
    probably give us a better price if available))
    - the quantity, positive if buy, negative if sell.
    '''
    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __str__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"

    def __repr__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"
    

class OrderDepth:
    '''
    A class that contains all 
    outstanding buy and sell orders.
    Orders are aggregated in a dict.
    
    Format {price:quantity,price:quantity,...}
    where the quantity is positive if on buy_orders side
    and negative if on sell_orders side
    
    E.g. if under buy_orders we have {9:5, 10:4}, 
    this means there are 5 pending orders at price 9 
    and 4 pending orders at price 10.
    if under sell_orders we have {12:-3, 11:-2},
    this means there are 3 pending sell orders at price 12
    and 2 pending sell orders at price 11.
    '''
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


class Trade:
    '''
    Trade class:
    Every 'trade' has a symbol, quantity, price,
    and identity of the buyer, and the seller,
    identity will be "SUBMISSION" if an algorithm was the buyer/seller
    '''
    def __init__(self, symbol: Symbol, price: int, quantity: int, buyer: UserId=None, seller: UserId=None, timestamp: int=0) -> None:
        self.symbol = symbol
        self.price: int = price
        self.quantity: int = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

    def __str__(self) -> str:
        return "(" + self.symbol + ", " + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(self.quantity) + ", " + str(self.timestamp) + ")"

    def __repr__(self) -> str:
        return "(" + self.symbol + ", " + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(self.quantity) + ", " + str(self.timestamp) + ")"


class TradingState(object):
    '''
    TradingState contains the current iteration's trading state
    - order_depths : the current order book
    - timestamp : current time
    - traderData : past data (if we need it)
    - own_trades : the trades our bot has executed (buyer/seller will be "SUBMISSION" to indicate which side we took)
    - market_trades : the trades that have been executed on the market
    - observations : 
    - position : our current positions in the market (dictionary of {product_name:net_position})
    '''
    def __init__(self,
                 traderData: str,
                 timestamp: Time,
                 listings: Dict[Symbol, Listing],
                 order_depths: Dict[Symbol, OrderDepth],
                 own_trades: Dict[Symbol, List[Trade]],
                 market_trades: Dict[Symbol, List[Trade]],
                 position: Dict[Product, Position],
                 observations: Observation):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations
        
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

    
class ProsperityEncoder(JSONEncoder):

        def default(self, o):
            return o.__dict__