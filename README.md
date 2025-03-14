# IMC-Trading

Formatting:
- When you want to download a log file and add it to our folder, name it {file_name}@{date}{time}.log


Summary of how IMC Trading algos will work:
- Submit 1 Python file
- Python file contains a Trader class with a "run" function
- Every iteration the run is provided with TradingState.
- Trading state contains all trades that have happened and existing orders. 
- run can then send orders to fully or partially match existing orders. If one of our orders is left oustanding, then the bots will trade on it. If this outstanding order is not traded on, it will not continue to the next iteration.
- Upon submission against their sample data, we get a log file which will contain any print statements we put in our code
- For every new product introduced, we get two CSV files, one containing market orders at every time step, one containing trades made that day.

Overview of classes:
- Trader : The class our python file needs to have
- TradingState : Contains the current state of the market for the current iteration (it is what is passed into our Trader's run function as a parameter)
- Order : Represents a trade order
- Trade : Represents a trade that has happened
- OrderDepth : Represents the orderbook

Trading constraints:
- We will have position limits for every product. If we try to submit an order that would push us over the limit, our order will automatically be rejected.

Libraries we can use:
The following libraries are supported in the simulation. Importing other external libraries is not supported.
- [pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [statistics](https://docs.python.org/3.9/library/statistics.html)
- [math](https://docs.python.org/3.9/library/math.html)
- [typing](https://docs.python.org/3.9/library/typing.html)
- [jsonpickle](https://pypi.org/project/jsonpickle/)