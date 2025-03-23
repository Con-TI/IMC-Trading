# IMC-Trading

Formatting:
- When you want to download a log file and add it to our folder, name it {file_name}@{date}{24 hr time}.log. E.g. tutorial1py@14_03_2025_1508.log
- utilities folder will contain all the stuff that is not related to the main project, like scripts to generate plots, etc.

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

Overview of log files:
- Every log file contains: Sandbox log, trade history
- How to visualize log files:
    Go to utilities streamlit_app.py
    Run "streamlit run streamlit_app.py" from an integrated terminal from the utilities folder.
    Go to the link that is shown in the terminal. This will open a streamlit app. 
    Rest should be straightforward based on the streamlit app.

Libraries we can use:
The following libraries are supported in the simulation. Importing other external libraries is not supported.
- [pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [statistics](https://docs.python.org/3.9/library/statistics.html)
- [math](https://docs.python.org/3.9/library/math.html)
- [typing](https://docs.python.org/3.9/library/typing.html)
- [jsonpickle](https://pypi.org/project/jsonpickle/)