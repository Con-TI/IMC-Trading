Tasks:
- Make backtesting code (Since a run on IMC prosperity's server takes a few minutes, it'd be faster to have a fully fledged backtesting system based off of some benchmark backtests (e.g. nothingbot))
    - [] Algorithms for benchmark backtests: ** Explanation: we'll look at the trade history to determine whether our algorithm's quotes would've been executed per timestamp. However, we'd need multiple references and not just the nothing bot to figure out how the IMC bots will trade given say our quotes were better/worse than theirs and how much of it would fill. This is a tough problem.**
        - [x] Nothing bot
        - [] ...
    - [] 
- [x] Make visualization (could copy from some git)
    - [x] Streamlit app to show charts and dataframes
    - [x] Log processor
    - [] Streamlist app has arrow keys to scroll through each order book at every timestamp
- [] Rounds (bots/algos to make):
    - [] All rounds:
        - [] Find a way to check against 2024 and 2023 data to see if we have an exact match. ** Explanation: If we do, would be safe to say that the bots they've made are working in the exact same way so we can exploit that since the price path roughly moves the same regardless of what our bot does, i.e. we can take advantage of look ahead bias. **
    - [] Tutorial round:
        - [] Bots to make:
            - [] Whale LP from algosoc competition
            - [] 
    