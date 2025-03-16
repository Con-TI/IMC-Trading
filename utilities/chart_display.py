from log_processor import LogProcessor
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt


class Display():
    def __init__(self, log_processor: LogProcessor):
        self.log_processor = log_processor
        
    def display_activites_log(self, symbol: str):
        fig, ax = plt.subplots(nrows=2,ncols=2)
        
        df = self.log_processor.activites_log
        df = df[df['product'] == symbol]
        price_data = df.drop(columns=['profit_and_loss'])
        
        # Profit_and_loss over time
        ax[0,0].set_title('Profit_and_loss over time')
        ax[0,0].plot(df['timestamp'],df['profit_and_loss'],color='k')       
        ax[0,0].fill_between(df['timestamp'], df['profit_and_loss'], 0, 
                where=(df['profit_and_loss'] >= 0), color='green', alpha=0.3)
        ax[0,0].fill_between(df['timestamp'], df['profit_and_loss'], 0, 
                        where=(df['profit_and_loss'] < 0), color='red', alpha=0.3) 
        ax[0,0].axhline(0, color='black', linewidth=1, linestyle='--')
        
        # Orderbook over time
        ax[1,0].set_title('Midprice, best bid, and best ask over time')
        ax[1,0].plot(price_data['timestamp'],price_data['bid_price_1'],color='red')
        ax[1,0].plot(price_data['timestamp'],price_data['ask_price_1'],color='lime')
        ax[1,0].plot(price_data['timestamp'],price_data['bid_price_2'],color='darkred')
        ax[1,0].plot(price_data['timestamp'],price_data['ask_price_2'],color='darkgreen')
        ax[1,0].plot(price_data['timestamp'],price_data['bid_price_3'],color='lightcoral')
        ax[1,0].plot(price_data['timestamp'],price_data['ask_price_3'],color='palegreen')
        ax[1,0].plot(price_data['timestamp'],price_data['mid_price'],color='k')

        plt.tight_layout()
        plt.show()
        
    


if __name__ == '__main__':
    log_processor = LogProcessor('TutorialRound/logs/tutorial1py@14_03_2025_1508.log')
    log_processor.read_log()
    display = Display(log_processor)
    display.display_activites_log('KELP')