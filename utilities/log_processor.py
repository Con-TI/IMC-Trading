'''
Processes the log files we get from IMC after we submit our algorithms.
'''
import re
import pandas as pd
from io import StringIO
import json

class LogProcessor:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.log_file = None
        self.sandbox_logs = None
        self.activities_log = None
        self.trade_history = None
        
    def change_file_path(self, log_file_path):
        self.log_file_path = log_file_path    
    
    def read_log(self):
        with open(self.log_file_path, 'r') as file:
            self.log_file = file.read()
            
        # Reads the sandbox logs
        self.sandbox_logs = re.search(r'Sandbox logs:(.*?)Activities log:', self.log_file, re.DOTALL)
        self.sandbox_logs = self.sandbox_logs.group(1).strip()
        self.sandbox_logs = re.findall(r'\{(?:.*?\n){4}.*?\}', self.sandbox_logs)
        self.sandbox_logs = [json.loads(match) for match in self.sandbox_logs]
        self.sandbox_logs = pd.DataFrame(self.sandbox_logs)
        self.sandbox_logs = self.sandbox_logs.set_index('timestamp')
        
        # Reads the activities_log csv
        self.activities_log = re.search(r'Activities log:(.*?)Trade History:', self.log_file, re.DOTALL)
        self.activities_log = self.activities_log.group(1).strip()
        self.activities_log = pd.read_csv(StringIO(self.activities_log), sep=";")
        
        # Reads the trade history list
        self.trade_history = re.search(r'Trade History:(.*?)]', self.log_file, re.DOTALL)
        self.trade_history = self.trade_history.group(1).strip()
        self.trade_history = self.trade_history + "]"
        self.trade_history = eval(self.trade_history)
        self.trade_history = pd.DataFrame(self.trade_history)
        
if __name__ == '__main__':
    '''
    Sample usage of the class
    log_file_path = {Round name}/logs/{filename}.log
    '''
    
    # log_processor = LogProcessor('TutorialRound/logs/tutorial1py@14_03_2025_1508.log')
    # log_processor.read_log()
    # print(log_processor.sandbox_logs)
    # print(log_processor.activites_log)
    # print(log_processor.trade_history)

    log_processor = LogProcessor('Round1/logs/nothing_bot@07_04_2025_1300.log')
    log_processor.read_log()
    print(log_processor.sandbox_logs)
    print(log_processor.activities_log)
    print(log_processor.trade_history)