# Pandas related imports
import pandas as pd
import pandas_ta as pta


# Date & time related imports 
import time 
import datetime as dt


# Imports for crypto exchanges
import ccxt
from ccxt.base.errors import BadSymbol # Symbol not supported @ exchange. 






class BotBrainV2:
    # Preset periods for indicators. Can be altered within function parameters.
    rsi_period = 14
    macd_short_period = 12
    macd_long_period = 26
    macd_period = 9
    bol_band_period = 14

    def __init__(self, market:str, exchange: str = "Coinbase", data_limit:int=100) -> None:
        
        # Class variables assigned from parameters
        self.market = market 
        self.exchange_name = exchange.lower() 
        self.data_exchange = getattr(ccxt, self.exchange_name)()
        self.trade_exchange = exchange
        self.data_limit = data_limit
        # Check that the data limit is not higher than 1000. 1-minute candle data is only offered 1000 max.
        if self.data_limit > 1000:
            self.data_limit = 1000

        # List to hold main list of tickers. 
        self.ticker_list = []

        # List to hold "Coins of Interest" (COI). 
        self.coi_list = []
    


    '''------------------------------------ Data Retrieval ------------------------------------'''
    '''------------------------------------'''
    def test(self):
        while True:
            print(f"TAG")
            time.sleep(10)
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
