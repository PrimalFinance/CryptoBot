# Pandas related imports.
import pandas as pd

# Imports for crypto exchanges
import ccxt
from ccxt.base.errors import BadSymbol # Symbol not supported @ exchange.







class BotDataRetrieval:
    def __init__(self, exchange: str = "Coinbase", source: str = "ccxt") -> None:
        self.exchange = exchange.lower()
        self.source = source.lower()
        
        # List of available sources. 
        available_sources = ["ccxt"]

        # Check if a valid source was passed. If not, default to ccxt. 
        if self.source not in available_sources:
            self.source = "ccxt"

        # If source is ccxt, create object. 
        if self.source == "ccxt":
            self.exchange_obj = getattr(ccxt, self.exchange)

    '''------------------------------------'''
    def get_OHLCV_data(self, ticker:str, market: str = "USD", timeframe:str='1m', limit:int=1000, convert_to_local: bool = True) -> pd.DataFrame:
        '''
        Get the Open, High, Low, Close, Volume data for the candle. 
        The length of the candle is specified by the parameter "timeframe". 
        '''
        if self.source == "ccxt":
            try:
                # Create a trading pair to retrieve. Ex: BTC/USDT
                trading_pair = f"{ticker}/{market}"
                # Column names for the dataframe.
                columns = ['time', 'open', 'high', 'low', 'close', 'volume']
                # Get the candles data at the desired timeframe. 
                candles = self.data_exchange.fetch_ohlcv(trading_pair, timeframe, limit=limit)
                # Convert list of lists to dataframe.
                candles_df = pd.DataFrame(candles)
                candles_df.columns = columns
                # Convert volume into local market currency. 
                candles_df['$volume'] = candles_df['volume'] * candles_df['close']
                # Format timestamps column. 
                candles_df['time'] = pd.to_datetime(candles_df['time'], unit='ms')
                # Convert timestamps to local time.
                if convert_to_local:
                    candles_df['time'] = [self.convert_to_local_timezone(i) for i in candles_df['time']] 

                return candles_df
            except BadSymbol as e:
                print(f"[OHLCV Error] {e}")

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
