import websocket, json
import asyncio
import pandas as pd 
import dateutil.parser
import datetime
from datetime import date, datetime, timedelta
from dateutil import tz
from concurrent.futures import ThreadPoolExecutor
import BotV2.bot_csv


alpaca_supported_coins = [
    "AAVE", "BAT", "BCH", # 0 - 2
    "BTC", "ETH", "GRT",  # 3 - 5
    "LINK", "LTC", "MKR", # 6 - 8
    "PAXG", "SHIB", "UNI" # 8 - 11
]



class StreamManager:
    def __init__(self, ticker_list: list, market: str = "USD") -> None:
        self.ticker_list = ticker_list
        self.market = market
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''
    def create_threads(self):
        market = "USD"
        # Create a ThreadPoolExecutor with the desired number of threads
        with ThreadPoolExecutor(max_workers=len(self.ticker_list)) as executor:
            # Submit the tasks to the executor
            futures = [executor.submit(StreamData(ticker, market).start_streaming) for ticker in self.ticker_list]
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()
    '''------------------------------------'''















class StreamData:
    def __init__(self, ticker: str, market: str = "USD", refresh: bool = True) -> None:
        self.ticker = ticker.upper()
        self.market = market.upper()
        self.trade_pair = f"{self.ticker}-{self.market}"
        self.path_to_csv = f"D:\\Coding\\VisualStudioCode\\Projects\\Python\\TA_Bot\\CandleStorage\\{self.ticker}_candles.csv"
        self.socket = 'wss://ws-feed.pro.coinbase.com'
        self.minutes_processed = {}
        self.minute_candlesticks = []
        self.current_tick = None
        self.previous_tick = None
        self.csv_handling = BotV2.bot_csv.CsvHandler(ticker)


    '''------------------------------------'''
    def start_streaming(self):
        ws = websocket.WebSocketApp(self.socket, on_open=self.on_open, on_message=self.on_message)
        ws.run_forever()
    '''------------------------------------'''
    def on_open(self, ws):
        print("Connection is opened")
        subscribe_msg = {
            "type": "subscribe",
            "channels": [
                {
                    "name": "ticker",
                    "product_ids": [
                        f"{self.ticker}-{self.market}"
                    ]
                }
            ]
        }

        ws.send(json.dumps(subscribe_msg))
    '''------------------------------------'''
    def on_message(self, ws, message):

        self.previous_tick = self.current_tick
        self.current_tick = json.loads(message)

        tick_datetime_object = dateutil.parser.parse(self.current_tick['time'])
        timenow = self.convert_to_local_timezone(timestamp=tick_datetime_object)
        
        tick_dt = timenow.strftime("%Y-%m-%d %H:%M")

        if not tick_dt in self.minutes_processed:
            print("This is a new candlestick")
            self.minutes_processed[tick_dt] = True

            if len(self.minute_candlesticks) > 0:
                self.minute_candlesticks[-1]['close'] = self.previous_tick['price']
            self.minute_candlesticks.append({
                'time': tick_dt,
                'open': self.current_tick['price'],
                'high': self.current_tick['price'],
                'low': self.current_tick['price'],
                "volume": self.current_tick["volume_24h"]
            })


            df = pd.DataFrame(self.minute_candlesticks[:-1])
            df = df.iloc[-1]
            #self.csv_handling.write_to_csv(df)
            
            #print(f"---------------------------------\nAppending: {appending_df}   Type: {type(appending_df)}\n\n")
            #appending_df.to_csv(self.path_to_csv, mode="a")

        if len(self.minute_candlesticks) > 0:
            current_candlestick = self.minute_candlesticks[-1]
            if self.current_tick['price'] > current_candlestick['high']:
                current_candlestick['high'] = self.current_tick['price']
            if self.current_tick['price'] < current_candlestick['low']:
                current_candlestick['low'] = self.current_tick['price']

    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------     Utilities          ------------------------------------'''
    '''------------------------------------'''
    def convert_to_local_timezone(self, timestamp):
        '''
        This function assumes your timestamps are in UTC format.
        It will convert the timestamp to your local timezone. '''

        # Assign from_zone to the UTC timezone, since the timestamps are expected to be in UTC format. 
        from_zone = tz.tzutc()
        to_zone = tz.tzlocal()
        # Add the timezone information to the timestamp.
        timestamp = timestamp.replace(tzinfo=from_zone)
        # Convert the timezone.
        converted_time = timestamp.astimezone(to_zone)
        return converted_time
    
    '''------------------------------------'''
    
    

#if __name__ == "__main__":
#    main()
