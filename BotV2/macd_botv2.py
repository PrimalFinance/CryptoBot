# Asynchronous related imports.
import asyncio

# Datetime related imports.
import datetime as dt

# Pandas related imports.
import pandas as pd

# Import the bot for the parent class. 
import Bot.bot_brain

# Streaming data from a websocket. 
import DataCollection.stream_data




class MACD_Bot(Bot.bot_brain.BotBrain):
    def __init__(self, tickers: list, market: str, exchange: str, trade_size: float = 100.0, trend_start_gap: int = 2, loop_duration: int = 60, data_limit: int = 100) -> None:
        '''
        ticker: Symbol for the cryptocurrency. 
        market: Use your native currency to determine which markets you want to trade in. 
                For example, in U.S. markets you can use USD, USDT, USDC, etc. and in European markets you can use EUR. 
        exchange: Name of the centralized exchange to pull data from. 
        trade_size: The amount that will be bought in a trade. 
        
        '''
        self.tickers = tickers
        self.market = market
        self.exchange = exchange
        self.trade_size = trade_size
        self.trend_start_gap = trend_start_gap
        self.loop_duration = loop_duration
        self.data_limit = data_limit
        self.max_trend_length = 5
        self.max_candle_distance = 5

        super().__init__(market=self.market, data_exchange=self.exchange, trade_exchange=self.exchange, data_limit=self.data_limit)


    '''------------------------------------ Setup ------------------------------------'''
    '''------------------------------------'''
    async def start_trading(self):
        
        task_single_trade = [self.single_trade_execution(t) for t in self.tickers]
        task_connect_socket = [self.connect_to_socket(self.tickers)]
        await asyncio.gather(*task_single_trade, *task_connect_socket)
        
    '''------------------------------------'''
    '''------------------------------------ Trade Execution ------------------------------------'''       
    '''------------------------------------'''
    async def single_trade_execution(self, ticker: str):
        print(f"TICKER: {ticker}")
        # Loop control
        trade_loop = True
        position = False
        # String for the bots name
        bot_name = f"{ticker} Bot"
        while trade_loop:
            # Get the current positions. 
            cur_pos = self.alpaca.get_current_positions(ticker, self.market)
            # Parse information from the current position. 
            cur_pos_index = cur_pos[3]
            cur_pos_status = cur_pos[2]

            # Get the candle data.Run asynchronously so multiple instances of this coroutine can collect data at the same time. 
            candle_data = await self.run_in_thread(self.get_OHLCV_data, ticker, "1m")
            date_timestamp, time_timestamp = str(dt.datetime.now()).split(" ")
            time_timestamp = time_timestamp.split(".")[0]
            print(f"{date_timestamp} {time_timestamp}")
            # Get the index of the most recent candle.
            mr_candle_index = candle_data.index[-1]
            # Add the MACD data to the dataframe.
            candle_data = self.get_MACD(candle_data)
            # Get the trend locations.
            trend_locations = self.find_MACD_trend_locations(candle_data)
            # Parse the most recent trend.
            mr_trend = trend_locations[-1]
            mr_trend_len = len(mr_trend)
            
            ################################################### SELL LOGIC ###################################################
            '''
            If there is a current position in the ticker pair. 
            Logic below will determine when the trend ends and sell. 
            '''
            if cur_pos_status:
                ''' 
                Check if the "most recent candle index" is in the "most recent trend".
                If it is, do not do anything. Wait for the trend to end to sell. 
                ''' 
                if mr_candle_index in mr_trend:
                    print(f"[{bot_name}] - Sell Wait | Trend in progress {mr_trend} {mr_candle_index}")
                
                # If the current candle is not in the most recent trend. 
                else:
                    # Assuming the candle is not in the trend, calculate the distance from the current candle and the last trend candle. 
                    # This will be used to determine if the trend just ended and it is time to sell. 
                    candle_distance = mr_candle_index - mr_trend[-1]
                    # Get information about the current position.
                    cur_pos_qty = float(cur_pos[0][cur_pos_index].qty)
                    # Check if the distance between the current candle, and the trend candle is close to the end of the recent trend. 
                    # Preferably, the bot will exit the candle the trend ends. 
                    if candle_distance > 0 and candle_distance < self.max_candle_distance:
                        print(f"[{bot_name}] - SELL EXECUTED {mr_trend} {mr_candle_index}")
                        self.alpaca.place_market_order(ticker, qty=cur_pos_qty, side="sell")
                    elif candle_distance > self.max_candle_distance:
                        print(f"[{bot_name}] - Sell Wait | Current Candle too far away from trend {mr_trend} {mr_candle_index}") 

            ################################################### Buy LOGIC ###################################################
            elif not cur_pos_status:
                # Check if the most recent index is in the most recent trend.
                if mr_candle_index in mr_trend:
                    # Check that the trend is longer that the "start gap" and that it is shorter than the "max length". 
                    # The start gap avoids entering trends that are short lived.
                    # The max length avoides entering trends that are already ongoing. 
                    if mr_trend_len >= self.trend_start_gap and mr_trend_len <= self.max_trend_length:
                        # Get the most recent candle closing price.
                        mr_price = candle_data["close"].iloc[-1]
                        # Calculate the quantity to buy based on the most recent closing price, and the "trade_size" class variable. 
                        calculated_quantity = self.trade_size / mr_price
                        print(f"[{bot_name}] - BUY EXECUTED | {mr_trend} {mr_candle_index}")
                        self.alpaca.place_market_order(ticker, qty=calculated_quantity, side="buy")
                    else:
                        if mr_trend_len < self.trend_start_gap:
                            print(f"[{bot_name}] - Buy Wait | Trend Too Short {mr_trend} {mr_candle_index}")    
                        elif mr_trend_len > self.max_trend_length:
                            print(f"[{bot_name}] - Buy Wait | Trend Too Long {mr_trend} {mr_candle_index}")
                
                # If the current candle is not in the current trend. 
                else:
                    print(f"[{bot_name}] - Buy Wait | No Trend")
            print("__________________________________________")
            await asyncio.sleep(self.loop_duration)


    '''------------------------------------'''

    '''------------------------------------ Thread Execution ------------------------------------'''       
    '''------------------------------------'''
    async def run_in_thread(self, func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
    '''------------------------------------'''
    '''------------------------------------'''
    '''------------------------------------'''

    '''------------------------------------ Utilities ------------------------------------'''
    '''------------------------------------'''
    def find_MACD_trend_locations(self, df:pd.DataFrame) -> list:
        # Loop through the recent data to get the trend locations. 
        index = 0
        trend_index = 0
        prev_row = None
        trend_locations = []
        for index, row in df.iterrows():
            if index == df.index[0]: 
                prev_row = row
            else:
                # Assign current row to variable.
                cur_row = row

                # Checks if the current row has a macd crossover but the previous row does not. 
                # This indicates the start of trend. 
                # If cur_row = True & prev_row = False
                if cur_row['MACD_Over'] and not prev_row['MACD_Over']:
                    # Store the index in a new list at the location of the trend_index within trend_locations
                    # We store the value in a new list because this indicates the start of a trend.
                    try: 
                        trend_locations[trend_index] = [index]
                    # Occurs if list is new. 
                    except IndexError:
                        trend_locations.append([index])
                # Checks if the trend is continuing. 
                # If cur_row = True & prev_row = True
                elif cur_row['MACD_Over'] and prev_row['MACD_Over']:
                    # Since we already created a list when the trend started, we are now appending new indexes to it. 
                    trend_locations[trend_index].append(index)
                # Checks if the trend stopped.
                # If cur_row = False & prev_row = True.
                elif not cur_row['MACD_Over'] and prev_row['MACD_Over']:
                    # If the current row is false, and the previous row is true, this indicates the end of the trend. 
                    # Since the current index (cur_row) is false, we will not append any values to the list. 
                    # However we will increment the "trend_index", so a new list can be created once the next trend is detected. 
                    trend_index += 1
                # If no trend is occuring 
                # If cur_row = False & prev_row = False
                elif not cur_row['MACD_Over'] and not prev_row['MACD_Over']:
                    # Since we are not interested in data without trends present, we pass. 
                    pass

                prev_row = row

        return trend_locations

