# Asynchronous and multiprocessing related imports
import asyncio
from concurrent.futures import ThreadPoolExecutor 
import multiprocessing


from DataCollection.stream_datav2 import StreamManager
from BotV2.bot_brainv2 import BotBrainV2


def start_streaming(tickers: list, market: str = "USD"):
    sm = StreamManager(ticker_list=tickers, market=market)
    sm.create_threads()

    """
    for ticker in tickers:
        process = multiprocessing.Process(target=StreamData(ticker, market).start_streaming)
        process.start()
        process.join()
    """

def start_bot(tickers: list, market: str = "USD"):
    bot = BotBrainV2(market)
    bot.test()







def test1():
    tickers = ["ERGO", "ADA"]
    market = "USD"

    process1 = multiprocessing.Process(target=start_streaming, args=(tickers, market))
    process2 = multiprocessing.Process(target=start_bot, args=(tickers, market))

    process1.start()
    process2.start()

    process1.join()
    process2.join()


def test2():
    pass

if __name__ == "__main__":
    test1()
