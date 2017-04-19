import numpy as np
import pandas as pd 
import datetime as dt 
from pandas_datareader import data,wb
import matplotlib.pyplot as plt 

ticker_list = {'INTC': 'Intel',
               'MSFT': 'Microsoft',
               'IBM': 'IBM',
               'BHP': 'BHP',
               'TM': 'Toyota',
               'AAPL': 'Apple',
               'AMZN': 'Amazon',
               'BA': 'Boeing',
               'QCOM': 'Qualcomm',
               'KO': 'Coca-Cola',
               'GOOG': 'Google',
               'SNE': 'Sony',
               'PTR': 'PetroChina'}

start = dt.datetime(2013, 1, 1)
end = dt.datetime.today()

price_change = {}

for ticker in ticker_list:
    prices = data.DataReader(ticker, 'yahoo', start, end)
    closing_prices = prices['Close']
    change = 100 * (closing_prices[-1] - closing_prices[0]) / closing_prices[0]
    name = ticker_list[ticker]
    price_change[name] = change

pc = pd.Series(price_change)
pc.sort_values(inplace=True)
fig, ax = plt.subplots(figsize=(10,8))
pc.plot(kind='bar', ax=ax)
plt.show()