"""
Origin: QE by John Stachurski and Thomas J. Sargent
Filename: wb_download.py
Authors: John Stachurski, Tomohito Okabe
LastModified: 29/08/2013

Dowloads data from the World Bank site on GDP per capita and plots result for
a subset of countries.
"""
import matplotlib.pyplot as plt
import requests
import pandas as pd

# == Get data and read into file gd.xls == #
wb_data_query = "http://api.worldbank.org/v2/en/indicator/gc.dod.totl.gd.zs?downloadformat=excel"
r = requests.get(wb_data_query)
with open('gd.xls', 'wb') as output:
    output.write(r.content)

# == Parse data into a DataFrame == #
govt_debt = pd.read_excel('gd.xls', sheetname='Data', skiprows=3, index_col=1)

# == Take desired values and plot == #
govt_debt = govt_debt.transpose()
govt_debt = govt_debt[['AUS', 'DEU', 'FRA', 'USA']]
govt_debt = govt_debt[38:]
govt_debt.plot(lw=2)
plt.show()
