"""
Origin: QE by John Stachurski and Thomas J. Sargent
Filename: wb_download.py
Authors: John Stachurski, Tomohito Okabe
LastModified: 29/08/2013

Dowloads data from the World Bank site on GDP per capita and plots result for
a subset of countries.

NOTE: This is not dually compatible with Python 3.  Python 2 and Python
3 call the urllib package differently.
"""
import sys
import matplotlib.pyplot as plt
from pandas.io.excel import ExcelFile

if sys.version_info[0] == 2:
    from urllib import urlretrieve
elif sys.version_info[0] == 3:
    from urllib.request import urlretrieve

# == Get data and read into file gd.xls == #
wb_data_query = "http://api.worldbank.org/v2/en/indicator/gc.dod.totl.gd.zs?downloadformat=excel"
urlretrieve(wb_data_query, "gd.xls")

# == Parse data into a DataFrame == #
gov_debt_xls = ExcelFile('gd.xls')
govt_debt = gov_debt_xls.parse('Data', index_col=1, na_values=['NA'], skiprows=3)

# == Take desired values and plot == #
govt_debt = govt_debt.transpose()
govt_debt = govt_debt[['AUS', 'DEU', 'FRA', 'USA']]
govt_debt = govt_debt[38:]
govt_debt.plot(lw=2)
plt.show()
