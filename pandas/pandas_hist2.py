import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/test_pwt.csv')
countries = df.pop('country')
df.index = countries
df['GDP percap'] = df['tcgdp'] / df['POP']
df = df.sort_values(by='GDP percap', ascending=False)
df['GDP percap'].plot(kind='bar')
plt.show()