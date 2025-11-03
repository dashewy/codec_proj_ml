import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

print(df.head())

prod_per_year = df.groupby('year').totalprod.mean().reset_index()

X = prod_per_year.year.values.reshape(-1,1)

y = prod_per_year.totalprod

regr = linear_model.LinearRegression().fit(X, y)

print(regr.coef_[0])
print(regr.intercept_)

y_predict = regr.predict(X)

X_future = np.arange(2013, 2050).reshape(-1, 1)

future_predict = regr.predict(X_future)


plt.scatter(X, y)
plt.plot(X, y_predict)
plt.plot(X_future, future_predict)
plt.show()
plt.clf()

# another project

import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:

df = pd.read_csv('tennis_stats.csv')
print(df.head())



plt.scatter(df.FirstServe, df.Wins)
plt.show()
plt.clf()
plt.scatter(df.Ranking, df.Wins)
plt.show()
plt.clf()