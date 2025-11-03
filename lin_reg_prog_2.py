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



# perform exploratory analysis here:
x = df[['FirstServe']]
y = df[['TotalPointsWon']]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 6)

lrg = LinearRegression()

model = lrg.fit(x_train, y_train)

y_predict = model.predict(x_test)
print(model.score(x_test, y_test))

plt.scatter(y_test, y_predict, alpha = 0.5)
plt.show()
plt.clf()

x_2 = df[['FirstServePointsWon', 'SecondServePointsWon']] 
y_2 = df[['TotalPointsWon']]

x_2train, x_2test, y_2train, y_2test = train_test_split(x_2, y_2, train_size = 0.8)

mlrg = LinearRegression()
mlrg.fit(x_2train, y_2train)

y_2_predict = mlrg.predict(x_2test)

print(mlrg.score(x_2test, y_2test))

plt.scatter(y_2test, y_2_predict, alpha = 0.5)
plt.show()
plt.clf()

x_3 = df[['Aces', 'TotalPointsWon', 'Wins', 'Ranking']]
y_3 = df[['Winnings']]

x_3train, x_3test, y_3train, y_3test = train_test_split(x_3, y_3, train_size = 0.8)

multi = LinearRegression()

multi.fit(x_3train, y_3train)
print(multi.score(x_3test, y_3test))

y_3predict = multi.predict(x_3test)

plt.scatter(y_3test, y_3predict, alpha = 0.5)
plt.show()
plt.clf()