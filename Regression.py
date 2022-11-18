import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("temperatures.csv")
x=df['YEAR']
y=df['ANNUAL']
plt.figure(figsize=(40,40))
plt.title('Temp plot ')
plt.xlabel('Year')
plt.ylabel('Annual Avg temp')
plt.scatter(x, y)
x=x.values
x=x.reshape(117,1)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)
print(regressor.coef_)
print(regressor.intercept_)
print(regressor.predict([[2024]]))

predicted = regressor.predict(x)
print(predicted)
y=y.values
print(y)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y, predicted))
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y, predicted))
from sklearn.metrics import r2_score
print(r2_score(y, predicted))

plt.title('Temp plot ')
plt.xlabel('Year')
plt.ylabel('Annual Avg temp')
plt.scatter(x, y, label='actual')
plt.plot(x, predicted, label='predicted')
plt.legend()
plt.show()