from sklearn.model_selection import train_test_split
import requests

# url = "https://apistocks.p.rapidapi.com/daily"

# querystring = {"symbol":"AAPL","dateStart":"2021-07-01","dateEnd":"2021-07-31"}

# headers = {
# 	"X-RapidAPI-Key": "c0d900ab2amsh144286cd4d115f7p1654a2jsn41bd444eaf61",
# 	"X-RapidAPI-Host": "apistocks.p.rapidapi.com"
# }

# response = requests.get(url, headers=headers, params=querystring)

# # print(response.json())

import pandas as pd
# df=pd.DataFrame(response.json()['Results'])
# df.to_csv('stockprice.csv')

df=pd.read_csv('stockprice.csv')


print(df.head())

print(df.info())
print(df.describe())
print(df.columns)

import matplotlib.pyplot as plt

x=df['Low']
y=df['Volume']
plt.scatter(x,y)
plt.show()




# Date: Represents the observation date.
# Open: Opening price of the stock.
# Close: Closing price of the stock.
# High: Highest price of the stock during the day.
# Low: Lowest price of the stock during the day.
# Volume: Total number of shares or contracts traded.
# AdjClose: Adjusted closing price, accounting for stock splits, dividends, etc.

df.rename(columns={
    'Date':'Stock Observation Date',
    'Open':'Opening Price of the Stock' ,
    'Close':'Closing Price of the Stock' ,
    'High':'Highest Price',
    'Low':'Lowest Price',
    'Volume':'Total number of Share trade',
    'AdjClose':'Adjusted Closing Price'
},inplace=True)


# print(df.columns)

x = df[[ 'Opening Price of the Stock',
        'Highest Price', 'Lowest Price', 'Total number of Share trade']]
y = df['Closing Price of the Stock']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=42, test_size=0.2)

# print(x_train)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)


df1 = pd.DataFrame({'ACTUAL': y_test.values.flatten(),
                   'prediction': y_pred.flatten()})
# print(df1)

from sklearn.metrics import r2_score

r2=r2_score(y_test,y_pred)
# print(r2)


openingprice=int(input('enter the opening price of the stock: '))
highestprice = int(input('enter the highest price of the stock: '))
lowestprice = int(input('enter the lowest price of the stock: '))
volume = int(input('enter the total amount of share of the stock: '))


# Create a new data point using the user input
new_data = pd.DataFrame({
    'Opening Price of the Stock':[openingprice],
    'Highest Price':[highestprice],
    'Lowest Price':[lowestprice],
    'Total number of Share trade':[volume],
})


new_prediction = lr.predict(new_data)

print("closing price of the stock:", new_prediction)
