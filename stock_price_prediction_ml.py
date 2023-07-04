from sklearn.model_selection import train_test_split
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Fetching stock price data from API
# url = "https://apistocks.p.rapidapi.com/daily"
# querystring = {"symbol": "AAPL", "dateStart": "2021-07-01", "dateEnd": "2021-07-31"}
# headers = {
#     "X-RapidAPI-Key": "c0d900ab2amsh144286cd4d115f7p1654a2jsn41bd444eaf61",
#     "X-RapidAPI-Host": "apistocks.p.rapidapi.com"
# }
# response = requests.get(url, headers=headers, params=querystring)

# Loading data from a CSV file
df = pd.read_csv('stockprice.csv')

# Displaying the first few rows of the DataFrame
print(df.head())

# Displaying information about the DataFrame
print(df.info())

# Displaying statistical summary of the DataFrame
print(df.describe())

# Displaying the column names of the DataFrame
print(df.columns)

# Scatter plot of 'Low' vs 'Volume'
x = df['Low']
y = df['Volume']
plt.scatter(x, y)
plt.show()

# Renaming column names to match the given names
df.rename(columns={
    'Date': 'Stock Observation Date',
    'Open': 'Opening Price of the Stock',
    'Close': 'Closing Price of the Stock',
    'High': 'Highest Price',
    'Low': 'Lowest Price',
    'Volume': 'Total number of Share trade',
    'AdjClose': 'Adjusted Closing Price'
}, inplace=True)

# Selecting input (x) and output (y) columns for the model
x = df[['Opening Price of the Stock', 'Highest Price',
        'Lowest Price', 'Total number of Share trade']]
y = df['Closing Price of the Stock']

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=42, test_size=0.2)

# Creating and training the linear regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Making predictions on the test set
y_pred = lr.predict(x_test)

# Creating a DataFrame with actual and predicted values
df1 = pd.DataFrame({'ACTUAL': y_test.values.flatten(),
                   'prediction': y_pred.flatten()})

# Calculating the R2 score
r2 = r2_score(y_test, y_pred)

# Prompting user for new data point
opening_price = int(input('Enter the opening price of the stock: '))
highest_price = int(input('Enter the highest price of the stock: '))
lowest_price = int(input('Enter the lowest price of the stock: '))
volume = int(input('Enter the total amount of shares of the stock: '))

# Creating a new data point using user input
new_data = pd.DataFrame({
    'Opening Price of the Stock': [opening_price],
    'Highest Price': [highest_price],
    'Lowest Price': [lowest_price],
    'Total number of Share trade': [volume],
})

# Making a prediction for the new data point
new_prediction = lr.predict(new_data)

print("Closing price of the stock:", new_prediction)
