import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv('house_price.csv')
print("\nFirst few rows of the dataset:")
print(df.head())
df = df.dropna()  
X = df[['SquareFootage']]  
y = df['Price']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error (MSE):", mse)
plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')  
plt.plot(X_test, y_pred, color='red', label='Predicted Prices')  
plt.title('Simple Linear Regression: House Price Prediction')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.legend()
plt.show()
