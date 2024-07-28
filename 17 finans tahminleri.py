import pandas as pd
from prophet import Prophet
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read the data from a CSV file
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/nvda_data.csv")
df = data.copy()

# Rename columns to fit Prophet requirements
df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

# Convert the 'ds' column to datetime format
df['ds'] = pd.to_datetime(df['ds'])

# Initialize the Prophet model
model = Prophet()
# Fit the model to the data
model.fit(df)

# Create a datafram
# e to hold future dates for prediction
feature = model.make_future_dataframe(periods=360)

# Make predictions on the future dataframe
tahmin = model.predict(feature)

# Plot the forecasted data
fig1 = model.plot(tahmin)
fig2 = model.plot_components(tahmin)

# Display the plots
plt.show()

# Print the dataframe
print(df)

# The following commented-out code seems to be for logistic regression, which is not used in time series forecasting.

"""
# Prepare the data for logistic regression
y = df["ds"]
x = df.drop(columns=["ds"])

# Split the data into training and testing sets
x_train , x_test , y_train , y_test = train_test_split(x, y, random_state=30, train_size=0.77)

# Initialize the logistic regression model
lr = LogisticRegression(max_iter=1000)

# Fit the model to the training data
model = lr.fit(x_train, y_train)

# Calculate the model's score on the test data
score = model.score(x_test, y_test)

# Print the score
print(score)
"""

