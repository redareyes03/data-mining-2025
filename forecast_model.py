import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Datasets/train.csv', sep=';')

# Convert 'y' to a numerical format (1 for 'yes', 0 for 'no')
df['y'] = df['y'].map({'yes': 1, 'no': 0})


month_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}
df['month_num'] = df['month'].map(month_map)
# Create a 'date' column
df['date'] = pd.to_datetime(df.apply(lambda row: f"2023-{row['month_num']}-{row['day']}", axis=1), errors='coerce')

df.dropna(subset=['date'], inplace=True)

daily_subscriptions = df.groupby('date')['y'].sum().reset_index()
daily_subscriptions.rename(columns={'y': 'subscriptions'}, inplace=True)

print("--- Daily Subscription Time Series ---")
print(daily_subscriptions.head())

# Feature Engineering for Time Series
daily_subscriptions['time_index'] = np.arange(len(daily_subscriptions))

# Create cyclical features
daily_subscriptions['month'] = daily_subscriptions['date'].dt.month
daily_subscriptions['month_sin'] = np.sin(2 * np.pi * daily_subscriptions['month'] / 12)
daily_subscriptions['month_cos'] = np.cos(2 * np.pi * daily_subscriptions['month'] / 12)

# Define features (X) and target (y)
features = ['time_index', 'month_sin', 'month_cos']
X = daily_subscriptions[features]
y = daily_subscriptions['subscriptions']

# Train the Linear Regression Model
model = LinearRegression()
model.fit(X, y)

print("\n--- Model Coefficients ---")
print(f"Intercept: {model.intercept_:.4f}")
for i, feature in enumerate(features):
    print(f"Coefficient for {feature}: {model.coef_[i]:.4f}")

# Predict/Forecast Future Data
# forecast for the next 30 days
last_time_index = X['time_index'].max()
last_date = daily_subscriptions['date'].max()

# Create future time indices and dates
future_time_indices = np.arange(last_time_index + 1, last_time_index + 31)
future_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, 31)])

# Create features for the future dates
future_months = future_dates.month
future_month_sin = np.sin(2 * np.pi * future_months / 12)
future_month_cos = np.cos(2 * np.pi * future_months / 12)

# Create the future feature set
X_future = pd.DataFrame({
    'time_index': future_time_indices,
    'month_sin': future_month_sin,
    'month_cos': future_month_cos
})

# Make predictions for the future
future_predictions = model.predict(X_future)

# Ensure predictions are non-negative
future_predictions[future_predictions < 0] = 0

# Create a DataFrame for the forecast
forecast_df = pd.DataFrame({
    'date': future_dates,
    'forecast': future_predictions
})

print("\n--- Forecast for the Next 30 Days ---")
print(forecast_df)

# Chart
plt.figure(figsize=(14, 7))
# Plot historical data
plt.plot(daily_subscriptions['date'], daily_subscriptions['subscriptions'], label='Historical Daily Subscriptions', color='blue')
# Plot the forecast
plt.plot(forecast_df['date'], forecast_df['forecast'], label='Forecasted Subscriptions', color='red', linestyle='--')

plt.title('Time Series Forecast of Daily Subscriptions')
plt.xlabel('Date')
plt.ylabel('Number of Subscriptions')
plt.legend()
plt.grid(True)
plt.show()
