import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


data = pd.read_csv('count_csv\\final_train.csv')  


average_flight_per_time = data.groupby('TimePeriod')['Flight_nums'].mean().reset_index()

average_flight_per_time.columns = ['TimePeriod', 'Average_Flight_nums']

average_flight_per_time.to_csv('count_csv\\average_flight_per_time.csv', index=False)

print(average_flight_per_time)

real_data = pd.read_csv('count_csv\\final_test.csv')

def mae(pred, true):
    return np.mean(np.abs(pred - true))

def mape(pred, true):
    true = np.where(true == 0, 1e-10, true)  # 避免除以零
    return np.mean(np.abs((pred - true) / true)) * 100


real_values = real_data[:95]['Flight_nums'].values
predicted_values = average_flight_per_time[:95]['Average_Flight_nums'].values

rmse = np.sqrt(mean_squared_error(real_values, predicted_values))

mae_value = mae(predicted_values, real_values)
mape_value = mape(predicted_values, real_values)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae_value:.2f}")
print(f"MAPE: {mape_value:.2f}%")

plt.figure(figsize=(12, 6))
plt.plot(predicted_values, label='Predicted (Average)', c='b')
plt.plot(real_values, label='Real', c='r')
plt.legend()
plt.show()
