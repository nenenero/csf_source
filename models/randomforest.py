import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
def calculate_hourly_rmse(y_true, y_pred, time_step=4):
   
    hourly_rmse = []
    for i in range(0, len(y_true), time_step):
        true_hour = y_true[i:i + time_step]
        pred_hour = y_pred[i:i + time_step]
        print('hour: ' ,i+1)
        print('true_hour ',true_hour)
        print('pred_hour ',pred_hour)
        if len(true_hour) == time_step:  
            rmse_value = np.sqrt(mean_squared_error(true_hour, pred_hour))
            hourly_rmse.append(rmse_value)
    return hourly_rmse

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


train_df = pd.read_csv('count_csv/final_train.csv')
test_df = pd.read_csv('count_csv/final_test.csv')



df1 = pd.read_csv('count_csv\\final_csv.csv')
df1 = shuffle(df1,random_state=42)

print('df: ',df1)


X = df1[['Day', 'TimePeriod', 'WeekDay']].values
y = df1['Flight_nums'].values
#test_size = 0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''
split_num = int(len(X)*(1-test_size))
X_train = X[:split_num]
X_test = X[split_num:]
y_train = y[:split_num]
y_test = y[split_num:]

print('y_test: ',y_test)
'''

rf_model = RandomForestRegressor(
    n_estimators=100,                  # 决策树数量
    criterion='squared_error',         # 节点分裂准则 MSE
    max_features=None,                 # 划分时考虑的最大特征比例
    max_depth=20,                      # 最大深度
    min_samples_split=2,               # 内部节点分裂的最小样本数
    min_samples_leaf=1,                # 叶子节点的最小样本数
    min_weight_fraction_leaf=0,        # 叶子节点中样本的最小权重
    max_leaf_nodes=512,                 # 叶子节点的最大数量
    min_impurity_decrease=0,           # 节点划分不纯度的阈值
    bootstrap=True,                    # 有放回采样
    oob_score=False,                   # 不使用袋外测试
    random_state=42
)

rf_model.fit(X_train, y_train)

y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)
hourly_rmse = calculate_hourly_rmse(y_test, y_test_pred, time_step=4)
train_mape = calculate_mape(y_train, y_train_pred)
test_mape = calculate_mape(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
print("Hourly RMSE:")
for hour, rmse_value in enumerate(hourly_rmse):
    print(f"Hour {hour + 1}: RMSE = {rmse_value:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(hourly_rmse) + 1), hourly_rmse, marker='o', label='Hourly RMSE')
plt.title("Hourly RMSE for Random Forest Predictions")
plt.xlabel("Hour")
plt.ylabel("RMSE")
plt.grid()
plt.legend()
plt.show()
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Train MAPE: {train_mape:.4f}%")
print(f"Test MAPE: {test_mape:.4f}%")
print(f"Train MAE: {train_mae:.4f}")
print(f"Test MAE: {test_mae:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), y_test, label='Real', color='blue')
plt.plot(range(len(y_test)), y_test_pred, label='Predicted', color='green')
plt.xlabel('Sample Index')
plt.ylabel('Flight Numbers')
plt.legend()
plt.title('Random Forest Prediction vs Real')
plt.show()

new_data = pd.read_csv('count_csv\\predict.csv')
new_features = new_data[['Day', 'TimePeriod', 'WeekDay']].values
predictions = rf_model.predict(new_features)

new_data['Predicted_Flight_nums'] = predictions

print(new_data)
#new_data.to_csv('count_csv\\rf_predicted1220.csv')

plt.figure(figsize=(12,6))

plt.plot(new_data['Predicted_Flight_nums'], label='Predicted Flight Numbers', color='r', linestyle='--', marker='x')

plt.title('Predicted Flight Numbers')
#plt.xlabel('Timeperiod')
plt.ylabel('Flight Numbers')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()