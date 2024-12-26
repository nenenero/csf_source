import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.utils.data as data

from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

train_df = pd.read_csv('count_csv\\final_train.csv')
test_df = pd.read_csv('count_csv\\final_test.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(train_df)

tr_timeseries = train_df[['Day','TimePeriod','WeekDay','Flight_nums']].values.astype('float32')
te_timeseries = test_df[['Day','TimePeriod','WeekDay','Flight_nums']].values.astype('float32')

new = pd.concat([train_df,test_df],axis=0).reset_index().drop('index',axis=1)
new_timeseries = new[['Day','TimePeriod','WeekDay','Flight_nums']].values.astype('float32')

#print(tr_timeseries)
#print(new_timeseries)

scaler = MinMaxScaler()
tr_timeseries = scaler.fit_transform(tr_timeseries)
te_timeseries = scaler.transform(te_timeseries)

def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[:,:3][i:i+lookback]
        target = dataset[:,3][i:i+lookback]
        X.append(feature)
        y.append(target)
    return torch.tensor(X).to(device), torch.tensor(y).to(device)
def mae(pred, true):
    return np.mean(np.abs(pred - true))

def mape(pred, true):
    true = np.where(true == 0, 1e-10, true)  
    return np.mean(np.abs((pred - true) / true)) * 100


lookback = 48
train, test = tr_timeseries, te_timeseries
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test,lookback=lookback)

X_train, X_test = X_train, X_test
y_train, y_test = y_train, y_test

loader = data.DataLoader(data.TensorDataset(X_train,y_train),batch_size = 16, shuffle=True)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3,
                            num_layers=2,
                            hidden_size=64,
                            batch_first=True,
                            bidirectional = False)
        self.dropout = nn.Dropout(0.5).to(device)
        #self.linear1 = nn.Linear(128*2, 64).to(device)
        #self.linear1 = nn.Linear(128, 64).to(device)
        self.linear2 = nn.Linear(64, 8).to(device)
        self.output_linear = nn.Linear(8,1).to(device)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        #x = self.linear1(x)
        x = self.linear2(x)
        x = self.output_linear(x)
        return x

model = LSTMModel().to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.00002, weight_decay = 0.0000005)

loss_fn = nn.MSELoss().to(device)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor = 0.7, patience=5, verbose=True)

class CustomEarlyStopping:
    def __init__(self, patience=20, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None
        self.best_y_pred = None
    
    def __call__(self, val_loss, model, X):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_state = model.state_dict()
            with torch.no_grad():
                self.best_y_pred = model(X)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}, score: {self.best_score}')
    
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_state = model.state_dict()
            with torch.no_grad():
                self.best_y_pred = model(X)
            self.counter = 0
            
early_stopping = CustomEarlyStopping(patience=15, verbose=True)

best_score = None
best_weights = None
best_train_preds = None
best_test_preds = None

n_epochs = 200

for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred.squeeze(),y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        y_pred = model(X_train)
        
        train_rmse = np.sqrt(loss_fn(y_pred.cpu(), y_train.cpu().unsqueeze(2)))
        train_preds = y_pred.clone().detach().cpu().numpy()
        
        y_pred = model(X_test) 
        test_rmse = np.sqrt(loss_fn(y_pred.cpu(), y_test.cpu().unsqueeze(2)))
        test_preds = y_pred.clone().detach().cpu().numpy()

        scheduler.step(test_rmse)

        if best_score is None or test_rmse < best_score:
            best_score = test_rmse
            best_weights = model.state_dict()
            best_train_preds = train_preds
            best_test_preds = test_preds

        early_stopping(test_rmse, model, X_test)

        if early_stopping.early_stop:
            print('Early stopping')
            break
    if epoch % 10 == 0:
        print('*'*10, 'Epoch: ', epoch, '\ train RMSE: ', train_rmse, '\ test RMSE', test_rmse)

if best_weights is not None:
    model.load_state_dict(best_weights)

    with torch.no_grad():
        y_pred_train = model(X_train).clone().detach().cpu().numpy()
        y_pred_test = model(X_test).clone().detach().cpu().numpy()
    
with torch.no_grad():
    train_plot = np.ones_like(new_timeseries) * np.nan
    train_plot[lookback:len(train)] = y_pred_train[:,-1,:]

    test_plot = np.ones_like(new_timeseries) * np.nan
    test_plot[len(train) + lookback: len(new_timeseries)] = y_pred_test[:,-1,:]

train_predictions = scaler.inverse_transform(train_plot)
test_predictions = scaler.inverse_transform(test_plot)

plt.figure(figsize=(12,6))
plt.plot(new_timeseries[:,3], c = 'b')
plt.plot(train_predictions[:,3], c='r')
plt.plot(test_predictions[:,3], c='g')

plt.show()


eval_df = pd.concat([test_df['Flight_nums'].reset_index(),
                     pd.Series(test_predictions[:,3][len(train):].reshape(-1).tolist())],axis=1).drop('index',axis=1)
#print(test_df.iloc[4:]['Flight_nums'])

eval_df.columns = ['real_flight_nums', 'pred_flight_nums']
print(eval_df.iloc[lookback:]['real_flight_nums'])
print(eval_df.iloc[lookback:]['pred_flight_nums'])
r_mse = np.sqrt(mean_squared_error(eval_df.iloc[lookback:]['real_flight_nums'], eval_df.iloc[lookback:]['pred_flight_nums']))
Mae = mae(eval_df.iloc[lookback:]['pred_flight_nums'].values, eval_df.iloc[lookback:]['real_flight_nums'].values)
Mape = mape(eval_df.iloc[lookback:]['pred_flight_nums'].values, eval_df.iloc[lookback:]['real_flight_nums'].values)
print('r_mase: ',r_mse)
print('Mae: ',Mae)
print('mape: ',Mape)
print(r_mse)


torch.save(model, 'models\mutlti_saved_model.pth')
print("Model saved to 'saved_model.pth'")
'''''
new_data = pd.read_csv('count_csv\\predict.csv')

# 对新数据进行归一化
scaled_new_data = scaler.transform(new_data[['Day', 'TimePeriod', 'WeekDay', 'Flight_nums']].values.astype('float32'))

# 创建推理数据集
def create_inference_dataset(dataset, lookback):
    X = []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:i + lookback, :3]  # 获取特征数据
        X.append(feature)
    return torch.tensor(X)

X_infer = create_inference_dataset(scaled_new_data, lookback)



# 初始化预测数组
predictions = []

# 对前 96 个数据进行预测
with torch.no_grad():
    for i in range(lookback, len(scaled_new_data)):
        # 获取当前时间窗口的数据
        input_seq = torch.tensor(scaled_new_data[i-lookback:i, :3]).unsqueeze(0).to(device)
        
        # 使用模型进行预测
        pred = model(input_seq)
        predicted_value = pred.squeeze().cpu().numpy()  # 获取预测的值
        
        predictions.append(predicted_value)

        # 将预测值加入到输入序列中，更新输入
        new_input = np.append(scaled_new_data[i, :3], predicted_value)
        scaled_new_data[i, :3] = new_input[:3]  # 更新窗口数据

# 逆归一化预测值
predictions = np.array(predictions).reshape(-1, 1)
predicted_values = scaler.inverse_transform(np.concatenate([scaled_new_data[:, :3], predictions], axis=1))[:, 3]

# 将预测值存入原数据
new_data['Predicted_Flight_nums'] = predicted_values

# 保存结果
new_data.to_csv('count_csv\\new_data_with_predictions.csv', index=False)
print("Predictions saved to 'count_csv\\new_data_with_predictions.csv'")

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(new_data.index, new_data['Flight_nums'], label='Real', color='blue')
plt.plot(new_data.index, new_data['Predicted_Flight_nums'], label='Predicted', color='orange')
plt.legend()
plt.show()
'''''
unknown_df = pd.read_csv('count_csv\\predict.csv')  # 未知数据文件
unknown_timeseries = unknown_df[['Day', 'TimePeriod', 'WeekDay', 'Flight_nums']].values.astype('float32')

# 使用相同的 scaler 进行归一化
unknown_timeseries = scaler.transform(unknown_timeseries)

# 推理的滚动预测
#lookback = 24  # 滑窗长度
current_window = unknown_timeseries[:lookback, :3]  # 初始化滑窗
predictions = []

with torch.no_grad():
    for i in range(len(unknown_timeseries) - lookback):
        # 将当前窗口转换为张量
        input_tensor = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 模型预测
        pred = model(input_tensor).squeeze().cpu().numpy()
        predictions.append(pred[-1])  # 取最新预测值

        # 更新滑窗，丢弃最旧值，加入最新预测
        current_window = np.vstack([current_window[1:], np.hstack([unknown_timeseries[lookback + i, :3], pred[-1]])[:3]])

# 将预测结果转回原始尺度
predictions = np.array(predictions).reshape(-1, 1)
extended_predictions = np.hstack([unknown_timeseries[lookback:len(predictions) + lookback, :3], predictions])
original_scale_predictions = scaler.inverse_transform(extended_predictions)[:, 3]
print(original_scale_predictions)
# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(range(len(unknown_timeseries)), unknown_timeseries[:, 3], label='Input Data', c='b')
plt.plot(range(lookback, lookback + len(original_scale_predictions)), original_scale_predictions, label='Predictions', c='r')
plt.legend()
plt.show()

# 保存预测结果到 CSV
result_df = pd.DataFrame({
    'TimeIndex': range(lookback, lookback + len(original_scale_predictions)),
    'Predicted_Flight_nums': original_scale_predictions
})
#result_df.to_csv('count_csv\\new_data_with_predictions.csv', index=False)