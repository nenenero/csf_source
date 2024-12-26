import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import csv


def set_random_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_random_seed(seed)

def mae(pre_y, true_y):
    return np.mean(np.abs(pre_y - true_y))

def mape(pre_y, true_y):
    true_y = np.where(true_y == 0, 1e-10, true_y)  
    return np.mean(np.abs((pre_y - true_y) / true_y)) * 100


train_df = pd.read_csv('count_csv\\final_csv.csv')
print(f"len(train_df):{len(train_df)}")

Flight_nums = train_df['Flight_nums'].values
plt.plot([i for i in range(len(Flight_nums))], Flight_nums)
plt.show()

scaler = MinMaxScaler()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Flight_nums = scaler.fit_transform(Flight_nums.reshape(-1, 1))

def build_data(data, time_step=36):
    dataX = []
    dataY = []
    for i in range(len(data) - time_step):
        dataX.append(data[i:i+time_step])
        dataY.append(data[i+time_step])
    dataX = np.array(dataX).reshape(len(dataX), time_step, -1)
    dataY = np.array(dataY)
    return dataX, dataY

dataX, dataY = build_data(Flight_nums, time_step= 36)
print(f"dataX.shape:{dataX.shape}, dataY.shape:{dataY.shape}")


def train_val_test_split(dataX, dataY, shuffle=True, train_percentage=0.8, val_percentage=0.1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)  
    if shuffle:
        random_num = np.arange(len(dataX))
        np.random.shuffle(random_num)
        dataX = dataX[random_num]
        dataY = dataY[random_num]
    
    train_split = int(len(dataX) * train_percentage)
    val_split = int(len(dataX) * (train_percentage + val_percentage))
    
    train_X = dataX[:train_split]
    train_Y = dataY[:train_split]
    
    val_X = dataX[train_split:val_split]
    val_Y = dataY[train_split:val_split]
    
    test_X = dataX[val_split:]
    test_Y = dataY[val_split:]
    
    return train_X, train_Y, val_X, val_Y, test_X, test_Y

train_X, train_Y, val_X, val_Y, test_X, test_Y = train_val_test_split(dataX, dataY, shuffle=True, train_percentage=0.8, val_percentage=0.1, random_state=seed)
print(f"train_X.shape:{train_X.shape}, train_Y.shape:{train_Y.shape}, val_X.shape:{val_X.shape}, val_Y.shape:{val_Y.shape}, test_X.shape:{test_X.shape}, test_Y.shape:{test_Y.shape}")

def rmse(pre_y, true_y):
    return np.sqrt(np.mean((pre_y - true_y) ** 2))


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


input_size = 1
hidden_size = 32
num_layers = 3
output_size = 1
num_epochs = 300
batch_size = 32
learning_rate = 0.0001

model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
criterion = nn.MSELoss()


train_losses = []
val_losses = []

print("Start Training")
for epoch in range(num_epochs):
    np.random.seed(seed) 
    random_num = np.arange(len(train_X))
    np.random.shuffle(random_num)
    train_X = train_X[random_num]
    train_Y = train_Y[random_num]

    for i in range(0, len(train_X) // batch_size):
        train_X1 = torch.Tensor(train_X[i * batch_size:(i + 1) * batch_size]).to(device)
        train_Y1 = torch.Tensor(train_Y[i * batch_size:(i + 1) * batch_size]).to(device)

        model.train()
        optimizer.zero_grad()
        output = model(train_X1)
        train_loss = criterion(output, train_Y1)
        train_loss.backward()
        optimizer.step()

    if epoch % 50 == 0:

        model.eval()
        with torch.no_grad():
            val_x1 = torch.Tensor(val_X).to(device)
            val_y1 = torch.Tensor(val_Y).to(device)
            val_output = model(val_x1)
            val_loss = criterion(val_output, val_y1)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        print(f"Epoch {epoch}, Train Loss: {train_loss.item()}, Validation Loss: {val_loss.item()}")


torch.save(model.state_dict(), 'models/saved_model.pth')
print("Model saved to 'models/saved_model.pth'")


plt.figure(figsize=(10, 6))
plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


test_X1 = torch.Tensor(test_X).to(device)
test_pre = model(test_X1).detach().cpu().numpy()
true_y = np.concatenate((test_Y))
true_y = scaler.inverse_transform(true_y.reshape(-1, 1)).T[0]
test_pre = scaler.inverse_transform(test_pre.reshape(-1, 1)).T[0]

print(f"rmse(pre_y,true_y):{rmse(test_pre, true_y)}")
mae_value = mae(test_pre, true_y)
mape_value = mape(test_pre, true_y)
print(f"MAE: {mae_value}")
print(f"MAPE: {mape_value:.2f}%")
plt.title("Test Results")
x = [i for i in range(len(true_y))]
plt.plot(x, test_pre, marker="o", markersize=1, label='Predicted')
plt.plot(x, true_y, marker="x", markersize=1, label="True")
plt.legend()
plt.show()
'''
train_df = pd.read_csv('count_csv\\final_csv.csv')
print(f"len(train_df):{len(train_df)}")

Flight_nums = train_df['Flight_nums'].values
plt.plot([i for i in range(len(Flight_nums))], Flight_nums)
plt.show()

scaler = MinMaxScaler()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Flight_nums = scaler.fit_transform(Flight_nums.reshape(-1, 1))
'''

pre_pred = train_df['Flight_nums'].values

pre_pred = pre_pred[-96*5:]
print('before scaler pre_pred', pre_pred)
pre_pred = scaler.fit_transform(pre_pred.reshape(-1,1))

future_steps = 96*8  


print('after scaler pre_pred', pre_pred)
input_seq = pre_pred  # (time_step, input_size)

predictions = [] 
model.eval()  
with torch.no_grad():
    for _ in range(future_steps):
        input_tensor = torch.Tensor(input_seq).unsqueeze(0).to(device)  
        pred = model(input_tensor)  #  (1, output_size)
        pred_value = pred.item()  
        predictions.append(pred_value)
        input_seq = np.roll(input_seq, -1, axis=0) 
        input_seq[-1, 0] = pred_value  

predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).T[0]
print('predictions: ', predictions)

plt.figure(figsize=(10, 6))
plt.title("LSTM Future Prediction")
x_history = range(len(Flight_nums))
plt.plot(x_history, scaler.inverse_transform(Flight_nums).T[0], label="Historical Data", marker='x', markersize=2)
x_future = range(len(Flight_nums), len(Flight_nums) + future_steps)
plt.plot(x_future, predictions, label="Predicted Data", marker='o', markersize=2, color='red')
plt.axvline(x=len(Flight_nums) - 1, color='gray', linestyle='--', label="Prediction Start")
plt.legend()
plt.show()

eff = 0
weekday_write = 7 
with open('test_csv\\lstm_with_val_pre1220.csv',mode='w', newline = '', encoding= 'utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Day', 'TimePeriod','Weekday','Flight_nums'])
    for day in range(1,9):
        time_period = 5
        weekday_write = weekday_write % 8
        if weekday_write == 0:
            weekday_write = 1
        for Flight_nums in predictions[eff:eff+96]:
            writer.writerow([day, time_period,weekday_write,Flight_nums])
            time_period += 15
        eff += 96
        weekday_write += 1

#predictions.to_csv('count_csv\\lstm_with_val_pre.csv')
#torch.save(model.state_dict(), 'models\saved_model.pth')
#print("Model saved to 'saved_model.pth'")