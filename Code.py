"""
Data-driven prediction of flow hemodyanmics in cardiovascular models with POD+LSTM
POD is a linear reduction method.
LSTM is a predictive nueral network.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import glob
import matplotlib.tri as tri
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import imageio
from typing import Tuple, Union
from torchsummary import summary
from tqdm import tqdm as tqdm
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import math
from scipy.interpolate import interp1d
from torch.utils.data import Dataset, DataLoader
import copy
from torchvision import transforms
import torchvision
import torch.nn.init as init
from scipy.linalg import norm
from skimage.transform import resize
import shutil
import cv2
from scipy.linalg import svd
import gzip


#Exercise 1
T  = 1
dt = 0.0005
dt_in_T = int(T / dt)
n_snapshots = 500

net  = 'POD'    # Reduction method
cond = 'Rest'   # physiologinal condition
var  = 'U'      # variable of interest

df = pd.read_csv('BC.csv')

#Velocity Function Interpolation
tv_values = df['t1'].tolist()
v_values = df['v'].tolist()
#Create interpolation function
interp_func = interp1d(tv_values, v_values)
# Generate 2000 uniform values
uniform_tv_values = np.linspace(min(tv_values), max(tv_values), n_snapshots)
interpolated_v_values = interp_func(uniform_tv_values)

#Pressure Function Interpolation
tp_values = df['t2'].tolist()
tp_values = [x for x in tp_values if not math.isnan(x)]
P_values = df['P'].tolist()
P_values = [x for x in P_values if not math.isnan(x)]
#Create interpolation function
interp_func = interp1d(tp_values, P_values)
#Generate 2000 uniform values
uniform_tp_values = np.linspace(min(tp_values), max(tp_values), n_snapshots)
interpolated_P_values = interp_func(uniform_tp_values)


### Scaling and load
state = 'load' #load | save | unnecessary
if state == 'save':

    scaler_U = MinMaxScaler()
    min_U = 0
    max_U = 255

    U = U.reshape(-1,1)
    U = scaler_U.fit_transform(U)
    variable = U.reshape(dt_in_T,1,h,w)
    np.save("S4_1channelImageScaled.npy", (variable))

elif state == 'load':
    variable = np.load('variable.npz', allow_pickle=True)['variable']
    print("1Channel (greyscale) image of (0~1) is read")

elif state == 'unnecessary':
    print("I wanna load directly the torch tensor, which contatins also  the rotated images")


ntt, ch, wid, hei = variable.shape

### Apply POD

variable = variable # (nt, ch, w, h)
variable = variable.reshape(len(variable), -1).T 
U, S, VT = svd(variable, full_matrices=False)

# Determine the number of modes to retain
num_modes = 9  # Adjust this based on your requirements

energy_retained = np.cumsum(S) / np.sum(S)
plt.plot(energy_retained[:50], marker='o')
plt.xlabel('# Modes')
plt.ylabel('Energy')
plt.show()
# Energy content calculation
total_energy = np.sum(S)
retained_energy = np.sum(S[:num_modes])
energy_content = retained_energy / total_energy * 100
print("With", num_modes, "number of modes, the energy content is:", energy_content)

np.savez('S_matrix.npz', S=S)
#data = np.load(f'/content/drive/MyDrive/Paper/POD/{cond}_{net}+LSTM_{var}/S_{cond}_{net}+LSTM_{var}.npz')
#S = data['S']

"""### Reconstruction"""

q = U[:,:num_modes].T @ variable

variable_reconst = U[:,:num_modes] @ q

RelErr1 = np.zeros((q.shape[1]))
for t in range(q.shape[1]):
        # print(t)
        RelErr1[t] = (norm(variable[:,t] - variable_reconst[:,t])/norm(variable[:,t]) ) * 100

plt.plot(RelErr1)
plt.xlabel('Snapshot')
plt.ylabel('L2 Norm Difference %')
plt.title('Reconstruction Error')
plt.show()


### LSTM

t = np.linspace(1, n_snapshots, num=n_snapshots)
inputt = np.concatenate((t.reshape(-1, 1), interpolated_v_values.reshape(-1, 1), interpolated_P_values.reshape(-1, 1), q.T), axis=1)
outputt = np.copy(q).T


#%% Latent Space Scaling
min_vals_input = np.min(inputt, axis=0)
max_vals_input = np.max(inputt, axis=0)
input_scaled = inputt - min_vals_input[np.newaxis, :]
input_scaled /= (max_vals_input - min_vals_input)[np.newaxis, :]

min_vals_output = np.min(outputt, axis=0)
max_vals_output = np.max(outputt, axis=0)
output_scaled = outputt - min_vals_output[np.newaxis, :]
output_scaled /= (max_vals_output - min_vals_output)[np.newaxis, :]

"""### Time sequencing"""

time_window = 100

x_train = []
y_train = []

for i in range(0,len(input_scaled) - time_window -1):
    x_train.append( input_scaled[i : (i+time_window) , :] )
    y_train.append( output_scaled[i+time_window,:])

total_x = np.array(x_train)
total_y = np.array(y_train)

"""### Splitting train and test"""

Test_split = 0.8

x_train = total_x[:int(Test_split * total_x.shape[0]),:,:]
y_train = total_y[:int(Test_split * total_y.shape[0])]

x_test = total_x[int(Test_split * total_x.shape[0]):, :,:]
y_test = total_y[int(Test_split * total_y.shape[0]):]

# Convert data to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

"""### Class LSTM"""

input_size = num_modes + 3 # 3 = (t & v & P)
hidden_size = 512
output_size = num_modes
sequence_length = time_window
learning_rate = 0.000005
num_epochs = 2000
batch_size = 20
early_stop_patience = 100
dropout_prob = 0.2

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True, dropout=dropout_prob)
        self.fc1 = nn.Linear(hidden_size, 9)

        # Initialize LSTM weights
        for name, param in self.lstm1.named_parameters():
            if 'weight' in name:
                init.xavier_normal_(param)
            elif 'bias' in name:
                init.constant_(param, 0.0)

        # Initialize linear layer weights
        init.xavier_normal_(self.fc1.weight)
        init.constant_(self.fc1.bias, 0.0)

    def forward(self, x):
        h01 = torch.zeros(3, x.size(0), self.hidden_size).requires_grad_()
        c01 = torch.zeros(3, x.size(0), self.hidden_size).requires_grad_()
        out1, (h01, c01) = self.lstm1(x, (h01.detach(), c01.detach()))
        out = nn.functional.relu(self.fc1(out1[:, -1, :]))
        return out

model_lstm = LSTMNet(input_size, hidden_size, output_size, dropout_prob)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=learning_rate)




best_loss = float('inf')
early_stop_count = 0

state_lstm = 'load'  # train | load

if state_lstm == 'train':
    for epoch in range(num_epochs):
        model_lstm.train()
        for i in range(0, x_train.shape[0], batch_size):
            batch_x = x_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            # Forward pass
            outputs = model_lstm(batch_x)

            # Compute loss and backpropagation
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate the model on training and test data
        with torch.no_grad():
            model_lstm.eval()
            train_outputs = model_lstm(x_train)
            train_loss = criterion(train_outputs, y_train)
            test_outputs = model_lstm(x_test)
            test_loss = criterion(test_outputs, y_test)

        # Print training and test loss every epoch
        print('Epoch [{}/{}], Train Loss: {:.6f}, Test Loss: {:.6f}'.format(epoch+1, num_epochs, train_loss.item(), test_loss.item()))

        # Check if the current test loss is the best so far
        if test_loss < best_loss:
            best_loss = test_loss
            early_stop_count = 0

        else:
            early_stop_count += 1
            if early_stop_count == early_stop_patience:
                print('Early stopping after {} epochs'.format(epoch+1))
                model_lstm.half()
                with gzip.open('my_lstm_model_compressed.pt', 'wb') as f:
                    torch.save(model_lstm.state_dict(), f)

                break

elif state_lstm == 'load':
    with gzip.open('my_lstm_model_compressed.pt', 'rb') as f:
        model_lstm.load_state_dict(torch.load(f))


with torch.no_grad():
    x = x_test[0:1,:,:]
    x = torch.tensor(x, dtype=torch.float32)
    model_lstm.eval()
    predict = model_lstm(x)
    pred_final = []

    for i in range (1,x_test.shape[0]):
        pred_final.append(predict)
        x = x_test[i:i+1, :, :]
        x = torch.tensor(x, dtype=torch.float32)
        predict = model_lstm(x)

pred_final = torch.cat(pred_final, dim=0).detach().numpy()