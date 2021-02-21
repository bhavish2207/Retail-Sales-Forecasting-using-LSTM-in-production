from __future__ import print_function
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
##from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#from sklearn.externals import joblib
import statistics
import pickle

dept_store_lvl_df = pd.read_csv('sales_forecasting.csv')
#print(dept_store_lvl_df.head())

### Creating week_nbr column
dept_store_lvl_df['Date'] = pd.to_datetime(dept_store_lvl_df.Date)
dept_store_lvl_df['week_nbr'] = dept_store_lvl_df.sort_values(['Date'],ascending=[True]).groupby(['Store', 'Dept'])\
             .cumcount() + 1
#print(dept_store_lvl_df.head())
#print(dept_store_lvl_df.shape)

### Creating a periodic week_nbr column to capture week of the year value between 1 & 52. 
### This column can capture the seasonality by holding the week of the year value

a = []
for el in dept_store_lvl_df.week_nbr:
    if el>52 and el<=104:
        a.append(el-52)
    elif el>104:
        a.append(el-104)
    else:
        a.append(el)
week_nbr_periodic = pd.Series(a)
cols = list(dept_store_lvl_df.columns)
cols.append('week_nbr_periodic')
dept_store_lvl_df = pd.concat([dept_store_lvl_df, week_nbr_periodic], axis = 1)
dept_store_lvl_df.columns = cols

### FUNCTION TO CREATE 'X' & 'Y' ARRAYs of the required sequence length. 
### x_arr will be a 3D Numpy object
def data_prep(xdata_arr,ydata_arr, input_seq_len):
    x_arr = []
    y_arr = []
    for i in range(len(xdata_arr)-input_seq_len):
        x_arr.append(xdata_arr[i:i+input_seq_len])
        y_arr.append(ydata_arr[i+input_seq_len])
    return np.array(x_arr), np.array(y_arr)

stores_dict_ips = {}
stores_dict_tgt3 = {}
stores_dict_df3 = {}
stores_dict3 = {}   ### Creating a dictionary containing the 45 stores

store_train_x = dept_store_lvl_df.iloc[:295867,].drop(columns=['Date','Store','Dept','week_nbr'])
store_train_y = dept_store_lvl_df.iloc[:295867,]['Weekly_Sales'].to_frame()
#store_test_x = dept_store_lvl_df.iloc[295867:,].drop(columns=['Date','Store','Dept','week_nbr'])
#store_test_y = dept_store_lvl_df.iloc[295867:,]['Weekly_Sales'].to_frame()

### Saving as Pickle File
#store_train_x.to_pickle('train_data.pkl')
#joblib.dump(store_train_x, 'train_data.pkl')
Pkl_Filename = "train_data.pkl"
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(store_train_x, file)
#store_train_y.to_pickle('train_data_y.pkl')
#joblib.dump(store_train_y, 'train_data_y.pkl')
Pkl_Filename = "train_data_y.pkl"
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(store_train_y, file)

#### Applying Scaler on entire Store Level Aggregated dataset
scaler = MinMaxScaler()
scaler = scaler.fit(store_train_x)
train_arr_x = scaler.transform(store_train_x)
#test_arr_x = scaler.transform(store_test_x)
    
scaler = scaler.fit(store_train_y)
train_arr_y = scaler.transform(store_train_y)
#test_arr_y = scaler.transform(store_test_y)

#### Creating Sliding Windows for Store 1 Dept 1
x_train, y_train = data_prep(train_arr_x[0:143],train_arr_y[0:143],10)
#x_test, y_test = data_prep(test_arr_x[0:143],test_arr_y[0:143],10)

### Creating Sliding windows for other stores 2 to 45 and concatenating them into a single numpy array
for i in range(143,len(train_arr_x),143):   
    ### Calling Data Prep Function to create multivariate sequences (10 lag moving window) Processed
    x_train_store, y_train_store = data_prep(train_arr_x[i:i+143],train_arr_y[i:i+143],10)
    x_train = np.concatenate((x_train, x_train_store), axis=0)
    y_train = np.concatenate((y_train, y_train_store), axis=0)
    
#for j in range(143,116907,143):  
#    x_test_store, y_test_store = data_prep(test_arr_x[j:j+143],test_arr_y[j:j+143],10)
#    x_test = np.concatenate((x_test, x_test_store), axis=0)
#    y_test = np.concatenate((y_test, y_test_store), axis=0)
    
### Creating Train & Test Loader Objects
train_target3 = torch.tensor(y_train.astype(np.float32))
train3 = torch.tensor(x_train.astype(np.float32)) 
train_tensor3 = torch.utils.data.TensorDataset(train3, train_target3) 
train_loader3 = torch.utils.data.DataLoader(dataset = train_tensor3, batch_size = 143, shuffle = False)
train_loader_full = torch.utils.data.DataLoader(dataset = train_tensor3, batch_size = len(train3), shuffle = False)

#test_target3 = torch.tensor(y_test.astype(np.float32))
#test3 = torch.tensor(x_test.astype(np.float32)) 
#test_tensor3 = torch.utils.data.TensorDataset(test3, test_target3) 
#test_loader3 = torch.utils.data.DataLoader(dataset = test_tensor3, batch_size = len(test3),shuffle = False)

 ### LSTM CLASS
class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers,seq_len):
        super(LSTM_Model, self).__init__()

        # Defining some parameters
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.seq_len = seq_len
        

        #Defining the layers
        # LSTM Layer
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_dim,num_layers = n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, 1)
    
    
    #def reset_hidden(self):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
    #    self.hidden = (torch.zeros(self.n_layers, self.seq_len, self.hidden_dim),
    #                  torch.zeros(self.n_layers, self.seq_len, self.hidden_dim))
    
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        #out, _ = self.lstm(x.view(len(x),self.seq_len,self.input_size), hidden)
        out, _ = self.lstm(x, hidden)
        #out_final = self.fc(out.view(self.seq_len,len(x),self.hidden_dim)[-1])
        out_final = self.fc(out[:,-1,:])
        return out_final
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim))
        return hidden

### TRAINING The LSTM Model
input_size = 11
hidden_dim = 10
n_layers = 1      ### 1 LAYER
seq_len = 10

model_lstm = LSTM_Model(input_size,hidden_dim,n_layers,seq_len)  ### Initiate the RNN class object named model_bptt
model_lstm = model_lstm.float()    
criterion = nn.MSELoss()  ### Define the loss function which is Mean Squared Error   
optimizer = optim.Adam(model_lstm.parameters(), lr=0.005) # Optimizers require the model parameters to optimize and a learning rate
epochs = 8 

    ### Training on the dataset.
for e in range(epochs):
    for inputs,targets in train_loader3:    
               # Training pass
        output = model_lstm(inputs.float())   ### Obtaining the outputs for every row of the batch using the model built in ann class
        loss = criterion(output, targets) ### Calculating the Loss for every row of the batch using the obtained output & target
        loss.backward()   ### Backpropogating based on the loss 
        optimizer.step()  ### Updating the weights based on the gradient using the optimizer.step() function
        optimizer.zero_grad() ### Erasing the gradient values after every weight update
  
### LOADING to PICKLE FILE      
#joblib.dump(model_lstm, 'model.pkl')
Pkl_Filename = "model.pkl"
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model_lstm, file)

        
#pickle.dump(model_lstm, open('model.pkl','wb'))
#model = pickle.load(open('model.pkl','rb'))
print("All three PICKLE FILEs CREATED")
