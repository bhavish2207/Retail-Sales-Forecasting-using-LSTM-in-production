from model import LSTM_Model
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
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
import statistics

app = Flask(__name__)


model = pickle.load(open('model.pkl', 'rb'))
train_df = pickle.load(open('train_data.pkl', 'rb'))
train_df_y = pickle.load(open('train_data_y.pkl', 'rb'))
#train_df = pd.read_pickle('train_data.pkl')
#train_df_y = pd.read_pickle('train_data_y.pkl')

@app.route('/')
def index():
    return render_template('public/index.html')

@app.route('/',methods=['POST'])
def upload_file():
    
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        input_df = pd.read_csv(uploaded_file.filename)
        store_test_y = input_df['Weekly_Sales'].to_frame()
        
        scaler = MinMaxScaler()
        scaler = scaler.fit(train_df)
        test_arr_x = scaler.transform(input_df)
    
        scaler = scaler.fit(train_df_y)
        test_arr_y = scaler.transform(store_test_y)
        
        x_test = []
        y_test = []
        for i in range(len(test_arr_x)-10):
            x_test.append(test_arr_x[i:i+10])
            y_test.append(test_arr_y[i+10])
        
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        
        test_target3 = torch.tensor(y_test.astype(np.float32))
        test3 = torch.tensor(x_test.astype(np.float32)) 
        test_tensor3 = torch.utils.data.TensorDataset(test3, test_target3) 
        test_loader3 = torch.utils.data.DataLoader(dataset = test_tensor3, batch_size = len(x_test),shuffle = False)
        
        for inputs,targets in test_loader3:
            output = model(inputs.float())
        output = (output[0][0]*(train_df_y["Weekly_Sales"].max()-train_df_y["Weekly_Sales"].min()))+train_df_y["Weekly_Sales"].min()
        output = round(float(output),2)
        args = True
        
    return render_template('public/index.html', args=args, prediction_text='Sales for the upcoming week is predicted to be ${}'.format(output))

if __name__ == "__main__":
    app.run()