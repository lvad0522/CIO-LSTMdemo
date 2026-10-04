import torch
import torch.nn as nn
import matplotlib as plt
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy import signal
from model import RNN
from year_split import split_leave_one_year_out
import torch


def pearson_correlation_loss(y_true, y_pred):
    y_true = y_true.float()
    y_pred = y_pred.float()
    y_true_mean = torch.mean(y_true)
    y_pred_mean = torch.mean(y_pred)
    y_true_centered = y_true - y_true_mean
    y_pred_centered = y_pred - y_pred_mean
    correlation = torch.sum(y_true_centered * y_pred_centered) / (
        torch.sqrt(torch.sum(y_true_centered ** 2)) *
        torch.sqrt(torch.sum(y_pred_centered ** 2))
    )
    return 1 - torch.mean(correlation)


class Predict:
    def __init__(self, pre_year, input_size=1, hidden_size=60, num_layers=2, seq_len=1, batch=356, output_size=1,
                 lr=0.001, learn_time=500, train_percentage=0.75, predict_num=0, dropout=0.2):
        self.pre_year = pre_year
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.batch = batch
        self.output_size = output_size
        self.lr = lr
        self.learn_time = learn_time
        self.train_percentage = train_percentage
        self.predict_num = predict_num
        self.dropout = dropout

    @staticmethod
    def band_pass_filter(data, dimension):
        b, c = signal.butter(dimension, [0.02, 0.1], 'bandpass')
        filtedData = signal.filtfilt(b, c, data)
        return filtedData

    def plot_fig(self, predict, real):
        plt.plot(predict, 'r', label='prediction')
        plt.plot(real, 'b', label='real')
        plt.show()

    def maxmin_norm(self, data):
        max_value = np.max(data)
        min_value = np.min(data)
        data = (data - min_value) / (max_value - min_value)
        return data

    def normalize(self, data):
        shape = data.shape
        if len(shape) == 1:
            data = self.maxmin_norm(data)
            return data
        for i in range(shape[0]):
            data[i] = self.maxmin_norm(data[i])
        return data

    def reshape_cio(self, data, batch):
        a, b = data.shape
        data1 = []
        data2 = []
        for i in range(0, b):
            data2.append(data[:, i].tolist())
            if ((i + 1) % batch == 0):
                data1.append(data2[:][:])
                data2 = []
        return np.array(data1)

    def predict(self, input_data, output_data):
        input_data = np.array(input_data)
        output_data = np.array(output_data)
        max_ = np.max(output_data)
        min_ = np.min(output_data)
        input_data = self.normalize(input_data)
        output_data = self.normalize(output_data)

        # Exclude only the target year, including when it is the first or last year.
        train_input, train_output, prediction_input, prediction_output = (
            split_leave_one_year_out(input_data, output_data, self.pre_year)
        )

        train_input = train_input[0:int(train_output.size / self.batch) * self.batch]
        train_output = train_output[0:int(train_output.size / self.batch) * self.batch]

        x_train = train_input.reshape(-1, self.batch, self.input_size)
        y_train = train_output.reshape(-1, self.batch, 1)
        x_prediction = prediction_input.reshape(-1, 1, self.input_size)
        y_prediction = prediction_output.reshape(-1, 1, 1)

        y_train = np.copy(y_train)
        y_prediction = np.copy(y_prediction)
        var_x = torch.tensor(x_train, dtype=torch.float32, device=self.device)
        var_y = torch.tensor(y_train, dtype=torch.float32, device=self.device)

        rnn = RNN(input_size=self.input_size, hidden_size=self.hidden_size, dropout=self.dropout).to(self.device)
        optimizer = torch.optim.Adam(rnn.parameters(), lr=self.lr)
        loss_func = nn.MSELoss()

        for i in range(self.learn_time):
            out = rnn(var_x)
            loss = loss_func(out, var_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        rnn = rnn.eval()
        var_x = x_prediction.reshape(1, -1, self.input_size)
        var_y = y_prediction.reshape(1, -1, 1)
        var_x = torch.tensor(var_x, dtype=torch.float32, device=self.device)
        var_y = torch.tensor(var_y, dtype=torch.float32, device=self.device)
        out = rnn(var_x)
        prediction = out.cpu().detach().numpy().reshape(-1)
        real = y_prediction.reshape(-1)

        prediction = np.array(prediction)
        real = np.array(real)
        x = prediction.shape
        for i in range(x[0]):
            if np.isnan(prediction[i]) or prediction[i] < 0:
                prediction[i] = 0

        r = pearsonr(prediction[:112], real[:112])
        prediction = prediction * (max_ - min_) + min_
        real = real * (max_ - min_) + min_
        return r[0], prediction, real
