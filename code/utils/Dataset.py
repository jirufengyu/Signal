import h5py
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import scipy.io


class Dataset():
    def __init__(self, name):
        self.path = './dataset/'
        self.name = name

    def load_data(self):
        data_path = self.path + self.name + '.mat'
        if 'ORL_3views' in self.name or 'COIL' in self.name or 'handwritten_6views' in self.name:
            dataset = scipy.io.loadmat(data_path)
            view_number = len(dataset)-4
            X = dict()
            for v_num in range(view_number):
                X[str(v_num)] = self.normalize(dataset['x' + str(v_num + 1)])
            y = dataset['gt']
            tmp = np.zeros(y.shape[0])
            y = np.reshape(y, np.shape(tmp))
            return X, y
        else:
            dataset = scipy.io.loadmat(data_path)
            x1, x2, y = dataset['x1'], dataset['x2'], dataset['gt']
            tmp = np.zeros(y.shape[0])
            y = np.reshape(y, np.shape(tmp))
            return x1, x2, y

    def normalize(self, x, min=0):
        # min_val = np.min(x)
        # max_val = np.max(x)
        # x = (x - min_val) / (max_val - min_val)
        # return x

        if min == 0:
            scaler = MinMaxScaler([0, 1])
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x
