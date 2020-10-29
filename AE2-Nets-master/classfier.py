'''
LastEditors: jirufengyu
Author: jirufengyu
'''
import h5py
import os
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
import numpy as np
import keras
from keras.layers import *
from sklearn.metrics import accuracy_score
import numpy as np
from keras.utils.np_utils import *
data_path="/home/stu2/Signal-1/H.mat"

dataset = scipy.io.loadmat(data_path)
H,gt0=dataset['H'],dataset['gt']

gt0=gt0[0]
print(gt0)
gt=to_categorical(gt0[:1600])
input_layer=Input(shape=(64,))
hide1_layer=Dense(30,activation="relu")
output_layer=Dense(10,activation="sigmoid")
h=hide1_layer(input_layer)
out=output_layer(h)
model=keras.Model(inputs=input_layer,outputs=out)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['mse'])
xx=H[:1600]
yy=gt[:1600]

model.fit(x=xx,y=yy,batch_size=10,epochs=50)
pre_result=model.predict(H[1600:])

pre = [np.argmax(one_hot)for one_hot in pre_result]


t=accuracy_score(gt0[1600:],pre)
print(t)

