import os
import numpy as np
from sklearn import manifold
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import matplotlib.pyplot as plt
from keras.layers import Input
from core.util import print_accuracy,LearningHandler
from core import Conv

import tensorflow as tf
def run_net(data, params):
    #
    # UNPACK DATA
    #

    x_train_unlabeled, y_train_unlabeled, x_val, y_val, x_test, y_test = data['spectral']['train_and_test']

    inputs_vae = Input(shape=(params['img_dim'], params['img_dim'], 1), name='inputs_vae')
    ConvAE = Conv.ConvAE(inputs_vae,params)
    ConvAE.vae.load_weights('/home/stu2/Signal-1/Deep-Spectral-Clustering-using-Dual-Autoencoder-Network-master/src/applications/vae_mnist.h5')

    lh = LearningHandler(lr=params['spec_lr'], drop=params['spec_drop'], lr_tensor=ConvAE.learning_rate,
                         patience=params['spec_patience'])

    lh.on_train_begin()


    losses_vae = np.empty((500,))
    for i in range(20):
        # if i==0:
        x_val_y = ConvAE.vae.predict(x_val)[2]  #得到y
        losses_vae[i] = ConvAE.train_vae(x_val,x_val_y, params['batch_size'])
        x_val_y = ConvAE.vae.predict(x_val)[2]
        y_sp = x_val_y.argmax(axis=1)
        print_accuracy(y_sp, y_val, params['n_clusters'])
        print("Epoch: {}, loss={:2f}".format(i, losses_vae[i]))

        if i>1:
            if np.abs(losses_vae[i]-losses_vae[i-1])<0.0001:
                print('STOPPING EARLY')
                break

    print("finished training")

    x_val_y = ConvAE.vae.predict(x_val)[2]
    # x_val_y = ConvAE.classfier.predict(x_val_lp)
    y_sp = x_val_y.argmax(axis=1)

    print_accuracy(y_sp, y_val, params['n_clusters'])
    from sklearn.metrics import normalized_mutual_info_score as nmi
    nmi_score1 = nmi(y_sp, y_val)
    print('NMI: ' + str(np.round(nmi_score1, 4)))



