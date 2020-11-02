'''
Author: jirufengyu
Date: 2020-11-02 05:55:22
LastEditTime: 2020-11-02 08:26:36
LastEditors: jirufengyu
Description: Nothing
FilePath: /Signal-1/DSNMF/dsnmf.py
'''
from scipy.sparse.linalg import svds
import numpy as np
import tensorflow as tf
def appr_seminmf(M, r):
    """
        Approximate Semi-NMF factorisation. 
        
        Parameters
        ----------
        M: array-like, shape=(n_features, n_samples)
        r: number of components to keep during factorisation
    """
    if r < 2:
        raise ValueError("The number of components (r) has to be >=2.")
    A, S, B = svds(M, r-1)
    S = np.diag(S)
    A = np.dot(A, S)
    m, n = M.shape
    for i in range(r-1):
        if B[i, :].min() < (-B[i, :]).min():
            B[i, :] = -B[i, :]
            A[:, i] = -A[:, i]
    if r == 2:
        U = np.concatenate([A, -A], axis=1)
    else:
        An = -np.sum(A, 1).reshape(A.shape[0], 1)
        U = np.concatenate([A, An], 1)
    V = np.concatenate([B, np.zeros((1, n))], 0)
    if r>=3:
        V -= np.minimum(0, B.min(0))
    else:
        V -= np.minimum(0, B)
    return U, V

def init_weights(X, num_components, svd_init=True):
    if svd_init:
        return appr_seminmf(X, num_components)

    Z = 0.08 * np.random.rand(X.shape[0], num_components)
    H = 0.08 * np.random.rand(num_components, X.shape[1])

    return Z, H

class DSNMF(object):
    def __init__(self, data, layers, verbose=False, l1_norms=[], pretrain=False, lr=1e-3):
        """
        Parameters
        ----------
        :param data: array-like, shape=(n_samples, n_features)
        :param layers: list, shape=(n_layers) containing the size of each of the layers
        :param verbose: boolean
        :param l1_norms: list, shape=(n_layers) the l1-weighting of each of the layers
        :param pretrain: pretrain layers using svd
        """
        H=data.T
        params=[]
        for i, l in enumerate(layers, start=1):
            print('Pretraining {}th layer [{}]'.format(i, l), end='\r')
            Z, H = init_weights(H, l, svd_init=pretrain)
            params.append(tf.Variable(Z,name='Z_%d'%(i)))
        params.append(tf.Variable(H,name='H_%d'%len(layers)))
        self.params=params
        self.layers=layers
    def train(self, data, layers, verbose=False, l1_norms=[], pretrain=True, lr=1e-3):
        
        
        H=data.T
        loss=tf.losses.mean_squared_error(data.T,self.get_h(-1))
        
        for norm,param in zip(l1_norms,self.params):
            loss+=((abs(param))*norm).sum()
        loss=tf.Variable(loss)
        H=tf.nn.relu(self.params[-1])
        updates=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss,var_list=self.params)
        #self.loss=loss
        init=tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(100):
                loss=sess.run(loss)
                sess.run(updates)
                print("loss:",epoch,loss)
                H=sess.run(H)
        self.H=H

    def get_h(self,layer_num,have_dropout=False):
        h=tf.nn.relu(self.params[-1])
        if have_dropout:
            h=tf.nn.dropout(h,0.5)
        for z in reversed(self.params[1:-1][:]):
            h=tf.nn.relu(tf.matmul(tf.cast(z,tf.float32),tf.cast(h,tf.float32)))
        if layer_num==-1:
            h=tf.matmul(tf.cast(self.params[0],tf.float32),tf.cast(h,tf.float32))
        return h