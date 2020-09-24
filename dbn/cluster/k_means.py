# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#print(train_images[0])
#取前一千条
print("len:",10000)
train_labels=train_labels[:10000]
train_images=train_images[:10000]
train_images=train_images.reshape(-1,784)
train_images=train_images/255.0
meandata=np.mean(train_images,axis=0)
train_images=train_images-meandata
cov=np.cov(train_images.transpose())
eigVals,eigVectors=np.linalg.eig(cov)
pca_mat=eigVectors[:,:50]   #*选前五十个向量
pca_data=np.dot(train_images,pca_mat)
#pca_data=pd.DataFrame



# %%

import sklearn
k=10
random_state=87
#!Kmeans
kmeans=KMeans(n_clusters=k,random_state=random_state)
cluster1=kmeans.fit_predict(train_images)

print("kmeans:",sklearn.metrics.accuracy_score(train_labels,cluster1))
# %%
#!GMM
gmm=GaussianMixture(n_components=k,covariance_type='full',random_state=random_state)
cluster2=gmm.fit(train_images).predict(train_images)
print("gmm:",sklearn.metrics.accuracy_score(train_labels,cluster2))


# %%
#!pca+Kmeans聚类

pca_data=np.asarray(pca_data).astype(float)
kmeans=KMeans(n_clusters=k,random_state=random_state)

cluster1=kmeans.fit_predict(pca_data)

print("pca+skmeans:",sklearn.metrics.accuracy_score(train_labels,cluster1))
# %%
#!pca+GMM聚类
gmm=GaussianMixture(n_components=k,covariance_type='full',random_state=random_state)
cluster2=gmm.fit(pca_data).predict(pca_data)
print("pca+gmm:",sklearn.metrics.accuracy_score(train_labels,cluster2))

# %%
class RBM(object):
    def __init__(self,input_size,output_size,learning_rate=1.0):
        self._input_size=input_size
        self._output_size=output_size
        self.w=np.zeros([input_size,output_size],np.float32)
        self.hb=np.zeros([output_size],np.float32)
        self.vb=np.zeros([input_size],np.float32)

        self.learning_rate=learning_rate
    # Fits the result from the weighted visible layer plus the bias into a sigmoid curve
    def prob_h_given_v(self, visible, w, hb):
        # Sigmoid
        return tf.math.sigmoid(tf.linalg.matmul(visible, w) + hb)
     # Fits the result from the weighted hidden layer plus the bias into a sigmoid curve
    def prob_v_given_h(self, hidden, w, vb):
        return tf.math.sigmoid(tf.linalg.matmul(hidden, tf.transpose(w)) + vb)
    def sample_prob(self,probs):
        return tf.nn.relu(tf.sign(probs-tf.random.uniform(tf.shape(probs))))
    
    def runG(self,batch,_w,_hb,_vb):
        v0=tf.cast(batch,tf.float32)
        
        h0=self.sample_prob(self.prob_h_given_v(v0,_w,_hb))
        v1=self.sample_prob(self.prob_v_given_h(h0,_w,_vb))
        h1=self.prob_h_given_v(v1,_w,_hb)
        
        positive_grad=tf.linalg.matmul(tf.transpose(v0),h0)
        negative_grad=tf.linalg.matmul(tf.transpose(v1),h1)

        update_w=_w+self.learning_rate*(positive_grad-negative_grad)/tf.cast(tf.shape(v0)[0],dtype=tf.float32)
        update_vb=_vb+self.learning_rate*tf.reduce_mean(v0-v1,0)
        update_hb=_hb+self.learning_rate*tf.reduce_mean(h0-h1,0)

        loss=tf.reduce_mean(tf.square(v0-v1))
        return update_w,update_hb,update_vb,loss
    def train(self,X,epochs=5,batchsize=100):
        prv_w=np.zeros([self._input_size,self._output_size],np.float32)
        prv_hb=np.zeros([self._output_size],np.float32)
        prv_vb=np.zeros([self._input_size],np.float32)
        
        cur_w=np.zeros([self._input_size,self._output_size],np.float32)
        cur_hb=np.zeros([self._output_size],np.float32)
        cur_vb=np.zeros([self._input_size],np.float32)    
        for epoch in range(epochs):
            for start,end in zip(range(0,len(X),batchsize),range(batchsize,len(X),batchsize)):
                batch=X[start:end]
                
                cur_w,cur_hb,cur_vb,_=self.runG(batch,prv_w,prv_hb,prv_vb)
                
                prv_w=cur_w
                prv_hb=cur_hb
                prv_vb=cur_vb
            _a,_b,_c,loss=self.runG(X,cur_w,cur_hb,cur_vb)        
            print('Epoch: %d' % epoch, 'loss: %f' % loss)

        self.w=prv_w
        self.hb=prv_hb
        self.vb=prv_vb
    
    def rbm_outpt(self, X):
        input_X = tf.constant(X,dtype=tf.float32)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        out = tf.math.sigmoid(tf.linalg.matmul(input_X, _w) + _hb)
        return out

# %%

rbm=RBM(784,500)
rbm.train(train_images,epochs=100)
out=rbm.rbm_outpt(train_images)

#!rbm+kmeans
kmeans1=KMeans(n_clusters=k,random_state=random_state)
cluster3=kmeans1.fit_predict(out)
print("rbm+kmeans:",sklearn.metrics.accuracy_score(train_labels,cluster3))
#!rbm+gmm
gmm1=GaussianMixture(n_components=k,covariance_type='full',random_state=random_state)
cluster4=gmm1.fit(out).predict(out)
print("rbm+gmm:",sklearn.metrics.accuracy_score(train_labels,cluster4))

# %%
RBM_hidden_sizes = [500, 200 , 50 ] #create 4 layers of RBM with size 785-500-200-50

#Since we are training, set input as training data
inpX2=inpX = train_images

#Create list to hold our RBMs
rbm_list = []

#Size of inputs is the number of inputs in the training set
input_size = inpX.shape[1]

#For each RBM we want to generate
for i, size in enumerate(RBM_hidden_sizes):
    print('RBM: ',i,' ',input_size,'->', size)
    rbm_list.append(RBM(input_size, size))
    input_size = size
#For each RBM in our list
for rbm in rbm_list:
    print('New RBM:')
    #Train a new one
    rbm.train(inpX,epochs=10) 
    #Return the output layer
    inpX = rbm.rbm_outpt(inpX)
eout=rbm_list[0].rbm_outpt(inpX2)
eout=rbm_list[1].rbm_outpt(eout)
eout=rbm_list[2].rbm_outpt(eout)

#!3rbm+kmeans
kmeans=KMeans(n_clusters=k,random_state=random_state)
cluster5=kmeans.fit_predict(eout)
print("3rbm+kmeans:",sklearn.metrics.accuracy_score(train_labels,cluster5))
#!3rbm+gmm
gmm=GaussianMixture(n_components=k,covariance_type='full',random_state=random_state)
cluster6=gmm.fit(eout).predict(eout)
print("3rbm+gmm:",sklearn.metrics.accuracy_score(train_labels,cluster6))

# %%
