#compat t1
import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.keras as keras
import math
tf1.disable_eager_execution()
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
    def train(self,X,epochs=2,batchsize=128):
        _w=tf1.placeholder("float",[self._input_size,self._output_size])
        _hb=tf1.placeholder("float",[self._output_size])
        _vb=tf1.placeholder("float",[self._input_size])

        prv_w=np.zeros([self._input_size,self._output_size],np.float32)
        prv_hb=np.zeros([self._output_size],np.float32)
        prv_vb=np.zeros([self._input_size],np.float32)

        cur_w=np.zeros([self._input_size,self._output_size],np.float32)
        cur_hb=np.zeros([self._output_size],np.float32)
        cur_vb=np.zeros([self._input_size],np.float32)
        v0=tf1.placeholder("float",[None,self._input_size])

        h0=self.sample_prob(self.prob_h_given_v(v0,_w,_hb))
        v1=self.sample_prob(self.prob_v_given_h(h0,_w,_vb))
        h1=self.prob_h_given_v(v1,_w,_hb)

        positive_grad=tf.linalg.matmul(tf.transpose(v0),h0)
        negative_grad=tf.linalg.matmul(tf.transpose(v1),h1)

        update_w=_w+self.learning_rate*(positive_grad-negative_grad)/tf.cast(tf.shape(v0)[0],dtype=tf.float32)
        update_vb=_vb+self.learning_rate*tf.reduce_mean(v0-v1,0)
        update_hb=_hb+self.learning_rate*tf.reduce_mean(h0-h1,0)

        err=tf.reduce_mean(tf.square(v0-v1))

        with tf1.Session() as sess:
            sess.run(tf1.global_variables_initializer())
            for epoch in range(epochs):
                for start,end in zip(range(0,len(X),batchsize),range(batchsize,len(X),batchsize)):
                    batch=X[start:end]
                    cur_w  = sess.run(update_w,  feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_hb = sess.run(update_hb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_vb = sess.run(update_vb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    prv_w=cur_w
                    prv_hb=cur_hb
                    prv_vb=cur_vb
                    print("prv_w",prv_w)
                error=sess.run(err,feed_dict={v0:X,_w: cur_w, _vb: cur_vb, _hb: cur_hb})


                print('Epoch: %d' % epoch, 'reconstruction error: %f' % error)

            self.w=prv_w
            self.hb=prv_hb
            self.vb=prv_vb
    
    def rbm_outpt(self, X):
        input_X = tf.constant(X,dtype=tf.float32)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        out = tf.math.sigmoid(tf.linalg.matmul(input_X, _w) + _hb)
        with tf1.Session() as sess:
            sess.run(tf1.global_variables_initializer())
            return sess.run(out)


mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#rbm=RBM(784,500)
#print(train_images[0])
train_images=train_images.reshape(-1,784)
train_images=train_images/255.0
#print("new:",train_images[0])
#rbm.train(train_images)
#print(rbm.rbm_outpt(train_images[:10]))

'''训练每一个RBM'''
RBM_hidden_sizes = [500, 200 , 50 ] #create 4 layers of RBM with size 785-500-200-50

#Since we are training, set input as training data
inpX = train_images

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
    rbm.train(inpX) 
    #Return the output layer
    inpX = rbm.rbm_outpt(inpX)