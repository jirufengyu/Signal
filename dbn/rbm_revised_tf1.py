import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
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
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)
    # Fits the result from the weighted hidden layer plus the bias into a sigmoid curve
    def prob_v_given_h(self, hidden, w, vb):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)
    def sample_prob(self,probs):
        return tf.nn.relu(tf.sign(probs-tf.random_uniform(tf.shape(probs))))
    
    def runG(self,batch,_w,_hb,_vb):
        v0=tf.cast(batch,tf.float32)
        
        h0=self.sample_prob(self.prob_h_given_v(v0,_w,_hb))
        v1=self.sample_prob(self.prob_v_given_h(h0,_w,_vb))
        h1=self.prob_h_given_v(v1,_w,_hb)
        
        positive_grad=tf.matmul(tf.transpose(v0),h0)
        negative_grad=tf.matmul(tf.transpose(v1),h1)

        update_w = _w + self.learning_rate * (positive_grad - negative_grad) / tf.to_float(tf.shape(v0)[0])
        update_vb=_vb+self.learning_rate*tf.reduce_mean(v0-v1,0)
        update_hb=_hb+self.learning_rate*tf.reduce_mean(h0-h1,0)

        loss=tf.reduce_mean(tf.square(v0-v1))
        return update_w,update_hb,update_vb,loss
    def train(self,X,epochs=5,batchsize=100):
        prv_w=self.w
        prv_hb=self.hb
        prv_vb=self.vb
        
        v0= tf.placeholder("float", [None, self._input_size])
        _w = tf.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.placeholder("float", [self._output_size])
        _vb = tf.placeholder("float", [self._input_size])
        cur_wt,cur_hbt,cur_vbt,_t=self.runG(v0,_w,_hb,_vb)

        #cur_w=np.zeros([self._input_size,self._output_size],np.float32)
        #cur_hb=np.zeros([self._output_size],np.float32)
        #cur_vb=np.zeros([self._input_size],np.float32)    
        with tf.Session() as sess:
            for epoch in range(epochs):
                for start,end in zip(range(0,len(X),batchsize),range(batchsize,len(X),batchsize)):
                    batch=X[start:end]
                    cur_w,cur_hb,cur_vb,_=sess.run([cur_wt,cur_hbt,cur_vbt,_t], feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    #cur_w,cur_hb,cur_vb,_=self.runG(batch,prv_w,prv_hb,prv_vb)
                    
                    prv_w=cur_w
                    prv_hb=cur_hb
                    prv_vb=cur_vb
                    print("!!!!!!!!!!",prv_w)
                   
                _a,_b,_c,loss=self.runG(X,cur_w,cur_hb,cur_vb)        
                print('Epoch: %d' % epoch)#, 'loss: %f' % loss[-1])

        self.w=prv_w
        self.hb=prv_hb
        self.vb=prv_vb
    
    def rbm_outpt(self, X):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)


mnist = input_data.read_data_sets("MNIST_data/")
trX, trY = mnist.train.images, mnist.train.labels
rbm=RBM(784,500)

rbm.train(trX)