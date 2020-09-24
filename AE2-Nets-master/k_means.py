# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import tensorflow as tf
import sklearn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import tensorflow.keras as keras
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
from utils.print_result import print_result

train_labels=train_labels[:10000]
train_images=train_images[:10000]
train_images=train_images.reshape(-1,784)
train_images=train_images/255.0


#mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
#trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
   # mnist.test.labels
class RBM_t1(object):
    def __init__(self, input_size, output_size):
        # Defining the hyperparameters
        self._input_size = input_size  # Size of input
        self._output_size = output_size  # Size of output
        self.epochs = 5  # Amount of training iterations
        self.learning_rate = 1.0  # The step used in gradient descent
        self.batchsize = 100  # The size of how much data will be used for training per sub iteration

        # Initializing weights and biases as matrices full of zeroes
        self.w = np.zeros([input_size, output_size], np.float32)  # Creates and initializes the weights with 0
        self.hb = np.zeros([output_size], np.float32)  # Creates and initializes the hidden biases with 0
        self.vb = np.zeros([input_size], np.float32)  # Creates and initializes the visible biases with 0

    # Fits the result from the weighted visible layer plus the bias into a sigmoid curve
    def prob_h_given_v(self, visible, w, hb):
        # Sigmoid
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    # Fits the result from the weighted hidden layer plus the bias into a sigmoid curve
    def prob_v_given_h(self, hidden, w, vb):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)

    # Generate the sample probability
    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    # Training method for the model
    def train(self, X):
        # Create the placeholders for our parameters
        _w = tf.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.placeholder("float", [self._output_size])
        _vb = tf.placeholder("float", [self._input_size])

        prv_w = np.zeros([self._input_size, self._output_size],
                         np.float32)  # Creates and initializes the weights with 0
        prv_hb = np.zeros([self._output_size], np.float32)  # Creates and initializes the hidden biases with 0
        prv_vb = np.zeros([self._input_size], np.float32)  # Creates and initializes the visible biases with 0

        cur_w = np.zeros([self._input_size, self._output_size], np.float32)
        cur_hb = np.zeros([self._output_size], np.float32)
        cur_vb = np.zeros([self._input_size], np.float32)
        v0 = tf.placeholder("float", [None, self._input_size])

        # Initialize with sample probabilities
        h0 = self.sample_prob(self.prob_h_given_v(v0, _w, _hb))
        v1 = self.sample_prob(self.prob_v_given_h(h0, _w, _vb))
        h1 = self.prob_h_given_v(v1, _w, _hb)

        # Create the Gradients
        positive_grad = tf.matmul(tf.transpose(v0), h0)
        negative_grad = tf.matmul(tf.transpose(v1), h1)

        # Update learning rates for the layers
        update_w = _w + self.learning_rate * (positive_grad - negative_grad) / tf.to_float(tf.shape(v0)[0])
        update_vb = _vb + self.learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_hb = _hb + self.learning_rate * tf.reduce_mean(h0 - h1, 0)

        # Find the error rate
        err = tf.reduce_mean(tf.square(v0 - v1))

        # Training loop
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # For each epoch
            for epoch in range(self.epochs):
                # For each step/batch
                for start, end in zip(range(0, len(X), self.batchsize), range(self.batchsize, len(X), self.batchsize)):
                    batch = X[start:end]
                    # Update the rates
                    cur_w = sess.run(update_w, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_hb = sess.run(update_hb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_vb = sess.run(update_vb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    prv_w = cur_w
                    prv_hb = cur_hb
                    prv_vb = cur_vb
                error = sess.run(err, feed_dict={v0: X, _w: cur_w, _vb: cur_vb, _hb: cur_hb})
                print('Epoch: %d' % epoch, 'reconstruction error: %f' % error)
            self.w = prv_w
            self.hb = prv_hb
            self.vb = prv_vb

    # Create expected output for our DBN
    def rbm_outpt(self, X):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)

class RBM_t2(object):
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
trX=train_images[:10000]
trY=train_labels[:10000]
print(trY[:100])
rbm=RBM_t2(784,500)
rbm.train(trX)
out=rbm.rbm_outpt(trX)

print_result(10, out, trY, count=10)


# %%
print(trX[0])
# %%
