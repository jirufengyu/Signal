import tensorflow as tf
#from tensorflow.contrib import layers
from tensorflow.keras import layers
from tensorflow import keras
'''
class Net_dg(object):
    def __init__(self, v, dims_net, activation, reg=None):
        """
        :param v: view number
        :param dims_net: nodes of dg net [H-layer,hidden-layer,.., output-layer]
        :param activation: activation function of each layer
        :param reg: coefficient of weight-decay
        """
        self.v = v
        self.dims_net = dims_net
        self.num_layers = len(self.dims_net)
        self.activation = activation
        self.reg = reg
        if activation in ['tanh', 'sigmoid']:
            self.initializer = None#layers.xavier_initializer()
        if activation == 'relu':
            self.initializer = None#layers.xavier_initializer()
            # self.initializer=layers.variance_scaling_initializer(mode='FAN_AVG')
        self.weights, self.netpara = self.init_weights()

    def init_weights(self):
        all_weights = dict()
        with tf.variable_scope("dgnet"):
            for i in range(1, self.num_layers):
                all_weights['dg' + str(self.v) + '_w' + str(i)] = tf.get_variable("dg" + str(self.v) + "_w" + str(i),
                                                                                  shape=[self.dims_net[i - 1],
                                                                                         self.dims_net[i]],
                                                                                  initializer=self.initializer,
                                                                                  regularizer=self.reg)
                all_weights['dg' + str(self.v) + '_b' + str(i)] = tf.Variable(
                    tf.zeros([self.dims_net[i]], dtype=tf.float32))

            dgnet = tf.trainable_variables()
        return all_weights, dgnet

    def degradation(self, h, weights):
        layer = tf.add(tf.matmul(h, weights['dg' + str(self.v) + '_w1']), weights['dg' + str(self.v) + '_b1'])
        if self.activation == 'sigmoid':
            layer = tf.nn.sigmoid(layer)
        if self.activation == 'tanh':
            layer = tf.nn.tanh(layer)
        if self.activation == 'relu':
            layer = tf.nn.relu(layer)
        for i in range(2, self.num_layers):
            layer = tf.add(tf.matmul(layer, weights['dg' + str(self.v) + '_w' + str(i)]),
                           weights['dg' + str(self.v) + '_b' + str(i)])
            # if i < self.num_layers-1:
            if self.activation == 'sigmoid':
                layer = tf.nn.sigmoid(layer)
            if self.activation == 'tanh':
                layer = tf.nn.tanh(layer)
            if self.activation == 'relu':
                layer = tf.nn.relu(layer)
        return layer

    def loss_degradation(self, h, z_half):
        g = self.degradation(h, self.weights)
        # loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(z_half, g), 2.0))
        loss = tf.losses.mean_squared_error(z_half, g)
        return loss

    def get_g(self, h):
        return self.degradation(h, self.weights)
'''
class SimpleDense(keras.layers.Layer):

  def __init__(self, units=32):
      super(SimpleDense, self).__init__()
      self.units = units

  def build(self, input_shape):  # Create the state of the layer (weights)
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(
        initial_value=w_init(shape=(input_shape[-1], self.units),
                             dtype='float32'),
        trainable=True)
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(
        initial_value=b_init(shape=(self.units,), dtype='float32'),
        trainable=True)

  def call(self, inputs):  # Defines the computation from inputs to outputs
      return tf.math.sigmoid(tf.linalg.matmul(inputs, self.w) + self.b)
class Net_dg(object):
    def __init__(self,z_dim,activation='sigmoid'):
        self.activation=activation
        self.z_dim=z_dim
        self.dense=SimpleDense(self.z_dim)
        self.dense.build(input_shape=[None,64])
        self.netpara=self.dense.weights
        
    def degradation(self,h):
        return self.dense(h)

    def loss_degradation(self,h,z):
        g=self.degradation(h)
        loss=keras.losses.MSE(z,g)
        return loss

    def get_g(self,h):
        return self.degradation(h)


class Net_Dg(keras.layers.Layer):
    def __init__(self,z_dim,h_dim,activation='sigmoid'):
        super().__init__()
        self.activation=activation
        self.z_dim=z_dim
        self.h_dim=h_dim
        
    def build(self, input_shape):
        b_init = tf.zeros_initializer()
        w_init = tf.random_normal_initializer()
        self.w=self.add_weight(initializer=w_init,shape=[input_shape,self.z_dim],dtype=tf.float32,trainable=True)
        self.b=self.add_weight(initializer=b_init,shape=self.z_dim,dtype=tf.float32,trainable=True)
        
        return super().build(input_shape)
        
    def degradation(self,h):
        return tf.math.sigmoid(tf.linalg.matmul(tf.cast(h,dtype=tf.float32), self.w) + self.b)

    def loss_degradation(self,h,z):
        g=self.degradation(h)
        loss=keras.losses.MSE(z,g)
        return loss

    def get_g(self,h):
        return self.degradation(h)
