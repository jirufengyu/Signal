import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
#from tensorflow.contrib import layers
'''
class Net_ae(object):
    def __init__(self, v, dims_encoder, para_lambda, activation, reg=None):
        """ 
        Building view-specific autoencoder network
        :param v:  view number
        :param dims_encoder: nodes of encoding layers, [input-layer, hidden-layer, ..., middle-layer]
        :param para_lambda: trade-off factor in objective
        :param activation: activation function of each layer
        :param reg: coefficient of weight-decay
        """
        self.v = v
        self.dims_encoder = dims_encoder
        self.dims_decoder = [i for i in reversed(dims_encoder)]
        self.num_layers = len(self.dims_encoder)
        self.para_lambda = para_lambda
        self.activation = activation
        self.reg = reg
        if activation in ['tanh', 'sigmoid']:
            self.initializer = None#layers.xavier_initializer()
        if activation == 'relu':
            self.initializer = None#layers.variance_scaling_initializer(mode='FAN_AVG')

        self.weights, self.netpara = self.init_weights()

    def init_weights(self):
        all_weights = dict()
        with tf.variable_scope("aenet"):
            for i in range(1, self.num_layers):
                all_weights['enc' + str(self.v) + '_w' + str(i)] = tf.get_variable("enc" + str(self.v) + "_w" + str(i),
                                                                                   shape=[self.dims_encoder[i - 1],
                                                                                          self.dims_encoder[i]],
                                                                                   initializer=self.initializer,
                                                                                   regularizer=self.reg)
                all_weights['enc' + str(self.v) + '_b' + str(i)] = tf.Variable(
                    tf.zeros([self.dims_encoder[i]], dtype=tf.float32))

            for i in range(1, self.num_layers):
                all_weights['dec' + str(self.v) + '_w' + str(i)] = tf.get_variable("dec" + str(self.v) + "_w" + str(i),
                                                                                   shape=[self.dims_decoder[i - 1],
                                                                                          self.dims_decoder[i]],
                                                                                   initializer=self.initializer,
                                                                                   regularizer=self.reg)
                all_weights['dec' + str(self.v) + '_b' + str(i)] = tf.Variable(
                    tf.zeros([self.dims_decoder[i]], dtype=tf.float32))
            aenet = tf.trainable_variables()
        return all_weights, aenet

    def encoder(self, x, weights):
        """
        :param x: input (feature)
        :param weights: weights of encoder
        :return: middle-layer feature(i.e., z_half)
        """
        layer = tf.add(tf.matmul(x, weights['enc' + str(self.v) + '_w1']), weights['enc' + str(self.v) + '_b1'])
        if self.activation == 'sigmoid':
            layer = tf.nn.sigmoid(layer)
        if self.activation == 'tanh':
            layer = tf.nn.tanh(layer)
        if self.activation == 'relu':
            layer = tf.nn.relu(layer)
        for i in range(2, self.num_layers):
            layer = tf.add(tf.matmul(layer, weights['enc' + str(self.v) + '_w' + str(i)]),
                           weights['enc' + str(self.v) + '_b' + str(i)])
            # if i < self.num_layers-1:
            if self.activation == 'sigmoid':
                layer = tf.nn.sigmoid(layer)
            if self.activation == 'tanh':
                layer = tf.nn.tanh(layer)
            if self.activation == 'relu':
                layer = tf.nn.relu(layer)
        return layer

    def decoder(self, z_half, weights):
        """
        :param z_half: middle-layer feature
        :param weights: weights of decoder
        :return: reconstruction of input(i.e., z)
        """
        layer = tf.add(tf.matmul(z_half, weights['dec' + str(self.v) + '_w1']), weights['dec' + str(self.v) + '_b1'])
        if self.activation == 'sigmoid':
            layer = tf.nn.sigmoid(layer)
        if self.activation == 'tanh':
            layer = tf.nn.tanh(layer)
        if self.activation == 'relu':
            layer = tf.nn.relu(layer)
        for i in range(2, self.num_layers):
            layer = tf.add(tf.matmul(layer, weights['dec' + str(self.v) + '_w' + str(i)]),
                           weights['dec' + str(self.v) + '_b' + str(i)])
            # if i < self.num_layers-1:
            if self.activation == 'sigmoid':
                layer = tf.nn.sigmoid(layer)
            if self.activation == 'tanh':
                layer = tf.nn.tanh(layer)
            if self.activation == 'relu':
                layer = tf.nn.relu(layer)

        return layer

    def loss_reconstruct(self, x):
        z_half = self.encoder(x, self.weights)
        z = self.decoder(z_half, self.weights)
        loss = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(x, z), 2.0))
        return loss

    def get_z_half(self, x):
        return self.encoder(x, self.weights)

    def get_z(self, x):
        z_half = self.encoder(x, self.weights)
        return self.decoder(z_half, self.weights)

    def loss_total(self, x, g):
        """
        :param x: input
        :param g: output of dg net
        :return: loss of objective
        """
        z_half = self.encoder(x, self.weights)
        z = self.decoder(z_half, self.weights)
        loss_recon = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(x, z), 2.0))
        loss_degra = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(z_half, g), 2.0))
        return loss_recon + self.para_lambda * loss_degra
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

class Net_ae(object):
    def __init__(self, input_dim,z_dim,para_lambda=1,activation="sigmoid"):
        """
        Building view-specific autoencoder network
        :param v:  view number
        :param dims_encoder: nodes of encoding layers, [input-layer, hidden-layer, ..., middle-layer]
        :param para_lambda: trade-off factor in objective
        :param activation: activation function of each layer
        :param reg: coefficient of weight-decay
        """
        self.input_dim=input_dim
        self.activation=activation
        self.z_dim=z_dim
        self.para_lambda=para_lambda
        self.dense1=SimpleDense(self.z_dim)
        self.dense2=SimpleDense(self.input_dim)
        self.dense1.build(input_shape=[None,self.input_dim])
        self.dense1.build(input_shape=[None,self.z_dim])
        #print("!!!!!!!!!!!!!!!",type(self.dense1.weights))
        self.netpara=[]
        self.netpara.extend(self.dense1.variables)
        self.netpara.extend(self.dense2.variables)
        #print("!!!!!!!!!!!!!!!netpara:",self.netpara)
        
    
    def encoder(self,x):
        return self.dense1(x)

    def decoder(self,h):
        return self.dense2(h)
        
    def get_z(self,x):
        return self.encoder(x)
    
    def loss_reconstruct(self,x):
        h=self.encoder(x)
        x_recon=self.decoder(h)
        loss=keras.losses.MSE(x,x_recon)
        #loss=0.5*tf.math.reduce_mean(tf.math.pow(tf.math.subtract(x,x_recon),2.0))
        return loss

    def loss_total(self,x,g):
        h=self.encoder(x)
        x_recon=self.decoder(h)
        #loss_recon=0.5*tf.math.reduce_mean(tf.math.pow(tf.math.subtract(x,x_recon),2.0))
        #loss_degra=0.5*tf.math.reduce_mean(tf.math.pow(tf.math.subtract(h,g),2.0))
        loss_recon=0.5*keras.losses.MSE(x,x_recon)
        loss_degra=0.5*keras.losses.MSE(h,g)
        return loss_recon+self.para_lambda*loss_degra


class Net_Ae(keras.layers.Layer):
    def __init__(self,input_dim,z_dim,para_lambda=1,activation="sigmoid" ):
        super().__init__()
        self.input_dim=input_dim
        self.activation=activation
        self.z_dim=z_dim
        self.para_lambda=para_lambda
        
    def build(self, input_shape):
        
        b_init = tf.zeros_initializer()
        w_init = tf.random_normal_initializer()
        self.w1=self.add_weight(initializer=w_init,shape=[input_shape,self.z_dim],dtype=tf.float32,trainable=True)
        self.b1=self.add_weight(initializer=b_init,shape=self.z_dim,dtype=tf.float32,trainable=True)
        self.w2=self.add_weight(initializer=w_init,shape=[self.z_dim,input_shape],dtype=tf.float32,trainable=True)
        self.b2=self.add_weight(initializer=b_init,shape=input_shape,dtype=tf.float32,trainable=True)
        return super().build(input_shape)

    def encoder(self,x):
        
        return tf.math.sigmoid(tf.linalg.matmul(tf.cast(x,dtype=tf.float32), self.w1) + self.b1)

    def decoder(self,h):
        return tf.math.sigmoid(tf.linalg.matmul(h, self.w2) + self.b2)
             
    def get_z(self,x):
       
        return self.encoder(x)
    
    def loss_reconstruct(self,x):
        h=self.encoder(x)
        x_recon=self.decoder(h)
        loss=0.5*keras.losses.MSE(x,x_recon)
        #loss=0.5*tf.math.reduce_mean(tf.math.pow(tf.math.subtract(x,x_recon),2.0))
        return loss

    def loss_total(self,x,g):
        h=self.encoder(x)
        x_recon=self.decoder(h)
        #loss_recon=0.5*tf.math.reduce_mean(tf.math.pow(tf.math.subtract(x,x_recon),2.0))
        #loss_degra=0.5*tf.math.reduce_mean(tf.math.pow(tf.math.subtract(h,g),2.0))
        loss_recon=0.5*keras.losses.MSE(x,x_recon)
        loss_degra=0.5*keras.losses.MSE(h,g)
        return loss_recon+self.para_lambda*loss_degra

    