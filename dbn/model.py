#%%
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU,LSTM
from tensorflow.keras.layers import UpSampling2D, Conv2D,Embedding
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import scipy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
class GAN():
    def __init__(self):
        self.img_rows=128
        self.img_cols=128
        self.channels=3
        self.img_shapes=(self.img_rows,self.img_cols,self.channels)
        self.IE_filters=32

        self.max_features=10000#词汇数量
        self.maxlen=500#
        self.embedding_dims=256
    #def T_Encoder(self,x):

    def I_Encoder(self,x=0):
        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            #d = InstanceNormalization()(d)#对单个对象进行正则化
            return d
        d0=Input(shape=self.img_shapes)
        d1=conv2d(d0,self.IE_filters)
        d2=conv2d(d1,self.IE_filters*2)
        d3 = conv2d(d2, self.IE_filters * 4)
        d4 = conv2d(d3, self.IE_filters * 8)
        flatten=Flatten()
        d4=flatten(d4)
       
        return d4#Model(d0,d4)
    def I_Decoder(self,x=0):
        def deconv2d(layer_input,  filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            #u = InstanceNormalization()(u)
            #u = Concatenate()([u, skip_input])//类似resnet把某些层跳过
            return u

        u1 = deconv2d(x, self.IE_filters * 4)
        u2 = deconv2d(u1,self.IE_filters * 2)
        u3 = deconv2d(u2, self.IE_filters)

        u4 = UpSampling2D(size=2)(u3)

        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return output_img#Model(x, output_img)

    def T_Encoder(self,x=0):
        '''

        :param x:文本
        :return: 编码后的文本，lstm后的文本与后面做loss使用
        '''
        embedding = Embedding(input_dim=self.max_features, output_dim=self.embedding_dims, input_length=self.maxlen)
        LSTM1 = LSTM(256, return_sequences=True, input_shape=(50, 300))
        drop1 = Dropout(0.5)

        flatten = Flatten()
        dense=Dense(64,activation='relu')

        x=embedding(x)

        out1=x=LSTM1(x)

        x=drop1(x)

        x=flatten(x)

        x=dense(x)

        return x,out1



    def C_Deconder(self,h_img,h_text,c_dim):
        '''

        :param h_img: 输入的图片
        :param h_text:
        :param c_dim: 中心编码器的隐藏层维度
        :return: 解码的总向量，图像向量，文本向量
        '''
        flatten=Flatten()
        i_shape=h_img.shape
        h_img=flatten(h_img)
        print(h_img.shape)
        I_inputdim=h_img.shape[1].value
        T_inputdim=h_text.shape[1].value
        h=tf.concat([h_img,h_text],0)
        ori_dim=h.shape[1].value
        dense0=Dense(c_dim,activation='relu')
        h=dense0(h)
        dense1=Dense(ori_dim,activation='relu')
        h=dense1(h)
        I_out,T_out=tf.split(h,[I_inputdim,T_inputdim],1)
        I_out=tf.reshape(I_out,i_shape)
        return h,I_out,T_out

    def T_Deconder(self,x):
        dense0=Dense(256*300)#256是编码器中的lstm维数，300是maxlen

        LSTM2=LSTM(256, return_sequences=True)
        x=dense0(x)
        print(x.shape)
        x=tf.reshape(x,[-1,300,256])
        x=LSTM2(x)
        print("t:", x.shape)
        return x


class Self_Conv2d(layers.Layer):#自定义conv2d
    def __init__(self,filters,f_size=4):
        super().__init__()
        self.conv2d=Conv2D(filters,kernel_size=f_size,strides=2,padding='same',
                             acitvation='relu')
        self.act=LeakyReLU(alpha=0.2)
    def call(self,input):
        d=self.conv2d(input)
        d=self.act(d)
        return d
class Self_Deconv2d(layers.Layer):
    def __init__(self,filters,f_size=4,dropout_rate=0):
        super().__init__()
        self.upsampling2d=UpSampling2D(size=2)
        self.conv2d=Conv2D(filters,kernel_size=f_size, strides=1, padding='same', activation='relu')
        self.dropout_rate=dropout_rate
        self.dropout=Dropout(dropout_rate)
    def call(self,input):
        u=self.upsampling2d(input)
        u=self.conv2d(u)
        if self.dropout_rate:
            u=self.dropout(u)
        return u
class I_Enconder(layers.Layer):
    def __init__(self,f_sizes=4):
        super().__init__(name='I_enconder')
        self.f_sizes=f_sizes
        self.IE_filters=32
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shapes = (self.img_rows, self.img_cols, self.channels)
        self.flatten=Flatten()
        self.conv2d_0=Self_Conv2d(self.IE_filters)      #Conv2D(filters=self.IE_filters,kernel_size=f_sizes,strides=2,padding='same',
                       #      acitvation='relu')
        self.conv2d_1=Self_Conv2d(self.IE_filters*2)#Conv2D(filters=self.IE_filters*2,kernel_size=f_sizes,strides=2,padding='same',
                       #      acitvation='relu')
        self.conv2d_2 = Self_Conv2d(self.IE_filters*4)#Conv2D(filters=self.IE_filters * 4, kernel_size=f_sizes, strides=2, padding='same',
                             #  acitvation='relu')
        self.conv2d_3 = Self_Conv2d(self.IE_filters*8)#Conv2D(filters=self.IE_filters * 8, kernel_size=f_sizes, strides=2, padding='same',
                               #acitvation='relu')
    def call(self,input):
        d0 = Input(shape=self.img_shapes)
        d1 = self.conv2d_0(d0)
        d2 = self.conv2d_1(d1)
        d3 = self.conv2d_2(d2)
        d4 = self.conv2d_3(d3)
        d4=self.flatten(d4)
        return d4
class I_Deconder(layers.Layer):
    def __init__(self):
        super().__init__(name='I_deconder')
        self.channels=3
        self.IE_filters = 32
        self.flatten = Flatten()
        self.deconv2d_0 = Self_Deconv2d(self.IE_filters*4)#Conv2D(filters=self.IE_filters, kernel_size=f_sizes, strides=2, padding='same',
                              # acitvation='relu')
        self.deconv2d_1 =Self_Deconv2d(self.IE_filters*2)# Conv2D(filters=self.IE_filters * 2, kernel_size=f_sizes, strides=2, padding='same',
                              # acitvation='relu')
        self.deconv2d_2 =Self_Deconv2d(self.IE_filters) #Conv2D(filters=self.IE_filters * 4, kernel_size=f_sizes, strides=2, padding='same',
                              # acitvation='relu')
        self.upsampling2d=UpSampling2D(size=2)  #Conv2D(filters=self.IE_filters * 8, kernel_size=f_sizes, strides=2, padding='same',
                              # acitvation='relu')
        self.outputConv2d=Conv2D(self.channels, kernel_size=4,
                                 strides=1, padding='same', activation='tanh')
    def call(self,input):
        u1=self.deconv2d_0(input)
        u2=self.deconv2d_1(u1)
        u3=self.deconv2d_2(u2)
        u4=self.upsampling2d(u3)
        output_img=self.outputConv2d(u4)
        return output_img
class T_Enconder(layers.Layer):
    def __init__(self):
        super().__init__(name='T_enconder')
        self.embedding = Embedding(input_dim=self.max_features,
                              output_dim=self.embedding_dims,
                              input_length=self.maxlen)
        self.LSTM1 = LSTM(256, return_sequences=True, input_shape=(50, 300))
        self.drop1 = Dropout(0.5)

        self.flatten = Flatten()
        self.dense = Dense(64, activation='relu')
    def call(self,input):
        x=self.embedding(input)
        out1=x=self.LSTM1(x)
        x=self.drop1(x)
        x=self.flatten(x)
        x=self.dense(x)
        return x,out1
class T_Deconder(layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense0 = Dense(256 * 300)  # 256是编码器中的lstm维数，300是maxlen

        self.LSTM2 = LSTM(256, return_sequences=True)
    def call(self,input):
        x=self.dense0(input)
        x=tf.reshape(x,[-1,300,256])
        x=self.LSTM2(x)
        return x

class GanModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        #判别器
        self.d_A=self.build_discriminator()#输入a的判别器
        self.d_B=self.build_discriminator()#输入b的判别器
        optimizer = Adam(0.0002, 0.5)

        self.d_A.compile(loss='mse',optimizer=optimizer,metrics=['accuracy'])
        self.d_B.compile(loss='mse',
                     optimizer=optimizer,
                     metrics=['accuracy'])
        #生成器
        self.g_AB=self.build_generator()
        self.g_BA=self.build_generator()
        img_A=Input(shape=self.img_shape)
        img_B=Input(shape=self.img_shape)
        #转换
        fake_B=self.g_AB(img_A)
        fake_A=self.g_BA(img_B)
        #重构
        reconstr_A=self.g_BA(fake_B)
        reconstr_B=self.g_AB(fake_A)
    def call(self,input):

        return
    def build_generator(self):
        x=I_Enconder(input)
        x=I_Deconder(x)
        return Model(input=input,output=x)
    def build_discriminator(self):
        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            #if normalization:
             #   d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

'''
#input=tf.random.normal([30,500], mean=0.0, stddev=1.0)
g=GAN()
img=plt.imread("img.png").astype(np.float)
img=g.I_Encoder(img)
#y=g.I_Decoder(m)
input=tf.random.uniform([50,300],minval=1,maxval=2)
txt,out1=g.T_Encoder(input)
p=g.C_Deconder(img,txt,32)
k=g.T_Deconder(txt)
'''
# %%
class AEModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.dense1=Dense(14)
        
        self.dense4=Dense(28)
    def call(self,input):
        x=self.dense1(input)
    
        x=self.dense4(input)
        return Model(input,x)
data=tf.keras.datasets.mnist()
(data,y),(_,_)=data.load_data()
model=AEModel()
model.compile(optimizer=tf.keras.optimizers.Adam,loss=tf.keras.losses.categorical_crossentropy)
model.fit(data,data)
out=model.evaluate(data[0])
    
