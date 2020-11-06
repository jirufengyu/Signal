import tensorflow as tf
import numpy as np
import scipy.io as scio
from utils.Net_ae import Net_ae
from utils.next_batch import next_batch
from keras import objectives, backend as K
import math
from sklearn.utils import shuffle
from keras.layers import *
from keras import Model
import timeit
from utils.print_result import print_result
from utils.Dataset import Dataset
from keras.optimizers import *
"""
#!适应版ae_mae,可以自定义维度
"""



def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                            minval=low, maxval=high,
                            dtype=tf.float32)
class MaeAEModel:
    def __init__(self,v1_aedims,v2_aedims,mae_dims,dis_dims,reg_lambda=0.05,latent_dim=200,h_dim=64,lr_ae=1e-3,lr_mae=1e-3,lamb=5):
        '''
        latent_dim : latent feature dim in outer auto encoder
        h_dim : representation in maeodal autocoder
        v1_aedims: 第一个视角各层网络维度
        v2_aedims: 第二个视角各层网络维度
        mae_dims: 多峰自编码器维度,一般的，多峰自编码器的维度不同的峰相同[[200,150,32],[200,150,32],[32,150,200],[32,150,200]]
                    前面两个是编码器维度，后面两个是解码器维度
        discr_dims: 判别器维度 [200,150,1]
        h_dim可能与mae_dims的维度不同,一般是两个模态的编码器维度相加
        #!v1_aedims和v2_aedims都是[[],[]]的列表，如[[240,200],[200,240]]
        '''        
        self.input1_shape=v1_aedims[0][0]
        self.input2_shape=v2_aedims[0][0]
        
        self.v1_latent_dim=v1_aedims[0][-1] 
        self.v2_latent_dim=v2_aedims[0][-1]

        self.v1_dims=v1_aedims
        self.v2_dims=v2_aedims
        self.h_dim=mae_dims[0][-1]+mae_dims[1][-1]
        self.mae_dims=mae_dims
        self.dis_dims=dis_dims
        self.reg_lambda = reg_lambda
        self.lr_ae=lr_ae
        self.lr_mae=lr_mae
        self.lamb=lamb
    def train_model(self,X1, X2, gt, epochs, batch_size):
        err_total = list()
        start = timeit.default_timer()
        n_clusters = len(set(gt))
        
        
        H = np.random.uniform(0, 1, [X1.shape[0], self.h_dim])

        x1_input = tf.placeholder(np.float32, [None, self.input1_shape])
        x2_input = tf.placeholder(np.float32, [None, self.input2_shape])
        x1_input0=Input([None, self.input1_shape],tensor=x1_input)
        x2_input0=Input([None, self.input2_shape],tensor=x2_input)
    
        self.encoder1=self.encoder(x1_input0,dims=self.v1_dims[0])
        self.encoder2=self.encoder(x2_input0,dims=self.v2_dims[0])

        z1=self.encoder1(x1_input0)
        z2=self.encoder2(x2_input0)

        z1_input=Input(shape=(self.v1_latent_dim,))
        z2_input=Input(shape=(self.v2_latent_dim,))

        self.decoder1=self.decoder(z1_input,dims=self.v1_dims[1])
        self.decoder2=self.decoder(z2_input,dims=self.v2_dims[1])

        vae_mse_loss, encoded = self.mae_encoder(z1_input,z2_input,dims1=self.mae_dims[0],dims2=self.mae_dims[1],h_dim=self.h_dim)

        self.maeencoder=Model(inputs=[z1_input,z2_input],outputs=encoded)

        encoded_input = Input(shape=(self.h_dim,))

        decoded_1,decoded_2=self.mae_decoder(encoded_input,dims1=self.mae_dims[2],dims2=self.mae_dims[3])
        self.maedecoder = Model(encoded_input, [decoded_1, decoded_2])
     
        decoder_output = self.maedecoder(encoded)
        lamb=self.lamb
        maeloss=lamb*K.sum(1*K.mean((z1_input-decoder_output[0])**2,0))+lamb*K.sum(1*K.mean((z2_input-decoder_output[1])**2,0))+vae_mse_loss
        update_mae = tf.train.AdamOptimizer(self.lr_mae).minimize(maeloss)

        x_recon1_withnoise=self.decoder1(z1)       #带噪声的loss
        x_recon2_withnoise=self.decoder2(z2)

        def shuffling(x):
            idxs = K.arange(0, K.shape(x)[0])
            idxs = K.tf.random_shuffle(idxs)
            return K.gather(x, idxs)

        z1_shuffle = Lambda(shuffling)(z1)
        z_z_1_true = Concatenate()([z1, z1])       # replicated feature vector
        z_z_1_false = Concatenate()([z1, z1_shuffle])    # drawn from another image

        z2_shuffle=Lambda(shuffling)(z2)
        z_z_2_true=Concatenate()([z2,z2])
        z_z_2_false=Concatenate()([z2,z2_shuffle])

        z1_in=Input(shape=(self.v1_latent_dim*2,))
        z2_in=Input(shape=(self.v2_latent_dim*2,))
        #z1_discr=self.discriminator(z1_in)
        #z2_discr=self.discriminator(z2_in)
        GlobalDiscriminator1=self.discriminator(z1_in,dims=self.dis_dims)   #Model(z1_in,z1_discr)
        GlobalDiscriminator2=self.discriminator(z2_in,dims=self.dis_dims)   #Model(z2_in,z2_discr)

        z_z_1_true_scores=GlobalDiscriminator1(z_z_1_true)
        z_z_1_false_scores=GlobalDiscriminator1(z_z_1_false)
        z_z_2_true_scores=GlobalDiscriminator2(z_z_2_true)
        z_z_2_false_scores=GlobalDiscriminator2(z_z_2_false)
        global_info_loss1=-K.mean(K.log(z_z_1_true_scores+1e-6)+K.log(1-z_z_1_false_scores+1e-6)) 
        global_info_loss2=-K.mean(K.log(z_z_2_true_scores+1e-6)+K.log(1-z_z_2_false_scores+1e-6))

        lamb=5 #5
        x1ent_loss=1*K.mean((x1_input-x_recon1_withnoise)**2,0)
        x2ent_loss=1*K.mean((x2_input-x_recon2_withnoise)**2,0)

        loss_vae1=lamb*K.sum(x1ent_loss)#+0.001*K.sum(global_info_loss1) #0.001
        loss_vae2=lamb*K.sum(x2ent_loss)#+0.001*K.sum(global_info_loss2)  #0.001
        loss_ae=loss_vae1+loss_vae2
        update_ae = tf.train.AdamOptimizer(self.lr_ae).minimize(loss_ae)
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())

        num_samples = X1.shape[0]
        num_batchs = math.ceil(num_samples / batch_size)
        for j in range(epochs):
            X1,X2,H,gt=shuffle(X1,X2,H,gt)
            for num_batch_i in range(int(num_batchs)-1):
                start_idx, end_idx = num_batch_i * batch_size, (num_batch_i + 1) * batch_size
                end_idx = min(num_samples, end_idx)
                batch_x1 = X1[start_idx: end_idx, ...]
                batch_x2 = X2[start_idx: end_idx, ...]
                batch_h = H[start_idx: end_idx, ...]               
                #ADM-step1 : optimize inner AEs 
                _,val_dg=sess.run([update_ae,loss_ae],feed_dict={x1_input:batch_x1,x2_input:batch_x2})
                                                                #fea1_latent:batch_g1,fea2_latent:batch_g2})
                batch_z_half1=sess.run(z1,feed_dict={x1_input:batch_x1})
                batch_z_half2=sess.run(z2,feed_dict={x2_input:batch_x2})

                

                #ADM-step2: optimize dg nets
                #_,val_dg=sess.run([update_dg,loss_dg],feed_dict={fea1_latent:batch_z_half1,
                    #                                    fea2_latent:batch_z_half2})
                _,val_dg=sess.run([update_mae,maeloss],feed_dict={z1_input:batch_z_half1,z2_input:batch_z_half2})
                #ADM-step3:update H
                
                #for k in range(epochs[2]):
                #    sess.run(update_h,feed_dict={fea1_latent:batch_z_half1,fea2_latent:batch_z_half2})
                h_get=sess.run(encoded,feed_dict={z1_input:batch_z_half1,z2_input:batch_z_half2})
                
                H[start_idx: end_idx, ...] = h_get

                

                #val_total = sess.run(loss_ae, feed_dict={x1_input: batch_x1, x2_input: batch_x2,
                    #                                    fea1_latent: batch_g1_new, fea2_latent: batch_g2_new})
                                                        
                #err_total.append(val_total)
                
                #output = "Epoch : {:.0f} -- Batch : {:.0f} ===> Total training loss = {:.4f} ".format((j + 1),
                 #                                                                                   (num_batch_i + 1),
                  #                                                                                  val_total)
                #print(output)
            print("epoch:",j+1)

            print_result(n_clusters, H, gt)
        elapsed = (timeit.default_timer() - start)
        print("Time used: ", elapsed)
        '''
        #?使用RBM  64->64
        rbm=RBM_t1(64,64)
        
        H=tf.cast(H,tf.float32)
        with tf.Session() as se:
            H=H.eval()
            H=H.tolist()
        rbm.train(H)
        H=rbm.rbm_outpt(H)
        '''
        scio.savemat('H.mat', mdict={'H': H, 'gt': gt, 'loss_total': err_total, 'time': elapsed,
                                        'x1': X1, 'x2': X2})
        return H, gt
                
    def encoder(self,x1,dims): #dims=[input_shape,200]
        h=x1
        for i in range(len(dims)-1):
            h=Dense(dims[i+1],activation="relu")(h)
        #h=Dense(200,activation="relu")(x1)
        #h=Dense(self.latent_dim)(h)
        return Model(x1,h)
        """
        z_mean=Dense(self.latent_dim)(h)
        z_log_var=Dense(self.latent_dim)(h)
        return Model(x1,[z_mean,z_log_var])
        """
    def decoder(self,z,dims): #dims=[200,input_shape]
        h=z
        for i in range(len(dims)-1): 
            h=Dense(dims[i+1],activation="relu")(h)
        #h=Dense(200,activation="relu")(h)
        #h=Dense(dim,activation="relu")(h)  #输出的维度与解码器输入的维度相同
        return Model(z,h)
    def discriminator(self,z,dims):   #判别器 dims=[latent_dims,1]
        h=z
        for i in range(len(dims)-1):
            h=Dense(dims[i],activation='relu')(h)
        #z1=Dense(self.latent_dim,activation='relu')(z1)
        #z1=Dense(self.latent_dim,activation='relu')(z1)
        h=Dense(1,activation='sigmoid')(h)
        return Model(z,h)
    def mae_decoder(self, encoded,dims1,dims2):   #dims1=[32,100,150,200],dims2=[32,100,150,200]
        h1=h2=encoded
        for i in range(len(dims1)-1):
            h1=Dense(dims1[i+1],activation='tanh',kernel_regularizer=regularizers.l2(self.reg_lambda))(h1)
        decoded1=Dense(dims1[-1],activation='sigmoid')(h1)
        #h1 = Dense(100, activation='tanh',name='mae_decoder11', kernel_regularizer=regularizers.l2(self.reg_lambda))(encoded)
        #h1 = Dense(150,  activation='tanh',name='mae_decoder12', kernel_regularizer=regularizers.l2(self.reg_lambda))(h1)
        #decoded1 = Dense(200, name='mae_decoder13',activation='sigmoid')(h1)
        
        #h2 = Dense(100, activation='tanh',name='mae_decoder21', kernel_regularizer=regularizers.l2(self.reg_lambda))(encoded)
        #h2 = Dense(150, activation='tanh',name='mae_decoder22', kernel_regularizer=regularizers.l2(self.reg_lambda))(h2)
        #decoded2 = Dense(200, name='mae_decoder23',activation='sigmoid')(h2)
        
        for i in range(len(dims2)-1):
            h2=Dense(dims2[i+1],activation='tanh',kernel_regularizer=regularizers.l2(self.reg_lambda))(h2)
        decoded2=Dense(dims2[-1],activation='sigmoid')(h2)

        return decoded1, decoded2
    def _build_fnd(self, encoded): #dims=[64,32,1]暂时不用这个函数

        h = Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(self.fnd_lambda))(encoded)
        h = Dense(32, activation='tanh', kernel_regularizer=regularizers.l2(self.fnd_lambda))(h)
        return Dense(1, activation='sigmoid', name='fnd_output')(h)
    def mae_encoder(self, input1, input2, dims1,dims2,h_dim=64):   #dims1=[200,100,32],dims2=[200,100,32],一般地，dims1[-1]+dims2[-1]=h_dim
        h1=input1
        h2=input2
        for i in range(len(dims1)-1):
            h1=Dense(dims1[i+1],activation='tanh',kernel_regularizer=regularizers.l2(self.reg_lambda))(h1)
        #h1 = Dense(100, activation='tanh', kernel_regularizer=regularizers.l2(self.reg_lambda))(input1)
        #h1 = Dense(32,  activation='tanh', kernel_regularizer=regularizers.l2(self.reg_lambda))(h1)

        #h2 = Dense(100,  activation='tanh', kernel_regularizer=regularizers.l2(self.reg_lambda))(input2)
        #h2 = Dense(32,  activation='tanh', kernel_regularizer=regularizers.l2(self.reg_lambda))(h2)
        for i in range(len(dims2)-1):
            h2=Dense(dims2[i+1],activation='tanh',kernel_regularizer=regularizers.l2(self.reg_lambda))(h2)
             
        h = Concatenate(axis=-1, name='concat')([h1, h2])
        
        h = Dense(h_dim, name='shared', activation='tanh', kernel_regularizer=regularizers.l2(self.reg_lambda))(h)
       
        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, h_dim), mean=0., stddev=0.01)
            return z_mean_ + K.exp(0.5 * z_log_var_) * epsilon

        z_mean = Dense(h_dim, name='z_mean', activation='linear')(h)
        z_log_var = Dense(h_dim, name='z_log_var', activation='linear')(h)
        
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        #return (vae_ce_loss, vae_mse_loss, Lambda(sampling, output_shape=(latent_dim,), name='lambda')([z_mean, z_log_var]))
        return kl_loss,Lambda(sampling, output_shape=(h_dim,), name='lambda')([z_mean, z_log_var])