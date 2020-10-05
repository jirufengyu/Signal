import tensorflow as tf
import numpy as np
import scipy.io as scio
from utils.Net_ae import Net_ae
from utils.Net_dg import Net_dg
from utils.next_batch import next_batch
import math
from sklearn.utils import shuffle
import timeit
from keras.layers import *
from keras.models import Model
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                            minval=low, maxval=high,
                            dtype=tf.float32)
class dualModel:
    def __init__(self,X1,X2,dims,batch_size,act,para_lambda,lr,epochs):
        self.dims=dims
        x1=X1
        self.latent_dim=100
        H = np.random.uniform(0, 1, [X1.shape[0], dims[2][0]])
        with tf.variable_scope("H"):
            h_input = tf.Variable(xavier_init(batch_size, dims[2][0]), name='LatentSpaceData')
            h_list = tf.trainable_variables()
        net_dg1 = Net_dg(1, dims[2], act[2])
        net_dg2 = Net_dg(2, dims[3], act[3])
        x1_input = tf.placeholder(np.float32, [None, dims[0][0]])
        x2_input = tf.placeholder(np.float32, [None, dims[1][0]])
        '''
        self.dims=dims
        x1=X1
        self.latent_dim=100
        h=Dense(200,activation="sigmoid")(x1)
        z_mean=Dense(self.latent_dim)(h)
        z_log_var=Dense(self.latent_dim)(h)
        self.encoder1=Model(x1,z_mean)
        z=Input(shape=(self.latent_dim,))
        h=z
        h=Dense(200,activation="sigmoid")(h)
        h=Dense(dims[0][0],activation="sigmoid")(h)
        self.decoder1=Model(z,h)
        x_recon1_noise=self.decoder1(z_mean)
        '''
        z_mean1,z_log_var1=self.encoder(x1_input)
        z_mean2,z_log_var2=self.encoder(x2_input)
        z1_input=Input(shape=(self.latent_dim,))
        z2_input=Input(shape=(self.latent_dim,))

        x_recon1=self.decoder(z1_input)
        x_recon2=self.decoder(z2_input)

        self.encoder1=Model(x1_input,z_mean1)
        self.encoder2=Model(x2_input,z_mean2)
        self.decoder1=Model(z1_input,x_recon1)
        self.decoder2=Model(z2_input,x_recon2)
        x_recon1_withnoise=self.decoder1(z_mean1)       #带噪声的loss
        x_recon2_withnoise=self.decoder2(z_mean2)
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
            return z_mean + K.exp(z_log_var / 2) * epsilon

        z1 = Lambda(sampling, output_shape=(latent_dim,))([z_mean1, z_log_var1])
        z2=Lambda(sampling,output_shape=(latent_dim,))([z_mean2,z_log_var2])
        x_recon1_true = self.decoder(z1)            #不带噪声的loss
        x_recon2_true=self.decoder(z2)

        #! fea_latent
        fea1_latent=tf.placeholder(np.float32, [None, dims[0][-1]])
        fea2_latent=tf.placeholder(np.float32, [None, dims[0][-1]])
        loss_degra1=0.5*tf.losses.mean_squared_error(z1, fea1_latent)
        loss_degra2=0.5*tf.losses.mean_squared_error(z2, fea2_latent)

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

        z1_in=Input(shape=(latent_dim*2))
        z2_in=Input(shape=(latent_dim*2))
        z1=self.discriminator(z1_in)
        z2=self.discriminator(z2_in)
        GlobalDiscriminator1=Model(z1_in,z1)
        GlobalDiscriminator2=Model(z2_in,z2)
        z_z_1_true_scores=GlobalDiscriminator1(z_z_1_true)
        z_z_1_false_scores=GlobalDiscriminator1(z_z_2_false)
        z_z_2_true_scores=GlobalDiscriminator2(z_z_2_true)
        z_z_2_false_scores=GlobalDiscriminator2(z_z_2_false)
        global_info_loss1=-K.mean(K.log(z_z_1_true_scores+1e-6)+K.log(1-z_z_1_false_scores+1e-6)) 
        global_info_loss2=-K.mean(K.log(z_z_2_true_scores+1e-6)+K.log(1-z_z_2_false_scores+1e-6))

        lamb=5
        x1ent_loss=1*K.mean((X1-x_recon1_true)**2,0)
        x2ent_loss=1*K.mean((X2-x_recon2_true)**2,0)
        x1ent1_loss=0.5*K.mean((x_recon1_withnoise-x_recon1_true)**2,0)
        x2ent1_loss=0.5*K.mean((x_recon2_withnoise-x_recon2_true)**2,0)
        loss_vae1=lamb*K.sum(x1ent_loss)+lamb*K.sum(x1ent1_loss)+0.001*K.sum(global_info_loss1)
        loss_vae2=lamb*K.sum(x2ent_loss)+lamb*K.sum(x2ent1_loss)+0.001*K.sum(global_info_loss2)
        loss_ae=loss_vae1+loss_vae2+loss_degra1+loss_degra2
        update_ae = tf.train.AdamOptimizer(1.0e-3).minimize(loss_ae)

        loss_dg = para_lambda * (
                net_dg1.loss_degradation(h_input, fea1_latent) 
                + net_dg2.loss_degradation(h_input, fea2_latent))
        update_dg = tf.train.AdamOptimizer(lr[2]).minimize(loss_dg, var_list=net_dg1.netpara.extend(net_dg2.netpara))

        update_h = tf.train.AdamOptimizer(lr[3]).minimize(loss_dg, var_list=h_list)
        g1 = net_dg1.get_g(h_input)
        g2 = net_dg2.get_g(h_input)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())

        num_samples = X1.shape[0]
        num_batchs = math.ceil(num_samples / batch_size)
        for j in range(epochs[1]):
            X1,X2,H,gt=shuffle(X1,X2,H,gt)
            for num_batch_i in range(int(num_batchs)-1):
                start_idx, end_idx = num_batch_i * batch_size, (num_batch_i + 1) * batch_size
                end_idx = min(num_samples, end_idx)
                batch_x1 = X1[start_idx: end_idx, ...]
                batch_x2 = X2[start_idx: end_idx, ...]
                batch_h = H[start_idx: end_idx, ...]

                batch_g1=sess.run(g1,feed_dict={h_input:batch_h})
                batch_g2=sess.run(g2,feed_dict={h_input:batch_h})

                #ADM-step1 : optimize inner AEs 





    def encoder(self,x1):
        h=Dense(200,activation="sigmoid")(x1)
        z_mean=Dense(self.latent_dim)(h)
        z_log_var=Dense(self.latent_dim)(h)
        return z_mean,z_log_var
    def decoder(self,z):
        h=z
        h=Dense(200,activation="sigmoid")(h)
        h=Dense(self.dims[0][0],activation="sigmoid")(h)
        return h
    def discriminator(self,z):
        z1=Dense(self.latent_dim,activation='relu')(z)
        z1=Dense(self.latent_dim,activation='relu')(z1)
        z1=Dense(self.latent_dim,activation='relu')(z1)
        z1=Dense(1,activation='sigmoid')(z1)
        return z1

    
def model(X1, X2, gt, para_lambda, dims, act, lr, epochs, batch_size):
    """
    Building model
    :rtype: object
    :param X1: data of view1
    :param X2: data of view2
    :param gt: ground truth
    :param para_lambda: trade-off factor in objective
    :param dims: dimensionality of each layer
    :param act: activation function of each net
    :param lr: learning rate
    :param epochs: learning epoch
    :param batch_size: batch size
    """
    start = timeit.default_timer()
    err_pre = list()
    err_total = list()

    # define each net architecture and variable(refer to framework-simplified)
    net_ae1 = Net_ae(1, dims[0], para_lambda, act[0])
    net_ae2 = Net_ae(2, dims[1], para_lambda, act[1])
    net_dg1 = Net_dg(1, dims[2], act[2])
    net_dg2 = Net_dg(2, dims[3], act[3])
    #z_mean=Dense(latent_dim)(h)
    H = np.random.uniform(0, 1, [X1.shape[0], dims[2][0]])
    x1_input = tf.placeholder(np.float32, [None, dims[0][0]])
    x2_input = tf.placeholder(np.float32, [None, dims[1][0]])

    with tf.variable_scope("H"):
        h_input = tf.Variable(xavier_init(batch_size, dims[2][0]), name='LatentSpaceData')
        h_list = tf.trainable_variables()
    fea1_latent = tf.placeholder(np.float32, [None, dims[0][-1]])
    fea2_latent = tf.placeholder(np.float32, [None, dims[1][-1]])

    loss_pre = net_ae1.loss_reconstruct(x1_input) + net_ae2.loss_reconstruct(x2_input)
    pre_train = tf.train.AdamOptimizer(lr[0]).minimize(loss_pre)

    loss_ae = net_ae1.loss_total(x1_input, fea1_latent) + net_ae2.loss_total(x2_input, fea2_latent)
    update_ae = tf.train.AdamOptimizer(lr[1]).minimize(loss_ae, var_list=net_ae1.netpara.extend(net_ae2.netpara))
    z_half1 = net_ae1.get_z_half(x1_input)
    z_half2 = net_ae2.get_z_half(x2_input)

    loss_dg = para_lambda * (
                net_dg1.loss_degradation(h_input, fea1_latent) 
                + net_dg2.loss_degradation(h_input, fea2_latent))
    update_dg = tf.train.AdamOptimizer(lr[2]).minimize(loss_dg, var_list=net_dg1.netpara.extend(net_dg2.netpara))

    update_h = tf.train.AdamOptimizer(lr[3]).minimize(loss_dg, var_list=h_list)
    g1 = net_dg1.get_g(h_input)
    g2 = net_dg2.get_g(h_input)
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    
    #writer = tf.summary.FileWriter("./mnist_nn_log",sess.graph)
    #writer.close()
    # init inner AEs
    
    for k in range(epochs[0]):
        X1, X2, gt = shuffle(X1, X2, gt)
        for batch_x1, batch_x2, batch_No in next_batch(X1, X2, batch_size):
            _, val_pre = sess.run([pre_train, loss_pre], feed_dict={x1_input: batch_x1, x2_input: batch_x2})
            err_pre.append(val_pre)

            output = "Pre_epoch : {:.0f}, Batch : {:.0f}  ===> Reconstruction loss = {:.4f} ".format((k + 1), batch_No,
                                                                                                    val_pre)
            print(output)

    # the whole training process(ADM)
    num_samples = X1.shape[0]
    num_batchs = math.ceil(num_samples / batch_size)  # fix the last batch
    for j in range(epochs[1]):
        X1, X2, H, gt = shuffle(X1, X2, H, gt)
        for num_batch_i in range(int(num_batchs) - 1):
            start_idx, end_idx = num_batch_i * batch_size, (num_batch_i + 1) * batch_size
            end_idx = min(num_samples, end_idx)
            batch_x1 = X1[start_idx: end_idx, ...]
            batch_x2 = X2[start_idx: end_idx, ...]
            batch_h = H[start_idx: end_idx, ...]

            batch_g1 = sess.run(g1, feed_dict={h_input: batch_h})
            batch_g2 = sess.run(g2, feed_dict={h_input: batch_h})

            # ADM-step1: optimize inner AEs and
            _, val_ae = sess.run([update_ae, loss_ae], feed_dict={x1_input: batch_x1, x2_input: batch_x2,
                                                                fea1_latent: batch_g1, fea2_latent: batch_g2})
            
            # get inter - layer features(i.e., z_half)
            batch_z_half1 = sess.run(z_half1, feed_dict={x1_input: batch_x1})
            batch_z_half2 = sess.run(z_half2, feed_dict={x2_input: batch_x2})

            sess.run(tf.assign(h_input, batch_h))
            # ADM-step2: optimize dg nets
            
            _, val_dg = sess.run([update_dg, loss_dg], feed_dict={fea1_latent: batch_z_half1,
                                                                fea2_latent: batch_z_half2})

            # ADM-step3: update H
            for k in range(epochs[2]):
                sess.run(update_h, feed_dict={fea1_latent: batch_z_half1, fea2_latent: batch_z_half2})

            batch_h_new = sess.run(h_input)
            
            H[start_idx: end_idx, ...] = batch_h_new

            # get latest feature_g for next iteration
            sess.run(tf.assign(h_input, batch_h_new))
            batch_g1_new = sess.run(g1, feed_dict={h_input: batch_h})
            batch_g2_new = sess.run(g2, feed_dict={h_input: batch_h})

            val_total = sess.run(loss_ae, feed_dict={x1_input: batch_x1, x2_input: batch_x2,
                                                    fea1_latent: batch_g1_new, fea2_latent: batch_g2_new})
                                                    
            err_total.append(val_total)
            
            output = "Epoch : {:.0f} -- Batch : {:.0f} ===> Total training loss = {:.4f} ".format((j + 1),
                                                                                                (num_batch_i + 1),
                                                                                                val_total)
            print(output)

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


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                            minval=low, maxval=high,
                            dtype=tf.float32)