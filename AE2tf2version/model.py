import tensorflow as tf
import numpy as np
import scipy.io as scio
from utils.Net_ae import Net_ae,Net_Ae
from utils.Net_dg import Net_dg,Net_Dg
from utils.next_batch import next_batch
import math
from sklearn.utils import shuffle
import timeit
from tensorflow import keras

'''
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
    #net_ae1 = Net_ae(1, dims[0], para_lambda, act[0])
    #net_ae2 = Net_ae(2, dims[1], para_lambda, act[1])
    #net_dg1 = Net_dg(1, dims[2], act[2])
    #net_dg2 = Net_dg(2, dims[3], act[3])
    net_ae1=Net_ae(input_dim=240,z_dim=200)
    net_ae2=Net_ae(input_dim=240,z_dim=200)
    net_dg1=Net_dg(z_dim=200)
    net_dg2=Net_dg(z_dim=200)

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
                net_dg1.loss_degradation(h_input, fea1_latent) + net_dg2.loss_degradation(h_input, fea2_latent))
    update_dg = tf.train.AdamOptimizer(lr[2]).minimize(loss_dg, var_list=net_dg1.netpara.extend(net_dg2.netpara))

    update_h = tf.train.AdamOptimizer(lr[3]).minimize(loss_dg, var_list=h_list)
    g1 = net_dg1.get_g(h_input)
    g2 = net_dg2.get_g(h_input)
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

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
    scio.savemat('H.mat', mdict={'H': H, 'gt': gt, 'loss_total': err_total, 'time': elapsed,
                                    'x1': X1, 'x2': X2})
    return H, gt

'''
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return keras.backend.random_uniform((fan_in, fan_out),
                            minval=low, maxval=high,
                            dtype=tf.float32)

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

    net_ae1=Net_Ae(input_dim=240,z_dim=200)
    net_ae2=Net_Ae(input_dim=216,z_dim=200)
    net_dg1=Net_Dg(z_dim=200,h_dim=64)
    net_dg2=Net_Dg(z_dim=200,h_dim=64)
    
    net_ae1.build(240)
    net_ae2.build(216)
    net_dg1.build(64)
    net_dg2.build(64)
    H = np.random.uniform(0, 1, [X1.shape[0], dims[2][0]])
    print(type(H))
    

    h_input=tf.Variable(xavier_init(batch_size, dims[2][0]),dtype=tf.float32, name='LatentSpaceData')
    
    pre_train = keras.optimizers.Adam(lr[0])#.minimize(loss_pre)
  
    update_ae = keras.optimizers.Adam(lr[1])#.minimize(loss_ae, var_list=net_ae1.netpara.extend(net_ae2.netpara))

    
    update_dg = keras.optimizers.Adam(lr[2])#.minimize(loss_dg, var_list=net_dg1.netpara.extend(net_dg2.netpara))

    update_h = keras.optimizers.Adam(lr[3])#.minimize(loss_dg, var_list=h_input)
    

    # init inner AEs
    
    #writer = tf.summary.create_file_writer("./log")
    #tf.summary.trace_on(graph=True, profiler=True)
    for k in range(epochs[0]):
        X1, X2, gt = shuffle(X1, X2, gt)
        for batch_x1, batch_x2, batch_No in next_batch(X1, X2, batch_size):
            
            temp=[]
            temp.extend(net_ae1.trainable_variables)
            temp.extend(net_ae2.trainable_variables)
            
            with tf.GradientTape() as tape:
                loss_pre = net_ae1.loss_reconstruct(batch_x1) + net_ae2.loss_reconstruct(batch_x2)
                
            grads = tape.gradient(loss_pre, temp)
            pre_train.apply_gradients(zip(grads,temp))
            
            err_pre.append(loss_pre)
            
            output = "Pre_epoch : {:.0f}, Batch : {:.0f}  ===> Reconstruction loss = {:.4f} ".format((k + 1), batch_No,
                                                                                                    loss_pre[-1])
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
            
            batch_g1= net_dg1.get_g(h_input)
            batch_g2= net_dg2.get_g(h_input)

            # ADM-step1: optimize inner AEs and
            temp=[]
            temp.extend(net_ae1.trainable_variables)
            temp.extend(net_ae2.trainable_variables)
            with tf.GradientTape() as tape:
                loss_ae = net_ae1.loss_total(batch_x1, batch_g1) + net_ae2.loss_total(batch_x2, batch_g2)
            print("!!!!!!!!!!!loss_ae",loss_ae[-1])
            grads = tape.gradient(loss_ae, temp)
            update_ae.apply_gradients(zip(grads,temp))

            batch_z_half1 = net_ae1.get_z(batch_x1)
            batch_z_half2 = net_ae2.get_z(batch_x2)
            tf.compat.v1.assign(h_input, batch_h)
          
            # !ADM-step2: optimize dg nets
            
            temp=[]
            temp.extend(net_dg1.trainable_variables)
            temp.extend(net_dg2.trainable_variables)
            with tf.GradientTape() as tape:
                val_dg=loss_dg = para_lambda * (
                            net_dg1.loss_degradation(h_input, batch_z_half1) + 
                            net_dg2.loss_degradation(h_input, batch_z_half2))
            grads = tape.gradient(loss_dg, temp)
            update_dg.apply_gradients(zip(grads,temp))

            # !ADM-step3: update H
            for k in range(epochs[2]):
                
                with tf.GradientTape() as tape:
                    val_dg=loss_dg = para_lambda * (
                                net_dg1.loss_degradation(h_input, batch_z_half1) + 
                                net_dg2.loss_degradation(h_input, batch_z_half2))
                grads = tape.gradient(loss_dg, [h_input])
                update_h.apply_gradients(zip(grads,[h_input]))
                
            t=keras.backend.eval(h_input)
            
            H[start_idx: end_idx,] = t

            batch_g1_new = net_dg1.get_g(h_input)#(batch_h)
            batch_g2_new = net_dg2.get_g(h_input)#(batch_h)
           
            val_total=loss_ae = net_ae1.loss_total(batch_x1, batch_g1_new) + net_ae2.loss_total(batch_x2, batch_g2_new)                                       
            err_total.append(val_total)
            #print("epoch:       ",j+1)
            output = "Epoch : {:.0f} -- Batch : {:.0f} ===> Total training loss = {:.4f} ".format((j + 1),
                                                                                                (num_batch_i + 1),
                                                                                                val_total[-1])
            print(output)
    #with writer.as_default():
    #    tf.summary.trace_export(
    #        name="my_func_trace",
    #        step=0,
    #        profiler_outdir="./log")
    elapsed = (timeit.default_timer() - start)
    print("Time used: ", elapsed)
    scio.savemat('H.mat', mdict={'H': H, 'gt': gt, 'loss_total': err_total, 'time': elapsed,
                                    'x1': X1, 'x2': X2})
    return H, gt