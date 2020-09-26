import tensorflow as tf
import numpy as np
import scipy.io as scio
from utils.Net_ae import Net_ae
from utils.Net_dg import Net_dg
from utils.next_batch import next_batch
import math
from sklearn.utils import shuffle
import timeit

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
    def reverse_runG(self,h,_w,_hb,_vb):
        h0=tf.cast(h,tf.float32)

        v0=self.sample_prob(self.prob_v_given_h(h0,_w,_vb))
        h1=self.sample_prob(self.prob_h_given_v(v0,_w,_hb))
        v1=self.prob_v_given_h(h1,_w,_vb)

        positive_grad=tf.matmul(tf.transpose(v0),h0)
        negative_grad=tf.matmul(tf.transpose(v1),h1)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #!
        #!
        # TODO:把positive_grad-negative_grad改为negative_grad-positive_grad试试效果
        #!
        #!
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        update_w=_w+self.learning_rate*(positive_grad-negative_grad)/tf.to_float(tf.shape(h0)[0])
        update_hb=_hb+self.learning_rate*tf.reduce_mean(h0-h1,0)
        update_vb=_vb+self.learning_rate*tf.reduce_mean(v0-v1,0)
        loss=tf.reduce_mean(tf.square(h0-h1))
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
            #for epoch in range(epochs):
                #for start,end in zip(range(0,len(X),batchsize),range(batchsize,len(X),batchsize)):
            batch=X#[start:end]
            cur_w,cur_hb,cur_vb,_=sess.run([cur_wt,cur_hbt,cur_vbt,_t], feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
            #cur_w,cur_hb,cur_vb,_=self.runG(batch,prv_w,prv_hb,prv_vb)
            
            prv_w=cur_w
            prv_hb=cur_hb
            prv_vb=cur_vb
            #print("!!!!!!!!!!",prv_w)
                   
                #_a,_b,_c,loss=self.runG(X,cur_w,cur_hb,cur_vb)        
                #print('Epoch: %d' % epoch)#, 'loss: %f' % loss[-1])

        self.w=prv_w
        self.hb=prv_hb
        self.vb=prv_vb
    def reverse_train(self,X,epochs=5,batchsize=100):
        prv_w=self.w
        prv_hb=self.hb
        prv_vb=self.vb
        
        v0= tf.placeholder("float", [None, self._output_size])
        _w = tf.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.placeholder("float", [self._output_size])
        _vb = tf.placeholder("float", [self._input_size])
        cur_wt,cur_hbt,cur_vbt,_t=self.reverse_runG(v0,_w,_hb,_vb)

        #cur_w=np.zeros([self._input_size,self._output_size],np.float32)
        #cur_hb=np.zeros([self._output_size],np.float32)
        #cur_vb=np.zeros([self._input_size],np.float32)    
        with tf.Session() as sess:
            #for epoch in range(epochs):
                #for start,end in zip(range(0,len(X),batchsize),range(batchsize,len(X),batchsize)):
            batch=X#[start:end]
            cur_w,cur_hb,cur_vb,_=sess.run([cur_wt,cur_hbt,cur_vbt,_t], feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
            #cur_w,cur_hb,cur_vb,_=self.runG(batch,prv_w,prv_hb,prv_vb)
            
            prv_w=cur_w
            prv_hb=cur_hb
            prv_vb=cur_vb
            #print("!!!!!!!!!!",prv_w)
                   
                #_a,_b,_c,loss=self.runG(X,cur_w,cur_hb,cur_vb)        
                #print('Epoch: %d' % epoch)#, 'loss: %f' % loss[-1])

        self.w=prv_w
        self.hb=prv_hb
        self.vb=prv_vb

    def rbm_output(self, X):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)
    
    def rbm_output_reverse(self,X):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _vb = tf.constant(self.vb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _vb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)

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
    net_dg1 = Net_dg(1, dims[2], act[2])
    net_dg2 = Net_dg(2, dims[3], act[3])

    rbm0=RBM(240,200)
    rbm1=RBM(216,200)

    H = np.random.uniform(0, 1, [X1.shape[0], dims[2][0]])
    x1_input = tf.placeholder(np.float32, [None, dims[0][0]])
    x2_input = tf.placeholder(np.float32, [None, dims[1][0]])

    with tf.variable_scope("H"):
        h_input = tf.Variable(xavier_init(batch_size, dims[2][0]), name='LatentSpaceData')
        h_list = tf.trainable_variables()
    fea1_latent = tf.placeholder(np.float32, [None, dims[0][-1]])
    fea2_latent = tf.placeholder(np.float32, [None, dims[1][-1]])

    #loss_pre = net_ae1.loss_reconstruct(x1_input) + net_ae2.loss_reconstruct(x2_input)
    #pre_train = tf.train.AdamOptimizer(lr[0]).minimize(loss_pre)

    #loss_ae = net_ae1.loss_total(x1_input, fea1_latent) + net_ae2.loss_total(x2_input, fea2_latent)
    #update_ae = tf.train.AdamOptimizer(lr[1]).minimize(loss_ae, var_list=net_ae1.netpara.extend(net_ae2.netpara))
    #z_half1 = net_ae1.get_z_half(x1_input)
    #z_half2 = net_ae2.get_z_half(x2_input)
    z_half1=rbm0.rbm_output(x1_input)
    z_half2=rbm1.rbm_output(x1_input)
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
    '''
    for k in range(epochs[0]):
        X1, X2, gt = shuffle(X1, X2, gt)
        for batch_x1, batch_x2, batch_No in next_batch(X1, X2, batch_size):
            _, val_pre = sess.run([pre_train, loss_pre], feed_dict={x1_input: batch_x1, x2_input: batch_x2})
            err_pre.append(val_pre)

            output = "Pre_epoch : {:.0f}, Batch : {:.0f}  ===> Reconstruction loss = {:.4f} ".format((k + 1), batch_No,
                                                                                                    val_pre)
            print(output)
'''
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
            #_, val_ae = sess.run([update_ae, loss_ae], feed_dict={x1_input: batch_x1, x2_input: batch_x2,
              #                                                  fea1_latent: batch_g1, fea2_latent: batch_g2})
            rbm0.train(batch_x1)
            rbm0.reverse_train(batch_g1)
            rbm1.train(batch_x2)
            rbm1.reverse_train(batch_g2)

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

            #val_total = sess.run(loss_ae, feed_dict={x1_input: batch_x1, x2_input: batch_x2,
             #                                       fea1_latent: batch_g1_new, fea2_latent: batch_g2_new})
                                                    
           # err_total.append(val_total)
            
            #output = "Epoch : {:.0f} -- Batch : {:.0f} ===> Total training loss = {:.4f} ".format((j + 1),
             #                                                                                   (num_batch_i + 1),
              #                                                                                  val_total)
            #print(output)

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