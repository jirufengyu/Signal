'''
Author: jirufengyu
Date: 2020-11-05 15:18:33
LastEditTime: 2020-11-05 15:28:30
LastEditors: jirufengyu
Description: Nothing
FilePath: /Signal-1/AE2-Nets-master/test_CUB.py
'''
from utils.Dataset import Dataset
from AE_BinAE_revise import MaeAEModel
from model import model
from utils.print_result import print_result
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
each net has its own learning_rate(lr_xx), activation_function(act_xx), nodes_of_layers(dims_xx)
ae net need pretraining before the whole optimization
'''
if __name__ == '__main__':
    data = Dataset('CUB_c10_2views')
    x1, x2, gt = data.load_data()
    x1 = data.normalize(x1, 0)
    x2 = data.normalize(x2, 0)
    n_clusters = len(set(gt))
    print(x1.shape)
    print(x2.shape)
    print(gt.shape)
    #act_ae1, act_ae2, act_dg1, act_dg2 = 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'
    v1_aedims_ = [[x1.shape[1], 512,256],[256,512,x1.shape[1]]]
    
    v2_aedims_ = [[x2.shape[1],  256,128],[128,256,x2.shape[1]]]
    #原来的
    mae_dims_=[[256,128,64],[128,128,64],[64,128,256],[64,128,128]]
    #现在用的
    #dims_dg1 = [64, 100]
    #dims_dg2 = [64, 100]
    dis_dims_=[256,128,1]
    para_lambda = 1
    batch_size = 100
    
    epochs = 200

    model=MaeAEModel(v1_aedims=v1_aedims_,v2_aedims=v2_aedims_,mae_dims=mae_dims_,dis_dims=dis_dims_)        #duaAE用的
    H, gt = model.train_model(x1, x2, gt, epochs, batch_size)
    #H,gt=model(x1, x2, gt, para_lambda, dims, act, lr, epochs, batch_size)
    print_result(n_clusters, H, gt)