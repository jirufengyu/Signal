import numpy as np
from utils.Dataset import Dataset
from model import model
from utils.print_result import print_result
from utils.cluster import cluster
import csv
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print(os.environ['CUDA_VISIBLE_DEVICES'])
'''
each net has its own learning_rate(lr_xx), activation_function(act_xx), nodes_of_layers(dims_xx)
ae net need pretraining before the whole optimizatoin
'''
if __name__ == '__main__':

    num = 1
    data = Dataset('coil_2views')
    x1, x2, gt = data.load_data()

    acc_H_all = np.zeros(num)
    nmi_H_all = np.zeros(num)
    RI_H_all = np.zeros(num)
    f1_H_all = np.zeros(num)

    act_ae1, act_ae2, act_dg1, act_dg2 = 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'
    dims_ae1 = [x1.shape[1], 1024, 512]
    dims_ae2 = [x2.shape[1], 1024, 512]
    dims_dg1 = [256, 512]
    dims_dg2 = [256, 512]

    para_lambda = 1
    batch_size = 50
    lr_pre = 1.0e-4
    lr_ae = 1.0e-4
    lr_dg = 1.0e-3
    lr_h = 1.0e-3
    epochs_pre = 50
    epochs_total = 200
    epochs_h = 100

    act = [act_ae1, act_ae2, act_dg1, act_dg2]
    dims = [dims_ae1, dims_ae2, dims_dg1, dims_dg2]
    lr = [lr_pre, lr_ae, lr_dg, lr_h]
    epochs = [epochs_pre, epochs_total, epochs_h]

    for j in range(num):
        data = Dataset('coil_2views')
        x1, x2, gt = data.load_data()
        x1 = data.normalize(x1, 0)
        x2 = data.normalize(x2, 0)
        n_clusters = len(set(gt))
        H, gt = model(x1, x2, gt, para_lambda, dims, act, lr, epochs, batch_size)

        acc_H_all[j], acc_H_std, nmi_H_all[j], nmi_H_std, RI_H_all[j], RI_std, f1_H_all[j], f1_std = cluster(n_clusters,
                                                                                                             H, gt,
                                                                                                             count=1)
        print('clustering h      : acc = {:.4f}, nmi = {:.4f}'.format(acc_H_all[j], nmi_H_all[j]))

    acc_mean = np.mean(acc_H_all)
    nmi_mean = np.mean(nmi_H_all)
    acc_std = np.std(acc_H_all)
    nmi_std = np.std(nmi_H_all)

    RI_mean = np.mean(RI_H_all)
    RI_std = np.std(RI_H_all)

    fs_mean = np.mean(f1_H_all)
    fs_std = np.std(f1_H_all)

    arg = ['lambda', para_lambda, 'batch_size', batch_size, 'lr_pre', lr_pre, 'lr_ae', lr_ae,
           'lr_dg', lr_dg, 'lr_h', lr_h, 'epoch_pre', epochs_pre, 'epochs_total', epochs_total, 'dim', dims]
    files = open('result_coil.csv', 'a')
    writer = csv.writer(files)
    writer.writerow(arg)

    data_w = ['sigmoid', acc_mean, acc_std, nmi_mean, nmi_std, RI_mean
        , RI_std, fs_mean, fs_std]

    print('clustering h      : acc = {:.4f}, nmi = {:.4f}'.format(acc_mean, nmi_mean))
    writer.writerow(data_w)
