'''
Author: jirufengyu
Date: 2020-10-29 09:12:25
LastEditTime: 2020-10-29 11:44:19
LastEditors: jirufengyu
Description: In User Settings Edit+
FilePath: /Signal-1/AE2-Nets-master/tSNE.py
'''
import h5py
import os
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
# 对样本进行预处理并画图
def plot_embedding(data, label, ):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)     # 对数据进行归一化处理
    fig = plt.figure()      # 创建图形实例
    ax = plt.subplot(111)       # 创建子图
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10),
                 fontdict={'weight' :  'bold' ,  'size' : 10})
    plt.xticks()        # 指定坐标的刻度
    plt.yticks()
    #plt.title(title, fontsize=14)
    # 返回值
    return fig
if __name__ == '__main__':
    data_path="/home/stu2/Signal-1/AE2-Nets-master/H.mat"

    dataset = scipy.io.loadmat(data_path)
    H,gt=dataset['H'],dataset['gt']
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    
    result = tsne.fit_transform(H)
    fig = plot_embedding(result,gt)
    plt.show(fig)

