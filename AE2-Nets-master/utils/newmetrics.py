import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import sklearn
from munkres import Munkres
def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels
def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:,j]) # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i,j]
            cost_matrix[j,i] = s-t
    return cost_matrix
def get_accuracy(cluster_assignments, y_true, n_clusters):
    '''
    Computes the accuracy based on the provided kmeans cluster assignments
    and true labels, using the Munkres algorithm

    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset

    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    '''
    y_pred, confusion_matrix = get_y_preds(cluster_assignments, y_true, n_clusters)
    # calculate the accuracy
    return np.mean(y_pred == y_true), confusion_matrix

def print_accuracy(cluster_assignments, y_true, n_clusters, extra_identifier=''):
    '''
    Convenience function: prints the accuracy
    '''
    # get accuracy
    accuracy, confusion_matrix = get_accuracy(cluster_assignments, y_true, n_clusters)
    # get the confusion matrix
    #print('confusion matrix{}: '.format(extra_identifier))
    #print(confusion_matrix)
    #print('standard_acc '.format(extra_identifier) + str(np.round(accuracy, 4)))
    return accuracy

def get_y_preds(cluster_assignments, y_true, n_clusters):
    '''
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)

    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset

    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    '''
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred, confusion_matrix

#以上辅助计算原论文的acc
def getMetrics(y_true, y_pred):
    
       
        
    NMI = normalized_mutual_info_score(y_true, y_pred)
    ARI = adjusted_rand_score(y_true, y_pred)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_pred)  # 样本总数
    assert len(y_pred) == len(y_true), "len(pred) not equal to len(y_true)"
    true = np.unique(y_true)

    pred = np.unique(y_pred)
    true_size = len(true)  # 真实值中有多少个类别
    pred_size = len(pred)  # 预测值中有多少个类别
    print(true_size)
    # 计算得到混淆矩阵
    a = np.ones((true_size, 1), dtype=int) * y_true  #
    b = true.reshape(true_size, 1) * np.ones((1, n), dtype=int)
    pid = (a == b) * 1  # true_size by n
    a = np.ones((pred_size, 1), dtype=int) * y_pred
    b = pred.reshape(pred_size, 1) * np.ones((1, n), dtype=int)
    cid = (a == b) * 1  # pred_size by n
    confusion_matrix = np.matmul(pid, cid.T)  # 计算得到混淆矩阵
    #        P    N
    #   P   TP   FN
    #   N   FP   TN
    # 计算得到准确率
    temp = np.max(confusion_matrix, axis=0)  # 选择正确数最多计算准确率
    Accuracy = np.sum(temp, axis=0) / float(n)

    # ------------------------计算得到f-score
    ci = np.sum(confusion_matrix, axis=0)  # [TP+FP,FN+TN]  预测
    pj = np.sum(confusion_matrix, axis=1)  # [TP+FN,FP+TN]  真实
    # 分情况交叉计算精确率
    precision = confusion_matrix / (np.ones((true_size, 1), dtype=float) * ci.reshape(1, len(ci)))
    # 分情况交叉计算召回率
    recall = confusion_matrix / (pj.reshape(len(pj), 1) * np.ones((1, pred_size), dtype=float))

    F = 2 * precision * recall / (precision + recall)
    F = np.nan_to_num(F)
    temp = (pj / float(pj.sum())) * np.max(F, axis=0)
    Fscore = np.sum(temp, axis=0)
    Accuracy=print_accuracy(y_pred, y_true, 10, extra_identifier='')
    return Fscore, Accuracy, NMI, ARI