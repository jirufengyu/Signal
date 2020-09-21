from utils.cluster import cluster
import warnings

warnings.filterwarnings('ignore')


def print_result(n_clusters, H, gt, count=10):
    acc_avg,nmi_avg,ari_avg,f1_avg = cluster(n_clusters, H, gt, count=count)
    print('clustering h      : acc = {:.4f}, nmi = {:.4f},ari = {:.4f}, f1 = {:.4f}'.format(acc_avg, nmi_avg,ari_avg, f1_avg))
