from sklearn.cluster import KMeans
import evaluation
import numpy as np
import scipy.io as sio
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph,NearestNeighbors

estimator = KMeans(n_clusters=5, n_init=20)  # 聚类数量
data_0 = sio.loadmat('./bdgp_view1.mat')
labels = sio.loadmat('./bdgp_label.mat')
labels = dict(labels)
labels = labels['Ytr'].reshape(-1)
data_dict = dict(data_0)
X_init = data_dict['Xtr'] # 2000*76

estimator.fit(X_init)
label_pred = estimator.labels_
y_pred = SpectralClustering(n_clusters=5,gamma=10,affinity='nearest_neighbors',n_neighbors=15).fit_predict(X_init)

nmi, ari, f, acc = evaluation.evaluate(labels, y_pred)
#nmi,ari,f,acc=evaluation.evaluate(labels,label_pred)
print(acc)
print(nmi)
data_0 = sio.loadmat('./bdgp_view1.mat')
data_dict = dict(data_0)
X_init = data_dict['Xtr'] #
connectivity = kneighbors_graph(X_init, n_neighbors=30,mode='distance',
                                            include_self=True,
                                            n_jobs=None)

affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
affinity_matrix_=affinity_matrix_.toarray()
sio.savemat('.aff.mat',{'S':affinity_matrix_})