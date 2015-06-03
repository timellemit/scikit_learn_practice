# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from benchmark import basic_benchmark

zoo_data = np.loadtxt("data/zoo.csv", delimiter=",") 
data = scale(zoo_data[:,:zoo_data.shape[1]-1])
labels = zoo_data[:,-1]

# величина окна, используемого ядерной фунцкией оценки плотности в 
# алгоритме Mean-Shift, может быть рассчитана автоматически
bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=50)

# bin seeding позволяет ускорить алгоритм и отсеять кластеры, состоящие из 
# малого количества точек. В пространстве объектов строится сетка, и объекты
# помещаются в ячейки. Процедура начинается только с тех ячеек, в которых 
# объектов не меньше определенного порога.
# http://sociograph.blogspot.ru/2011/11/accessible-introduction-to-mean-shift.html   
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
basic_benchmark(ms, "Mean-Shift", data, labels)

# отображение будет проводиться для двух главных 
# компонент данных
reduced_data = PCA(n_components=2).fit_transform(data)
bandwidth = estimate_bandwidth(reduced_data, quantile=0.2, n_samples=50)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(reduced_data)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
labels_unique = np.unique(labels)
# число кластеров на выходе
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

# отображение двух главных компонент данных
import pylab as pl
from itertools import cycle

pl.figure(1)
pl.clf()
colors = cycle('bgrmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    pl.plot(reduced_data[my_members, 0], reduced_data[my_members, 1], col + '.')
    pl.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=14)
pl.title('Estimated number of clusters: %d' % n_clusters_)
pl.show()