# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from benchmark import basic_benchmark

##############################################################################
# генерация синтетических данных
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)
# шкалирование выборки
X = StandardScaler().fit_transform(X)

##############################################################################
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
# плотно-связанные (с каким-либо соседом) точки 
core_samples = db.core_sample_indices_
# print len(core_samples)
labels = db.labels_

# Количество кластеров в массиве labels, не учитывая выбросы (метка -1)
# конструктор set(ar) оставляет только уникальные значения списка ar.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# оценка характеристических параметров кластеризации
basic_benchmark(db, "DBSCAN", X, labels_true)

##############################################################################
# Plot result
import pylab as pl

unique_labels = set(labels)
# цвета взяты из библиотеки цветовых карт (color maps)
colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    # -1 - метка для выбросов (черный цвет)
    if k == -1:
        # для выбросов используется черный цвет
        col = 'k'
        markersize = 6
    # индексы тех позиций массива labels, где стоит метка k
    class_members = [index[0] for index in np.argwhere(labels == k)]
#     print len(class_members)
    # плотно-связанные точки
    cluster_core_samples = [index for index in core_samples
                            if labels[index] == k]
#     print len(cluster_core_samples)
    for index in class_members:
        x = X[index]
        # используются разные маркеры для плотно-связанных и
        # граничных точек
        if index in core_samples and k != -1:
            markersize = 14
        else:
            markersize = 6
        pl.plot(x[0], x[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=markersize)

pl.title('Estimated number of clusters: %d' % n_clusters_)
pl.show()