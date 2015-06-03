# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from benchmark import basic_benchmark

# Для воспроизводимости случайного выбора центроидов из множества
# объектов в случае вызова kmeans с параметром init='random'
np.random.seed(42)

zoo_data = np.loadtxt("data/zoo.csv", delimiter=",") 
data = scale(zoo_data[:,:zoo_data.shape[1]-1]) 

# Для оценки оценки качества кластеризации
labels = zoo_data[:,-1]

n_samples, n_features = data.shape
n_classes = len(np.unique(labels))

print("n_classes: %d, \t n_samples %d, \t n_features %d"
      % (n_classes, n_samples, n_features))

print(79 * '_')
# 'k-means++' выбирает начальные центры кластеров таким образом, что это
# ускоряет сходимость метода
basic_benchmark(KMeans(init='k-means++', n_clusters=n_classes, n_init=10),
              name="k-means++", data=data, labels=labels)
# начальные центроиды выбираются случайно из объектов
basic_benchmark(KMeans(init='random', n_clusters=n_classes, n_init=10),
              name="random", data=data, labels=labels)
# K-means, инициализированный главными компонентами данных
pca = PCA(n_components=n_classes).fit(data)
basic_benchmark(KMeans(init=pca.components_, n_clusters=n_classes, n_init=1),
              name="PCA-based",
              data=data, labels=labels)



###############################################################################
# Визуализация двух выделенных компонент (с помощью PCA) в данных

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_classes, n_init=10)
kmeans.fit(reduced_data)
# Сетка с( шагом h в прямоугольнике [x_min, x_max][y_min, y_max]
# для раскрашивания в соответствии с меткой класса
h = .01
x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Z - вектор меток класса для узлов сетки
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
pl.figure(1)
pl.clf()

# Метки классов используются для окраски фона
pl.imshow(Z, interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
          cmap=pl.cm.Paired,
          aspect='auto', origin='lower')

# Отображение точек, соответствующих двум главным компонентам
# начальных данных 
pl.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

# центроиды отображаются белыми крестами
centroids = kmeans.cluster_centers_
pl.scatter(centroids[:, 0], centroids[:, 1],
           marker='x', s=169, linewidths=3,
           color='w', zorder=10)
pl.title('K-means clustering on the zoo dataset (PCA-reduced data)\n'
         'Centroids are marked with white crosses.')
pl.xlim(x_min, x_max)
pl.ylim(y_min, y_max)
pl.xticks(())
pl.yticks(())
pl.show()