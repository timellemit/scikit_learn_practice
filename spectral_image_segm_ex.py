# -*- coding: utf-8 -*-
# Authors:  Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#           Gael Varoquaux <gael.varoquaux@normalesup.org>
# License: BSD 3 clause

import numpy as np
import pylab as pl

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

###############################################################################
l = 100
# x и y - матрицы 100x100, первая строка из нулей, вторая - из единиц и т.д.
x, y = np.indices((l, l))

center1 = (28, 24)
center2 = (40, 50)
center3 = (67, 58)
center4 = (24, 70)

radius1, radius2, radius3, radius4 = 16, 14, 15, 14

# задаются 4 круга в виде 100x100 булевых матриц
circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3 ** 2
circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4 ** 2

###############################################################################
# 2 circles
# массив 100x100 с единицами на тех позициях, которые попадают
# в один из кругов circle1 или circle2
img = circle1 + circle2
mask = img.astype(bool)
img = img.astype(float)

# вносится "шум" 
img += 1 + 0.2 * np.random.randn(*img.shape)
# кластеризовать будем только область, включающую два
# круга - это задается параметром mask
# graph - взвешенный граф, вершины которого пикселы, на ребрах - 
# градиентная разность цветов соответствующих пикселов 
graph = image.img_to_graph(img, mask=mask)
# print graph
# нормализация весов на ребрах графа
graph.data = np.exp(-graph.data / graph.data.std())
# print graph.data
# pl.hist( graph.data) 
labels = spectral_clustering(graph, n_clusters=2, eigen_solver='arpack')

# заполняем весь "фон" числами, отличными от 0 или 1 - например, -1
label_im = -np.ones(mask.shape)
# пиксели, попавшие в один из кругов, помечаем меткой кластера - 0 или 1
label_im[mask] = labels

pl.matshow(img)
pl.matshow(label_im)

###############################################################################
# 4 круга
img = circle1 + circle2 + circle3 + circle4
mask = img.astype(bool)
img = img.astype(float)
img += 1 + 0.2 * np.random.randn(*img.shape)
 
graph = image.img_to_graph(img, mask=mask)
graph.data = np.exp(-graph.data / graph.data.std())
 
labels = spectral_clustering(graph, n_clusters=4, eigen_solver='arpack')
label_im = -np.ones(mask.shape)
label_im[mask] = labels
 
pl.matshow(img)
pl.matshow(label_im)

pl.show()