# -*- coding: utf-8 -*-
from time import time
from sklearn import metrics

# Оценка качества кластеризации
# inertia - значение минимизируемого фунцкионала - суммы квадратов расстояний
# между объектами одного кластера 
# homogeneity_score - от 0 до 1. Насколько кластеры содержат объекты 
# из разных классов
# completeness_score - от 0 до 1. Наоборот: насколько точно объекты одного 
# класса попадают в один и тот же кластер
# v_measure_score - среднее гармоническое homogeneity_score и 
# completeness_score. 
# v = 2 * (homogeneity * completeness) / (homogeneity + completeness)
# adjusted_rand_score: Rand Index (RI) показывает, насколько две разные 
# кластеризации одного и того же набора данных похожи. Сравнение идет по 
# всем парам объектов - находятся ли они в одинаковых кластерах при обеих
# кластеризациях. 
# Adjusted Rand Index: ARI  = (RI - Expected_RI) / (max(RI) - Expected_RI).
# 0 - для случайной кластеризации, 1 - для совпадающих.
# adjusted_mutual_info_score: Mutual Info (MI) - мера сходства двух 
# кластеризаций, учитывающая вероятности отнесения объекта к определенному
# кластеру. Adjusted Mutual Information - нормировка, нивелирующая то, что 
# для больших кластеров MI обычно выше.
# silhouette_score - насколько среднее расстояние до объектов своего кластера
# отличается от среднего расстояния до объектов других кластеров
# ss = (b - a) / max(a,b). b - расстояние от объекта до ближайшего кластера, к 
# которому он не принадлежит, a - среднее расстояние до объектов из своего 
# кластера.
def bench_k_means(estimator, name, data, labels):
    t0 = time()
    estimator.fit(data)
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
            % (name, (time() - t0), estimator.inertia_,
            metrics.homogeneity_score(labels, estimator.labels_),
            metrics.completeness_score(labels,estimator.labels_),
            metrics.v_measure_score(labels, estimator.labels_),
            metrics.adjusted_rand_score(labels, estimator.labels_),
            metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
            metrics.silhouette_score(data, estimator.labels_,
                                              metric='euclidean')))