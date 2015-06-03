from random import shuffle, random
import numpy as np
import matplotlib.pylab as pl
from sklearn.cluster import spectral_clustering

# n = 1000;
# x = randperm(n);
# p1 = 0.5;
# p2 = 0.4;
# pb = 0.1;
# gs = 450;
# group1 = x(1:gs);
# group2 = x(gs+1:end);
# A(group1, group1) = rand(gs,gs) < p1;
# A(group2, group2) = rand(n-gs,n-gs) < p2;
# A(group1, group2) = rand(gs,n-gs) < pb;
# A  = triu(A,1);
# A = A'+A;
# spy(A);
# D = diag(sum(A));
# L = D-A;
# [V,D] = eigs(L,2,'SA');
# %plot(sort(V(:,2)),'-');
# [x,i] = sort(V(:,2));
# %spy(A(i,i));

n = 10
x = range(n)
shuffle(x)
p1, p2, p_betw  = 0.5, 0.4, 0.1
group1_size = 4
group1_ind, group2_ind = x[:group1_size], x[group1_size:]
A = np.zeros(shape=(n,n))
for i in group1_ind:
    for j in group1_ind:
        A[i, j] = random() < p1
for i in group2_ind:
    for j in group2_ind:
        A[i, j] = random() < p2
for i in group1_ind:
    for j in group2_ind:
        A[i, j] = random() < p_betw

A = np.tril(A.T) + np.triu(A,1)
labels = spectral_clustering(A, n_clusters=4)
print labels
print A
pl.spy(A)
pl.matshow(A)
pl.show()