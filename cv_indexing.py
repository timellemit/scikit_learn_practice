import numpy as np
from sklearn.cross_validation import LeaveOneOut, KFold
X = np.array([[0., 0.], [1., 1.], [-1., -1.], [2., 2.]])
Y = np.array([0, 1, 0, 1])

loo = LeaveOneOut(len(Y))
print "Leave-One-Out indices"
for train, test in loo:
    print("%s %s" % (train, test))

kf = KFold(len(Y), n_folds=2)
print "Kfold indices"
for train, test in kf:
    print("%s %s" % (train, test))
