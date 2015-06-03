# -* - coding : utf -8 -* -
"""
ML & DM Homework #3 Loptev
"""
import multiprocessing
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

CV_FOLDS = 4
TEST_FRACTION = 0.2
GRID_SEARCH_SCORE ='accuracy'

if __name__ =='__main__':
    data = loadmat ('data/digits.mat')
    x, y = data ['X'], np.squeeze ( data ['y'])
    tests = [(KNeighborsClassifier(), {'n_neighbors': range (3, 7) }),
             (LogisticRegression(), [{'C': 10.0 ** np.arange (-2, 3)},
                                      {'C': np.linspace(0.1, 1.5, 5)}]),
             (SVC(kernel ='rbf'), [{'kernel': ['rbf'],'gamma': 10.0 ** np.arange (-4, 1),
                                        'C': 10.0 ** np.arange (-2, 8)},
                                   {'kernel': ['rbf'], 'gamma': np.linspace (0.05, 0.1, 5),
                                        'C': 10.0 ** np.arange ( -2, 8)},]),
             (SVC(kernel ='poly'), {'kernel': ['poly'],'degree': [3, 4],
                                     'C': 10.0 ** np.arange ( -2, 8)}),]
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                        test_size = TEST_FRACTION, random_state =0)
    bests = []
    for estimator, param_space in tests :
        print 'Tuning parameters of', type(estimator ).__name__
        clf = GridSearchCV(estimator, param_space, cv = CV_FOLDS, scoring =
                           GRID_SEARCH_SCORE, n_jobs = multiprocessing.cpu_count(), verbose =1)
        clf.fit(x_train, y_train)
        print 'Best estimator :', clf.best_estimator_,'\n'
        bests.append(clf.best_estimator_)
        print 'Scores of all estimators :'
        for params, mean_score, scores in clf.grid_scores_:
            print " %s : %0.4f (+/ -%0.03f ) for %r " % (GRID_SEARCH_SCORE, 
                                                         mean_score, scores.std()/2, params)
        
    print 'Testing winners'
    for estimator in bests :
        print 'Testing', estimator,', result:'
        print classification_report(y_test, estimator.predict(x_test))