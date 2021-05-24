import sklearn.datasets as dsts
import numpy as np
from mondrian import MondrianTreeClassifier,MondrianForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PolyaMondrian import MondrianPolyaTree,MondrianPolyaForest

from sklearn import metrics
# from skgarden import MondrianTreeClassifier as MondrianTreeClassifierSk
import warnings
import h5py
import time
import scipy.io
import cProfile
import matplotlib.pyplot as plt

def main2():
    
    with h5py.File("usps.h5", 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]

    # X = dsts.load_wine().data
    # y = dsts.load_wine().target

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # mt = MondrianForestClassifier(n_estimators=100)
    # # mt = MondrianTreeClassifier()

    # s = time.time()

    # mt.fit(X_tr,y_tr)


    
    # print(time.time()-s)
    # X_train = np.array([[0.0,0.0],[0.25,0.25],[0.4,0.8],[1.0,1.0]])



    
    mat = scipy.io.loadmat('./data/thyroid.mat')
    X_train = mat['X']
    y = mat['y']
    mpt = MondrianPolyaTree()

    mpt.fit(X_train)
    
    scores = mpt.score_all()

    np.count_nonzero(scores > 0.003)
    
    
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
    print(metrics.auc(fpr, tpr))

    plt.plot(np.sort(scores))
    # plt.plot(y_new)
    plt.show()



    


    # print(accuracy_score(y_te, mt.predict(X_te)))

if __name__ == "__main__":
    main2()