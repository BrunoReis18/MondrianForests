import sklearn.datasets as dsts
import numpy as np
from mondrian import MondrianTreeClassifier,MondrianForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from PolyaMondrian import MondrianPolyaTree,MondrianPolyaForest

from sklearn import metrics
# from skgarden import MondrianTreeClassifier as MondrianTreeClassifierSk
import warnings
import h5py
import time
import scipy.io
import cProfile
import matplotlib.pyplot as plt
from skmultiflow.data import WaveformGenerator,FileStream,DataStream
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import AnomalySineGenerator
from skmultiflow.anomaly_detection import HalfSpaceTrees
# from river import anomaly
# from river import compose
# from river import datasets
# from river import metrics
# from river import preprocessing
# from river import evaluate



def main5():
    model = MondrianForestClassifier()


    
    X = [0.5, 0.45, 0.43, 0.44, 0.445, 0.45, 0.0]


    for x in X[:3]:
        hst = model.learn_one(x)  # Warming up

    for x in X:
     
        hst = model.learn_one(x)
        print(f'Anomaly score for x={x:.3f}: {hst.score_one(x):.3f}')

def main3():
    
        
    with h5py.File("usps.h5", 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]
    

    # stream = DataStream(np.append(X_tr,X_te,0),np.append(y_tr,y_te,0))
    # 1. Create a stream
    # stream = FileStream("./datasets/elec.csv")
   
    # stream = FileStream("./datasets/weather.csv")
    stream = FileStream("./datasets/sea_a.csv")

    # 2. Instantiate the HoeffdingTreeClassifier
    # ht = HoeffdingTreeClassifier()


    # ht = MondrianTreeClassifier()
    ht = MondrianForestClassifier(n_estimators=50)

    # 3. Setup the evaluator
    evaluator = EvaluatePrequential(show_plot=True,
                                    pretrain_size=3000,
                                    max_samples=15000,
                                    batch_size=100,
                                    metrics = ["accuracy", "precision","recall"])
    # 4. Run evaluation
    evaluator.evaluate(stream=stream, model=ht)

def main2():

    with h5py.File("usps.h5", 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]

    mt = MondrianForestClassifier(n_estimators=10)
    # # mt = MondrianTreeClassifier()


    mt.fit(X_tr,y_tr)

    print(mt.forest[0])

    # mt.partial_fit(X_tr2,y_tr2)
    
    # # print(time.time()-s)

    # s = time.time()
    print(accuracy_score(y_te, mt.predict(X_te)))
    # print(time.time()-s)
    # X_train = np.array([[0.0,0.0],[0.25,0.25],[0.4,0.8],[1.0,1.0]])




    
    # mat = scipy.io.loadmat('./data/thyroid.mat')
    # X_train = mat['X']
    # y = mat['y']
    # mpt = MondrianPolyaTree()

    # mpt.fit(X_train)
    
    # scores = mpt.score_all()

    
    
    
    # fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
    # print(metrics.auc(fpr, tpr))

    # plt.plot(np.sort(scores))
    # # plt.plot(y_new)
    # plt.show()



    



if __name__ == "__main__":
    main2()
