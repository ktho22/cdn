import numpy as np
import ipdb

def genData():
    nData = 1000

    m1 = [3,3]
    s1 = [[0.5,0],[0,0.5]]
    m2 = [0,0]
    s2 = [[0.2,0.1],[0.1,0.2]]

    X = np.concatenate(
        (np.random.multivariate_normal(m1,s1,nData),
        np.random.multivariate_normal(m2,s2,nData)),
        axis=0)
    Y = np.concatenate(
        (np.zeros(nData),
        np.ones(nData)))

    return X, Y

def plotData(X,Y=None):
    if Y==None: Y=np.ones(X.shape)
    import matplotlib.pyplot as plt

    Y_dom = np.unique(Y)
    for y in Y_dom:
        idx = np.argwhere(Y==y)
        plt.scatter(X[idx,0],X[idx,1],c=np.random.random(3,))
    plt.show()

X,Y = genData()


plotData(X,Y)
ipdb.set_trace()
