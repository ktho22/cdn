import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plotcent(cent):
    '''
        cent.shape should be (nCent, nDim)
    '''
    def con(a,b):
        return np.concatenate((a,b),axis=1)
    #cent = np.swapaxes(cent,0,2)
    #cent = np.swapaxes(cent,1,2)
    cent2=reduce(con,cent)
    plt.imshow(cent2,cmap = plt.get_cmap('gray'))
    
def genData():
    nData = 1000

    m1 = [3,3]
    s1 = [[0.5,0],[0,0.5]]
    m2 = [0,0]
    s2 = [[0.2,0.1],[0.1,0.2]]
    m3 = [-3,3]
    s3 = [[0.1,0.1],[0.1,0.2]]

    X = np.concatenate(
        (np.random.multivariate_normal(m1,s1,nData),
        np.random.multivariate_normal(m2,s2,nData),
        np.random.multivariate_normal(m3,s3,nData)),
        axis=0)
    Y = np.concatenate(
        (np.zeros(nData),
        np.ones(nData),
        np.ones(nData)*2))

    return X, Y
