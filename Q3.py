import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

df = pd.read_csv('C:/Users/Jingwei/Desktop/input.dat', header=None, sep='\s+')
df.columns = ['x1', 'x2']
X = df[['x1', 'x2']].values
df2 = pd.read_csv('C:/Users/Jingwei/Desktop/output.dat', header=None, sep='\s+')
df2.columns = [['y']]
Y = df2[['y']].values


def plot_decision_regions(XX, YY, B, abs_error,X_test,y_test, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'o', 'x', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(YY))])
    y = []
    # tmpy = YY[:,1].split('\n')
    for elem in np.transpose(YY)[0, :]:
        y.append(elem)
    # plot the decision surface
    # X = np.mat(XX)

    # ytt=YY[:,np.newaxis]
    # y = np.transpose(ytt)
    # y = np.mat(YY)
    # x1_min =min(X[:, 0])-1
    x1_min, x1_max = min(X[:, 0]) - 1, max(X[:, 0]) + 1
    x2_min, x2_max = min(X[:, 1]) - 1, max(X[:, 1]) + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    myx = np.array([xx1.ravel(), xx2.ravel()]).T
    print(myx.shape)
    bb = B*B
    penalty = 0.1*sum(abs(B*B))
    Z = myx.dot(B) + abs_error+penalty

    myztmp = []
    for tmp in Z:
        if tmp > 0.5:
            myztmp.append(1)
        else:
            myztmp.append(0)
    myztmp = np.mat(myztmp)
    Z = myztmp.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        x_ = XX[y == cl, 0]
        y_ = XX[y == cl, 1]
        plt.scatter(x_, y_, alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
        # plt.scatter(1, -4.5, alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    #highlight the test sample
    myX_test, myy_test = X_test, y_test
    #judge if the test set is empty or not
    if myX_test.size != 0:
        plt.scatter(X_test[:,0], X_test[:,1], c='',alpha=1.0, linewidths=1, marker='o',s=55,label='test set')


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=10000, random_state=0)

lr.fit(X, Y)
print(lr.coef_)
print(lr.intercept_)

YYYY = lr.predict(X)

from sklearn.metrics import r2_score

print(r2_score(YYYY, Y))

# print(lr.coef_)
# print(lr.intercept_)

'''

R = []
for tmp in Y:
        R.append(tmp*(1-tmp))
R=np.dot(R,np.diag(1))

class MyLogisticRegression():
        def __init__(self):
               self.coef = np.mat([0,0])
               self.intercept = 0

        def fit(self,X,Y,R):
            index = 20
            while(index>0):
                Y_Pred = X*np.transpose(self.coef)+self.intercept
                z=X*np.transpose(self.coef)-np.linalg.inv(R)*(Y_Pred-Y)
                self.coef = np.dot(np.linalg.inv(np.dot(np.linalg.inv(X),R,X)),np.transpose(X),R,z)
                R = []
                for tmp in Y_Pred:
                        R.append(tmp * (1 - tmp))
                index=index-1
            #print(self.coef)
            #print(self.intercept)

'''
from numpy import array, diag, dot, maximum, empty, repeat, ones, sum
from numpy.linalg import inv
from numpy import zeros


def IRLS(y, X, maxiter, w_init=1, d=0.0001, tolerance=0.001):
    n, p = X.shape
    # 生成数组，其中repeat函数生成n个d，reshape函数将一维数组变成二维数组，
    # 一行n列的数组
    delta = array(repeat(d, n)).reshape(1, n)
    # w是n个1的数组
    w = repeat(1, n)
    # W是对角线上为w的对角矩阵
    W = diag(w)
    z = inv(W).dot(y)
    B = dot(inv(X.T.dot(W).dot(X)),
            (X.T.dot(W).dot(z)))
    for _ in range(maxiter):
        _B = B
        _w = abs(y - X.dot(B)).T
        # w = float(1) / maximum(delta, _w)
        tmpx = X.dot(B)

        tmpxx = tmpx * (1 - tmpx)
        tmpxxx = tmpxx.reshape(1, 99)
        W = diag(tmpxxx[0])
        z = X.dot(B) - inv(W).dot(X.dot(B) - y)
        B = dot(inv(X.T.dot(W).dot(X)),
                (X.T.dot(W).dot(z)))
        tol = sum(abs(B - _B))
        print("Tolerance = %s" % tol)
        if tol < tolerance:
            return B
    return B


'''


mlr = MyLogisticRegression()
mlr.__init__()
mlr.fit(X=X,Y=Y,R=R)
'''
B = IRLS(y=Y, X=X, maxiter=10)
abs_error = sum(Y - X.dot(B)) / Y.shape[0]
print("B is")
print(B)
print("intercept is")
print(abs_error)
print(r2_score(X.dot(B) + abs_error, Y))
# plot_decision_regions(XX=X,YY=Y,classifier=lr)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,Y,test_size=0.3,random_state=0)
X_test
plot_decision_regions(XX=X, YY=Y, B=B,X_test=X_test,y_test=y_test, abs_error=abs_error)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.show()
