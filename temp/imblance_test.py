from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.svm import SVC
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import scipy.stats


def classification_result(n0, n1, title=""):
    rv1 = sp.stats.multivariate_normal(mean=[-1, 0], cov=[[1, 0], [0, 1]])
    rv2 = sp.stats.multivariate_normal(mean=[+1, 0], cov=[[1, 0], [0, 1]])
    X0 = rv1.rvs(n0, random_state=0)
    X1 = rv2.rvs(n1, random_state=0)

    print(X0.shape, X1.shape)

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n0), np.ones(n1)])

    print(X.shape, y.shape)
    print()

    x1min = -4;
    x1max = 4
    x2min = -2;
    x2max = 2
    xx1 = np.linspace(x1min, x1max, 1000)
    xx2 = np.linspace(x2min, x2max, 1000)
    X1, X2 = np.meshgrid(xx1, xx2)

    plt.contour(X1, X2, rv1.pdf(np.dstack([X1, X2])), levels=[0.05], linestyles="dashed")
    plt.contour(X1, X2, rv2.pdf(np.dstack([X1, X2])), levels=[0.05], linestyles="dashed")

    model = SVC(kernel="linear", C=1e4, random_state=0).fit(X, y)

    pred = model.predict(np.array([X1.ravel(), X2.ravel()]).T)
    Y = np.reshape(pred, X1.shape)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='x', label="Class - 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', label="Class - 1")

    plt.contour(X1, X2, Y, colors='k', levels=[0.5])

    y_pred = model.predict(X)
    plt.xlim(-4, 4)
    plt.ylim(-3, 3)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)

    return model, X, y, y_pred


plt.subplot(121)
model1, X1, y1, y_pred1 = classification_result(200, 200, "Balanced Data (5:5)")

plt.subplot(122)
model2, X2, y2, y_pred2 = classification_result(200, 20, "Imbalanced Data (9:1)")

plt.tight_layout()
plt.show()

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y1, y_pred1))
print(classification_report(y2, y_pred2))