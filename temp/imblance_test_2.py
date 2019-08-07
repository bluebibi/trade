from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, classification_report, f1_score
from sklearn.svm import SVC
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import scipy.stats


from imblearn.under_sampling import *

n0 = 1000; n1 = 20

rv1 = sp.stats.multivariate_normal([-1, 0], [[1, 0], [0, 1]])
rv2 = sp.stats.multivariate_normal([+1, 0], [[1, 0], [0, 1]])
X0 = rv1.rvs(n0, random_state=0)
X1 = rv2.rvs(n1, random_state=0)
X_imb = np.vstack([X0, X1])
y_imb = np.hstack([np.zeros(n0), np.ones(n1)])

model_samp = SVC(kernel="linear", C=1e4, random_state=0).fit(X_imb, y_imb)
print("Original")
print(classification_report(y_imb, model_samp.predict(X_imb)))
print()

X_samp, y_samp = RandomUnderSampler(random_state=0).fit_sample(X_imb, y_imb)
model_samp = SVC(kernel="linear", C=1e4, random_state=0).fit(X_samp, y_samp)
print("RandomUnderSampler")
print(classification_report(y_samp, model_samp.predict(X_samp)))

print()

X_samp, y_samp = TomekLinks(random_state=0).fit_sample(X_imb, y_imb)
model_samp = SVC(kernel="linear", C=1e4, random_state=0).fit(X_samp, y_samp)
print("TomekLinks")
print(classification_report(y_samp, model_samp.predict(X_samp)))
print()

X_samp, y_samp = CondensedNearestNeighbour(random_state=0).fit_sample(X_imb, y_imb)
model_samp = SVC(kernel="linear", C=1e4, random_state=0).fit(X_samp, y_samp)
print("CondensedNearestNeighbour")
print(classification_report(y_samp, model_samp.predict(X_samp)))
print()

X_samp, y_samp = NeighbourhoodCleaningRule(kind_sel="all", n_neighbors=5, random_state=0).fit_sample(X_imb, y_imb)
model_samp = SVC(kernel="linear", C=1e4, random_state=0).fit(X_samp, y_samp)
print("NeighbourhoodCleaningRule")
print(classification_report(y_samp, model_samp.predict(X_samp)))
print()

X_samp, y_samp = SMOTEENN(random_state=0).fit_sample(X_imb, y_imb)
model_samp = SVC(kernel="linear", C=1e4, random_state=0).fit(X_samp, y_samp)
print("SMOTEENN")
print(classification_report(y_samp, model_samp.predict(X_samp)))
print()

X_samp, y_samp = SMOTETomek(random_state=4).fit_sample(X_imb, y_imb)
model_samp = SVC(kernel="linear", C=1e4, random_state=0).fit(X_samp, y_samp)
print("SMOTETomek")
print(classification_report(y_samp, model_samp.predict(X_samp)))
print()
