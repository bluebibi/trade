import numpy as np
from sklearn.datasets import make_classification
from torch import nn
import torch.nn.functional as F

from skorch import NeuralNetClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

X, y = make_classification(1000, 20, n_informative=10, random_state=0)
print(type(X), type(y))

X = X.astype(np.float32)
y = y.astype(np.int64)

class MyModule(nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu):
        super(MyModule, self).__init__()

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X), dim=-1)
        return X


if __name__ == "__main__":
    net = NeuralNetClassifier(
        MyModule,
        max_epochs=100,
        lr=0.1,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )

    params = {
        'lr': [0.01, 0.02],
        'max_epochs': [10, 20],
        'module__num_units': [10, 20],
    }

    gs = GridSearchCV(net, params, refit=True, cv=3, scoring='accuracy')

    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('gs', gs),
    ])

    pipe.fit(X, y)

    print(gs.best_params_)

    y_proba = pipe.predict_proba(X)

    # for prob in y_proba:
    #     print("{0:6.4f} {1:6.4f}".format(prob[0], prob[1]))
