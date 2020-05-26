import os
import argparse
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib


CLF_NAMES = ["mlp", "logreg"]
CLFS = [
    MLPClassifier(alpha=0.1, hidden_layer_sizes=(64, 64, 64), learning_rate_init=0.001, max_iter=300),
    LogisticRegression(),
]

import pickle
info_dict = None
with open("features", 'rb') as f:
    info_dict = pickle.load(f)

features= info_dict['features']
labels = info_dict['labels']
# print(features)
features = np.array(features)
print(features.shape)
for name, clf in zip(CLF_NAMES, CLFS):
    clf.fit(features, labels)
    print(name)
    print(clf.predict(features))
    # joblib.dump(clf, os.path.join(output_path, name + '.pkl'))