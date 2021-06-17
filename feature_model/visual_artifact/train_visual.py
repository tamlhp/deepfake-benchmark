import os
import argparse
import pandas as pd
from sklearn.svm import SVC

import pickle

def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data', dest='input',default='',
                        help='Path to pkl data file.')
    parser.add_argument('-o', '--model', dest='output', help='path to save model.',
                        default='./output')
    args = parser.parse_args()
    return args

def train_visual(data,model_file):
    svclassifier_r = SVC()
    with open(data, 'rb') as f:
        info_dict = pickle.load(f)
    features= info_dict['features']
    labels = info_dict['labels']
    svclassifier_r.fit(features, labels)

    with open(model_file, 'wb') as f:
        pickle.dump(svclassifier_r, f)
if __name__ == "__name__":
    args_in = parse_args()
    train_visual(args_in.data,args_in.model)