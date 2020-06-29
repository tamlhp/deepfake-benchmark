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

def eval_visual(data,model_file):
    with open(data, 'rb') as f:
        info_dict = pickle.load(f)
    features_= info_dict['features']
    labels_ = info_dict['labels']

    with open(model_file, 'rb') as f:
        svclassifier_r = pickle.load(f)
    SVM_score = svclassifier_r.score(features_, labels_)
    print("SVM: " + str(SVM_score))
if __name__ == "__name__":
    args_in = parse_args()
    eval_visual(args_in.data,args_in.model)