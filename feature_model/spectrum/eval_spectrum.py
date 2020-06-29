import argparse
from sklearn.svm import SVC
import pickle
# load feature file

def eval_spectrum(data,model_file):
    pkl_file = open(data, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    X = data["data"]
    y = data["label"]
    with open(model_file, 'rb') as f:
        svclassifier_r = pickle.load(f)
    SVM_score = svclassifier_r.score(X, y)
    print("accuracy: " + str(SVM_score))

def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data', dest='input',default='',
                        help='Path to pkl data file.')
    parser.add_argument('-o', '--model', dest='output', help='path to save model.',
                        default='./output')
    args = parser.parse_args()
    return args

if __name__ == "__name__":
    args_in = parse_args()
    eval_spectrum(args_in.data,args_in.model)