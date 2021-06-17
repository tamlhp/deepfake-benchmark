import argparse
from sklearn.svm import SVC
import pickle
# load feature file

def train_spectrum(data,model_file):
    pkl_file = open(data, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    X = data["data"]
    y = data["label"]
    svclassifier_r = SVC(C=6.37, kernel='rbf', gamma=0.86)
    svclassifier_r.fit(X, y)
    with open(model_file, 'wb') as f:
        pickle.dump(svclassifier_r, f)


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
    train_spectrum(args_in.data,args_in.model)