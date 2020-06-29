from sklearn import svm
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import pickle
import argparse
import numpy as np

def process_training_data(data):
    videos_real = []
    videos_fake = []
    video_list = []
    label_list = []

    R_vec_feat = []
    R_mat_feat = []
    R_mat_full_feat = []
    t_vec_feat = []

    for key, value in data.items():
        label = value['label']
        if label == 'real':
            label_id = 0
            videos_real.append(key)
        else:
            label_id = 1
            videos_fake.append(key)

        print(key)
        R_c_list = value['R_c_vec']
        R_c_matrix_list = value['R_c_mat']
        t_c_list = value['t_c']

        R_a_list = value['R_a_vec']
        R_a_matrix_list = value['R_a_mat']
        t_a_list = value['t_a']

        # Compute diff
        delta_R_vec_list = [R_c_list[i][:, -1] - R_a_list[i][:, -1] for i in range(len(R_c_list)) if R_c_list[i] is not None]
        delta_t_vec_list = [t_c_list[i][:, -1] - t_a_list[i][:, -1] for i in range(len(t_c_list)) if t_c_list[i] is not None]
        delta_R_mat_list = [R_c_matrix_list[i][:, -1] - R_a_matrix_list[i][:, -1] for i in range(len(R_c_matrix_list)) if R_c_matrix_list[i] is not None]
        delta_R_full_mat_list = [(R_c_matrix_list[i] - R_a_matrix_list[i]).flatten() for i in range(len(R_c_matrix_list)) if R_c_matrix_list[i] is not None]

        R_vec_feat += delta_R_vec_list
        R_mat_feat += delta_R_mat_list
        t_vec_feat += delta_t_vec_list
        R_mat_full_feat += delta_R_full_mat_list

        label_list += [label_id] * len(delta_R_mat_list)
        video_list += [key] * len(delta_R_mat_list)

    return sorted(set(videos_real)), sorted(set(videos_fake)), video_list, label_list, R_vec_feat, R_mat_feat, R_mat_full_feat, t_vec_feat


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data', dest='input',default='',
                        help='Path to pkl data file.')
    parser.add_argument('-o', '--model', dest='output', help='path to save model.',
                        default='./output')
    args = parser.parse_args()
    return args

def eval_headposes(data,model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    with open(data, 'rb') as f:
        data_ = pickle.load(f)

    videos_real, videos_fake, video_list, label_list, R_vec_feat, R_mat_feat, R_mat_full_feat, t_vec_feat \
        = process_training_data(data_)
    features = [np.concatenate([R_mat_full_feat[i], t_vec_feat[i]]) for i in range(len(R_mat_feat))]
    X_test, y_test = shuffle(features, label_list, random_state=0)
    model_acc = model[0].score(X_test, y_test)
    print(model_acc)
if __name__ == "__name__":
    args_in = parse_args()
    eval_headposes(args_in.data,args_in.model)