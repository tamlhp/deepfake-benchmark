import os
from utils.face_proc import FaceProc
import argparse
import pickle
from forensic_test import exam_img, exam_video


def main(args):
    all_paths = os.listdir(args.input_dir)
    proba_list = []

    # initiate face process class, used to detect face and extract landmarks
    face_inst = FaceProc()

    # initialize SVM classifier for face forensics
    with open(args.classifier_path, 'rb') as f:
        model = pickle.load(f)
    classifier = model[0]
    scaler = model[1]

    for f_name in all_paths:
        f_path = os.path.join(args.input_dir, f_name)
        print('_'*20)
        print('Testing: ' + f_name)
        suffix = f_path.split('.')[-1]
        if suffix.lower() in ['jpg', 'png', 'jpeg', 'bmp']:
            proba, optout = exam_img(args, f_path, face_inst, classifier, scaler)
        elif suffix.lower() in ['mp4', 'avi', 'mov', 'mts']:
            proba, optout = exam_video(args, f_path, face_inst, classifier, scaler)
        print('fake_proba: {},   optout: {}'.format(str(proba), optout))
        tmp_dict = dict()
        tmp_dict['file_name'] = f_name
        tmp_dict['probability'] = proba
        tmp_dict['optout'] = optout
        proba_list.append(tmp_dict)
    pickle.dump(proba_list, open(args.save_file, 'wb'))
    print(proba_list)


if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="headpose forensics")
   parser.add_argument('--input_dir', type=str, default='debug_data')
   parser.add_argument('--markID_c', type=str, default='18-36,49,55', help='landmark ids to estimate CENTRAL face region')
   parser.add_argument('--markID_a', type=str, default='1-36,49,55', help='landmark ids to estimate WHOLE face region')
   parser.add_argument('--classifier_path', type=str, default='svm_model.p')
   parser.add_argument('--save_file', type=str, default='proba_list.p')
   args = parser.parse_args()
   main(args)