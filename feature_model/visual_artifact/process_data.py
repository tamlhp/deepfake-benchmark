import os
import argparse
from collections import defaultdict
import dlib
import cv2
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from feature_model.visual_artifact.pipeline.eyecolor import extract_eyecolor_features
from feature_model.visual_artifact.pipeline.face_utils import *
from feature_model.visual_artifact.pipeline import pipeline_utils
from feature_model.visual_artifact.pipeline.texture import extract_features_eyes,extract_features_faceborder,extract_features_mouth,extract_features_nose
import glob
import random
from PIL import ImageEnhance,Image
import pickle
def load_facedetector():
    """Loads dlib face and landmark detector."""
    # download if missing http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    if not os.path.isfile('shape_predictor_68_face_landmarks.dat'):
        print ('Could not find shape_predictor_68_face_landmarks.dat.')
        exit(-1)
    face_detector = dlib.get_frontal_face_detector()
    sp68 = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    return face_detector, sp68

face_detector, sp68 = load_facedetector()
import dlib.cuda as cuda
print(cuda.get_num_devices())
print(dlib.DLIB_USE_CUDA)




final_score_clf = 0.0
final_score_HSV = 0.0
final_feature_vector = None
final_valid_seg = False
scale = 768
def extract_visual_artifact(img):
    face_crop_list, landmarks_list = get_crops_landmarks(face_detector, sp68, img)
    scale = 768

    if (len(face_crop_list) == 1):

        face_crop = face_crop_list[0]
        landmarks = landmarks_list[0].copy()

        out_size = pipeline_utils.new_size(face_crop.shape[1], face_crop.shape[0], large_dim=scale)
        scale_x = float(out_size[1]) / face_crop.shape[1]
        scale_y = float(out_size[0]) / face_crop.shape[0]

        landmarks_resize = landmarks.copy()
        landmarks_resize[:, 0] = landmarks_resize[:, 0] * scale_x
        landmarks_resize[:, 1] = landmarks_resize[:, 1] * scale_y

        face_crop_resize = cv2.resize(face_crop, (int(out_size[1]), int(out_size[0])), interpolation=cv2.INTER_LINEAR)

        feature_eyecolor, distance_HSV, valid_seg = extract_eyecolor_features(landmarks_resize, face_crop_resize)
        features_eyes = extract_features_eyes(landmarks, face_crop, scale=scale)
        features_mounth = extract_features_mouth(landmarks, face_crop, scale=scale)
        features_nose = extract_features_nose(landmarks, face_crop, scale=scale)
        features_face = extract_features_faceborder(landmarks, face_crop, scale=scale)
        feature = np.concatenate([feature_eyecolor, features_eyes, features_mounth, features_nose, features_face], axis=0)
    else:
        feature = np.array([0])
#     print(feature)
    return feature


def main(input_real,input_fake, output_path,number_iter):

    features = []
    labels = []
    cont = 0
    video_dir_dict = {}
    video_dir_dict['real'] = input_real
    video_dir_dict['fake'] = input_fake
    for tag in video_dir_dict:
        if tag == 'real':
            label = 0
        else:
            label = 1
        input_path = video_dir_dict[tag]
        list_df_path = glob.glob(input_path + "/*.png")
        random.shuffle(list_df_path)
        for vid_path in list_df_path:
            img = cv2.imread(vid_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = img.astype("uint8")
            contrast = ImageEnhance.Contrast(Image.fromarray(img))
            img = contrast.enhance(2.0)
            brightness = ImageEnhance.Brightness(img)
            img = brightness.enhance(1.0)
            img = np.array(img, dtype='uint8')

            feature = extract_visual_artifact(img)
            if len(feature) < 3:
                print("aaaa")
                continue
            features.append(feature)
            labels.append([label])
            cont += 1
            if cont == number_iter:
                break
    info_dict = {'features':features,'labels':labels}
    # print(info_dict)
    with open(output_path, 'wb') as f:
        pickle.dump(info_dict, f)

def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-ir', '--input_real', dest='input_real',default='',
                        help='Path to input image or folder containting multiple images.')
    parser.add_argument('-if', '--input_fake', dest='input_fake',default='',
                        help='Path to input image or folder containting multiple images.')
    parser.add_argument('-o', '--output', dest='output', help='Path to save outputs.',
                        default='./output')
    parser.add_argument('-n', '--number_iter', default=100,help='number image process')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args_in = parse_args()
    main(args_in.input_real,args_in.input_fake, args_in.output,args_in.number_iter)

