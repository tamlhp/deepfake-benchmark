import os
import argparse
from collections import defaultdict
import dlib
import cv2
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from pipeline.eyecolor import extract_eyecolor_features
from process_data import load_facedetector
from pipeline.face_utils import *
from pipeline import pipeline_utils
from pipeline.texture import extract_features_eyes,extract_features_faceborder,extract_features_mouth,extract_features_nose
import glob


# img = cv2.imread("aa.jpeg")
# img = cv2.imread("190.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
face_detector, sp68 = load_facedetector()
# face_crop_list, landmarks_list = get_crops_landmarks(face_detector, sp68, img)
# print(landmarks_list)
import dlib.cuda as cuda
print(cuda.get_num_devices())
print(dlib.DLIB_USE_CUDA)
# cv2.imshow("aaa",face_crop_list[0])
# cv2.show()
# cv2.waitKey()

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
        print(feature_eyecolor)
        print(features_eyes)
        print(features_mounth)
        print(features_nose)
        print(features_face)
    else:
        feature = np.array([0])
    return feature


df_path = '../../extract_raw_img_test/df'
real_path = '../../extract_raw_img_test/real'

df_list = os.listdir(df_path)
features = []
labels = []
for vid_path in glob.glob(df_path+"/*.jpg"):
    # vid_path = os.path.join(df_path, vid_name)
    print(vid_path)

    try:
        img = cv2.imread(vid_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        feature = extract_visual_artifact(img)
        # print(feature)
        if len(feature) < 3:
            print("aaaa")
            continue
        # print(feature)
        features.append(feature)
        labels.append([1])
    except:
        continue

for vid_path in glob.glob(real_path+"/*.jpg"):
    # vid_path = os.path.join(real_path, vid_name)
    print(vid_path)

    try:
        img = cv2.imread(vid_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        feature = extract_visual_artifact(img)
        if len(feature) < 3:
            continue
        features.append(feature)
        labels.append([0])
    except:
        continue

import pickle
info_dict = {'features':features,'labels':labels}
with open("features", 'wb') as f:
    pickle.dump(info_dict, f)

# print("feature_vector  ",feature_eyecolor)
# print("feature_vector 0 ",features_eyes)
# print("feature_vector 1 ",features_mounth)
# print("feature_vector 2 ",features_nose)
# print("feature_vector 3 ",features_face)
# # print("final_score_clf, final_score_HSV, final_feature_vector, final_valid_seg" , final_score_clf, final_score_HSV, final_feature_vector, final_valid_seg)
# # feature_vector, distance_HSV, valid_seg = extract_eyecolor_features(landmarks_resize, face_crop_resize)
# feature = np.concatenate([feature_eyecolor,features_eyes,features_mounth,features_nose,features_face],axis=0)
# print(len(feature))