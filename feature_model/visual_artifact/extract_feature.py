import os
import argparse
from collections import defaultdict
import dlib
import cv2
from feature_model.visual_artifact.pipeline.eyecolor import extract_eyecolor_features
from feature_model.visual_artifact.process_data import load_facedetector
from feature_model.visual_artifact.pipeline.face_utils import *
from feature_model.visual_artifact.pipeline import pipeline_utils
from feature_model.visual_artifact.pipeline.texture import extract_features_eyes,extract_features_faceborder,extract_features_mouth,extract_features_nose
import glob


# img = cv2.imread("../prnu/camera.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
face_detector, sp68 = load_facedetector()
# face_crops, final_landmarks = get_crops_landmarks(face_detector, sp68, img)

# feature_vector, distance_HSV, valid_seg = extract_eyecolor_features(landmarks_resize, face_crop_resize)

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
    return np.concatenate([feature_eyecolor,features_eyes,features_mounth,features_nose,features_face],axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--train_set', default="data/train/", help='path to train data ')
    parser.add_argument('--val_set', default="data/test/", help='path to test data ')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--image_size', type=int, default=256,
                        help='the height / width of the input image to network')
    parser.add_argument('--workers', type=int, default=4, help='number wokers for dataloader ')
    parser.add_argument('--checkpoint', default=None, required=True, help='path to checkpoint ')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id ')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint ')
    parser.add_argument('--print_every', type=int, default=5000, help='Print evaluate info every step train')
    parser.add_argument('--loss', type=str, default="bce", help='Loss function use')

    args = parser.parse_args()

    for i in range(len(glob.glob())):
        pass