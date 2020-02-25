import os
import argparse
import random
from os.path import isfile, join
import json
from tqdm import tqdm
import cv2
import numpy as np
import pickle
from facenet_pytorch import MTCNN
from operator import itemgetter
import glob
import matplotlib.pyplot as plt

# from pytorch_model.train import *
# from tf_model.train import *
def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--datapath', default="data/train/", help='path to train data ')
    parser.add_argument('--outputpath', default="data/test/", help='path to test data ')
    parser.add_argument('--workers', type=int, default=1, help='number wokers for dataloader ')
    parser.add_argument('--resume',type=int, default = 0, help='Resume from checkpoint ')

    return parser.parse_args()

detector = MTCNN()


def extract_face(data,margin=0.2,save_interval=7,output="output/"):

    # data = glob.glob("/hdd/tam/FaceForensics/data/original_sequences/youtube/c23/videos/*.mp4")

    for vi in tqdm(data):
        vi = vi.split("/")[-1].split(".")[0]
        video = cv2.VideoCapture(join("/hdd/tam/FaceForensics/data/original_sequences/youtube/c23/videos/", vi))

        success= True
        image = None
        i = 0
        while success:
            for i in range(save_interval):
                success, image = video.read()
                if not success:
                    break
            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                continue
            #         face_positions = face_recognition.face_locations(img)
            face_positions = detector.detect_faces(image)
            try:
                face_position = face_positions[0][0]
            except:
                continue
            x, y, x2, y2 = face_position
            x, y, w, h = int(x), int(y), int(x2 - x), int(y2 - y)
            offsetx = round(margin * (w))
            offsety = round(margin * (h))
            y0 = round(max(y - offsety, 0))
            x1 = round(min(x + w + offsetx, image.shape[1]))
            y1 = round(min(y + h + offsety, image.shape[0]))
            x0 = round(max(x - offsetx, 0))
            #         print(x0,x1,y0,y1)
            face = image[y0:y1, x0:x1]
            plt.imsave(join(output, vi + "_" + str(i) + ".jpg"), face, format='jpg')
            i+=1

if __name__ == "__main__":
    args = parse_args()
