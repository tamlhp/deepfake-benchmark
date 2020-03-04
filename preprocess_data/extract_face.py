# import os
import argparse
# import random
from os.path import isfile, join
# import json
from tqdm import tqdm
import cv2
# import numpy as np
# import pickle
from facenet_pytorch import MTCNN
# from operator import itemgetter
import glob
import matplotlib.pyplot as plt
import torch
torch.multiprocessing.set_start_method('spawn')

# from pytorch_model.train import *
# from tf_model.train import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--inp', default="data/train/", help='path to train data ')
    parser.add_argument('--output', default="data/test/", help='path to test data ')
    parser.add_argument('--workers', type=int, default=4, help='number wokers for dataloader ')
    parser.add_argument('--duration',type=int, default = 4, help='Resume from checkpoint ')

    return parser.parse_args()

detector = MTCNN(device=device)


def extract_face(vi):
    output = args.output
    margin = 0.2
    duration = args.duration
    # data = glob.glob("/hdd/tam/FaceForensics/data/original_sequences/youtube/c23/videos/*.mp4")
    # print(vi)
    # for vi in tqdm(data):
    video = cv2.VideoCapture(join(vi))
    name_vi = vi.split("\\")[-1].split("/")[-1].split(".")[0]
    # video = cv2.VideoCapture(join("/hdd/tam/FaceForensics/data/original_sequences/youtube/c23/videos/", vi))
    # print(vi)
    success= True
    image = None
    id_frame = 0
    while success:
        for i in range(duration):
            success, image = video.read()
            if not success:
                break
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            # print(image)
            continue
        #         face_positions = face_recognition.face_locations(img)
        face_positions = detector.detect(image)
        try:
            face_position = face_positions[0][0]
        except:
            # print(face_positions)
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
        plt.imsave(join(output, name_vi + "_" + str(id_frame) + ".jpg"), face, format='jpg')
        id_frame+=1
# import multiprocessing

from concurrent.futures import ThreadPoolExecutor

args = parse_args()
if __name__ == "__main__":

    paths = glob.glob(args.inp+ "/*.mp4")
    # print(paths[0])
    # extract_face(paths[0])
    print(len(paths))
    # pool = multiprocessing.Pool(args.workers)
    # pool.map(extract_face, tqdm(paths))
    # def process_file(i):
    #     extract_face
    #     return y_pred


    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        predictions = ex.map(extract_face, tqdm(paths))