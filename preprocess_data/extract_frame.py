# import os
import argparse
# import random
from os.path import isfile, join
# import json
from tqdm import tqdm
import cv2
import numpy as np
# import pickle
from facenet_pytorch import MTCNN
# from operator import itemgetter
import glob
import matplotlib.pyplot as plt
import torch

# from pytorch_model.train import *
# from tf_model.train import *

def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--in_dir', default='../../../data/download_df_2', help='path to train data ')
    parser.add_argument('--out_dir', default='../../../data/frame_download', help='path to test data ')
    parser.add_argument('--workers', type=int, default=4, help='number wokers for dataloader ')
    parser.add_argument('--duration',type=int, default = 4, help='Resume from checkpoint ')

    return parser.parse_args()

def extract_face(vi):
    output = args.out_dir

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

        plt.imsave(join(output, name_vi + "_" + str(id_frame) + ".jpg"), image, format='jpg')
        id_frame+=1
# import multiprocessing

from concurrent.futures import ThreadPoolExecutor

args = parse_args()
if __name__ == "__main__":
    paths = []
    types = ('/*.mp4', '/*.avi')
    # paths = glob.glob(args.inp+ "/*.mp4")
    for files in types:
        paths.extend(glob.glob(args.in_dir+ files))
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