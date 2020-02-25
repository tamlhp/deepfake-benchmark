import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import random
from os import listdir
from os.path import isfile, join

import numpy as np

import json
import matplotlib.pyplot as plt
import cv2
margin = 0.2
from tqdm import tqdm
# from mtcnn import MTCNN
import pickle

path_root = "/data/tam/kaggle/video"
path0 = "/data/tam/kaggle/video/dfdc_train_part_0"
path1 = "/data/tam/kaggle/video/dfdc_train_part_1"
path2 = "/data/tam/kaggle/video/dfdc_train_part_2"
path3 = "/data/tam/kaggle/video/dfdc_train_part_3"
path4 = "/data/tam/kaggle/video/dfdc_train_part_4"
path5 = "/data/tam/kaggle/video/dfdc_train_part_5"
path6 = "/data/tam/kaggle/video/dfdc_train_part_6"
path7 = "/data/tam/kaggle/video/dfdc_train_part_7"
path8 = "/data/tam/kaggle/video/dfdc_train_part_8"
path9 = "/data/tam/kaggle/video/dfdc_train_part_9"
path10 = "/data/tam/kaggle/video/dfdc_train_part_10"
path11 = "/data/tam/kaggle/video/dfdc_train_part_11"
path12 = "/data/tam/kaggle/video/dfdc_train_part_12"
path13 = "/data/tam/kaggle/video/dfdc_train_part_13"
path14 = "/data/tam/kaggle/video/dfdc_train_part_14"
path15 = "/data/tam/kaggle/video/dfdc_train_part_15"

path16 = "/data/tam/kaggle/video/dfdc_train_part_16"
path17 = "/data/tam/kaggle/video/dfdc_train_part_17"
path18 = "/data/tam/kaggle/video/dfdc_train_part_18"
path19 = "/data/tam/kaggle/video/dfdc_train_part_19"
path20 = "/data/tam/kaggle/video/dfdc_train_part_20"

path21 = "/data/tam/kaggle/video/dfdc_train_part_21"
path22 = "/data/tam/kaggle/video/dfdc_train_part_22"
path23 = "/data/tam/kaggle/video/dfdc_train_part_23"
path24 = path_root + "/dfdc_train_part_24"
path25 = path_root +"/dfdc_train_part_25"


path26 = path_root +"/dfdc_train_part_26"
path27 = path_root +"/dfdc_train_part_27"
path28 = path_root +"/dfdc_train_part_28"
path29 = path_root +"/dfdc_train_part_29"
path30 = path_root +"/dfdc_train_part_30"

path31 = path_root +"/dfdc_train_part_31"
path32 = path_root +"/dfdc_train_part_32"
path33 = path_root +"/dfdc_train_part_33"
path34 = path_root +"/dfdc_train_part_34"
path35 = path_root +"/dfdc_train_part_35"

path36 = path_root +"/dfdc_train_part_36"
path37 = path_root +"/dfdc_train_part_37"
path38 = path_root +"/dfdc_train_part_38"
path39 = path_root +"/dfdc_train_part_39"
path40 = path_root +"/dfdc_train_part_40"

path41 = path_root +"/dfdc_train_part_41"
path42 = path_root +"/dfdc_train_part_42"
path43 = path_root +"/dfdc_train_part_43"
path44 = path_root +"/dfdc_train_part_44"
path45 = path_root +"/dfdc_train_part_45"
path46 = path_root +"/dfdc_train_part_46"
path47 = path_root +"/dfdc_train_part_47"

path48 = path_root + "/dfdc_train_part_48"
path49 = path_root +  "/dfdc_train_part_49"

# paths = [path0,path1,path2,path3,path4,path5,path6,path7,path8,path9,path10,path11,path12,path13,path14,path15,path16,path17,path18,path19,\        path20,path21,path22,path23,path24,path25,path26,path27,path28,path29,path30,path31,path32,path33,path34,path35,path36,path37,path38\
#          ,path39,path40,path41,path42,path43,path44,path45,path46,path47,path48,path49]
# paths = [path0,path1,path2,path3,path4,path5,path6,path7,path8,path9,path10,path11,path12,path13,\
#          path20,path21,path22,path23,path24,path25,path26,path27,path28,path29,path30,path31,path32,path36\
#          ,path40,path44,path48]

# paths = [path14,path15,path16,path17]
# paths = [path18,path19,path33,path34]
# paths = [path14,path15,path16,path17,path18,path19,path33,path34,path35,path37,path38,path39,path41,path42,\
#          path43,path45,path46,path47,path49]

# paths = [path7,path21,path23,path25,path26,path27,path29,path30,path31,path44,path48]
paths = [path7,path21,path23,path25,path26,path27]
# paths = [path29,path30,path31,path44,path48]

save_interval = 7

def extract_face(path):
    print(path)
    import keras.backend as K
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.05
    # config.gpu_options.visible_device_list = "0" #only the gpu 0 is allowed
    set_session(tf.Session(config=config))
    from mtcnn import MTCNN
    from operator import itemgetter

    detector = MTCNN()

    data = json.load(open(join(path, "metadata.json")))
#     print(data)
    for vi in tqdm(data):
#         print(vi)
        if data[vi]['label'] == "FAKE":
            if os.path.exists(join("/data/tam/kaggle/raw_img/df", vi +".pkl")):
                continue
        if data[vi]['label'] == 'REAL':
            if os.path.exists(join("/data/tam/kaggle/raw_img/real", vi+".pkl")):
                continue
        video = cv2.VideoCapture(join(path, vi))
#         print(video)
        data_videos = []
        success, image = video.read()
#         print(success)
        while success:
#             print("while aa")
            try:
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            except:
                success, image = video.read()
        #         face_positions = face_recognition.face_locations(img)
#             print("chuyen di color")
#             print(image)
            face_positions = detector.detect_faces(image)

#             print("face_positions  " ,face_positions)
            if len(face_positions) == 0:
                success, image = video.read()
                continue
            face_position = max(face_positions, key=itemgetter('confidence'))
#             face_position =face_positions[0]['box']
            x,y,w,h = face_position['box']
            offsetx = round(margin * (w))
            offsety = round(margin * (h))
            y0 = round(max(y - offsety, 0))
            x1 = round(min(x + w + offsetx, image.shape[1]))
            y1 = round(min(y+ h + offsety, image.shape[0]))
            x0 = round(max(x - offsetx, 0))
    #         print(x0,x1,y0,y1)
            face = image[y0:y1,x0:x1]
#             print(face)

#             face = cv2.resize(face,(IMGWIDTH,IMGWIDTH))
    #         plt.imshow(face)
    #         plt.show()
            data_videos.append(face)
            success, image = video.read()
            for i in range(save_interval):
                success, image = video.read()
                if not success:
                    break
        data_videos = np.array(data_videos)
        if data[vi]['label'] == "FAKE":
            output = open(join("/data/tam/kaggle/raw_img/df", vi +".pkl"),'wb')
            pickle.dump(data_videos, output)
            output.close()
        if data[vi]['label'] == 'REAL':
            output = open(join("/data/tam/kaggle/raw_img/real", vi+".pkl"),'wb')
            pickle.dump(data_videos, output)
            output.close()
    return
import multiprocessing
pool = multiprocessing.Pool(6)
pool.map(extract_face,paths)