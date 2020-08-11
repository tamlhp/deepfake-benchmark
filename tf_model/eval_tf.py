# from collections import Counter
import numpy as np
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
import torchtoolbox.transform as transforms
import cv2
import keras
import os
from PIL import ImageEnhance,Image
import tensorflow as tf
from albumentations.augmentations.functional import image_compression

def get_random_eraser(p=1.0, s_l=0.2, s_h=0.20001, r_1=1.0, r_2=1.000001, v=0, pixel_level=False):
    def eraser(input_img):
        img_h, img_w,_ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break


        input_img[top:top + h, left:left + w,:] = v

        return input_img
    return eraser
def image_contrast_adjusment(img):
    # print(img.shape)   # 256,256,3
    # print(type(img))   # <class 'numpy.ndarray'>
    # print(img)
    # # [[[128. 118. 119.]
    # #   [127. 117. 118.]
    # #  [127. 117. 118.]
    # print(np.max(img))   # 255.0
    #
    # print(img.astype(int).astype("int16"))
    # # [[[128 118 119]
    # #   [127 117 118]
    # #  [127 117 118]

    # img = img.astype("uint8")
    # print(img)
    # print(np.min(img))  # 0
    # print(np.max(img))  # 255
    # contrast = ImageEnhance.Contrast(Image.fromarray(img))
    # img = contrast.enhance(0.0)
    # img = transforms.Compose([transforms.RandomGaussianNoise(p=0.0)
    #                     ])(img)


    # print(img)  # <PIL.Image.Image image mode=RGB size=256x256 at 0x1DB637F0438>
    # print(np.min(img))  #0
    # print(np.max(img))  # 255

    # img = np.array(img,dtype='float64')
    # img = get_random_eraser()(img)

    #img  = transforms.RandomGaussianNoise(p=0.0,mean=0,std=76)(img)
    #img = cv2.resize(img,(64,64))
    #img = cv2.resize(img,(256,256))

    # img = image_compression(img,70,image_type=".jpg")
    return img
def get_generate(val_set,image_size,batch_size,adj_brightness=1.0,adj_contrast=1.0):
    dataGenerator = ImageDataGenerator(rescale=1./255,
                        brightness_range = (adj_brightness - 1e-6, adj_brightness + 1e-6),
                       preprocessing_function = image_contrast_adjusment
    )


    generator_val = dataGenerator.flow_from_directory(
        val_set,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')

    return generator_val

class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(model.predict(X_val))

        y_val = np.argmax(y_val, axis=1)
        y_predict = np.argmax(y_predict, axis=1)

        self._data.append({
            'val_recall': recall_score(y_val, y_predict),
            'val_precision': precision_score(y_val, y_predict),
        })
        print(self._data)
        return

    def get_data(self):
        return self._data

def eval_cnn(model,loss,val_set ='../../extract_raw_img',image_size=256,batch_size=16,adj_brightness=1.0,adj_contrast=1.0):

    #### Load data
    generator_val = get_generate(val_set,image_size,batch_size,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
    ### Compile model
    metrics = Metrics()

    model.compile(optimizer="adam", loss=loss, metrics=['accuracy',keras.metrics.Recall(),keras.metrics.Precision()])
    #model.compile(optimizer="adam", loss=loss, metrics=['accuracy'])
    result = model.evaluate_generator(generator_val, len(generator_val),verbose=1, callbacks=[metrics])
    print(result)

def eval_gan(val_set ='../../extract_raw_img',checkpoint="checkpoint",total_val_img=1000,show_time=False, adj_brightness=1.0, adj_contrast=1.0):
    import tf_model.gan_fingerprint.config as config
    import tf_model.gan_fingerprint.tfutil as tfutil
    from tf_model.gan_fingerprint import misc


    assert checkpoint != ' ' and val_set != ' '
    tfutil.init_tf(config.tf_config)

    if val_set[-1] == '/':
        val_set = val_set[:-1]
    idx = val_set.rfind('/')
    config.data_dir = val_set[:idx]

    config.validation_set = config.EasyDict(tfrecord_dir=val_set[idx + 1:], max_label_size='full')

    app = config.EasyDict(func='tf_model.gan_fingerprint.run.eval_classifier', model_path=checkpoint,
                          total_val_img=total_val_img,show_time=show_time)

    tfutil.call_func_by_name(**app)

if __name__ == "__main__":
    generator_val = get_generate("../../data/extract_raw_img", 256, 1)
    print(generator_val.__next__())
