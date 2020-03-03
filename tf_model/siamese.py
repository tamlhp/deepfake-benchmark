# import os
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# config.gpu_options.visible_device_list = "0" #only the gpu 0 is allowed
set_session(tf.Session(config=config))

# https://github.com/hlamba28/One-Shot-Learning-with-Siamese-Networks/blob/master/Siamese%20on%20Omniglot%20Dataset.ipynb
import numpy as np

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model

from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.regularizers import l2
from keras import backend as K

def initialize_weights(shape, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)


def initialize_bias(shape, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def get_siamese_model(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
                     kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu',
                     kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer=initialize_weights, bias_initializer=initialize_bias))

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    #     L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_layer = Lambda(lambda tensors: K.l2_normalize(tensors[0] - tensors[1]))

    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)

    # Connect the inputs with the outputs
    # siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    ####################
    model.add(Dense(1,activation='sigmoid', bias_initializer=initialize_bias))
    class_1 = model(left_input)
    class_2 = model(right_input)
    siamese_net = Model(inputs=[left_input, right_input], outputs=[prediction,class_1,class_2])
    # return the model
    return siamese_net


import random
import glob

from PIL import Image
import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, path, batch_size=32, image_size=256, shuffle=True):
        'Initialization'
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.df_path = glob.glob(path + "/df/*.jpg")
        self.real_path = glob.glob(path + "/real/*.jpg")
        self.indexes = range(min(len(self.df_path), len(self.real_path)))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(min(len(self.df_path), len(self.real_path)) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        rr = random.randint(0, 1)
        if rr == 0:
            list_IDs_temp = [self.df_path[k] for k in indexes]
        else:
            list_IDs_temp = [self.real_path[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp, rr)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.df_path)
            np.random.shuffle(self.real_path)

    def __data_generation(self, list_IDs_temp, rr):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X_l = np.empty((self.batch_size, self.image_size, self.image_size, 3))
        X_r = np.empty((self.batch_size, self.image_size, self.image_size, 3))
        y = np.empty((self.batch_size), dtype=int)
        y1 = np.empty((self.batch_size), dtype=int)
        y2 = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = Image.open(ID).resize((self.image_size, self.image_size))
            y1[i] = int(rr)
            X_l[i,] = np.array(img)
            rr2 = random.randint(0, 1)
            y2[i] = int(rr2)
            ID2 = ""
            if rr2 == 0:
                ID2 = random.choice(self.df_path)
            else:
                ID2 = random.choice(self.real_path)
            img2 = Image.open(ID2).resize((self.image_size, self.image_size))
            X_r[i,] = np.array(img2)
            # Store class
            y[i] = 1 if rr == rr2 else 0
        X = [X_l, X_r]
        y = [y,y1,y2]
        return X, y


generator_train = DataGenerator(path='/hdd/tam/kaggle/extract_raw_img', batch_size=16)
generator_val = DataGenerator(path='/hdd/tam/kaggle/extract_raw_img_test', batch_size=16)

model = get_siamese_model((256, 256, 3))
model.summary()
model.load_weights("siamese/checkpoint_0014.pth")

model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=['accuracy'])

tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./siamese")
checkpoints = keras.callbacks.ModelCheckpoint("./siamese/checkpoint_{epoch:04d}.pth", monitor='val_loss', verbose=0,
                                              save_best_only=False, period=1)

model.fit_generator(generator_train, validation_data=generator_val, epochs=50, workers=8,
                    callbacks=[tensorboard_callback, checkpoints], initial_epoch=14)