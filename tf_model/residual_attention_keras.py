import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow.keras.backend as K
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
# config.gpu_options.visible_device_list = "0" #only the gpu 0 is allowed
set_session(tf.Session(config=config))

from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import Add
from keras.layers import Multiply
from keras.layers import Lambda

from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import Model
from keras.regularizers import l2



import keras
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


def residual_block(input, input_channels=None, output_channels=None, kernel_size=(3, 3), stride=1):
    """
    full pre-activation residual block
    https://arxiv.org/pdf/1603.05027.pdf
    """
    if output_channels is None:
        output_channels = input.get_shape()[-1].value
    if input_channels is None:
        input_channels = output_channels // 4

    strides = (stride, stride)

    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, (1, 1))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, kernel_size, padding='same', strides=stride)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(output_channels, (1, 1), padding='same')(x)

    if input_channels != output_channels or stride != 1:
        input = Conv2D(output_channels, (1, 1), padding='same', strides=strides)(input)

    x = Add()([x, input])
    return x


def attention_block(input, input_channels=None, output_channels=None, encoder_depth=1):
    """
    attention block
    https://arxiv.org/abs/1704.06904
    """

    p = 1
    t = 2
    r = 1

    if input_channels is None:
        input_channels = input.get_shape()[-1].value
    if output_channels is None:
        output_channels = input_channels

    # First Residual Block
    for i in range(p):
        input = residual_block(input)

    # Trunc Branch
    output_trunk = input
    for i in range(t):
        output_trunk = residual_block(output_trunk)

    # Soft Mask Branch

    ## encoder
    ### first down sampling
    output_soft_mask = MaxPool2D(padding='same')(input)  # 32x32
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)

    skip_connections = []
    for i in range(encoder_depth - 1):

        ## skip connections
        output_skip_connection = residual_block(output_soft_mask)
        skip_connections.append(output_skip_connection)
        # print ('skip shape:', output_skip_connection.get_shape())

        ## down sampling
        output_soft_mask = MaxPool2D(padding='same')(output_soft_mask)
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)

            ## decoder
    skip_connections = list(reversed(skip_connections))
    for i in range(encoder_depth - 1):
        ## upsampling
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)
        output_soft_mask = UpSampling2D()(output_soft_mask)
        ## skip connections
        output_soft_mask = Add()([output_soft_mask, skip_connections[i]])

    ### last upsampling
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)
    output_soft_mask = UpSampling2D()(output_soft_mask)

    ## Output
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)

    # Attention: (1 + output_soft_mask) * output_trunk
    output = Lambda(lambda x: x + 1)(output_soft_mask)
    output = Multiply()([output, output_trunk])  #

    # Last Residual Block
    for i in range(p):
        output = residual_block(output)

    return output

def AttentionResNet92(shape=(256, 256, 3), n_channels=64, n_classes=1,
                      dropout=0, regularization=0.01):
    """
    Attention-92 ResNet
    https://arxiv.org/abs/1704.06904
    """
    regularizer = l2(regularization)

    input_ = Input(shape=shape)
    x = Conv2D(n_channels, (7, 7), strides=(2, 2), padding='same')(input_) # 112x112
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # 56x56

    x = residual_block(x, output_channels=n_channels * 4)  # 56x56
    x = attention_block(x, encoder_depth=3)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 8, stride=2)  # 28x28
    x = attention_block(x, encoder_depth=2)  # bottleneck 7x7
    x = attention_block(x, encoder_depth=2)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 16, stride=2)  # 14x14
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 32, stride=2)  # 7x7
    x = residual_block(x, output_channels=n_channels * 32)
    x = residual_block(x, output_channels=n_channels * 32)

    pool_size = (x.get_shape()[1].value, x.get_shape()[2].value)
    x = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
    x = Flatten()(x)
    if dropout:
        x = Dropout(dropout)(x)
    output = Dense(n_classes, kernel_regularizer=regularizer, activation='sigmoid')(x)

    model = Model(input_, output)
    return model

def AttentionResNet56(shape=(256, 256, 3), n_channels=64, n_classes=1,
                      dropout=0, regularization=0.01):
    """
    Attention-56 ResNet
    https://arxiv.org/abs/1704.06904
    """

    regularizer = l2(regularization)

    input_ = Input(shape=shape)
    x = Conv2D(n_channels, (7, 7), strides=(2, 2), padding='same')(input_) # 112x112
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # 56x56

    x = residual_block(x, output_channels=n_channels * 4)  # 56x56
    x = attention_block(x, encoder_depth=3)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 8, stride=2)  # 28x28
    x = attention_block(x, encoder_depth=2)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 16, stride=2)  # 14x14
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 32, stride=2)  # 7x7
    x = residual_block(x, output_channels=n_channels * 32)
    x = residual_block(x, output_channels=n_channels * 32)

    pool_size = (x.get_shape()[1].value, x.get_shape()[2].value)
    x = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
    x = Flatten()(x)
    if dropout:
        x = Dropout(dropout)(x)
    output = Dense(n_classes, kernel_regularizer=regularizer, activation='sigmoid')(x)

    model = Model(input_, output)
    return model


def AttentionResNetCifar10(shape=(256, 256, 3), n_channels=32, n_classes=1):
    """
    Attention-56 ResNet for Cifar10 Dataset
    https://arxiv.org/abs/1704.06904
    """
    input_ = Input(shape=shape)
    # input_ = Input(tensor=image)
    x = Conv2D(n_channels, (5, 5), padding='same')(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)  # 16x16

    x = residual_block(x, input_channels=32, output_channels=128)
    x = attention_block(x, encoder_depth=2)

    x = residual_block(x, input_channels=128, output_channels=256, stride=2)  # 8x8
    x = attention_block(x, encoder_depth=1)

    x = residual_block(x, input_channels=256, output_channels=512, stride=2)  # 4x4
    x = attention_block(x, encoder_depth=1)

    x = residual_block(x, input_channels=512, output_channels=1024)
    x = residual_block(x, input_channels=1024, output_channels=1024)
    x = residual_block(x, input_channels=1024, output_channels=1024)

    x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1))(x)  # 1x1
    x = Flatten()(x)
    output = Dense(n_classes, activation='sigmoid')(x)

    model = Model(input_, output)
    return model

from collections import Counter
from sklearn.utils import class_weight
import numpy as np

if __name__ == "__main__":
    IMAGE_SHAPE = 256
    model = AttentionResNet92(shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3),n_channels=32, n_classes=1)

    model.compile(optimizer = "adam", loss = 'binary_crossentropy',metrics = ['accuracy'])

    from keras.preprocessing.image import ImageDataGenerator
    batch_size = 32
    dataGenerator = ImageDataGenerator(rescale=1./255,rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,shear_range=0.05)
    generator = dataGenerator.flow_from_directory(
            '/data/tam/kaggle/extract_raw_img/',
            target_size=(IMAGE_SHAPE, IMAGE_SHAPE),
            batch_size=batch_size,
            class_mode='binary',
            subset='training',shuffle=True)
    test_generator = dataGenerator.flow_from_directory(
            '/data/tam/kaggle/extract_raw_img_test/',
            target_size=(IMAGE_SHAPE, IMAGE_SHAPE),
            batch_size=batch_size,
            class_mode='binary',
            subset='training',shuffle=True)

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./log_residual_attention_keras")
    lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=7, min_lr=10e-7, epsilon=0.01, verbose=1)
    checkpoints = keras.callbacks.ModelCheckpoint("./log_residual_attention_keras/checkpoint_newdata_{epoch:04d}.pth", monitor='val_loss', verbose=0, save_best_only=False, period=1)
    early_stopper = EarlyStopping(monitor='val_acc', min_delta=0, patience=15, verbose=1)
    callbacks= [tensorboard_callback,checkpoints,lr_reducer, early_stopper]


    counter = Counter(generator.classes)
    max_val = float(max(counter.values()))
    class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

    class_weights_2 = class_weight.compute_class_weight(
                   'balanced',
                    np.unique(generator.classes),
                    generator.classes)

    model.fit_generator(generator,validation_data=test_generator, steps_per_epoch=int(1142792/batch_size), epochs=30,workers=1,validation_steps=14298/batch_size,class_weight = class_weights,callbacks = callbacks)
