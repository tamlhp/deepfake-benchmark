# from collections import Counter
import numpy as np
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
import keras
import os

def get_generate(val_set,image_size,batch_size):
    dataGenerator = ImageDataGenerator(rescale=1./255)


    generator_val = dataGenerator.flow_from_directory(
        val_set,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')

    return generator_val

def eval_cnn(model,loss,val_set ='../../extract_raw_img',image_size=256,batch_size=16):

    #### Load data
    generator_val = get_generate(val_set,image_size,batch_size)
    ### Compile model
    model.compile(optimizer="adam", loss=loss, metrics=['accuracy'])
    result = model.evaluate_generator(generator_val, len(generator_val),verbose=1)
    print(result)

def eval_gan(val_set ='../../extract_raw_img',checkpoint="checkpoint",total_val_img=1000,show_time=False):
    import tf_model.gan_fingerprint.config as config
    import tf_model.gan_fingerprint.tfutil as tfutil
    from tf_model.gan_fingerprint import misc


    assert checkpoint != ' ' and val_set != ' '
    if val_set[-1] == '/':
        val_set = val_set[:-1]
    idx = val_set.rfind('/')
    config.data_dir = val_set[:idx]

    config.validation_set = config.EasyDict(tfrecord_dir=val_set[idx + 1:], max_label_size='full')

    app = config.EasyDict(func='tf_model.gan_fingerprint.run.eval_classifier', model_path=checkpoint,
                          total_val_img=total_val_img,show_time=show_time)

    tfutil.call_func_by_name(**app)