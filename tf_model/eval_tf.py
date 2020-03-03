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

def eval_gan(val_set ='../../extract_raw_img',checkpoint="checkpoint",output = "checkpoint"):
    import tf_model.gan_fingerprint.config as config
    import tf_model.gan_fingerprint.tfutil as tfutil
    from tf_model.gan_fingerprint import misc


    assert checkpoint != ' ' and val_set != ' ' and output != ' '
    misc.init_output_logging()
    print('Initializing TensorFlow...')
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)
    app = config.EasyDict(func='util_scripts.classify', model_path=checkpoint,
                          testing_data_path=val_set, out_fingerprint_dir=output)

    tfutil.call_func_by_name(**app)