# from collections import Counter
import numpy as np
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
import keras

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
