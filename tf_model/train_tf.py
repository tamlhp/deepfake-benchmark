# from collections import Counter
import numpy as np
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
import keras
import os
from keras.optimizers import Adam
from PIL import ImageEnhance,Image
import matplotlib.pyplot as plt
def image_contrast_adjusment(img):
    # print(img.shape)
    # print(type(img))
    # print(img)
    # print(np.max(img))
    # print(img.astype(int).astype("int16"))
    img = img.astype("uint8")
    # print(img)
    # print(np.min(img))
    # print(np.max(img))
    contrast = ImageEnhance.Contrast(Image.fromarray(img))
    img = contrast.enhance(1.0)
    img = np.array(img,dtype='float64')
    # print(img)
    return img
def get_generate(train_set,val_set,image_size,batch_size,adj_brightness=1.0, adj_contrast=1.0):
    dataGenerator = ImageDataGenerator(rescale=1. / 255, rotation_range=5,
                                       width_shift_range=0.05,
                                       height_shift_range=0.05,
                                       horizontal_flip=True, shear_range=0.05,
                                       brightness_range=[adj_brightness-1e-6, adj_brightness+1e-6],
                                       preprocessing_function = image_contrast_adjusment
                                       )
    generator_train = dataGenerator.flow_from_directory(
            train_set,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='binary',
            subset='training',shuffle=True)

    generator_val = dataGenerator.flow_from_directory(
        #         '/data/tam/kaggle/test_imgs/',
        val_set,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')

    return generator_train,generator_val

def train_cnn(model,loss,train_set = '../../../data/extract_raw_img',val_set ='../../../data/extract_raw_img',image_size=256,\
              batch_size=16,num_workers=1,checkpoint="checkpoint",resume="",epochs=20, \
              adj_brightness=1.0, adj_contrast=1.0):

    #### Load data
    generator_train, generator_val = get_generate(train_set,val_set,image_size,batch_size,adj_brightness=adj_brightness,\
                                                  adj_contrast=adj_contrast)
    if resume != "":
        model.load_weights(os.path.join(checkpoint, resume))
    # counter = Counter(generator_train.classes)
    # max_val = float(max(counter.values()))
    # class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}
    class_weights = class_weight.compute_class_weight(
        'balanced',
        np.unique(generator_train.classes),
        generator_train.classes)

    ### Compile model
    # model.compile(optimizer="adam", loss=BinaryFocalLoss(gamma=2), metrics=['accuracy'])
    model.compile(optimizer="adam", loss=loss, metrics=['accuracy'])

    tensorboard = keras.callbacks.TensorBoard(log_dir=checkpoint)
    checkpoints = keras.callbacks.ModelCheckpoint(checkpoint + "/checkpoint_{epoch:04d}.pth", monitor='val_loss', verbose=0, save_best_only=False, period=2)
    # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0)
    callbacks = [tensorboard, checkpoints]
    model.fit_generator(generator_train, validation_data=generator_val, steps_per_epoch=len(generator_train),
                        epochs=epochs, workers=num_workers, validation_steps=len(generator_val), class_weight=class_weights,
                        callbacks=callbacks)


def train_siamese(model,loss,train_set = '../../extract_raw_img',val_set ='../../extract_raw_img',image_size=256,\
                  batch_size=16,num_workers=1,checkpoint="checkpoint",resume="",epochs=20, adj_brightness=1.0, adj_contrast=1.0):
    from tf_model.siamese import DataGenerator
    generator_train = DataGenerator(path=train_set, batch_size=batch_size,image_size=image_size)
    generator_val = DataGenerator(path=val_set, batch_size=batch_size,image_size=image_size)
    print(len(generator_train))
    # model = get_siamese_model((256, 256, 3))
    # model.summary()
    # model.load_weights("siamese/checkpoint_0014.pth")
    if resume != "":
        model.load_weights(os.path.join(checkpoint, resume))
    model.compile(loss=loss, optimizer=Adam(lr=0.001), metrics=['accuracy'])

    tensorboard = keras.callbacks.TensorBoard(log_dir=checkpoint)
    checkpoints = keras.callbacks.ModelCheckpoint(checkpoint + "/checkpoint_{epoch:04d}.pth", monitor='val_loss', verbose=0, save_best_only=False, period=2)
    # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0)
    callbacks = [tensorboard, checkpoints]

    model.fit_generator(generator_train,validation_data=generator_val,steps_per_epoch=len(generator_train), validation_steps=len(generator_val),epochs=epochs, workers=num_workers,verbose=1,callbacks=callbacks)
    # model.fit_generator(generator_train, validation_data=generator_val, epochs=50, workers=1,)
                        # callbacks=[tensorboard_callback, checkpoints])
def train_gan(train_set = 'checkpoint/data/test',val_set ='checkpoint/data/test',training_seed=0,image_size=256,\
              batch_size=16,num_workers=1,checkpoint="checkpoint",resume="",epochs=20,total_train_img = 15000,\
              total_val_img = 1000, adj_brightness=1.0, adj_contrast=1.0):
    import tf_model.gan_fingerprint.config as config
    import tf_model.gan_fingerprint.tfutil as tfutil
    from tf_model.gan_fingerprint import misc


    assert train_set != ' ' and checkpoint != ' '
    if val_set == ' ':
        val_set = train_set
    misc.init_output_logging()
    np.random.seed(training_seed)
    print('Initializing TensorFlow...')
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)
    if train_set[-1] == '/':
        train_set = train_set[:-1]
    idx = train_set.rfind('/')
    config.data_dir = train_set[:idx]
    config.training_set = config.EasyDict(tfrecord_dir=train_set[idx + 1:], max_label_size='full')
    if val_set[-1] == '/':
        val_set = val_set[:-1]
    idx = val_set.rfind('/')
    config.validation_set = config.EasyDict(tfrecord_dir=val_set[idx + 1:], max_label_size='full')

    config.sched.minibatch_base = batch_size
    config.sched.lod_initial_resolution = image_size

    app = config.EasyDict(func='tf_model.gan_fingerprint.run.train_classifier', lr_mirror_augment=True, ud_mirror_augment=False,
                          total_kimg=total_train_img/1000.0,epochs = epochs,total_val_img = total_val_img,)
    config.result_dir = checkpoint
    # elif app == 'test':
    #     assert model_path != ' ' and val_set != ' ' and out_fingerprint_dir != ' '
    #     misc.init_output_logging()
    #     print('Initializing TensorFlow...')
    #     os.environ.update(config.env)
    #     tfutil.init_tf(config.tf_config)
    #     app = config.EasyDict(func='util_scripts.classify', model_path=model_path,
    #                           testing_data_path=val_set, out_fingerprint_dir=out_fingerprint_dir)

    tfutil.call_func_by_name(**app)

if __name__ == "__main__":
    from mesonet.model import Meso4
    from train_tf import train_cnn
    model = Meso4(image_size=256).model
    loss = "binary_crossentropy"
    train_cnn(model,loss)