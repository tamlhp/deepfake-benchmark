# from collections import Counter
import numpy as np
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
import keras
import os

def get_generate(train_set,val_set,image_size,batch_size):
    dataGenerator = ImageDataGenerator(rescale=1. / 255, rotation_range=5,
                                       width_shift_range=0.05,
                                       height_shift_range=0.05,
                                       horizontal_flip=True, shear_range=0.05)
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

def train_cnn(model,loss,train_set = '../../extract_raw_img',val_set ='../../extract_raw_img',image_size=256,batch_size=16,num_workers=1,checkpoint="checkpoint",epochs=20):

    #### Load data
    generator_train, generator_val = get_generate(train_set,val_set,image_size,batch_size)


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

def train_gan(train_set = '../../extract_raw_img',val_set ='../../extract_raw_img',training_seed=0,checkpoint="checkpoint"):
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
        training_data_dir = train_set[:-1]
    idx = train_set.rfind('/')
    config.data_dir = train_set[:idx]
    config.training_set = config.EasyDict(tfrecord_dir=val_set[idx + 1:], max_label_size='full')
    if val_set[-1] == '/':
        validation_data_dir = val_set[:-1]
    idx = validation_data_dir.rfind('/')
    config.validation_set = config.EasyDict(tfrecord_dir=val_set[idx + 1:], max_label_size='full')
    app = config.EasyDict(func='run.train_classifier', lr_mirror_augment=True, ud_mirror_augment=False,
                          total_kimg=25000)
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