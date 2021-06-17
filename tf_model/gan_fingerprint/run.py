# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licen sed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import time
import numpy as np
import tensorflow as tf

from tf_model.gan_fingerprint import config
from tf_model.gan_fingerprint import tfutil
from tf_model.gan_fingerprint import dataset
from tf_model.gan_fingerprint import misc
from sklearn import metrics
from tqdm import tqdm
import argparse
from sklearn.metrics import recall_score,accuracy_score,precision_score,log_loss,classification_report

#----------------------------------------------------------------------------
# Choose the size and contents of the image snapshot grids that are exported
# periodically during training.

#----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.

def process_reals(x, lod, lr_mirror_augment, ud_mirror_augment, drange_data, drange_net):
    with tf.name_scope('ProcessReals'):
        with tf.name_scope('DynamicRange'):
            x = tf.cast(x, tf.float32)
            x = misc.adjust_dynamic_range(x, drange_data, drange_net)
        if lr_mirror_augment:
            with tf.name_scope('MirrorAugment'):
                s = tf.shape(x)
                mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
                mask = tf.tile(mask, [1, s[1], s[2], s[3]])
                x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[3]))
        if ud_mirror_augment:
            with tf.name_scope('udMirrorAugment'):
                s = tf.shape(x)
                mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
                mask = tf.tile(mask, [1, s[1], s[2], s[3]])
                x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[2]))
        with tf.name_scope('FadeLOD'): # Smooth crossfade between consecutive levels-of-detail.
            s = tf.shape(x)
            y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2])
            y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
            y = tf.tile(y, [1, 1, 1, 2, 1, 2])
            y = tf.reshape(y, [-1, s[1], s[2], s[3]])
            x_fade = tfutil.lerp(x, y, lod - tf.floor(lod))
            x_orig = tf.identity(x)
        with tf.name_scope('UpscaleLOD'): # Upscale to match the expected input/output size of the networks.
            s = tf.shape(x)
            factor = tf.cast(2 ** tf.floor(lod), tf.int32)
            x_fade = tf.reshape(x_fade, [-1, s[1], s[2], 1, s[3], 1])
            x_fade = tf.tile(x_fade, [1, 1, 1, factor, 1, factor])
            x_fade = tf.reshape(x_fade, [-1, s[1], s[2] * factor, s[3] * factor])
            x_orig = tf.reshape(x_orig, [-1, s[1], s[2], 1, s[3], 1])
            x_orig = tf.tile(x_orig, [1, 1, 1, factor, 1, factor])
            x_orig = tf.reshape(x_orig, [-1, s[1], s[2] * factor, s[3] * factor])
        return x_fade, x_orig

#----------------------------------------------------------------------------
# Class for evaluating and storing the values of time-varying training parameters.

class TrainingSchedule:
    def __init__(
        self,
        cur_nimg,
        training_set,
        lod_initial_resolution  = 128,        # Image resolution used at the beginning.
        lod_training_kimg       = 1500,     # Thousands of real images to show before doubling the resolution.
        lod_transition_kimg     = 1500,     # Thousands of real images to show when fading in new layers.
        minibatch_base          = 16,       # Maximum minibatch size, divided evenly among GPUs.
        minibatch_dict          = {},       # Resolution-specific overrides.
        max_minibatch_per_gpu   = {},       # Resolution-specific maximum minibatch size per GPU.
        lrate_base              = 0.001,    # Learning rate for AutoEncoder.
        lrate_dict              = {},       # Resolution-specific overrides.
        tick_kimg_base          = 1,        # Default interval of progress snapshots.
        tick_kimg_dict          = {}):      # Resolution-specific overrides.

        # Training phase.
        self.kimg = cur_nimg / 1000.0
        phase_dur = lod_training_kimg + lod_transition_kimg
        phase_idx = int(np.floor(self.kimg / phase_dur)) if phase_dur > 0 else 0
        phase_kimg = self.kimg - phase_idx * phase_dur

        # Level-of-detail and resolution.
        self.lod = training_set.resolution_log2
        self.lod -= np.floor(np.log2(lod_initial_resolution))
        self.lod -= phase_idx
        if lod_transition_kimg > 0:
            self.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
        self.lod = max(self.lod, 0.0)
        self.resolution = 2 ** (training_set.resolution_log2 - int(np.floor(self.lod)))

        # Minibatch size.
        self.minibatch = minibatch_dict.get(self.resolution, minibatch_base)
        self.minibatch -= self.minibatch % config.num_gpus
        if self.resolution in max_minibatch_per_gpu:
            self.minibatch = min(self.minibatch, max_minibatch_per_gpu[self.resolution] * config.num_gpus)

        # Other parameters.
        self.lrate = lrate_dict.get(self.resolution, lrate_base)
        self.tick_kimg = tick_kimg_dict.get(self.resolution, tick_kimg_base)

#----------------------------------------------------------------------------
# Main training script.
# To run, comment/uncomment appropriate lines in config.py and launch train.py.

def train_classifier(
    smoothing               = 0.999,        # Exponential running average of encoder weights.
    minibatch_repeats       = 4,            # Number of minibatches to run before adjusting training parameters.
    reset_opt_for_new_lod   = True,         # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
    total_kimg              = 25000,        # Total length of the training, measured in thousands of real images.
    lr_mirror_augment       = True,        # Enable mirror augment?
    ud_mirror_augment       = False,        # Enable up-down mirror augment?
    drange_net              = [-1,1],       # Dynamic range used when feeding image data to the networks.
    image_snapshot_ticks    = 10,           # How often to export image snapshots?
    save_tf_graph           = False,        # Include full TensorFlow computation graph in the tfevents file?
    save_weight_histograms  = False,       # Include weight histograms in the tfevents file?
    epochs = 10,
        total_val_img = 5000,
    ):
    maintenance_start_time = time.time()
    training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.training_set)
    validation_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.validation_set)

    # Construct networks.
    with tf.device('/gpu:0'):
        try:
            network_pkl = misc.locate_network_pkl()
            resume_kimg, resume_time = misc.resume_kimg_time(network_pkl)
            print('Loading networks from "%s"...' % network_pkl)
            EG, D_rec, EGs = misc.load_pkl(network_pkl)
        except:
            print('Constructing networks...')
            resume_kimg = 0.0
            resume_time = 0.0
            EG = tfutil.Network('EG', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **config.EG)
            D_rec = tfutil.Network('D_rec', num_channels=training_set.shape[0], resolution=training_set.shape[1], **config.D_rec)
            EGs = EG.clone('EGs')
        EGs_update_op = EGs.setup_as_moving_average_of(EG, beta=smoothing)
    EG.print_layers(); D_rec.print_layers()

    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'):
        lod_in          = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in        = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_in    = tf.placeholder(tf.int32, name='minibatch_in', shape=[])
        minibatch_split = minibatch_in // config.num_gpus
        reals, labels   = training_set.get_minibatch_tf()
        reals_split     = tf.split(reals, config.num_gpus)
        labels_split    = tf.split(labels, config.num_gpus)
    EG_opt = tfutil.Optimizer(name='TrainEG', learning_rate=lrate_in, **config.EG_opt)
    D_rec_opt = tfutil.Optimizer(name='TrainD_rec', learning_rate=lrate_in, **config.D_rec_opt)
    for gpu in range(config.num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            EG_gpu = EG if gpu == 0 else EG.clone(EG.name + '_shadow_%d' % gpu)
            D_rec_gpu = D_rec if gpu == 0 else D_rec.clone(D_rec.name + '_shadow_%d' % gpu)
            reals_fade_gpu, reals_orig_gpu = process_reals(reals_split[gpu], lod_in, lr_mirror_augment, ud_mirror_augment, training_set.dynamic_range, drange_net)
            labels_gpu = labels_split[gpu]
            with tf.name_scope('EG_loss'):
                EG_loss = tfutil.call_func_by_name(EG=EG_gpu, D_rec=D_rec_gpu, reals_orig=reals_orig_gpu, labels=labels_gpu, **config.EG_loss)
            with tf.name_scope('D_rec_loss'):
                D_rec_loss = tfutil.call_func_by_name(EG=EG_gpu, D_rec=D_rec_gpu, D_rec_opt=D_rec_opt, minibatch_size=minibatch_split, reals_orig=reals_orig_gpu, **config.D_rec_loss)
            EG_opt.register_gradients(tf.reduce_mean(EG_loss), EG_gpu.trainables)
            D_rec_opt.register_gradients(tf.reduce_mean(D_rec_loss), D_rec_gpu.trainables)
    EG_train_op = EG_opt.apply_updates()
    D_rec_train_op = D_rec_opt.apply_updates()

    print('Setting up result dir...')
    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)

    est_fingerprints = np.transpose(EGs.vars['Conv_fingerprints/weight'].eval(), axes=[3,2,0,1])
    misc.save_image_grid(est_fingerprints, os.path.join(result_subdir, 'est_fingerrints-init.png'), drange=[np.amin(est_fingerprints), np.amax(est_fingerprints)], grid_size=[est_fingerprints.shape[0],1])

    summary_log = tf.summary.FileWriter(result_subdir)
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        EG.setup_weight_histograms(); D_rec.setup_weight_histograms()

    print('Training...')
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    train_start_time = tick_start_time - resume_time
    prev_lod = -1.0
    total_val_iter = int(total_val_img/config.sched.minibatch_base)
    text_writer = open(os.path.join(config.result_dir, 'train.csv'), 'a')
    print(int(total_kimg * 1000/config.sched.minibatch_base))
    for i in range(epochs):
        # while cur_nimg < total_kimg * 1000:
        cur_nimg = 0
        for jtrain in tqdm(range(int(total_kimg * 1000/config.sched.minibatch_base))):

            # Choose training parameters and configure training ops.
            sched = TrainingSchedule(cur_nimg, training_set, **config.sched)
            training_set.configure(config.sched.minibatch_base, sched.lod)
            # if reset_opt_for_new_lod:
            #     if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
            #         EG_opt.reset_optimizer_state(); D_rec_opt.reset_optimizer_state()
            # prev_lod = sched.lod

            # Run training ops.
            for repeat in range(minibatch_repeats):
                tfutil.run([D_rec_train_op], {lod_in: sched.lod, lrate_in: sched.lrate, minibatch_in: config.sched.minibatch_base})
                tfutil.run([EG_train_op], {lod_in: sched.lod, lrate_in: sched.lrate, minibatch_in: config.sched.minibatch_base})
                tfutil.run([EGs_update_op], {})
            cur_nimg += config.sched.minibatch_base

        # Perform maintenance tasks once per tick.
        cur_tick += 1
        cur_time = time.time()
        tick_start_nimg = cur_nimg
        tick_time = cur_time - tick_start_time
        total_time = cur_time - train_start_time
        maintenance_time = tick_start_time - maintenance_start_time
        maintenance_start_time = cur_time

        # Report progress.
        print(
            'tick %-5d kimg %-8.1f time %-12s sec/tick %-7.1f maintenance %.1f' % (
                tfutil.autosummary('Progress/tick', cur_tick),
                tfutil.autosummary('Progress/kimg', cur_nimg / 1000.0),
                misc.format_time(tfutil.autosummary('Timing/total_sec', total_time)),
                tfutil.autosummary('Timing/sec_per_tick', tick_time),
                tfutil.autosummary('Timing/maintenance_sec', maintenance_time)))
        tfutil.autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
        tfutil.autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))
        tfutil.save_summaries(summary_log, cur_nimg)

        y_label = []
        y_pred = []
        y_pred_label = []
        for jtest in range(total_val_iter):
            real, label = validation_set.get_minibatch_np(config.sched.minibatch_base)
            real = misc.adjust_dynamic_range(real, training_set.dynamic_range, drange_net)
            rec, fingerprint, logits = EGs.run(real, minibatch_size=config.sched.minibatch_base, num_gpus=1, out_dtype=np.float32)
            idx = np.argmax(np.squeeze(logits),axis=1)
            y_pred_label.extend(idx)
            y_label.extend(np.argmax(np.squeeze(label), axis=1))
            y_pred.extend(logits)
            # print(logits)
            # print("438 idx: ", idx)
        # acc_test = metrics.accuracy_score(idxs, labels)
        acc_test = np.float32(np.sum(np.array(y_pred_label) == np.array(y_label))) / np.float32(len(y_label))
        log_loss_metric = log_loss(y_label, y_pred, labels=np.array([0., 1.]))
        print("Epoch  %d :  loss : %f   accuracy : %f " %(i,log_loss_metric,acc_test))
        text_writer.write("Epoch  %d : loss : %f    accuracy : %f " %(i,log_loss_metric,acc_test))
        text_writer.flush()
        misc.save_pkl((EG, D_rec, EGs),
                      os.path.join(result_subdir, 'network-snapshot-%06d.pkl' % (i)))

    # Write final results.
    misc.save_pkl((EG, D_rec, EGs), os.path.join(result_subdir, 'network-final.pkl'))
    summary_log.close()
    open(os.path.join(result_subdir, '_training-done.txt'), 'wt').close()
from scipy.special import softmax
def eval_classifier(
    drange_net              = [-1,1],       # Dynamic range used when feeding image data to the networks.
        total_val_img = 5000,
        model_path="checkpoint",
        show_time=False
    ):
    maintenance_start_time = time.time()
    validation_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.validation_set)

    # Construct networks.
    with tf.device('/gpu:0'):
        try:
            EG, D_rec, EGs = misc.load_network_pkl(model_path)
            print('Loading networks from "%s"...' % model_path)
        except:
            print('Khong load duoc model...')
            return

    EG.print_layers(); D_rec.print_layers()

    print('Eval...')
    total_val_iter = int(total_val_img/config.sched.minibatch_base)

    y_label = []
    y_pred = []
    y_pred_label = []
    y_softmax = []
    begin = time.time()
    for jtest in range(total_val_iter):
        #begin = time.time()
        real, label = validation_set.get_minibatch_np(config.sched.minibatch_base)
        real = misc.adjust_dynamic_range(real, validation_set.dynamic_range, drange_net)
        rec, fingerprint, logits = EGs.run(real, minibatch_size=config.sched.minibatch_base, num_gpus=1, out_dtype=np.float32)
        idx = np.argmax(np.squeeze(logits),axis=1)
        #if show_time:
        #    print("Time:  ",time.time()-begin)
        y_pred_label.extend(idx)
        y_label.extend(np.argmax(np.squeeze(label), axis=1))
        y_pred.extend(logits)
        y_softmax.extend(softmax(logits,axis=1))
        # print(logits)
        # print("438 idx: ", idx)
    # acc_test = metrics.accuracy_score(idxs, labels)
    if show_time:
        print("Time:  ",time.time()-begin)
    acc_test = np.float32(np.sum(np.array(y_pred_label) == np.array(y_label))) / np.float32(len(y_label))
    log_loss_metric = log_loss(y_label, y_softmax, labels=np.array([0., 1.]))
    print("loss : %f   accuracy : %f " % (log_loss_metric, acc_test))
    print(acc_test)
    print(f"Test log_loss: {log_loss(y_label,y_softmax,labels=np.array([0.,1.])):.3f}\n" +
          f"Test accuracy_score: {accuracy_score(y_label,y_pred_label):.3f}\n" +
          f"Test precision_score: {precision_score(y_label,y_pred_label):.3f}\n" +
          f"Test recall: {recall_score(y_label,y_pred_label):.3f}\n")
    print(classification_report(y_label,y_pred_label))

#----------------------------------------------------------------------------
# Main entry point.
# Calls the function indicated in config.py.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--app', type=str, default=' ')
    #------------------- training arguments -------------------
    parser.add_argument('--training_data_dir', type=str, default=' ') # The prepared training dataset directory that can be efficiently called by the code
    parser.add_argument('--validation_data_dir', type=str, default=' ') # The prepared validation dataset directory that can be efficiently called by the code
    parser.add_argument('--out_model_dir', type=str, default=' ') # The output directory containing trained models, training configureation, training log, and training snapshots
    parser.add_argument('--training_seed', type=int, default=1000) # The random seed that differentiates training instances
    #------------------- image generation arguments -------------------
    parser.add_argument('--model_path', type=str, default=' ') # The pre-trained GAN model
    parser.add_argument('--testing_data_path', type=str, default=' ') # The path of testing image file or the directory containing a collection of testing images
    parser.add_argument('--out_fingerprint_dir', type=str, default=' ') # The output directory containing model fingerprints, image fingerprints, and image fingerprints masked(re-weighted) by each model fingerprint

    args = parser.parse_args()
    if args.app == 'train':
        assert args.training_data_dir != ' ' and args.out_model_dir != ' '
        if args.validation_data_dir == ' ':
            args.validation_data_dir = args.training_data_dir
        misc.init_output_logging()
        np.random.seed(args.training_seed)
        print('Initializing TensorFlow...')
        os.environ.update(config.env)
        tfutil.init_tf(config.tf_config)
        if args.training_data_dir[-1] == '/':
            args.training_data_dir = args.training_data_dir[:-1]
        idx = args.training_data_dir.rfind('/')
        config.data_dir = args.training_data_dir[:idx]
        config.training_set = config.EasyDict(tfrecord_dir=args.training_data_dir[idx+1:], max_label_size='full')
        if args.validation_data_dir[-1] == '/':
            args.validation_data_dir = args.validation_data_dir[:-1]
        idx = args.validation_data_dir.rfind('/')
        config.validation_set = config.EasyDict(tfrecord_dir=args.validation_data_dir[idx+1:], max_label_size='full')
        app = config.EasyDict(func='run.train_classifier', lr_mirror_augment=True, ud_mirror_augment=False, total_kimg=25000)
        config.result_dir = args.out_model_dir
    elif args.app == 'test':
        assert args.model_path != ' ' and args.testing_data_path != ' ' and args.out_fingerprint_dir != ' '
        misc.init_output_logging()
        print('Initializing TensorFlow...')
        os.environ.update(config.env)
        tfutil.init_tf(config.tf_config)
        app = config.EasyDict(func='util_scripts.classify', model_path=args.model_path, testing_data_path=args.testing_data_path, out_fingerprint_dir=args.out_fingerprint_dir)
    
    tfutil.call_func_by_name(**app)
