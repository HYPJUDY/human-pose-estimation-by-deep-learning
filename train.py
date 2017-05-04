#!/usr/bin/env python
# encoding: utf-8

from datetime import datetime
import os
import random

import tensorflow as tf
import numpy as np

import cpm
import read_data


class Config():

  # =================== modify parameters ==================
  TAG = "mix_half" # used for uniform filename
  FLAG = "cleaned_3" # "cleaned": train cleaned data;
                   # "origin": train origin data
  batch_size = 20
  initialize = True # True: train from scratch;
                     # False: restore from pretrained model
  steps = "30000"   # if 'initialize = False', set steps to 
                     # where you want to restore
  toDistort = False
  # iterations config
  max_iteration = 30000
  checkpoint_iters = 2000
  summary_iters = 100
  validate_iters = 2000
  # ========================================================

  annos_path = "./labels/txt/input/train_annos_" +\
   TAG + "_" + FLAG + ".txt"
  data_path = "./data/input/train_imgs_" + TAG + "_" + FLAG + "/"
  gpu = '/gpu:0'

  # checkpoint path and filename
  logdir = "./log/train_log/"
  params_dir = "./params/" + FLAG + "_" + TAG + "/"
  load_filename = "cpm" + '-' + steps
  save_filename = "cpm"

  # image config
  points_num = 15
  fm_channel = points_num + 1
  origin_height = 212
  origin_width = 256
  img_height = 216
  img_width = 256
  is_color = False

  # feature map config
  fm_width = img_width >> 1
  fm_height = img_height >> 1
  sigma = 2.0
  alpha = 1.0
  radius = 12

  # random distortion
  degree = 8

  # solver config
  wd = 5e-4
  stddev = 5e-2
  use_fp16 = False
  moving_average_decay = 0.999

def main():

    config = Config()
    with tf.Graph().as_default():
        # create a reader object
        reader = read_data.PoseReader(config.annos_path,
            config.data_path, config)

        # create a model object
        model = cpm.CPM(config)

        # feedforward
        predicts = model.build_fc(True)

        # return the loss
        loss = model.loss()

        # training operation
        train_op = model.train_op(loss, model.global_step)
        # Initializing operation
        init_op = tf.global_variables_initializer()

        saver = tf.train.Saver(max_to_keep = 100)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:

            # initialize parameters or restore from previous model
            if not os.path.exists(config.params_dir):
                os.makedirs(config.params_dir)
            if os.listdir(config.params_dir) == [] or config.initialize:
                print "Initializing Network"
                sess.run(init_op)
            else:
                sess.run(init_op)
                model.restore(sess, saver, config.load_filename)

            merged = tf.summary.merge_all()
            logdir = os.path.join(config.logdir,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

            writer = tf.summary.FileWriter(logdir, sess.graph)

            # start training
            for idx in xrange(config.max_iteration):
                with tf.device("/cpu:0"):
                  imgs, fm, coords, begins, filename_list = \
                  reader.get_random_batch(distort=config.toDistort)

                # feed data into the model
                feed_dict = {
                    model.images: imgs,
                    model.coords: coords,
                    model.labels: fm
                    }
                with tf.device(config.gpu):
                    # run the training operation
                    sess.run(train_op, feed_dict=feed_dict)

                with tf.device('/cpu:0'):
                  # write summary
                  if (idx + 1) % config.summary_iters == 0:
                      print ("iter#", idx + 1)
                      tmp_global_step = model.global_step.eval()
                      summary = sess.run(merged, feed_dict=feed_dict)
                      writer.add_summary(summary, tmp_global_step)
                  # save checkpoint
                  if (idx + 1) % config.checkpoint_iters == 0:
                      tmp_global_step = model.global_step.eval()
                      model.save(sess, saver, config.save_filename, tmp_global_step)


if __name__ == "__main__":
    main()

# added by hypjudy
# $ tensorboard --logdir ./log/train_log/
# visit: http://172.18.181.94:6006/
# Q: ERROR:tensorflow:Tried to connect to port 6006, but address is in use.
# $ ps aux | grep tensorboard
# $ kill -9 pid
