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

    # == modify parameters ===========================
    TAG = "mix_half" # used for uniform filename
    FLAG = "cleaned_3" # "cleaned": train cleaned data;
                     # "origin": train origin data
    steps = "30000"
    # ================================================

    # test_num = 50
    # annos_path = "./labels/txt/input/test_annos.txt"
    # data_path = "./data/input/test_imgs"
    # annos_write_path = "./labels/txt/output/test_annos_" +\
    #  TAG + "_" + FLAG + ".txt"
    test_num = 50
    annos_path = "./labels/txt/input/train_annos_" +\
     TAG + "_" + FLAG + ".txt"
    data_path = "./data/input/train_imgs_mix_half_cleaned"
    annos_write_path = "./labels/txt/output/train_annos_" +\
     TAG + "_" + FLAG + ".txt"
    batch_size = 20
    initialize = False
    gpu = '/gpu:0'

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
    degree = 15

    # solver config
    wd = 5e-4
    stddev = 5e-2
    use_fp16 = False
    moving_average_decay = 0.999

    # checkpoint path and filename
    logdir = "./log/test_log/"
    params_dir = "./params/" + FLAG + "_" + TAG + "/"

    load_filename = "cpm" + '-' + steps
    save_filename = "cpm"


def main():
    config = Config()
    with tf.Graph().as_default():

        # create a reader object
        reader = read_data.PoseReader(config.annos_path,
            config.data_path, config)

        # create a model object
        model = cpm.CPM(config)

        # feedforward
        predicts = model.build_fc(False) # False: is not training

        # Initializing operation
        init_op = tf.global_variables_initializer()

        saver = tf.train.Saver(max_to_keep=100)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            sess.run(init_op)
            model.restore(sess, saver, config.load_filename)

            # write the predict coordinates to file
            with open(config.annos_write_path, 'w') as fw:
                # start testing
                for idx in xrange(config.test_num):
                    with tf.device("/cpu:0"):
                        imgs, fm, coords, begins, filename_list = \
                        reader.get_random_batch(distort=False)
                    # feed data into the model
                    feed_dict = {
                        model.images: imgs,
                        model.coords: coords,
                        model.labels: fm
                        }
                    with tf.device(config.gpu):
                        # run the testing operation
                        predict_coords_list = sess.run(predicts, feed_dict=feed_dict)
                        # print predict_coords_list
                        for filename, predict_coords in zip(filename_list, predict_coords_list):
                            print (filename, predict_coords)
                            fw.write(filename + ' ')
                            for i in xrange(config.points_num):
                                # w = predict_coords[i*2] * config.img_width
                                # h = predict_coords[i*2 + 1] * config.img_height
                                # item = str(int(w)) + ',' + str(int(h))
                                w = predict_coords[i*2]
                                h = predict_coords[i*2 + 1]
                                item = str(w) + ',' + str(h)
                                fw.write(item + ',')
                            fw.write('1\n')


if __name__ == "__main__":
    print "Begin testing..."
    main()
    print "Finished testing."

# added by hypjudy
# $ tensorboard --logdir ./log/test_log/
# visit: http://172.18.181.94:6006/
# Q: ERROR:tensorflow:Tried to connect to port 6006, but address is in use.
# $ ps aux | grep tensorboard
# $ kill -9 pid
# or
# $ tensorboard --logdir ./log/test_log/ --port=8008
# visit: http://172.18.181.94:8008/
