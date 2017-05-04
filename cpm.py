#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf
from datetime import datetime
import os
import numpy as np
import read_data

class CPM:
    def __init__(self, config):
        # self.global_step = tf.get_vari(0, trainable=False, name="global_step")
        self.global_step = tf.get_variable("global_step", initializer=0,
                    dtype=tf.int32, trainable=False)
        self.wd = config.wd
        self.stddev = config.stddev
        self.batch_size = config.batch_size
        self.use_fp16 = config.use_fp16
        self.points_num = config.points_num
        self.fm_channel = config.fm_channel
        self.moving_average_decay = config.moving_average_decay
        self.params_dir = config.params_dir

        self.fm_height = config.fm_height
        self.fm_width = config.fm_width

        self.images = tf.placeholder(
                dtype = tf.float32,
                shape = (self.batch_size, config.img_height, config.img_width, 1)
                )
        self.labels = tf.placeholder(
                dtype = tf.float32,
                shape = (self.batch_size, config.fm_height, config.fm_width, self.points_num))
        self.coords = tf.placeholder(
                dtype = tf.float32,
                shape = (self.batch_size, self.points_num * 2))


    def build_fc(self, is_train):
      fc_is_train = is_train & True

      with tf.name_scope("original_images") as scope:
        self._image_summary(self.images, 1)
      out_fc = self.cnn_fc(self.images, fc_is_train, 'fc')
      self.add_to_euclidean_loss(self.batch_size, out_fc, self.coords, 'fcn')

      return out_fc

    def cnn_fc(self, input_, is_train, name):
      trainable = is_train
      is_BN = False

      with tf.variable_scope(name) as scope:
        conv1_1 = self.conv_layer(input_, 3, 64,
            'conv1_1', is_BN, trainable)
        conv1_2 = self.conv_layer(conv1_1, 3, 64,
            'conv1_2', is_BN, trainable)
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding="SAME", name="pool1")

        conv2_1 = self.conv_layer(pool1, 3, 128,
            'conv2_1', is_BN, trainable)
        conv2_2 = self.conv_layer(conv2_1, 3, 128,
            'conv2_2', is_BN, trainable)
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding="SAME", name="pool2")

        conv3_1 = self.conv_layer(pool2, 3, 256,
            'conv3_1', is_BN, trainable)
        conv3_2 = self.conv_layer(conv3_1, 3, 256,
            'conv3_2', is_BN, trainable)
        conv3_3 = self.conv_layer(conv3_2, 3, 256,
            'conv3_3', is_BN, trainable)
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding="SAME", name="pool3")

        conv4_1 = self.conv_layer(pool3, 3, 512,
            'conv4_1', is_BN, trainable)
        conv4_2 = self.conv_layer(conv4_1, 3, 512,
            'conv4_2', is_BN, trainable)
        conv4_3 = self.conv_layer(conv4_2, 3, 512,
            'conv4_3', is_BN, trainable)
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding="SAME", name="pool4")

        conv5_1 = self.conv_layer(pool4, 3, 512,
            'conv5_1', is_BN, trainable)
        conv5_2 = self.conv_layer(conv5_1, 3, 512,
            'conv5_2', is_BN, trainable)
        conv5_3 = self.conv_layer(conv5_2, 3, 512,
            'conv5_3', is_BN, trainable)
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding="SAME", name="pool5")

        fc6 = self.fc_layer(pool5, 4096, 'fc6', is_BN, trainable)
        if is_train:
            fc6 = tf.nn.dropout(fc6, 0.5)
        fc7 = self.fc_layer(fc6, 4096, 'fc7', is_BN, trainable)
        if is_train:
            fc7 = tf.nn.dropout(fc7, 0.5)
        fc8 = self.final_fc_layer(fc7, \
            self.points_num * 2, 'fc8', trainable)

      return fc8


    def final_fc_block(self, input_, is_train, name):
      trainable = is_train
      is_BN = True

      with tf.variable_scope(name) as scope:
        final_fc = self.final_fc_layer(input_,
            self.points_num * 2, 'final_fc', trainable)

      return final_fc

    def loss(self):
      return tf.add_n(tf.get_collection('losses'), name = "total_loss")

    def add_to_euclidean_loss(self, batch_size, predicts, labels, name):
        flatten_labels = tf.reshape(labels, [batch_size, -1])
        flatten_predicts = tf.reshape(predicts, [batch_size, -1])

        with tf.name_scope(name) as scope:
            euclidean_loss = tf.sqrt(tf.reduce_sum(
              tf.square(tf.subtract(flatten_predicts, flatten_labels)), 1))
            euclidean_loss_mean = tf.reduce_mean(euclidean_loss,
                name='euclidean_loss_mean')

        tf.add_to_collection("losses", euclidean_loss_mean)

    def train_op(self, total_loss, global_step):
        self._loss_summary(total_loss)

        optimizer = tf.train.AdamOptimizer()
        grads = optimizer.compute_gradients(total_loss)

        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        variable_averages = tf.train.ExponentialMovingAverage(
                self.moving_average_decay, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
            train_op = tf.no_op(name = "train")

        return train_op

    def save(self, sess, saver, filename, global_step):
        path = saver.save(sess, self.params_dir+filename, global_step=global_step)
        print "Save params at " + path

    def restore(self, sess, saver, filename):
        print "Restore from previous model: ", self.params_dir+filename
        saver.restore(sess, self.params_dir+filename)

    def fc_layer(self, bottom, out_num, name, is_BN, trainable):
        flatten_bottom = tf.reshape(bottom, [self.batch_size, -1])
        with tf.variable_scope(name) as scope:
            weights = self._variable_with_weight_decay(
                    "weights",
                    shape = [flatten_bottom.get_shape()[-1], out_num],
                    stddev = self.stddev,
                    wd = self.wd,
                    trainable=trainable)
            mul = tf.matmul(flatten_bottom, weights)
            biases = self._variable_on_cpu('biases', [out_num],
                    tf.constant_initializer(0.0), trainable)
            pre_activation = tf.nn.bias_add(mul, biases)
            if is_BN:
                bn_activation = tf.layers.batch_normalization(pre_activation)
                top = tf.nn.relu(bn_activation, name=scope.name)
            else:
                top = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(top)
        return top

    def final_fc_layer(self, bottom, out_num, name, trainable):
        flatten_bottom = tf.reshape(bottom, [self.batch_size, -1])
        with tf.variable_scope(name) as scope:
            weights = self._variable_with_weight_decay(
                    "weights",
                    shape = [flatten_bottom.get_shape()[-1], out_num],
                    stddev = self.stddev,
                    wd = self.wd,
                    trainable=trainable)
            mul = tf.matmul(flatten_bottom, weights) # Multiplies matrix `a` by matrix `b`, producing `a` * `b`.
            biases = self._variable_on_cpu('biases', [out_num],
                    tf.constant_initializer(0.0), trainable)
            top = tf.nn.bias_add(mul, biases) # Returns: A `Tensor` with the same type as `value`.
            self._activation_summary(top)
        return top

    def conv_layer(self, bottom, kernel_size, out_channel, name, is_BN, trainable):
        with tf.variable_scope(name) as scope:
            kernel = self._variable_with_weight_decay(
                    "weights",
                    shape = [kernel_size, kernel_size, bottom.get_shape()[-1],
                      out_channel],
                    stddev = self.stddev,
                    wd = self.wd,
                    trainable=trainable)
            conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding="SAME")
            biases = self._variable_on_cpu('biases', [out_channel],
                    tf.constant_initializer(0.0), trainable)
            pre_activation = tf.nn.bias_add(conv, biases)
            if is_BN:
                bn_activation = tf.layers.batch_normalization(pre_activation)
                top = tf.nn.relu(bn_activation, name=scope.name)
            else:
                top = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(top)
        return top

    def final_conv_layer(self, bottom, kernel_size, out_channel, name, trainable):
        with tf.variable_scope(name) as scope:
            kernel = self._variable_with_weight_decay(
                    "weights",
                    shape = [kernel_size, kernel_size, bottom.get_shape()[-1],
                      out_channel],
                    stddev = self.stddev,
                    wd = self.wd,
                    trainable=trainable)
            conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding="SAME")
            biases = self._variable_on_cpu('biases', [out_channel],
                    tf.constant_initializer(0.0), trainable)
            top = tf.nn.bias_add(conv, biases)
            self._activation_summary(top)
        return top

    def _variable_on_cpu(self, name, shape, initializer, trainable):
        with tf.device('/cpu:0'):
            dtype = tf.float16 if self.use_fp16 else tf.float32
            # trainable: If `True` also add the variable to the graph collection
            var = tf.get_variable(name, shape, initializer=initializer,
                    dtype=dtype, trainable=trainable)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd, trainable):
        dtype = tf.float16 if self.use_fp16 else tf.float32
        var = self._variable_on_cpu(name, shape,
                tf.truncated_normal_initializer(stddev=stddev, dtype=dtype),
                trainable)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                name='weights_loss')
            tf.add_to_collection("losses", weight_decay)
        return var

    def _activation_summary(self, x):
        name = x.op.name
        tf.summary.histogram(name + '/activations', x)
        tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(x))

    def _image_summary(self, x, channels):
        def sub(batch, idx):
            name = x.op.name
            tmp = x[batch, :, :, idx] * 255
            tmp = tf.expand_dims(tmp, axis = 2)
            tmp = tf.expand_dims(tmp, axis = 0)
            tf.summary.image(name + '-' + str(idx), tmp, max_outputs = 100)
        if (self.batch_size > 1):
          for idx in xrange(channels):
            # the first batch
            sub(0, idx)
            # the last batch
            sub(-1, idx)
        else:
          for idx in xrange(channels):
            sub(0, idx)

    def _loss_summary(self, loss):
        tf.summary.scalar(loss.op.name + " (raw)", loss)

    def _fm_summary(self, predicts):
      with tf.name_scope("fcn_summary") as scope:
          self._image_summary(self.labels, self.points_num)
          tmp_predicts = tf.nn.relu(predicts)
          self._image_summary(tmp_predicts, self.points_num)


def main():
    pass

if __name__ == "__main__":
    main()
