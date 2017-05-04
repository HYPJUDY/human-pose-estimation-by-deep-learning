#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import cv2
import sys
import os
import re
import random
import math

class PoseReader():

    def __init__(self, annos_path, data_path, config):
        self.records = list()
        self.batch_size = config.batch_size
        self.points_num = config.points_num
        self.fm_channel = config.fm_channel
        self.img_width = config.img_width
        self.img_height = config.img_height
        self.origin_width = config.origin_width
        self.origin_height = config.origin_height
        self.record_len = self.points_num * 2 + 2
        self.data_path = data_path
        self.line_idx = 0

        self.fm_width = config.fm_width
        self.fm_height = config.fm_height
        self.sigma = config.sigma
        self.alpha = config.alpha
        self.radius = config.radius

        # self.float_max = 1.0 - 1.0 / self.img_width
        self.float_max = 1.0

        self.degree = config.degree

        if config.is_color:
            self.color_mode = 1
        else:
            self.color_mode = 0
        with open(annos_path, 'rb') as fr:
            for line in fr:
                tmp = re.split(',| ', line.strip())
                if(len(tmp) != self.record_len):
                    print "Length Error: ", len(tmp)
                    sys.exit(0)
                filename = tmp[0]
                coords = [int(x) for x in tmp[1:self.record_len - 1]]
                begin = int(tmp[-1])
                self.records.append((filename, np.array(coords), begin))
        self.size = len(self.records)


    def random_batch(self):
        rand = random.sample(xrange(self.size), self.batch_size)
        filename_list = list()
        coords_list = list()
        begins_list = list()
        for idx in rand:
            filename_list.append(self.records[idx][0])
            coords_list.append(self.records[idx][1])
            begins_list.append(self.records[idx][2])

        img_list = list()
        for filename in filename_list:
            img = cv2.imread(os.path.join(self.data_path, filename),
                    self.color_mode)
            img = cv2.resize(img, (self.img_width, self.img_height))
            img_list.append(img)

        out_imgs = self._img_preprocess(np.stack(img_list))
        out_labels = self._label_preprocess(np.stack(coords_list))
        out_begins = np.stack(begins_list)

        return out_imgs, out_labels, out_begins, filename_list


    def batch(self, line_idx=None):
        if line_idx is not None:
            self.line_idx = line_idx
        end_idx = self.line_idx + self.batch_size
        idxs = range(self.line_idx, end_idx)
        for idx in xrange(len(idxs)):
            if idxs[idx] >= self.size:
                idxs[idx] %= self.size
        if end_idx < self.size:
            self.line_idx = end_idx
        else:
            self.line_idx = end_idx % self.size

        filename_list = list()
        coords_list = list()
        begins_list = list()
        for idx in idxs:
            filename_list.append(self.records[idx][0])
            coords_list.append(self.records[idx][1])
            begins_list.append(self.records[idx][2])

        img_list = list()
        for filename in filename_list:
            img = cv2.imread(os.path.join(self.data_path, filename),
                    self.color_mode)
            img = cv2.resize(img, (self.img_width, self.img_height))
            img_list.append(img)

        out_imgs = self._img_preprocess(np.stack(img_list))
        out_labels = self._label_preprocess(np.stack(coords_list))
        out_begins = np.stack(begins_list)

        return out_imgs, out_labels, out_begins, filename_list

    def _img_preprocess(self, imgs):
        if self.color_mode == 0:
            output = np.reshape(imgs, [-1, self.img_height, self.img_width, 1])
        elif self.color_mode == 1:
            output = np.reshape(imgs, [-1, self.img_height, self.img_width, 3])
        else:
            raise Exception ("color_mode error.")

        output = output.astype(np.float32) * (1. / 255) - 0.5
        return output

    def _label_preprocess(self, label):
        output = np.reshape(label, [-1, self.points_num * 2]).astype(np.float32)
        output[:, ::2] /= self.origin_width
        output[:, 1::2] /= self.origin_height
        return output

    def label2fm(self, label):
        def get_point(a_list, idx):
            w, h = a_list[idx * 2: idx * 2 + 2]
            return int(w * self.fm_width), int(h * self.fm_height)

        def _gaussian2d(x, y, x0, y0, a, sigmax, sigmay):
            xx = (float(x) - x0)** 2 / 2 / sigmax **2
            yy = (float(y) - y0)** 2 / 2 / sigmay **2
            return a * math.exp(- xx - yy)

        def draw(img, center):
            w0, h0 = center
            height, width = img.shape
            # for h in xrange(height):
                # for w in xrange(width):
                    # if(math.fabs(h - h0) + math.fabs(w - w0) < self.radius):
            for w in xrange(max(0, w0-self.radius), min(width, w0+self.radius+1)):
                for h in xrange(max(0, h0-self.radius), min(height, h0+self.radius+1)):
                    if(math.fabs(h - h0) + math.fabs(w - w0) < self.radius):
                        img[h, w] = _gaussian2d(w, h, w0, h0, self.alpha, self.sigma,
                                self.sigma)

        fm_label = np.zeros((label.shape[0], self.fm_height, self.fm_width, self.points_num))
        for batch_idx in xrange(len(fm_label)):
            for ii in xrange(self.points_num):
                w, h = get_point(label[batch_idx], ii)
                draw(fm_label[batch_idx, :, :, ii], (w, h))
        return fm_label.astype(np.float32)

    def label2sm_fm(self, label):
        def get_point(a_list, idx):
            w, h = a_list[idx * 2: idx * 2 + 2]
            return int(w * self.fm_width), int(h * self.fm_height)

        def p8_distance(h1, h2, w1, w2):
            return max(math.fabs(h1 - h2), math.fabs(w1 - w2))

        def p4_distance(h1, h2, w1, w2):
            return math.fabs(h1 - h2) + math.fabs(w1 - w2)

        def draw(img, center, idx):
            w0, h0 = center
            height, width = img.shape
            for w in xrange(max(0, w0-self.radius), min(width, w0+self.radius+1)):
                for h in xrange(max(0, h0-self.radius), min(height, h0+self.radius+1)):
                    if(p8_distance(h, h0, w, w0) < self.radius):
                        img[h, w] = idx + 1
        fm_label = np.zeros((label.shape[0], self.fm_height, self.fm_width))
        for batch_idx in xrange(len(fm_label)):
            for ii in xrange(self.points_num):
                w, h = get_point(label[batch_idx], ii)
                draw(fm_label[batch_idx], (w, h), ii)
        # fm_label = fm_label.astype(np.int32)
        return fm_label.astype(np.int32)

    def _visualize(self, imgs, merge):
        import matplotlib.pyplot as plt
        if (merge):
            imgs = np.amax(imgs, axis = 2)
        imgs = np.squeeze(imgs)
        plt.imshow(imgs, cmap='gray')
        plt.show()

    def _draw_imgs(self, imgs, coords):
        def get_point(a_list, idx):
            w, h = a_list[idx * 2: idx * 2 + 2]
            return int(w * self.img_width), int(h * self.img_height)

        import matplotlib.pyplot as plt
        for idx in xrange(len(imgs)):
            img = np.squeeze(imgs[idx])
            coord = coords[idx]
            for ii in xrange(self.points_num):
                w, h = get_point(coord, ii)
                cv2.circle(img, (w, h), 1, 1)
            plt.imshow(img, cmap='gray')
            plt.show()

    def _rotate(self, origin, angle):
        x, y = origin
        o_y = 0.5 + (y - 0.5) * math.cos(angle) + (x - 0.5) * math.sin(angle)
        o_x = 0.5 + (x - 0.5) * math.cos(angle) - (y - 0.5) * math.sin(angle)
        return o_x, o_y

    def _random_roate(self, images, labels, degree):
        if(images.shape[0] != labels.shape[0]):
            raise Exception("Batch size Error.")
        degree = degree * math.pi / 180
        rand_degree = np.random.uniform(-degree, degree, images.shape[0])

        o_images = np.zeros_like(images)
        o_labels = np.zeros_like(labels)
        for idx in xrange(images.shape[0]):
            theta = rand_degree[idx]

            # labels
            for ii in xrange(self.points_num):
                o_labels[idx, 2*ii: 2*ii+2] = self._rotate(labels[idx, ii*2: 2*ii+2], theta)

            # image
            M = cv2.getRotationMatrix2D((self.img_width/2,self.img_height/2),-theta*180/math.pi,1)
            o_images[idx] = np.expand_dims(cv2.warpAffine(images[idx],M,(self.img_width,self.img_height)), axis=2)

        return o_images, o_labels

    def _batch_random_roate(self, images, labels, degree):
        if(images.shape[0] != labels.shape[0]):
            raise Exception("Batch size Error.")
        degree = degree * math.pi / 180
        rand_degree = np.random.uniform(-degree, degree)

        o_images = np.zeros_like(images)
        o_labels = np.zeros_like(labels)
        for idx in xrange(images.shape[0]):
            theta = rand_degree

            # labels
            for ii in xrange(self.points_num):
                o_labels[idx, 2*ii: 2*ii+2] = self._rotate(labels[idx, ii*2: 2*ii+2], theta)

            # image
            M = cv2.getRotationMatrix2D((self.img_width/2,self.img_height/2),-theta*180/math.pi,1)
            o_images[idx] = np.expand_dims(cv2.warpAffine(images[idx],M,(self.img_width,self.img_height)), axis=2)

        return o_images, o_labels

    def _random_flip_lr(self, images, labels):
        if(images.shape[0] != labels.shape[0]):
            raise Exception("Batch size Error.")
        rand_u = np.random.uniform(0.0, 1.0, images.shape[0])
        rand_cond = rand_u > 0.5

        o_images = np.zeros_like(images)
        o_labels = np.zeros_like(labels)

        for idx in xrange(images.shape[0]):
            condition = rand_cond[idx]
            if condition:
                # "flip"
                o_images[idx] = np.fliplr(images[idx])
                o_labels[idx, ::2] = self.float_max - labels[idx, ::2]
                o_labels[idx, 1::2] = labels[idx, 1::2]
            else:
                # "origin"
                o_images[idx] = images[idx]
                o_labels[idx] = labels[idx]

        return o_images, o_labels

    def _batch_random_flip_lr(self, images, labels):
        if(images.shape[0] != labels.shape[0]):
            raise Exception("Batch size Error.")
        rand_u = np.random.uniform(0.0, 1.0)
        rand_cond = rand_u > 0.5

        o_images = np.zeros_like(images)
        o_labels = np.zeros_like(labels)

        for idx in xrange(images.shape[0]):
            condition = rand_cond
            if condition:
                # "flip"
                o_images[idx] = np.fliplr(images[idx])
                o_labels[idx, ::2] = self.float_max - labels[idx, ::2]
                o_labels[idx, 1::2] = labels[idx, 1::2]
            else:
                # "origin"
                o_images[idx] = images[idx]
                o_labels[idx] = labels[idx]

        return o_images, o_labels

    def get_random_batch(self, distort=True):

        imgs, labels, begins, filename_list = self.random_batch()
        if distort:
            imgs, labels = self._random_flip_lr(imgs, labels)
            imgs, labels = self._random_roate(imgs, labels, self.degree)
        fm = self.label2fm(labels)

        return (imgs.reshape([self.batch_size, self.img_height, self.img_width, 1]),
                fm.reshape([self.batch_size, self.fm_height, self.fm_width, self.points_num]),
                labels.reshape([self.batch_size, self.points_num * 2]),
                begins,
                filename_list)

    def get_batch(self, distort=True, line_idx=None):

        imgs, labels, begins, filename_list = self.batch(
                line_idx=line_idx)
        if distort:
            imgs, labels = self._batch_random_flip_lr(imgs, labels)
            imgs, labels = self._batch_random_roate(imgs, labels, self.degree)
        fm = self.label2fm(labels)

        return (imgs.reshape([self.batch_size, self.img_height, self.img_width, 1]),
                fm.reshape([self.batch_size, self.fm_height, self.fm_width, self.points_num]),
                labels.reshape([self.batch_size, self.points_num * 2]),
                begins,
                filename_list)



def main():
    pass


if __name__ == "__main__":
    main()

