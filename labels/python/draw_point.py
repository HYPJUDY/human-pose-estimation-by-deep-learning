#!/usr/bin/env python
# encoding: utf-8

import cv2
import numpy as np
import os
import sys
import re
import matplotlib.pyplot as plt


def get_point(a_list, idx):
    # w, h = a_list[idx * 2: idx * 2 + 2]
    w = float(a_list[idx * 2]) * 256
    h = float(a_list[idx * 2 + 1]) * 212
    return int(float(w)), int(float(h))


def draw_imgs(src_dir, dst_dir, annos_file):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    count = 0
    with open(annos_file, 'rb') as fr:
        for line in fr:
            count += 1
            if (count % 100 == 0):
                print ("Processing ", count, " images.")

            tmp = re.split(" |,", line.strip())
            if(len(tmp) != 32): # 40
                print(len(tmp))
                print ("Length of Data Error")
                sys.exit(0)
            filename = tmp[0]
            coors = tmp[1:39]
            img = cv2.imread(src_dir + filename, 0)
            for idx in xrange(15): # 19
                w, h = get_point(coors, idx)
                cv2.circle(img, (w, h), 2, 255)
            #plt.imshow(img, cmap='gray')
            #plt.show()
            cv2.imwrite(dst_dir + filename, img)

def main():
    # =========== Specify parameters =====================
    TAG = "_demo" # used for uniform filename
                  # "_demo": train with demo images
                  # "": (empty) train with ~60000 images
    PHASE = "train"  # "train" or "test"
    # ====================================================

    src_dir = "../../data/input/" + PHASE + "_imgs" + TAG + "/"
    dst_dir = "../../data/output/" + PHASE + "_imgs" + TAG + "/"
    annos_file = "../txt/output/" + PHASE + "_annos" + TAG + ".txt"
    
    print ("start")
    draw_imgs(src_dir, dst_dir, annos_file)
    print ("end")


if __name__ == "__main__":
    main()
