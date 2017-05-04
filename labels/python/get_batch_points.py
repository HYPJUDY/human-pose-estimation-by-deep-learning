#!/usr/bin/env python
# encoding: utf-8

import re
import sys

def _get_point(from_list, idx):
    return [from_list[idx*2], from_list[idx*2 + 1]]

def get_batch_points(src_file, dst_file):
    # get half of data
    count = 0
    with open(src_file, 'rb') as fr, open(dst_file, 'wb') as fw:
        for line in fr:
            if count == 0:
                fw.write(line)
                count = 1
            else:
                count = 0

def main():
    src_file = "../txt/input/train_annos_mix_cleaned.txt"
    dst_file = "../txt/input/train_annos_mix_half_cleaned.txt"
    get_batch_points(src_file, dst_file)

if __name__ == "__main__":
    main()
