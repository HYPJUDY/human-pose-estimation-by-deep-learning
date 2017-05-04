#!/usr/bin/env python
# encoding: utf-8

import re
import sys

def _get_point(from_list, idx):
    return [from_list[idx*2], from_list[idx*2 + 1]]

def delete_points(src_file, dst_file):
    def del_points(from_list):
        delete = [2, 4, 7, 11]
        to_list = list()
        for idx in xrange(19):
            if idx in delete:
                continue
            to_list += _get_point(from_list, idx)
        return to_list

    with open(src_file, 'rb') as fr, open(dst_file, 'wb') as fw:
        for line in fr:
            tmp = re.split(" |,", line.strip())
            if(len(tmp) != 40):
                print len(tmp)
                print ("Length of Data Error.")
                sys.exit(0)
            filename = tmp[0]
            coords = tmp[1:39]
            begin = tmp[39]
            coords = del_points(coords)

            fw.write(filename + ' ')
            for item in coords:
                fw.write(item + ',')
            fw.write(begin + '\n')

def main():
    src_file = "../txt/input/7.txt"
    dst_file = "../txt/input/train_annos_7.txt"
    delete_points(src_file, dst_file)

if __name__ == "__main__":
    main()
