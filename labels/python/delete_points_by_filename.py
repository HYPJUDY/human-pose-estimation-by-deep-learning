#!/usr/bin/env python
# encoding: utf-8

import re
import sys

def delete_points(src_file, dst_file, delete_list_file):
    with open(src_file, 'rb') as fr, open(dst_file, 'wb') as fw, \
         open(delete_list_file, 'rb') as fd:
        # form a delete filename list
        delete_list = []
        for line in fd:
            filename_line = re.split("-", line.strip())
            if(len(filename_line) == 1): # a filename
                delete_list.append(filename_line[0])
            elif(len(filename_line) == 2): # two or more continuous filenames
                # e.g. "01061471.png-01061540.png"
                f1 = int(filename_line[0][0:-4]) # e.g. 1061471
                f2 = int(filename_line[1][0:-4])
                file_ext = filename_line[0][-4:] # e.g. '.png'
                filename_len = len(filename_line[0]) - 4
                for f in xrange(f1, f2 + 1):
                    delete_list.append(str(f).zfill(filename_len) + file_ext)
            else:
                print line
                print ("Format of delete filename list should be either \
                 'filename' or 'filename#1-filename#N'.")
                sys.exit(0)

        # write each line except those to be deleted in src to dest
        for line in fr:
            tmp = re.split(" |,", line.strip())
            if(len(tmp) != 32):
                print len(tmp)
                print ("Length of Data Error.")
                sys.exit(0)
            filename = tmp[0]
            if filename not in delete_list:
                fw.write(line)

def main():
    src_file = "../txt/input/train_annos_6.txt"
    dst_file = "../txt/input/train_annos_6_cleaned.txt"
    delete_list_file = "../txt/input/train_annos_6_to_delete.txt"
    delete_points(src_file, dst_file, delete_list_file)

if __name__ == "__main__":
    main()
