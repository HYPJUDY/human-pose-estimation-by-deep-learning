import os
import sys
import re
import string

#clear_labels = open("./labels/txt/clear_annos.txt", 'r')
#clear_txt = open("./data/clear_4.txt")

old_labels = open("./labels/txt/4.txt")
new_labels = open("./labels/txt/validate_annos.txt", 'w')

# for line in clear_txt:
#     tmp = re.split("-", line.strip())
#     #print tmp, len(tmp)

#     if len(tmp) == 1:
#         print tmp[0]
#         clear_labels.write(tmp[0] + '\n')
#     else:
#         start = string.atoi(tmp[0][4:8])
#         end = string.atoi(tmp[1][4:8])
#         #print start, end

#         while (start != end+1):
#             indexStr = str(start)
#             l = len(indexStr)
#             for i in range(4-l):
#                 indexStr = '0' + indexStr
            
#             filename = tmp[0][0:4] + indexStr + tmp[0][8:12]
#             clear_labels.write(filename + '\n')
            
#             #print filename
#             start += 1

for line in old_labels:
    tmp = re.split(" |,", line.strip())
    filename = tmp[0] + '\n'

    flag = True

    clear_labels = open("./labels/txt/clear_annos.txt", 'r')
    for clear_line in clear_labels:
        if filename[0:len(filename)] == clear_line[0:len(clear_line)]:
           #print filename, clear_line
           flag = False
           break
    
    if flag:
        new_labels.write(line.strip() + '\n')
