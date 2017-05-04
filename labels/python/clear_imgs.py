import os
import sys
import re
import string

clear_txt = open("./data/clear_4.txt")

for line in clear_txt:
    tmp = re.split("-", line.strip())
    #print tmp, len(tmp)

    if len(tmp) == 1:
        print tmp[0]
    else:
        start = string.atoi(tmp[0][4:8])
        end = string.atoi(tmp[1][4:8])
        #print start, end
        while (start != end+1):
            indexStr = str(start)
            l = len(indexStr)
            for i in range(4-l):
                indexStr = '0' + indexStr
            
            filename = tmp[0][0:4] + indexStr + tmp[0][8:12]
            print filename
            os.remove('./data/clear_train_imgs/' + filename)

            start += 1
