import random
import os
import sys
import time
import numpy as np


def genertor():
    Point=[random.uniform(123.449169,123.458654),random.uniform(41.740567,41.743705)]
    arr = []
    for i in range(1, random.randint(0, 500)):
        bias = np.random.randn() * pow(10,-4)
        bias = round(bias,4)
        X = Point[0] + bias
        bias1 = np.random.randn() * pow(10,-4)
        bias1 = round(bias,4)
        Y = Point[1] + bias
        time_str=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
        arr.append(['13888888888'+'\t',str(X)+',', str(Point[1])+'\t',time_str])
    return arr


if __name__ == '__main__':
    path = sys.argv[1]
    if not os.path.isfile(path):
        open(path, 'w')
    with open(path,'a') as f:
        while True:
            arr = genertor()
            for i in range(len(arr)):
                f.writelines(arr[i])
                f.write('\n')
            time.sleep(5)
