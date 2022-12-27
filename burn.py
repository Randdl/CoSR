# use this script to detect the gpu memory and buring the 3 most free gpus
import os
cmd = "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pynvml"
os.system(cmd)
import pynvml
import time
import os
import numpy as np
pynvml.nvmlInit()
import torch


def detect_memory(gpu_id):
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_id))
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total = meminfo.total/(1024*1024)
    free = meminfo.free/(1024*1024)
    free_rate = free/total
    return free_rate

if __name__ == '__main__':
    while True:
        mean_rate=detect_memory(0)
        # rate1=detect_memory(1)
        # rate2=detect_memory(2)
        # rate3=detect_memory(3)
        # rate4=detect_memory(4)
        # rate5=detect_memory(5)
        # rate6=detect_memory(6)
        # rate7=detect_memory(7)
        # rate_list = [rate0]
        # mean_rate = np.mean(rate_list)
        print("mean_rate", mean_rate)
        # sorted_nums = sorted(enumerate(rate_list), key=lambda x: x[0])
        # idx = [i[0] for i in sorted_nums]
        # nums = [i[1] for i in sorted_nums]
        if mean_rate > 0.2:
            print('buring gpu')
            a1=torch.zeros(4024,4024).cuda()
            while(True):
                b1=torch.matmul(a1,a1)
        time.sleep(1800)
#
