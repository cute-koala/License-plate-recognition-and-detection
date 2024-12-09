import os
import numpy as np
from basicsr.utils import scandir
import random
from tqdm import tqdm

# base_dir = 'E:\experiment\python\plate_id\dataset\images'
# context_list = list(scandir(base_dir,recursive=True))
base_dir = r'E:\experiment\python\plate_id\dataset\ImageSets\val_all.txt'
context_list = np.loadtxt(base_dir,delimiter='\t',dtype=str)
random.seed(0)
num = len(context_list)
train_data_index = random.sample(list(range(num)), int(0.3*num))
with open(r'E:\experiment\python\plate_id\dataset\ImageSets\val.txt','w') as f:
    for index in train_data_index:
        f.write(context_list[index]+'\n')

# random.seed(0)
# num = len(context_list)
# train_data_index = random.sample(range(num), int(0.1*num))
# train_data_file = open(r'E:\experiment\python\plate_id\dataset\ImageSets\train.txt','w')
# val_data_file = open(r'E:\experiment\python\plate_id\dataset\ImageSets\val.txt','w')
# for index,context in enumerate(tqdm(context_list)):
#     if index in train_data_index:
#         train_data_file.write('.\dataset\images\\'+context+'\n')
#     else:
#         val_data_file.write('.\dataset\images\\'+context+'\n')
# train_data_file.close()
# val_data_file.close()

# 补充ccpd中的split文件
# base_dir = 'E:\experiment\python\plate_id\dataset\images\ccpd_np'
# context_list = list(scandir(base_dir,recursive=True))
# with open(r'E:\experiment\python\plate_id\dataset\splits\ccpd_np.txt','w') as f:
#     for line in context_list:
#         f.write("ccpd_np/"+line+'\n')
