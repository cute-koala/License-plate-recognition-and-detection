# 作者：水果好好吃哦
# 日期：2023/8/19
# ccpd_base: 21000
# ccpd_blur: 2100
# ccpd_challenge: 2100
# ccpd_db: 2100
# ccpd_fn: 2100
# ccpd_rotate: 2100
# ccpd_tilt: 2100
# ccpd_weather: 2100

from shutil import copyfile
import os
import random


def select_data(src_path, dst_train_path_1, dst_val_path_2):
    dirs = os.listdir(src_path)
    num = len(dirs)
    random.seed(0)
    train_data_index = random.sample(range(len(dirs)), int(num*0.1))
    for index,image in enumerate(dirs):
        if index in train_data_index:
            copyfile(src_path + f"{image}", dst_train_path_1 + f"{image}")
        else:
            copyfile(src_path + f"{image}", dst_val_path_2 + f"{image}")


if __name__ == "__main__":
    # CCPD数据集文件夹位置
    root = r"E:/Desktop/CCPD2019/"
    base_path = root + "ccpd_base/"
    blur_path = root + "ccpd_blur/"
    challenge_path = root + "ccpd_challenge/"
    db_path = root + "ccpd_db/"
    fn_path = root + "ccpd_fn/"
    rotate_path = root + "ccpd_rotate/"
    tilt_path = root + "ccpd_tilt/"
    weather_path = root + "ccpd_weather/"
    dic = {base_path: 21000, blur_path: 2100, challenge_path: 2100, db_path: 2100, fn_path: 2100, rotate_path: 2100,
           tilt_path: 2100, weather_path: 2100}
    # 训练集路径
    dst_train_path = r"E:\experiment\python\plate_id\dataset\CCPD_data\train\images\\"
    # 评估集路径
    dst_val_path = r"E:\experiment\python\plate_id\dataset\CCPD_data\val\images\\"
    for path in dic:
        print(path)
        select_data(path, dst_train_path_1=dst_train_path, dst_val_path_2=dst_val_path)
