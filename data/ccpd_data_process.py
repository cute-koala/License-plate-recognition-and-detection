# 作者：水果好好吃哦
# 日期：2023/8/19
import numpy as np
import os
import cv2
from PIL import Image
from tqdm import tqdm

def imgname2bbox(images_path, labels_path_):
    # dirs = os.listdir(images_path)
    ccpd_base_dir = 'E:\experiment\python\plate_id\dataset\splits'
    ccpds = os.listdir(ccpd_base_dir)
    for d in ccpds:
        if d not in ['ccpd_base.txt']:
        # if d not in ['ccpd_weather.txt']:
            continue
        print(d)
        dirs = np.loadtxt(os.path.join(ccpd_base_dir,d),delimiter='\t',dtype=str)

        for image_ in tqdm(dirs):
            dir = image_.split("/")[0]
            if not os.path.exists(os.path.join(labels_path_,dir)):
                os.mkdir(os.path.join(labels_path_,dir))
            labels_path = os.path.join(labels_path_,dir)+"/"
            image = image_.split("/")[-1]
            image_name = image.split(".")[0]
            box = image_name.split("-")[2]
            # 边界框信息
            box = box.split("_")
            box = [list(map(int, i.split('&'))) for i in box]
            # 图片信息
            image_path = f"{images_path}/{image_}"
            # img = cv2.imread(image_path)
            with Image.open(image_path) as img:
                w,h = img.size
            with open(labels_path + image_name+".txt", "w") as f:
                x_min, y_min = box[0]
                x_max, y_max = box[1]
                x_center = (x_min + x_max) / 2 / w
                y_center = (y_min + y_max) / 2 / h
                width = (x_max - x_min) / w
                height = (y_max - y_min) / h
                f.write(f"0 {x_center:.6} {y_center:.6} {width:.6} {height:.6}")


def ccpd_data2ccpd_plate_data(images_path, plate_images_path):
    dirs = os.listdir(images_path)
    for image in tqdm(dirs):
        # 读取图片
        img = cv2.imread(f"{images_path}\\{image}")
        # 图片名字
        image_name = image.split(".")[0]
        # 车牌的四个角点信息
        points = image_name.split("-")[3]
        points = points.split('_')
        points = [list(map(int, i.split('&'))) for i in points]
        # 将关键点的顺序变为从左上顺时针开始
        points = points[-2:] + points[:2]
        # 在图像上绘制车牌的四个角点
        # for i, pt in enumerate(points):
        #    cv2.circle(img, pt, 5, (0, 222, 0), -1)
        # 原车牌角点数组
        pst1 = np.float32(points)
        # 变换后的车牌角点数组
        x_min, x_max = min(pst1[:, 0]), max(pst1[:, 0])
        y_min, y_max = min(pst1[:, 1]), max(pst1[:, 1])
        pst2 = np.float32([(0, 0), (x_max - x_min, 0), (x_max - x_min, y_max - y_min), (0, y_max - y_min)])
        matrix = cv2.getPerspectiveTransform(pst1, pst2)
        plate = cv2.warpPerspective(img, matrix, (int(x_max - x_min), int(y_max - y_min)))
        cv2.imwrite(os.path.join(plate_images_path,f"plate_{image}"), plate)


# 省份列表，index对应ccpd
province_list = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "西", "陕", "甘", "青", "宁", "新"]
# 字母数字列表，index对应ccpd
word_list = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def imgname2plate_label(images_path, plate_labels_path):
    dirs = os.listdir(images_path)
    lst = []
    for image in dirs:
        # 图片名字
        image_name = image.split(".")[0]
        # 车牌的文字信息
        label = image_name.split("-")[4]
        # 读取车牌号
        label = label.split("_")
        province = province_list[int(label[0])]
        words = [word_list[int(i)] for i in label[1:]]
        # 车牌号
        label = province + "".join(words)
        lst.append(f"{image}---{label}")
    with open(os.path.join(plate_labels_path, "imgnames_labels.txt"), "w") as f:
        for line in lst:
            f.write(f"plate_{line}\n")


if __name__ == "__main__":
    images_train_path = "E:\Desktop\dataset\images\ccpd_base" # 图片添加
    # labels_train_path = "E:\experiment\python\plate_id\crnn_ctc\data" # 标签要保存的位置

    plate_images_train_path = "E:\Desktop\dataset\crnn_images"
    plate_labels_train_path = "E:\Desktop\dataset\crnn_labels"

    # 从图片名字中提取ccpd的边界框信息，即(c, x, y, w, h)
    # dic_images = {0: images_train_path, }
    # dic_labels = {0: labels_train_path, }
    # for i in dic_images:
    #     imgname2bbox(dic_images[i], dic_labels[i])


    # 从ccpd数据集中提取车牌数据集 # 车辆数据集转车牌数据集
    dic_images = {0: images_train_path}
    dic_plate_images = {0: plate_images_train_path}
    for i in dic_images:
        ccpd_data2ccpd_plate_data(dic_images[i], dic_plate_images[i])

    dic_images = {0: images_train_path}
    dic_plate_labels = {0: plate_labels_train_path}
    for i in dic_images:
        imgname2plate_label(dic_images[i], dic_plate_labels[i])

