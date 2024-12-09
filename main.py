# coding=utf-8

from ultralytics import YOLO
import cv2,torch
from torchvision import transforms
from crnn_ctc.lib import convert,alphabets
from crnn_ctc.net.CRNN_Net import CRNN
from matplotlib.animation import FFMpegWriter
from PIL import Image,ImageDraw,ImageFont
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from celluloid import Camera
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
import time

# load weight
yolo_weight = r'E:\experiment\python\plate_id\yolov11\runs\best.pt'
ocr_weight = r"E:\experiment\python\plate_id\crnn_ctc\runs\train\best_weights.pth"
# context = r"E:\Desktop\plate_identification-main\test\images\000.jpg"
context = r"E:\Desktop\plate_identification-main\test\images\000.jpg"
output = r'E:\experiment\python\plate_id\1.mp4'
device = 'cuda:0'

image = cv2.imread(context)

# Load model
yolo_model = YOLO(yolo_weight,task='detect')
model_ocr = CRNN(class_num=len(alphabets.alphabets)+1).to(device)
model_ocr.load_state_dict(torch.load(ocr_weight, map_location=device))
model_ocr.eval()

start = time.time()
# results = yolo_model('rtsp://admin:admin123456@192.168.11.65',stream=True,imgsz=2048)
results = yolo_model(r"E:\Desktop\111.mp4",stream=True,imgsz=4096,conf=0.6)

for re in results:
    img = re.orig_img
    if len(re.boxes.xyxy)==0:
        continue
    for i in range(len(re.boxes.xyxy)):
        x1,y1,x2,y2 = re.boxes.xyxy[i].long().cpu()
        # boxes_xywh = re.boxes.xywh.long().cpu()
        conf = re.boxes.conf[i].cpu()
        cls = re.boxes.cls[i].cpu()

        crop_img = img[y1:y2,x1:x2,:]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        transformer = transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize((32, 100)),
                                          transforms.Normalize(0.418, 0.141)])
        crop_img = transformer(crop_img)
        crop_img = crop_img.to(device)
        crop_img = crop_img.view(1, 1, 32, 100)
        output = model_ocr(crop_img)

        converter = convert.StrLabelConverter(alphabets.alphabets)
        preds_size = torch.IntTensor([output.size(0)] * output.size(1))
        _, preds = output.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        plate = converter.ocr_decode(preds.data, preds_size.data)

        # plot
        x1, y1, x2, y2 = re.boxes.xyxy[i].cpu().numpy().astype('int32')

        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],color='r')
        plt.text(x1, y1-10, plate,color='r')

    plt.pause(0.005)
    plt.clf()