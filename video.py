# coding=utf-8
# 提供视频流模式与视频模式，视频流模式中按q结束
from ultralytics import YOLO
import cv2,torch
from torchvision import transforms
from crnn_ctc.lib import convert,alphabets
from crnn_ctc.net.CRNN_Net import CRNN
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
import time

def create_video_writer(video_cap, output_filename):
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, 20,(frame_width, frame_height))
    return writer

# load weight
yolo_weight = r'E:\experiment\python\plate_id\yolov11\runs\best.pt' # yolov11权重
ocr_weight = r"E:\experiment\python\plate_id\crnn_ctc\runs\best_weights.pth" # crnn权重
# context = r"E:\Desktop\plate_identification-main\test\images\000.jpg"
context = r"E:\experiment\python\plate_id\test\111.mp4" # 测试视频
output = r'E:\experiment\python\plate_id\2.mp4' # 输出视频位置
device = 'cuda:0'

image = cv2.imread(context)

# Load model
yolo_model = YOLO(yolo_weight,task='detect')
model_ocr = CRNN(class_num=len(alphabets.alphabets)+1).to(device)
model_ocr.load_state_dict(torch.load(ocr_weight, map_location=device))
model_ocr.eval()

# start = time.time()
# # results = yolo_model('rtsp://admin:admin123456@192.168.11.65',stream=True,imgsz=2048)
# results = yolo_model(r"E:\Desktop\111.mp4",stream=True,imgsz=4096,conf=0.6)

# cap = cv2.VideoCapture(context) # 输入视频模式
cap = cv2.VideoCapture('rtsp://admin:admin123456@192.168.11.65') # 视频流模式
writer = create_video_writer(cap, output)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    h,w,c = frame.shape
    frames = frame[4*h//5:,:,:] # cropped
    start = time.time()
    results = yolo_model(frames,imgsz=2048,conf=0.7) # detect

    for re in results:
        img = re.orig_img
        orig_img = frame
        if len(re.boxes.xyxy) == 0:
            # img = cv2.resize(frame, (w // 2, h // 2))
            writer.write(frame.astype(np.uint8))
            cv2.imshow('1', frame)
            continue
        for i in range(len(re.boxes.xyxy)):
            x1, y1, x2, y2 = re.boxes.xyxy[i].long().cpu()
            # boxes_xywh = re.boxes.xywh.long().cpu()
            conf = re.boxes.conf[i].cpu()
            cls = re.boxes.cls[i].cpu()

            crop_img = img[y1:y2, x1:x2, :]
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
            y1 +=4*h//5
            y2 +=4*h//5

            orig_img = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            # 创建一个可以在给定图像上绘图的对象
            draw = ImageDraw.Draw(orig_img)
            # 字体的格式
            fontStyle = ImageFont.truetype("ZpixEX2_SS.ttf", 25, encoding="utf-8")
            # 绘制文本
            draw.text((x1,y1-30), plate, (255,0,0), font=fontStyle)
            # 转换回OpenCV格式
            orig_img = cv2.cvtColor(np.asarray(orig_img), cv2.COLOR_RGB2BGR)
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        # orig_img = cv2.resize(orig_img,(w//2,h//2))
        writer.write(orig_img.astype(np.uint8))
        cv2.imshow('1',orig_img)
    print(1/(time.time()-start),'FPS')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        writer.release()
        cap.release()

writer.release()
cap.release()
