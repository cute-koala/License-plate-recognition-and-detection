# 文件结构
总体上可以分为两个部分：车牌检测部分（yolov11),车牌识别部分（crnn_ctc）。总体过程是先使用yolo检测车牌所在区域，然后使用crnn网络识别车牌上面的号码。

main.py 与 video.py都可以直接输入视频，区别在于video.py还可以输入摄像头的rtsp流地址，以及将检测识别结果保存成视频，建议使用video.py文件。文件中包含已经训练好的yolov11权重以及crnn网络权重。

所需要的库主要包含ultralytics,pytorch,opencv

# License-plate-recognition-and-detection
