from ultralytics import YOLO

# Load a model
model = YOLO(r"E:\experiment\python\plate_id\yolov11\runs\best.pt",task='detect')

# # Train the model
# train_results = model.train(
#     data="CCPD_data.yaml",  # path to dataset YAML
#     epochs=30,  # number of training epochs
#     imgsz=640,  # training image size
#     device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
#     workers=0,
#     batch=16 ,
#     lr0=0.001,
#     save_period=1,
#     plots=True,
#     amp=False
# )

# Evaluate model performance on the validation set
# metrics = model.val()

# Perform object detection on an image
results = model(r"C:\Users\gg\Web\CaptureFiles\2024-12-06\192.168.11.65_01_20241206114512618.jpg",imgsz=2048)

for r in results:
    r.show()


# Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model

