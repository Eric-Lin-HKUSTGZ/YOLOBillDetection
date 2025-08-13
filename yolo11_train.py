from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # Load a pretrained YOLOv8 model
results = model.train(data="/hpc2hdd/home/qxiao183/linweiquan/detection_train_data/invoice_data/data.yaml", epochs=100, imgsz=640)  # Train the model