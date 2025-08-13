from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # Load a pretrained YOLOv8 model

results = model.predict(source="3-4恰饭小票.jpg", show=True, conf=0.5, save=True)  # Predict on webcam
