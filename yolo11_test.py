from ultralytics import YOLO
import cv2
import os

model = YOLO("/hpc2hdd/home/qxiao183/linweiquan/yolo_ultralytics/runs/detect/train/weights/best.pt")  # Load a pretrained YOLOv8 model

results = model(['/hpc2hdd/home/qxiao183/linweiquan/yolo_ultralytics/test1.jpg'])

orig_img = cv2.imread('/hpc2hdd/home/qxiao183/linweiquan/yolo_ultralytics/test1.jpg')
for result in results:
    boxes = result.boxes
    # print("boxes:", boxes.xyxy)
    for i, box in enumerate(boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        h, w = orig_img.shape[:2]
        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(w,x2)
        y2 = min(h,y2)

        # 裁剪
        crop = orig_img[y1:y2, x1:x2]
        # 保存
        save_path = f"/hpc2hdd/home/qxiao183/linweiquan/yolo_ultralytics/result_box_{i}.jpg"
        cv2.imwrite(save_path, crop)
        print(f"Saved: {save_path}")
    result.show()
    result.save(filename="result.jpg")