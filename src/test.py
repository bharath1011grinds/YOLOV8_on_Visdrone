from ultralytics import YOLO


model = YOLO('runs/detect/runs/train/visdrone_yolov8n/weights/best.pt')

img_path = 'dataset/images/val/0000001_03999_d_0000007.jpg'

conf = [0.10, 0.15, 0.20, 0.25, 0.30]

for c in conf:

   result = model.predict(img_path, conf=c, save = True, save_txt = True)
   print(f"{c} : {len(result[0].boxes)} detections")