# Sim
YOLO
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'C:\ultralytics-main(original)\ultralytics\cfg\models\v8\yolov8.yaml').load(r'yolov8n.pt')
    model.train(data=r"C:\ultralytics-main(original)\ultralytics\cfg\datasets\coco128.yaml",
        epochs=2000,
        batch=64,                                             
        imgsz=640,
        lr0=0.01,
        lrf=0.01,
        fliplr=0,
        patience=100,)
