from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('yolov8n.pt') 
    results = model.train(
        data=r"C:\Users\mahad\Desktop\Vrishin's Office\python\baitsense-byop\yolo\dataset.yaml", 
        epochs=50, 
        imgsz=640, 
        batch=16, 
        name='my_custom_yolo')