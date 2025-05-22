from ultralytics import YOLO
import cv2

model = YOLO("runs\\detect\\train\\weights\\best.pt")

def predict(image : cv2.Mat) -> cv2.Mat:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict(image, conf = 0.5)
    annotated = image.copy()
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(annotated, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return annotated
