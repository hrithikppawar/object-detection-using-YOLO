import cv2
from detection.detection import Yolo

weights_path = 'yolov3-608.weights'
conf_path = 'yolov3.cfg'
label_path = 'coco.names'
yolo = Yolo(weights_path, conf_path, label_path, YoloShape=(608, 608), threshold_confidence=0.80)

# Real Time Object Detection---

cap = cv2.VideoCapture(0)
while True:
    success, image = cap.read()
    objects = yolo.detect_objects(image)
    post_image = yolo.draw_objects(image, objects)
    cv2.imshow('Image Window', post_image)
    cv2.waitKey(True)

cap.release()
cv2.destroyAllWindows()
