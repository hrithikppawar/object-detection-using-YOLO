import cv2
from detection.detection import Yolo

# Object Detection From image
weights_path = 'yolov3-608.weights'
conf_path = 'yolov3.cfg'
label_path = 'coco.names'
yolo = Yolo(weights_path, conf_path, label_path, YoloShape=(608, 608), threshold_confidence=0.80)


image = cv2.imread('./chair.jpg')
objects = yolo.detect_objects(image)
print(objects)
post_image = yolo.draw_objects(image, objects)
cv2.imshow('Image window', post_image)
cv2.waitKey(10000)
cv2.destroyAllWindows()
