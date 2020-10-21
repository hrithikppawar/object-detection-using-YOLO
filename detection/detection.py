import cv2
import numpy as np
#from numba import jit, cuda


class Yolo:
    def __init__(self, weightsPath, confPath, labelPath, YoloShape=(416, 416), threshold_confidence = 0.95):
        self.Network = cv2.dnn.readNetFromDarknet(confPath, weightsPath)
        self.ln = self.Network.getUnconnectedOutLayersNames()
        self.labels = open(labelPath).read().strip().split('\n')
        self.YoloShape = YoloShape
        self.threshold_confidence = threshold_confidence

#    @jit(_target='cuda')
    def detect_objects(self, image):
        H, W = image.shape[:2]
        object_location = {}
        pre_image = cv2.dnn.blobFromImage(image, 1 / 255.0, self.YoloShape, swapRB=True)
        self.Network.setInput(pre_image)
        layer_output = self.Network.forward(self.ln)
        for output in layer_output:
            for detection in output:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if float(confidence) > self.threshold_confidence:
                    object = self.labels[classId]
                    (centerX, centerY, width, height) = (detection[:4] * np.array([W, H, W, H])).astype("int")
                    object_location[object] = [centerX, centerY, width, height]

        return object_location

    def draw_objects(self, image, object_location):
        for key, value in object_location.items():
            X, Y = int(value[0] - (value[2] / 2)), int(value[1] - (value[3] / 2))
            cv2.rectangle(image, (X, Y), (X + value[2], Y + value[3]), (0, 0, 255), 2)
            cv2.putText(image, key, (X, Y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
        return image


if __name__ == '__main__':
    weights_path = 'yolov3-608.weights'
    conf_path = 'yolov3.cfg'
    label_path = 'coco.names'
    yolo = Yolo(weights_path, conf_path, label_path, YoloShape=(608, 608), threshold_confidence=0.80)

    ##### Real Time ---

    cap = cv2.VideoCapture(0)
    while True:
        success, image = cap.read()
        objects = yolo.detect_objects(image)
        post_image = yolo.draw_objects(image, objects)
        cv2.imshow('Image Window', post_image)
        cv2.waitKey(True)

    cap.release()
    cv2.destroyAllWindows()

    ###### Using Image ---

    # image = cv2.imread('./chair.jpg')
    # objects = yolo.detect_objects(image)
    # print(objects)
    # post_image = yolo.draw_objects(image, objects)
    # cv2.imshow('Image window', post_image)
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()


