import cv2
import numpy as np
import tensorflow as tf
import csv
import os
from HandDetection import HandDetector
import SkinSegmentation




if __name__ == '__main__':
    curr_dir = os.getcwd()
    palm_model_path = curr_dir + "/models/palm_detection.tflite"
    anchors_path = curr_dir + "/models/anchors.csv"

    detector = HandDetector(palm_model_path, anchors_path)
    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints, center = detector(image)

        if keypoints is not None:
            for x, y in keypoints:
                x, y = int(x), int(y)
                frame = cv2.circle(frame, (x, y), 5, (0, 0, 255))

            frame = cv2.circle(frame, (int(center[0]),int(center[1])),5, (255, 0, 255))
            (x, y, w, h) = detector.getBBox(keypoints, center, 3)
            frame = cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

            points = np.vstack([keypoints, center])
            mask = SkinSegmentation.mean_colors_hsv(image, points)
            frame = cv2.bitwise_and(frame,frame, mask=mask)

        cv2.imshow('video', frame)

        if cv2.waitKey(1) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()