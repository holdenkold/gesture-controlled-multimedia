import cv2
import numpy as np
import tensorflow as tf
import csv
import os
from HandDetection import HandDetector
import SkinSegmentation
import GestureRecognition


if __name__ == '__main__':
    curr_dir = os.getcwd()
    palm_model_path = curr_dir + "/models/palm_detection.tflite"
    anchors_path = curr_dir + "/models/anchors.csv"

    #load model
    detector = HandDetector(palm_model_path, anchors_path)

    capture = cv2.VideoCapture(0)  

    hasBackground = False

    while True:
        #get camera feed
        ret, frame = capture.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #save background
        if not hasBackground:
            background = frame
            hasBackground = True

        #detect hand keypoints
        keypoints, center = detector(image)

        source = np.copy(frame)

        if keypoints is not None:
            #visualize detection
            for x, y in keypoints:
                x, y = int(x), int(y)
                source = cv2.circle(source, (x, y), 5, (0, 0, 255))
            source = cv2.circle(source, (int(center[0]),int(center[1])),5, (255, 0, 255))
            (x, y, w, h) = detector.getBBox(keypoints, center, 4)
            source = cv2.rectangle(source, (x, y), (x + w, y + h), (0, 255, 0), 2)

            #extract skin
            handImage = SkinSegmentation.getSkinBackground(frame, background, x, y, w, h, 256)
            if handImage is not None:
                mask = SkinSegmentation.getSkinMask(handImage, 50)
                cont = GestureRecognition.getContour(mask)
                masked = cv2.bitwise_and(handImage,handImage,mask = mask)
                with_cont = cv2.drawContours(masked, [cont], -1, (0,255,0), 3)
                cv2.imshow('hand', with_cont)
        
        cv2.imshow('source', source)

        #get key code if pressed
        key = cv2.waitKey(1)

        if key == 27: #esc
            break

        if key == 32: #space
            background = frame

    capture.release()
    cv2.destroyAllWindows()