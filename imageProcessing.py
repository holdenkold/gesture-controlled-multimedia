import cv2
import numpy as np
import tensorflow as tf
import csv
import os
from datetime import datetime
from pathlib import Path
from HandDetection import HandDetector
import SkinSegmentation


if __name__ == '__main__':
    curr_dir = os.getcwd()
    palm_model_path = curr_dir + "/models/palm_detection.tflite"
    anchors_path = curr_dir + "/models/anchors.csv"
    Path(curr_dir, 'dataset').mkdir(exist_ok=True)

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
        mask = None

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
                mask = SkinSegmentation.getSkinMask(handImage, 60)
                cv2.imshow('hand', mask)
        
        cv2.imshow('source', source)

        #get key code if pressed
        key = cv2.waitKey(1)

        if key == 27: #esc
            break

        if key == 32: #space
            background = frame

        # Save frame on click
        if mask is not None and key != -1:
            date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = '{:0>3}_{}.png'.format(key, date_time)
            path = os.path.join(curr_dir, 'dataset', filename)
            cv2.imwrite(path, mask)
            print("Frame saved! " + path)

    capture.release()
    cv2.destroyAllWindows()