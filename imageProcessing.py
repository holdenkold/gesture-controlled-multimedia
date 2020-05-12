import cv2
import numpy as np
import tensorflow as tf
import csv
import os
from datetime import datetime
from pathlib import Path
from HandDetection import HandDetector
import SkinSegmentation

def nothing(arg):
    pass

if __name__ == '__main__':
    curr_dir = os.getcwd()
    palm_model_path = curr_dir + "/models/palm_detection.tflite"
    anchors_path = curr_dir + "/models/anchors.csv"
    Path(curr_dir, 'dataset').mkdir(exist_ok=True)

    #load model
    detector = HandDetector(palm_model_path, anchors_path)

    capture = cv2.VideoCapture(0)  

    hasBackground = False
    mask = None
    
    cv2.namedWindow('source')
    cv2.createTrackbar('Threshhold','source',60,254,nothing)

    cv2.createTrackbar('Label','source',0,6,nothing)

    photo_counter = 0

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
                thresh = cv2.getTrackbarPos('Threshhold','source')
                mask = SkinSegmentation.getSkinMask(handImage, thresh)
                cv2.imshow('hand', mask)
        
        cv2.imshow('source', source)

        #get key code if pressed
        key = cv2.waitKey(1)

        if key == 27: #esc
            break

        if key == 32: #space
            background = frame

        # Save frame on click
        if mask is not None and key != -1 and key != 32:
            date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            lbl = cv2.getTrackbarPos('Label','source')
            filename = '{}_{}.png'.format(lbl, date_time)
            path = os.path.join(curr_dir, 'dataset', filename)
            cv2.imwrite(path, mask)
            photo_counter+= 1
            print(f"Frame saved! nr {photo_counter}" + path)

    capture.release()
    cv2.destroyAllWindows()