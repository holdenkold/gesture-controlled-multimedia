import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import csv
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

    # Load the cascade
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

    #load model
    detector = HandDetector(palm_model_path, anchors_path)
    gesture_model = keras.models.load_model('models/model_v1')

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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #save background
        if not hasBackground:
            background = frame
            hasBackground = True

        #detect hand keypoints
        keypoints, center = detector(image)

        #detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        source = np.copy(frame)

        if keypoints is not None:
            #get hand box
            (x, y, w, h) = detector.getBBox(keypoints, center, 4)
            #check if not face
            if not HandDetector.checkIfFace(x, y, w, h, faces, 0.3):
                #visualize detection
                for px, py in keypoints:
                    px, py = int(px), int(py)
                    source = cv2.circle(source, (px, py), 5, (0, 0, 255))
                source = cv2.circle(source, (int(center[0]),int(center[1])),5, (255, 0, 255))
                source = cv2.rectangle(source, (x, y), (x + w, y + h), (0, 255, 0), 2)

            #extract skin
            handImage = SkinSegmentation.getSkinBackground(frame, background, x, y, w, h, 256)
            if handImage is not None:
                thresh = cv2.getTrackbarPos('Threshhold','source')
                mask = SkinSegmentation.getSkinMask(handImage, thresh)
                cv2.imshow('hand', mask)

                # Prediction
                img_shape = (28, 28)
                mask_norm = mask // 255
                im = cv2.resize(mask_norm, img_shape)
                rshp = np.reshape(im, (1, 28, 28, 1))
                pred = gesture_model.predict(rshp)

                argm = np.argmax(pred[0])

                y0, dy = 50, 20
                for i, line in enumerate(pred[0]):
                    y = y0 + i*dy
                    
                    maxind = '   '
                    if i == argm:
                        maxind = 'MAX'

                    txt = '{} {} {:f}'.format(maxind, i, line)
                    cv2.putText(source, txt, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, 4)

                print(pred)

        
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