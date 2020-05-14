import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import csv
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image
import urllib.request
from pathlib import Path
from HandDetection import HandDetector
from spotifyIntegration import SpotifyClient
import SkinSegmentation
from GestureRecognition import GestureAccepter

SHOW_SPOTIFY_INFO = False
CONNECT_TO_SPOTIFY = False
CREATE_DATA_SET = False
SHOW_MODEL_PREDICTIONS = True

def nothing(arg):
    pass

def drawtext(img, osd_list, bgracolor=(255,255,255,0)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    font_text = ImageFont.truetype('Arial.ttf', 24, encoding="utf-8")
    for txt, coords in osd_list:
        draw.text(coords, txt, fill=bgracolor, font=font_text)
    img = np.array(img_pil)
    return img

if __name__ == '__main__':
    curr_dir = os.getcwd()
    palm_model_path = curr_dir + "/models/palm_detection.tflite"
    anchors_path = curr_dir + "/models/anchors.csv"

    # Load the cascade
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

    #load model
    detector = HandDetector(palm_model_path, anchors_path)
    gesture_model = keras.models.load_model('models/model_v1')

    #load GestureAccepter
    gesture_accepter = GestureAccepter(5, 15)

    # load Spotify client
    if CONNECT_TO_SPOTIFY:
        spclient = SpotifyClient()
        me = spclient.me()
        st = spclient.status()
        if(st is None):
            raise ConnectionError("Can't connect to Spotify")

    capture = cv2.VideoCapture(0)  

    hasBackground = False
    mask = None
    
    cv2.namedWindow('source')
    cv2.createTrackbar('Threshhold','source',60,254,nothing)
    
    if CREATE_DATA_SET:
        cv2.createTrackbar('Label','source',0,6,nothing)
        Path(curr_dir, 'dataset').mkdir(exist_ok=True)

    photo_counter = 0

    album_cover = None
    album_cover_src = None

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
        osd = []

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

                    # Model output (array of classes probabilities)
                    pred = gesture_model.predict(rshp)

                    recognised_gesture = gesture_accepter.recognise_gesture(pred[0])

                    if recognised_gesture is not None:
                        osd.append((recognised_gesture, (450, 50)))


                    # Index of maximum probability
                    argmax = np.argmax(pred[0])

                    # Display the output
                    if SHOW_MODEL_PREDICTIONS:
                        y0, dy = 50, 20
                        for i, line in enumerate(pred[0]):
                            y = y0 + i*dy
                            maxind = 'MAX' if i == argmax else '   '
                            txt = '{} {} {:f}'.format(maxind, i, line)
                            osd.append((txt, (50, y)))
        
        # Apply Spotify info
        if (CONNECT_TO_SPOTIFY and SHOW_SPOTIFY_INFO):
            name = me['display_name']
            st = spclient.status()
            playing = st['is_playing']
            track_name = st['item']['name']
            artist_name = st['item']['artists'][0]['name']
            album_cover_src_new = st['item']['album']['images'][2]

            if album_cover_src_new != album_cover_src:
                album_cover_src = album_cover_src_new
                album_cover_pil = Image.open(urllib.request.urlopen(album_cover_src['url']))
                album_cover = np.array(album_cover_pil.convert('RGB'))

            x_offset=source.shape[1]-50-album_cover.shape[1]
            y_offset=50

            source[y_offset:y_offset+album_cover_src['height'], x_offset:x_offset+album_cover_src['width']] = album_cover

            x_pos = source.shape[1]-450
            y_pos = 50
            dy = 30

            osd.append((track_name, (x_pos, y_pos)))
            osd.append((artist_name, (x_pos, y_pos+dy)))
            osd.append(("Logged as: "+name, (x_pos, y_pos+2*dy)))

        source = drawtext(source, osd)
        cv2.imshow('source', source)

        #get key code if pressed
        key = cv2.waitKey(1)

        if key == 27: #esc
            break

        if key == 32: #space
            background = frame

        # Save frame on click
        if CREATE_DATA_SET and mask is not None and key != -1 and key != 32:
            date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            lbl = cv2.getTrackbarPos('Label','source')
            filename = '{}_{}.png'.format(lbl, date_time)
            path = os.path.join(curr_dir, 'dataset', filename)
            cv2.imwrite(path, mask)
            photo_counter+= 1
            print(f"Frame saved! nr {photo_counter}" + path)

    capture.release()
    cv2.destroyAllWindows()