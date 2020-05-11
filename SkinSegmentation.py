import cv2
import numpy as np

def mean_colors_hsv(image, keypoints):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv_image)

    mask = getPixelMask(hsv_image, int(image.shape[0]/2), int(image.shape[0]/2), 2)

    for x, y in keypoints:
        kp_mask = getPixelMask(hsv_image, int(x), int(y), 10)
        mask = mask | kp_mask

    kernel = np.ones((3,3),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 2)
    #mask = cv2.erode(mask,kernel,iterations = 1)
    #mask = cv2.GaussianBlur(mask,(3,3),50)
    return mask

def getPixelMask(image_hsv, x,y, hd):
    point = image_hsv[x,y]
    lower = np.array([point[0] - hd, 20, 70])
    upper = np.array([point[0] + hd, 255, 255])
    mask = cv2.inRange(image_hsv, lower, upper)
    kernel = np.ones((3,3),np.uint8)
    #mask = cv2.dilate(mask,kernel,iterations = 2)
    #mask = cv2.erode(mask,kernel,iterations = 2)
    return mask

def getSkinMask(image, thresh):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray,thresh,255,cv2.THRESH_BINARY)[1]
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def getSkinBackground(image, background, x, y, w, h, outsize = 128):
    if x <0 or y <0:
        return None
    difference = cv2.absdiff(image, background)

    crop = difference[y:y+h, x:x+w]

    handImage = cv2.resize(crop, (outsize, outsize))
    return handImage