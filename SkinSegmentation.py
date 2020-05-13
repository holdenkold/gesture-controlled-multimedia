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
    size = image.shape
    new_x = x
    new_y = y
    temp_image = image
    temp_background = background
    if x <0 or y <0 or x + w > size[1] or y + h > size[0]:
        pad_length = max(-x, -y, (x + w) - size[1], (y + h) - size[0])
        temp_image = cv2.copyMakeBorder(image, pad_length, pad_length, pad_length,pad_length, cv2.BORDER_CONSTANT, (0, 0, 0))
        temp_background = cv2.copyMakeBorder(background, pad_length, pad_length, pad_length,pad_length, cv2.BORDER_CONSTANT, (0, 0, 0))
        new_x = x + pad_length
        new_y = y + pad_length
    difference = cv2.absdiff(temp_image, temp_background)

    crop = difference[new_y:new_y+h, new_x:new_x+w]

    handImage = cv2.resize(crop, (outsize, outsize))
    return handImage