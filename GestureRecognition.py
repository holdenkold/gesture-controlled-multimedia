import cv2
import numpy as np

def getContour(mask):
    mask_copy = np.copy(mask)
    contours, hierarchy = cv2.findContours(mask_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)
    return c
