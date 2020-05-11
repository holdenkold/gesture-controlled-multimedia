import cv2
import numpy as np

def mean_colors_hsv(image, keypoints):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H,S,V = cv2.split(hsv_image)

    keypoints_ok = [point for point in keypoints 
                    if point[0] >= 0 and point[0] < image.shape[1] 
                    and point[1] >= 0 and point[1] < image.shape[0]]

    hh = sorted(H[int(y)][int(x)] for x, y in keypoints_ok)
    ss = sorted(S[int(y)][int(x)] for x, y in keypoints_ok)
    vv = sorted(V[int(y)][int(x)] for x, y in keypoints_ok)

    lower = np.array([hh[0], ss[0], 0])
    upper = np.array([hh[-1], ss[-1], 255])

    mask = cv2.inRange(hsv_image, lower, upper)
    return mask