import cv2
import numpy as np
import tensorflow as tf
import csv
import os

class HandDetector():
    def __init__(self, palm_model, anchors_path):
        self.interpreter = tf.lite.Interpreter(model_path = palm_model)
        self.interpreter.allocate_tensors()
        
        with open(anchors_path, "r") as csv_f:
            self.anchors = np.r_[[x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]]

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.imageTargetSize = (256, 256)
    
    def preProcessImage(self, image):
        resized = cv2.resize(image, self.imageTargetSize)
        return resized
    
    @staticmethod
    def _sigm(x):
        return 1 / (1 + np.exp(-x) )

    def resizeOriginal(self, keypoints, center, original_width, original_height):
        kp_orig = np.copy(keypoints)
        cnt_orig = np.copy(center)

        (w, h) = self.imageTargetSize

        kp_orig[:,0] *= original_width/w
        kp_orig[:,1] *= original_height/h

        cnt_orig[0] *= original_width/w
        cnt_orig[1] *= original_height/h

        return (kp_orig, cnt_orig)

    def getCenter(self, keypoints):
        return np.mean(keypoints, axis= 0)

    @staticmethod
    def getBBox(keypoints, center, side_multiplier = 3):
        maxes = np.amax(keypoints, axis = 0)
        mins = np.amin(keypoints, axis = 0)
        sides = np.abs(maxes - mins)
        max_side = np.amax(sides)
        side = side_multiplier * max_side
        x = center[0] - side/2
        y = center[1] - side/2
        return (int(x), int(y), int(side), int(side))

    @staticmethod
    def getHandImage(image, keypoints, x, y, w, h, outsize = 128):
        if x <0 or y <0:
            return (None, None)
        begin = np.array([x,y])
        newKeypoints = keypoints - begin
        newKeypoints[:, 0] *= outsize/w
        newKeypoints[:, 1] *= outsize/h
        newKeypoints = newKeypoints.astype(int)
        crop = image[y:y+h, x:x+w]
        handImage = cv2.resize(crop, (outsize, outsize))
        return (handImage, newKeypoints)

    @staticmethod
    def checkIfFace(x, y, w, h, faces, perc_of_cover):
        area_hand = w * h
        for (xf, yf, wf, hf) in faces:
            dx = min(x + w, xf + wf) - max(x, xf)
            dy = min(y + h, yf + hf) - max(y, yf)
            if (dx>=0) and (dy>=0):
                area_inter = dx * dy
                area_hand = wf * hf
                perc = area_inter/area_hand
                if perc > perc_of_cover or abs(area_hand - area_inter) < 10:
                    return True
        return False


    def __call__(self, image):
        preProcessed = self.preProcessImage(image)
        input_img = tf.expand_dims(tf.image.convert_image_dtype(preProcessed, dtype=tf.float32), 0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_img)

        self.interpreter.invoke()

        out_reg = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        out_clf = self.interpreter.get_tensor(self.output_details[1]['index'])[0,:,0]

        detecion_mask = self._sigm(out_clf) > 0.7

        candidate_detect = out_reg[detecion_mask]

        candidate_anchors = self.anchors[detecion_mask]

        if candidate_detect.shape[0] != 0: 
            max_idx = np.argmax(candidate_detect[:, 3])

            dx,dy,w,h = candidate_detect[max_idx, :4]
            center_wo_offst = candidate_anchors[max_idx,:2] * 256

            keypoints = center_wo_offst + candidate_detect[max_idx,4:].reshape(-1,2)
            
            center = self.getCenter(keypoints)

            (res_kp, res_cnt) = self.resizeOriginal(keypoints, center, image.shape[1], image.shape[0])
            return (res_kp, res_cnt)
        return (None, None)

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
            (x, y, w, h) = detector.getBBox(keypoints, center, 4)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('video', frame)

        if cv2.waitKey(1) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
