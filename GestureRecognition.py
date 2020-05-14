import cv2
import numpy as np
from collections import Counter

class GestureAccepter():
    def __init__(self, k, capacity):
        self.acceptance_level = k
        self.last_gestures = list()
        self.capacity = capacity
        self.stats = [0 for _ in range(7)]
        self.gest_map = {0:'None', 1: 'play / pause', 2:'mute', 3:'previous track', 4:'next track', 5:'volume up', 6:'volume down'}

    def recognise_gesture(self, gesture):
        self.last_gestures.append(np.argmax(gesture))
        for i in range(len(gesture)):
            self.stats[i]+= gesture[i]

        if len(self.last_gestures) == self.capacity:
            return self.accept_gesture()

    def accept_gesture(self):
        max_gesture = self.stats.index(max(self.stats))
        print('1 way')
        print(self.gest_map[max_gesture])
        print()

        c = Counter(self.last_gestures)
        max_gesture = max(c, key=c.get)
        print('2 way')
        print(self.gest_map[max_gesture])
        print()

        self.last_gestures = list()
        self.stats = [0 for _ in range(7)]

        if max_gesture != 0:
            return self.gest_map[max_gesture]
        


    
        





