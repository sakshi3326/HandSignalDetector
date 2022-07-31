import time

import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgsize = 300

folder = "Data/C"
counter =0
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y , w, h = hand['bbox']

        imgWhite = np.ones((imgsize,imgsize,3),np.uint8)*255

        imgcrop = img[y-offset:y+h+offset,x-offset:x+w+offset]



        aspectRAtio = h/w
        if aspectRAtio>1:
            k = imgsize/h
            wcal = math.ceil(k*w)
            imgresize = cv2.resize(imgcrop,(wcal,imgsize))
            wGap = math.ceil((imgsize-wcal)/2)

            imgWhite[:, wGap:wcal+wGap] = imgresize
        else:
            k = imgsize / w
            hcal = math.ceil(k * h)
            imgresize = cv2.resize(imgcrop, (imgsize,hcal))
            hGap = math.ceil((imgsize - hcal) / 2)

            imgWhite[hGap:hcal + hGap, :] = imgresize

        cv2.imshow("imageCrop", imgcrop)
        cv2.imshow("imageWhite", imgWhite)


    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)



