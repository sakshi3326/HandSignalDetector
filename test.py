import time

import cv2
import numpy as np
import math
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

classifier = Classifier("model/keras_model.h5", "model/labels.txt")
offset = 20
imgsize = 300

folder = "Data/C"
counter =0

labels = ["A", "B","C"]
while True:
    success, img = cap.read()
    imgout = img.copy()
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
            prediction, index =classifier.getPrediction(imgWhite,draw=False)
            print(prediction,index)

        else:
            k = imgsize / w
            hcal = math.ceil(k * h)
            imgresize = cv2.resize(imgcrop, (imgsize,hcal))
            hGap = math.ceil((imgsize - hcal) / 2)
            imgWhite[hGap:hcal + hGap, :] = imgresize
            prediction, index = classifier.getPrediction(imgWhite,draw=False)
        cv2.putText(imgout,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        cv2.rectangle(imgout,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)
        cv2.imshow("imageCrop", imgcrop)
        cv2.imshow("imageWhite", imgWhite)


    cv2.imshow("image", imgout)
    key = cv2.waitKey(1)




