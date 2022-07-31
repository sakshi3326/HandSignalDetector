import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgsize = 300
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y , w, h = hand['bbox']

        imgWhite = np.ones((imgsize,imgsize,3),np.uint8)*255

        imgcrop = img[y-offset:y+h+offset,x-offset:x+w+offset]

        imgWhite[0:imgcrop.shape[0], 0:imgcrop.shape[1]] = imgcrop
        cv2.imshow("imageCrop", imgcrop)
        cv2.imshow("imageWhite", imgWhite)


    cv2.imshow("image", img)
    cv2.waitKey(1)


