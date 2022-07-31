import cv2
from cvzone.HandTrackingModule import HandDetector
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    cv2.imshow("image", img)
    cv2.waitKey(1)
