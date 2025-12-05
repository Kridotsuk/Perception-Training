import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    horse = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(horse,299,168)
    _, edges = cv.threshold(edges, 150, 255, cv.THRESH_BINARY_INV)
    
    cv.imshow('horse', edges)
    if cv.waitKey(1) == ord('q'):
        break
