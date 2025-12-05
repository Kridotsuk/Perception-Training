import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

square = cv.imread('shape.jpg')

img = cv.cvtColor(square, cv.COLOR_BGR2GRAY)

_, blurred = cv.threshold(img,240,255,cv.THRESH_BINARY_INV)
kernel = np.ones((5,5), np.float32)/25
#blurred = cv.bilateralFilter(thresh, 9, 75,75)
#blurred = cv.Canny(thresh, 200, 100)

contours, _ = cv.findContours(blurred,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

minArea = 1000
shapes = []
shape_names = []

for cnt in contours:
    if cv.contourArea(cnt) > minArea:
        epsilon = 5
        approx = cv.approxPolyDP(cnt,epsilon,True)
        cv.drawContours(square, [approx], 0, (0,255,0), 3)
        shapes.append(approx)
        x,y,w,h = cv.boundingRect(cnt)
        M = cv.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        regular = False
        rectangle = False
        ellipse = False
        if 0.9 <= w/h <= 1.1:
            regular = True
        if 0.9 <= w*h/cv.contourArea(cnt) <= 1.1:
            rectangle = True
        if 0.9 <= (w*h/cv.contourArea(cnt))/(4/np.pi) <=1.1:
            ellipse = True            
        if len(approx) == 3:
            shape_names.append("triangle")
        elif len(approx) == 4:
            if rectangle and regular:
                shape_names.append("square")
            elif rectangle:
                shape_names.append("rectangle")
            else:
                shape_names.append("rhombus")
        elif len(approx) == 5:
            shape_names.append("pentagon")
        elif len(approx) == 6:
            shape_names.append("hexagon")
        elif ellipse:
            if regular:
                shape_names.append("circle")
            else:
                shape_names.append("ellipse")
        else:
            shape_names.append("unknown")
                
        cv.putText(square, shape_names[-1], (cx,cy), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0))


cv.imshow('contours', square)
cv.imshow('canny', blurred)
