import cv2 as cv
import numpy as np

cap = cv.VideoCapture('messi.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    kernel = np.ones((20,20),np.float32)/400
    gray_frame = cv.filter2D(gray_frame,-1,kernel)
    #thresh_frame = cv.adaptiveThreshold(gray_frame, 255, cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY, 7,10)
    _, thresh_frame = cv.threshold(gray_frame, 150, 255, cv.THRESH_BINARY)
    
    circles = cv.HoughCircles(
        gray_frame,
        cv.HOUGH_GRADIENT,
        dp = 1,
        minDist = 100,
        param1 = 100,
        param2 = 40,
        minRadius = 30,
        maxRadius = 60)

    if circles is not None:
        x, y, r = circles[0][0]
        x = int(x)
        y = int(y)

        cv.putText(frame, "BALL", (x, y),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.7,
                   (255, 0, 0), 2)
        print("LOOK")
            
    cv.imshow('Original Frame', frame)

    if cv.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
