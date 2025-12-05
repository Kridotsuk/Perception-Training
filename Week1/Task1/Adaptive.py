

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def adaptiveThresholding():
    earth = cv.imread("earth.jpg")
    earthGray = cv.cvtColor(earth, cv.COLOR_BGR2GRAY)

    plt.subplot(1,3,1)
    plt.imshow(earthGray, cmap = 'gray')

    _, earthThres = cv.threshold(earthGray, 60, 255, cv.THRESH_BINARY)

    plt.subplot(1,3,2)
    plt.imshow(earthThres, cmap = 'gray')

    earthThres = cv.adaptiveThreshold(earthGray, 255, cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY, 7,10)
    plt .subplot(1,3,3)
    plt.imshow(earthThres, cmap = 'gray')

    plt.show()

if __name__ == '__main__':
    adaptiveThresholding()
