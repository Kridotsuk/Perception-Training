import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('sudoku.jpg', cv.IMREAD_GRAYSCALE)

laplacian = cv.Laplacian(image, cv.CV_64F)
sobelx = cv.Sobel(image, cv.CV_64F,1,0,5)
sobely = cv.Sobel(image, cv.CV_64F,0,1,5)

plt.subplot(2,2,1)
plt.axis('off')
plt.imshow(image, cmap='gray')

plt.subplot(2,2,2)
plt.axis('off')
plt.imshow(laplacian, cmap='gray')

plt.subplot(2,2,3)
plt.axis('off')
plt.imshow(sobelx, cmap='gray')

plt.subplot(2,2,4)
plt.axis('off')
plt.imshow(sobely, cmap='gray')

plt.show()
