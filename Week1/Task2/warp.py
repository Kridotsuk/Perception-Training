import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

image = cv.imread('image.jpg')

rows,cols,_ = image.shape
M = np.float32([[1,0,100],[0,1,100]])

for i in range(1, 11):
    ax = plt.subplot(2,5,i)
    ax.axis('off')

M = np.float32([[1,0,100],[0,1,100]])
dst = cv.warpAffine(image,M,(cols,rows))
dst = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
plt.subplot(2,5,1)
plt.title('Translated by 100,100)', fontsize = 4)
plt.imshow(dst)

M = cv.getRotationMatrix2D(((cols-1)/2,(rows-1)/2),-30,1)
dst = cv.warpAffine(image,M,(cols,rows))
dst = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
plt.subplot(2,5,2)
plt.title('Clockwise rotated by 30 deg about center', fontsize = 4)
plt.imshow(dst)

M = cv.getRotationMatrix2D((0,(rows-1)/2),-90,1)
dst = cv.warpAffine(image,M,(cols,rows))
dst = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
plt.subplot(2,5,3)
plt.title('Clockwise rotated by 90 deg about center of left side', fontsize = 4)
plt.imshow(dst)

M = cv.getRotationMatrix2D(((cols-1)/2,(rows-1)/2),60,4)
dst = cv.warpAffine(image,M,(cols,rows))
dst = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
plt.subplot(2,5,4)
plt.title('Anticlockwise 60 and scaled to 4 times', fontsize = 4)
plt.imshow(dst)

#253 346

zoom = 4.3
M = np.float32([[zoom,0,253-253*zoom],[0,zoom,363 - 363*zoom]])
dst = cv.warpAffine(image,M,(cols,rows))
dst = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
plt.subplot(2,5,5)
plt.title('zoomed into box', fontsize = 4)
plt.imshow(dst)

zoom = 4.3
M = np.float32([[zoom,0,253-253*zoom],[0,zoom,363 - 363*zoom]])
dst = cv.warpAffine(image,M,(cols,rows))
dst = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
_, dst = cv.threshold(dst, 150, 255, cv.THRESH_BINARY)
plt.subplot(2,5,6)
plt.title('zoomed into box and binary threshold', fontsize = 4)
plt.imshow(dst, cmap = 'gray')

kernel = np.ones((10,10), np.float32)/100
dst = cv.filter2D(image,-1,kernel)
dst = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
plt.subplot(2,5,7)
plt.title('10x10 2D convolution', fontsize = 4)
plt.imshow(dst)

kernel = cv.getGaussianKernel(5,1)
dst = cv.filter2D(image,-1,kernel)
dst = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
plt.subplot(2,5,8)
plt.title('5x5 Gaussian convolution', fontsize = 4)
plt.imshow(dst)

pts1 = np.float32([[199,297],[303,302],[299,400],[196,392]])
pts2 = np.float32([[0,0],[300,0],[300,300],[0,300]])
M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(image, M, (300,300))
dst = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
plt.subplot(2,5,9)
plt.title('Perpective transformation to the square',fontsize = 4)
plt.imshow(dst,cmap ='gray')

pts1 = np.float32([[237,328],[265,330],[265,358],[237,356]])
pts2 = np.float32([[0,0],[300,0],[300,300],[0,300]])
M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(image, M, (300,300))
dst = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
plt.subplot(2,5,10)
plt.title('Perpective transform to text',fontsize = 4)
plt.imshow(dst, cmap='gray')

plt.show()



