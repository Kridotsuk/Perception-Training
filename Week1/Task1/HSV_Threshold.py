import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("nemo.jpg")
if img is None:
    raise ValueError("Image not found")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

h1 = 1
s1 = 190
v1 = 100

h2 = 18
s2 = 255
v2 = 255

lower = (h1, s1, v1)
upper = (h2, s2, v2)

mask = cv2.inRange(img_hsv, lower, upper)
result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Segmented Result")
plt.imshow(result)
plt.axis("off")

plt.show()
