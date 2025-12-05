import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors

nemo = cv2.imread("nemo.jpg")

nemo = cv2.resize(nemo, (200, 200))

nemo_hsv = cv2.cvtColor(nemo, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(nemo_hsv)

nemo_rgb = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB) / 255.0
pixel_colors = nemo_rgb.reshape((-1, 3))

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()
