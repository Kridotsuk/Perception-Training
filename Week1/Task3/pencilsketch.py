import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

horse = cv.imread('honse.jpeg', cv.IMREAD_GRAYSCALE)

edges = cv.Canny(horse,200,100)
_, edges = cv.threshold(edges, 150, 255, cv.THRESH_BINARY_INV)

cv.imshow('horse', edges)
