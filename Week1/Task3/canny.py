import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

horse = cv.imread('horse.jpg')

edges = cv.Canny(horse,299,168)

cv.imshow('horse', edges)
