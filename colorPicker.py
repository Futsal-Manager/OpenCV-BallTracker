#-*- coding: utf-8 -*-

import numpy as np
import cv2
from collections import deque

centerArr = []

img = None

def nothing(x):
    pass

def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONUP:
        copyImg = img.copy()
        height, width, channels = img.shape
        print '클릭한 좌표: ',x, y,'색상: ', copyImg[y, x]

# Make Window
cv2.namedWindow('result')

while(1):
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        # cv2.waitKey(1)
        break
    elif k == ord('a'):
        print mouseX,mouseY

    # 1. Image 읽기
    # imgPath = 'test.png'
    imgPath = 'test.png' # ball_fieldsample.jpg
    img = cv2.imread(imgPath, 1)
    cv2.setMouseCallback('result', draw_circle)
    cv2.imshow('result', img)

    # 2. 이미지 리사이



    # cv2.imshow('origin', originImg)

cv2.destroyAllWindows()