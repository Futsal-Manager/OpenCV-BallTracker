#-*- coding: utf-8 -*-

import numpy as np
import cv2
# print cv2.__version__

# Algorithm
# 1. Image 읽기
# 2. 이미지 리사이즈
# 3. 블러처리 (노이즈 제거)
# 4. 2진화, 색상 검출(흰색)
# 5. 팽창(dilation) or 부식(erosion)
# 6. Canny Edge Detect
# 7. Find Contours
# 8. Contours 중에서 가장큰 사각형 검출


# Todo: 그냥 꼭지점에 '스티커' 붙이자...

# Variable
X_DIMENSION = 640
Y_DIMENSION = 480
DILATION = 10
BINARY_THRESH_HOLD = 200
CANNY_MIN_THRESH_HOLD = 100
CANNY_MAX_THRESH_HOLD = 200
GAUSIAN_BLUR_SIZE = 7 # 홀수가 되어야 함

def nothing(x):
    pass

def printText(img, text):
    y0, dy = 100, 30
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        cv2.putText(img, line, (0, y), cv2.FONT_HERSHEY_SIMPLEX, 1, 20, 2)

# def _onTrackBar(GAUSIAN_BLUR_SIZE):
#     if GAUSIAN_BLUR_SIZE % 2 == 0:
#         GAUSIAN_BLUR_SIZE += 1
#         pass

# Make Window
cv2.namedWindow('result')

# create trackbars for color change
cv2.createTrackbar('GAUSIAN_BLUR_SIZE','result',GAUSIAN_BLUR_SIZE,255,nothing)
cv2.createTrackbar('BINARY_THRESH_HOLD','result',BINARY_THRESH_HOLD,255,nothing)
cv2.createTrackbar('DILATION','result',DILATION,100,nothing)
cv2.createTrackbar('CANNY_MIN_THRESH_HOLD','result',CANNY_MIN_THRESH_HOLD,255,nothing)
cv2.createTrackbar('CANNY_MAX_THRESH_HOLD','result',CANNY_MAX_THRESH_HOLD,255,nothing)


while(1):
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.waitKey(1)
        break

    # get current positions of four trackbars
    DILATION = cv2.getTrackbarPos('DILATION', 'result')
    BINARY_THRESH_HOLD = cv2.getTrackbarPos('BINARY_THRESH_HOLD', 'result')
    GAUSIAN_BLUR_SIZE = cv2.getTrackbarPos('GAUSIAN_BLUR_SIZE', 'result')
    CANNY_MIN_THRESH_HOLD = cv2.getTrackbarPos('CANNY_MIN_THRESH_HOLD', 'result')
    CANNY_MAX_THRESH_HOLD = cv2.getTrackbarPos('CANNY_MAX_THRESH_HOLD', 'result')

    # 1. Image 읽기
    # Image List
    # 'set/IMG_0638.jpg'
    # 'set/IMG_0649.jpg'
    # 'set/IMG_0652.jpg'
    # 'set/IMG_0652.jpg'
    imgPath = 'set/IMG_0652.jpg'

    original = cv2.imread(imgPath, 1)
    original = cv2.resize(original, (X_DIMENSION, Y_DIMENSION))
    img = cv2.imread(imgPath, 0)

    # 2. 이미지 리사이즈
    img = cv2.resize(img, (X_DIMENSION, Y_DIMENSION))
    # cv2.imshow('resized_image', img)

    # 3. 블러처리 (노이즈 제거)
    if GAUSIAN_BLUR_SIZE % 2 == 0:
        GAUSIAN_BLUR_SIZE += 1
    blur = cv2.GaussianBlur(img, (GAUSIAN_BLUR_SIZE, GAUSIAN_BLUR_SIZE), 0)
    # cv2.imshow('blur', blur)

    # 4. 2진화, 색상 검출(흰색)
    ret, thresh4 = cv2.threshold(blur, BINARY_THRESH_HOLD, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresh', thresh4)

    # 5. 팽창 or 부식
    kernel = np.ones((DILATION, DILATION), np.uint8)  # 팽창을 위한, 1x1 => 3x3 pixel로 팽창
    dilation = cv2.dilate(thresh4, kernel, iterations=1)
    # erosion = cv2.erode(thresh4, kernel, iterations= 1) 부식
    cv2.imshow('dilation', dilation)

    # 6. Canny Edge Detect
    edges = cv2.Canny(dilation, CANNY_MIN_THRESH_HOLD, CANNY_MAX_THRESH_HOLD)
    cv2.imshow('Canny edge', edges)

    # 7. Find Contours
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imgCopy = original
    cv2.drawContours(imgCopy, contours, -1, (255, 0, 0), 3)
    cv2.imshow('contour', imgCopy)

    # 8. Contours 중에서 가장큰 사각형 검출
    areaArray = []
    count = 1

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)

    # array를 영역별로 정렬
    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

    # find the nth largest contour [n-1][1], in this case 1
    if sorteddata:
        secondlargestcontour = sorteddata[0][1]

        # draw it
        x, y, w, h = cv2.boundingRect(secondlargestcontour)
        # cv2.drawContours(original, secondlargestcontour, -1, (255, 0, 0), 2)
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show Result
        # cv2.putText(original, 'test', (0, Y_DIMENSION), cv2.FONT_HERSHEY_PLAIN, 10, 20, 5)
        printText(original,
                  'DILATION: ' + str(DILATION) + '\n' +
                  'BINARY_THRESH: ' + str(BINARY_THRESH_HOLD) + '\n'
                  'GAUSIAN_BLUR_SIZE: ' + str(GAUSIAN_BLUR_SIZE) + '\n'
                   )
    cv2.imshow('result', original)



    # Todo: 반복문 돌면서 가장 큰 사각형이 있는 1. thresh hold를 찾음
    #
    # ref: http://stackoverflow.com/questions/22240746/recognize-open-and-closed-shapes-opencv

cv2.destroyAllWindows()
