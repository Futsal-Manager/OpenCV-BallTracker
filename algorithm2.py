#-*- coding: utf-8 -*-

import numpy as np
import cv2
# print cv2.__version__

# Algorithm

# 골대검출
# 1. Image 읽기
# 2. 이미지 리사이즈
# 3. 블러처리 (노이즈 제거)
# 4. cvtColor to HSV, 색상 검출(4가지 색상)
# 5. 4가지 꼭지점을 사각형으로 만들기

# Todo: 마우스 이벤트 달아서 HSV값 직접 추출해서 변환한 값과 다른지 검출
# Todo: 빨(우상) 파(좌상) 노(좌하) 보(우하)
# Todo: point 중앙점이 4개인 순간 검출

centerArr = []

# Variable
X_DIMENSION = 640
Y_DIMENSION = 480
GAUSIAN_BLUR_SIZE = 11
CANNY_MIN_THRESH_HOLD = 100
CANNY_MAX_THRESH_HOLD = 200

# Color Variable
'''
RGB

우상 빨강: 255, 0, 0
좌상 파랑:  0, 90, 234
좌하 노랑: 248, 244, 2
우하 보라:  222, 34, 185

골 주황: 243, 121, 61

을 변환툴로 변환하면 (ref: https://www.google.co.kr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwjkruq37MbTAhWCnpQKHW0rB78QFgglMAA&url=http%3A%2F%2Fwww.rapidtables.com%2Fconvert%2Fcolor%2Frgb-to-hsv.htm&usg=AFQjCNHbmMYfaCowVxiWVkL2U85oBO3N0g&sig2=U7Jpr7zOv1LG0L1EgTnsow&cad=rjt)

우상 빨강: 255, 0, 0
좌상 파랑:  0, 90, 234
좌하 노랑: 248, 244, 2
우하 보라:  222, 34, 185


######################################
원래 HSV
H: 0 ~ 360
S: 0 ~ 100%
V: 0 ~ 100%

을 아래로 변환해야 함

OpenCV HSV
HUE: 0 ~ 180 
SATURATION: 0 ~ 255
Value: 0 ~ 255
######################################
'''

# S, V는 100으로 나누고 255 곰합

BLACK = (0, 0, 0)




def hsvConverter(color):
    h = float(color[0]) / 2
    s = float(color[1]) / 100 * 255
    v = float(color[2]) / 100 * 255
    return (h, s, v)

def hsvInverter(color):
    h = float(color[0]) * 2
    s = float(color[1]) / 255 * 100
    v = float(color[2]) / 255 * 100
    return (h, s, v)

def findSpotColor(lower, upper, isBall = False):
    mask = cv2.inRange(hsv, hsvConverter(lower), hsvConverter(upper))  ## Todo: 찾은 색상코드를 opencv 원 주위에 넣기
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    c = max(cnts, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    # 원 색칠하기.
    if isBall:
        cv2.circle(img, (int(x), int(y)), int(radius), (0, 0, 0), 2)
        ballCenter = list(center)
        npBallCenter = np.array(ballCenter)
        emptyCenter = np.array([])
        print 'founded ball center 2D: ' + str(npBallCenter)
        cv2.convertPointsToHomogeneous(npBallCenter, emptyCenter)
        # arr3d = np.array([])
        # print(arr3d)
        # print

    else:
        centerArr.append(center)
        if len(centerArr) == 4:  # 만약 4개의 점을 모두 검출하면 정렬,
            print 'goalpost' + str(centerArr)
            _makeLine(centerArr)
            # 공을 추적하는 로직 Orange_ball.py를 불러다 쓸 것.

def _makeLine(arr): # arr의 length는 항상 4
    arr.sort()
    leftCenter = arr[0:2]
    rightCneter = arr[2:4]

    leftCenter.sort(key=lambda x: x[1])
    rightCneter.sort(key=lambda x: x[1])

    cv2.line(img, leftCenter[0], leftCenter[1], (0,0,0), 3) # 좌상 좌하 잇고
    cv2.line(img, leftCenter[0], rightCneter[0], (0, 0, 0), 3) # 좌상 우상 잇고
    cv2.line(img, rightCneter[0], rightCneter[1], (0, 0, 0), 3) # 우상 우하 잇고
    cv2.line(img, leftCenter[1], rightCneter[1], (0, 0, 0), 3) # 좌하 우하 잇고


# 공이 중심을 찾고, 공이 들어왔는지 검사

# 좌상 (파랑) (217, 100, 91.8)
blueLower = (210, 90, 90)
blueUpper = (230, 110, 110)

# 우상 (빨강) (0, 100, 100)
redLower = (0, 90, 90)
redUpper = (0, 100, 100)

# 좌하 (노랑) (59, 99.2, 97.3)
yellowLower = (54, 94.2, 92.3)
yellowUpper = (64, 100, 100)

# 우하 (보라) (312, 84.7, 87,1)
purpleLower = (307, 79.7, 82,1)
purpleUpper = (317, 100, 100)

# 공 (주황)
# 20, 74.9, 95.3
orangeLower = (15, 69.9, 90)
orangeUpper = (25, 79.9, 100)




# 1. Image 읽기
imgPath = 'sample.jpg'
imgPath = 'ball_fieldsample.jpg' # ball_fieldsample.jpg
img = cv2.imread(imgPath, 1)

# 2. 이미지 리사이즈
img = cv2.resize(img, (X_DIMENSION, Y_DIMENSION))

# 원본 이미지 복사
originImg = img.copy()

# 3. 블러처리 (노이즈 제거)
blur = cv2.GaussianBlur(img, (GAUSIAN_BLUR_SIZE, GAUSIAN_BLUR_SIZE), 0)

# mask = cv2.erode(mask, None, iterations=2)
# mask = cv2.dilate(mask, None, iterations=2)

# 3. 색 공간 변경
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
cv2.imshow('image', hsv)

# Mask Array
maskArr = []

# Red
cntsArr = []

# 빨강을 위한 mask
findSpotColor(redLower, redUpper)

# 파랑을 위한 mask
findSpotColor(blueLower, blueUpper)

# 노랑을 위한 mask
findSpotColor(yellowLower, yellowUpper)

# 보라을 위한 mask
findSpotColor(purpleLower, purpleUpper)

# 주황(공) 위한 mask
findSpotColor(orangeLower, orangeUpper, True)

# 6. Canny Edge Detect
# cv2.drawContours(img, cnts, -1, (255, 0, 0), 3)
cv2.imshow('result', img)
cv2.imshow('origin', originImg)

k = cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.waitKey(1)
    cv2.destroyAllWindows()