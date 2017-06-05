#-*- coding: utf-8 -*-

import numpy as np
import cv2
from collections import deque

# print cv2.__version__

# Algorithm

# 골대검출
# 1. Image 읽기
# 2. 이미지 리사이즈
# 3. 블러처리 (노이즈 제거)
# 4. cvtColor to HSV, 색상 검출(4가지 색상)
# 5. 4가지 꼭지점을 사각형으로 만들기

# Todo: 마우스 이벤트 달아서 HSV값 직접 추출해서 변환한 값과 다른지 검출
# Prev Todo: 공 크기를 구하기 (골대의 가로 길이 * 0.07) (x)
# Todo 1: 볼 Min Max 오차 스크롤 만들기 (x)
# Todo 2: 골대 인식 범위 스크롤 만들기 (x)
# Todo 3: 골대 범위 만들기 (x)
# Todo 4: 범위 내에 공이 있는지 없는지 알아내기
# Todo 5: 골인시 알고리즘 만들기(공이 크기가 어느정도 수준일때 (ex 지름이 10px~ 15px)이고 골대의 박스안에 들어갈때 => 어느정도의 속도 => 영상의 잘리는 범위 => 소리 크기???)
# Todo 6: 영상에 적용

## Todo: After Field Test
## Todo: 클릭해서 색 검출

centerArr = []

def nothing(x):
    pass

# Variable
X_DIMENSION = 480
Y_DIMENSION = 480
GAUSIAN_BLUR_SIZE = 11
CANNY_MIN_THRESH_HOLD = 100
CANNY_MAX_THRESH_HOLD = 200
BALL_SIZE = 10 # Todo: 공의 크기가 골대에 의해 계산되도록 재정의 (공식: 골대 가로길이 * 0.07)
BALL_ERROR_RANGE = 5 # Todo: Need to Scroll
BALL_MIN_RADIUS = BALL_SIZE - BALL_ERROR_RANGE
BALL_MAX_RADIUS = BALL_SIZE + BALL_ERROR_RANGE
GOAL_POST_RANGE = 30 # Todo: Need to Scroll
BALL_POST_RATIO = 0.07
mouseX = 0
mouseY = 0

# Make Window
cv2.namedWindow('result')

# create trackbars for color change
cv2.createTrackbar('BALL_ERROR_RANGE','result',BALL_ERROR_RANGE,10,nothing)
cv2.createTrackbar('GOAL_POST_RANGE','result',GOAL_POST_RANGE,70,nothing)

# Data Structure
pts = deque(maxlen=64)

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


def findGoalPostSpotColor(lower, upper):
    findSpotResult = _findSpot(lower, upper)
    if(findSpotResult != None):
        (mask, cnts, c, ((x, y), radius), center) = findSpotResult
        centerArr.append(center)
        if len(centerArr) == 4:  # 만약 4개의 점을 모두 검출하면 정렬,
            _makeLine(centerArr)
            # 공을 추적하는 로직 Orange_ball.py를 불러다 쓸 것.

def findBallColor(lower, upper):
    findSpotResult = _findSpot(lower, upper)
    print 'findSpotResult,', findSpotResult
    if(findSpotResult != None):
        (mask, cnts, c, ((x, y), radius), center) = _findSpot(lower, upper)
        print str(BALL_MIN_RADIUS) + ' ~ ' + str(BALL_MAX_RADIUS)

        if BALL_MIN_RADIUS < radius and radius < BALL_MAX_RADIUS:
            # print 'ball founded'
            cv2.circle(img, (int(x), int(y)), int(radius), (0, 0, 0), 2)
            ballCenter = list(center)
            _drawBallTrackLine(center) # 볼의 선을 그려서 경로 알아봄
            npBallCenter = np.array(ballCenter)
            emptyCenter = np.array([])
            # print 'founded ball center 2D: ' + str(npBallCenter)
        return radius

def _findSpot(lower, upper):
    if(lower == (210, 68.9, 69.4)): print hsvConverter(lower), hsvConverter(upper) # Blue일때
    mask = cv2.inRange(hsv, hsvConverter(lower), hsvConverter(upper))  ## Todo: 찾은 색상코드를 opencv 원 주위에 넣기
    cv2.imshow('mask' + str(lower) + str(upper), mask)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return (mask, cnts, c, ((x, y), radius), center)
    else: return None


def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONUP:
        copyImg = img.copy()
        height, width, channels = img.shape
        print '클릭한 좌표: ',x, y,'색상: ', copyImg[y, x]

def _makeLine(arr): # arr의 length는 항상 4
    arr.sort()
    leftCenter = arr[0:2]
    rightCneter = arr[2:4]

    leftCenter.sort(key=lambda x: x[1])
    rightCneter.sort(key=lambda x: x[1])

    # print 'leftGoalPost' + str(leftCenter)
    # print 'rightGoalPost' + str(rightCneter)

    # 공 크기 설정
    _setBallSize((rightCneter[0][0] - leftCenter[0][0]) * BALL_POST_RATIO)

    # 좌상 x-,y-
    leftCenter[0] = (leftCenter[0][0] - GOAL_POST_RANGE, leftCenter[0][1] - GOAL_POST_RANGE)

    # 좌하 x-,y+
    leftCenter[1] = (leftCenter[1][0] - GOAL_POST_RANGE, leftCenter[1][1] + GOAL_POST_RANGE)

    # 우상 x+, y+
    rightCneter[0] = (rightCneter[0][0] + GOAL_POST_RANGE, rightCneter[0][1] - GOAL_POST_RANGE)

    # 우하 x+, y-
    rightCneter[1] = (rightCneter[1][0] + GOAL_POST_RANGE, rightCneter[1][1] + GOAL_POST_RANGE)



    cv2.line(img, leftCenter[0], leftCenter[1], (0,0,0), 3) # 좌상 좌하 잇고
    cv2.line(img, leftCenter[0], rightCneter[0], (0, 0, 0), 3) # 좌상 우상 잇고
    cv2.line(img, rightCneter[0], rightCneter[1], (0, 0, 0), 3) # 우상 우하 잇고
    cv2.line(img, leftCenter[1], rightCneter[1], (0, 0, 0), 3) # 좌하 우하 잇고

def _setBallSize(size):
    print 'setted', size
    global BALL_SIZE
    BALL_SIZE = size


def _drawBallTrackLine(center):
    # 포인트 큐를 업데이트
    pts.appendleft(center)

    # Todo: 이부분에 else if 문으로 외곽(contours)가 여러개일때의 처리가 필요함

    # 추적좌표들의 집합을 반복
    for i in xrange(1, len(pts)):
        # 추적지점이 하나도 없으면 loop 무시
        if pts[i - 1] is None or pts[i] is None:
            continue

        # 추적지점이 있으면 두께를 계산해서 그림
        # 연결된 선을 그림
        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
        cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), thickness)

# 공이 중심을 찾고, 공이 들어왔는지 검사


## HSV
# 좌상 (파랑) 177  78  59
blueLower = (200, 50, 50)
blueUpper = (230, 70, 75)

# 우상 (빨강) V
redLower = (350, 60, 90)
redUpper = (360, 70, 100)

# 좌하 (하늘) V
skyLower = (170, 60, 80)
skyUpper = (230, 80, 90)

# 우하 (보라) 301, 58.9, 59.2
purpleLower = (290, 45, 55)
purpleUpper = (320, 65, 75)

# 공 (주황)
# 20, 74.9, 95.3
orangeLower = (0, 85, 0)
orangeUpper = (45, 110, 255)

##################################################
############### Main Function ####################
##################################################

while(1):
    k = cv2.waitKey(0) & 0xFF
    centerArr = []
    if k == 27:
        # cv2.waitKey(1)
        break
    elif k == ord('a'):
        print mouseX,mouseY

    # get current positions of four trackbars
    BALL_ERROR_RANGE = cv2.getTrackbarPos('BALL_ERROR_RANGE', 'result')
    # print 'BALL_ERROR_RANGE: ' + str(BALL_ERROR_RANGE)
    BALL_MIN_RADIUS = BALL_SIZE - BALL_ERROR_RANGE
    BALL_MAX_RADIUS = BALL_SIZE + BALL_ERROR_RANGE
    GOAL_POST_RANGE = cv2.getTrackbarPos('GOAL_POST_RANGE', 'result')
    # print 'GOAL_POST_RANGE: ' + str(GOAL_POST_RANGE)

    # 1. Image 읽기
    # imgPath = 'test.png'
    imgPath = 'test.png' # ball_fieldsample.jpg
    img = cv2.imread(imgPath, 1)

    # 2. 이미지 리사이즈
    img = cv2.resize(img, (X_DIMENSION, Y_DIMENSION))

    cv2.putText(img, 'Range: ' + str(BALL_MIN_RADIUS) + ' ~ ' + str(BALL_MAX_RADIUS), (0, 480), cv2.FONT_HERSHEY_SIMPLEX, 1, 20, 2)

    # 원본 이미지 복사
    originImg = img.copy()

    # 3. 블러처리 (노이즈 제거)
    blur = cv2.GaussianBlur(img, (GAUSIAN_BLUR_SIZE, GAUSIAN_BLUR_SIZE), 0)

    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)

    # 3. 색 공간 변경
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # cv2.imshow('image', hsv)

    # Mask Array
    maskArr = []

    # Red
    cntsArr = []

    # 빨강을 위한 mask
    findGoalPostSpotColor(redLower, redUpper)

    # 파랑을 위한 mask
    findGoalPostSpotColor(blueLower, blueUpper)

    # 하늘을 위한 mask
    findGoalPostSpotColor(skyLower, skyUpper)

    # 보라을 위한 mask
    findGoalPostSpotColor(purpleLower, purpleUpper)

    # 주황(공) 위한 mask
    radius = findBallColor(orangeLower, orangeUpper)


    # 6. Canny Edge Detect
    # cv2.drawContours(img, cnts, -1, (255, 0, 0), 3)
    cv2.putText(img, 'real ball pixel' + str(radius), (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, 20, 2)
    print 'BALL_SIZE is', BALL_SIZE
    cv2.putText(img, '0.07 convert: ' + str(BALL_SIZE), (0, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, 20, 2)

    # 'BALL_SIZE'
    cv2.setMouseCallback('result', draw_circle)
    cv2.imshow('result', img)



    # cv2.imshow('origin', originImg)

cv2.destroyAllWindows()