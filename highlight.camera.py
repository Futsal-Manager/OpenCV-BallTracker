# -*- coding: utf-8 -*-
# 위 코드로 한글주석 처리가 가능해짐
# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

# Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
help="max buffer size")
args = vars(ap.parse_args())

# Algorithm
# Todo: TBD

yellowCenterArr = []
orangeCenterArr = []

def nothing(x):
    pass

## Note
# 골대 가로길이: 3m
# 골대 세로길이: 2m
# 비율 3:2
##

# Variable
X_DIMENSION = 480
Y_DIMENSION = 480
GAUSIAN_BLUR_SIZE = 11
CANNY_MIN_THRESH_HOLD = 100
CANNY_MAX_THRESH_HOLD = 200
BALL_SIZE = 15 # Todo: 공의 크기가 골대에 의해 계산되도록 재정의 (공식: 골대 가로길이 * 0.07)
BALL_ERROR_RANGE = 5 # Todo: Need to Scroll
BALL_MIN_RADIUS = BALL_SIZE - BALL_ERROR_RANGE
BALL_MAX_RADIUS = BALL_SIZE + BALL_ERROR_RANGE
GOAL_POST_RANGE = 0 # Todo: Need to Scroll
BALL_POST_RATIO = 0.07

# For Scren Click Event
mouseX = 0
mouseY = 0

originImg = None
prevCenter = None

# Make Window
cv2.namedWindow('result')

# create trackbars for color change
cv2.createTrackbar('BALL_ERROR_RANGE','result',BALL_ERROR_RANGE,10,nothing)
cv2.createTrackbar('GOAL_POST_RANGE','result',GOAL_POST_RANGE,70,nothing)

# Data Structure deque for ball path
pts = deque(maxlen=64)

# Direction
DIRECTION_LEFT = 'LEFT'
DIRECTION_RIGHT = 'RIGHT'

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

# 비디오가 제공되지 않으면, 영상을 캡쳐
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# 비디오가 제공되면 영상에서 동작 수행
else:
    camera = cv2.VideoCapture(args["video"])


# (180, 255, 255)이 최대: openCV 색공간
def hsvConverter(color):
    h = float(color[0]) / 2
    s = float(color[1]) / 100 * 255
    v = float(color[2]) / 100 * 255
    return (h, s, v)

# (360, 100, 100)이 최대: 원래 HSV
def hsvInverter(color):
    h = float(color[0]) * 2
    s = float(color[1]) / 255 * 100
    v = float(color[2]) / 255 * 100
    return (h, s, v)

# 골대 찾기
def findGoalPostByColorAndDirection(firstColor, secondColor, direction):
    firstColorLower = firstColor[0]
    firstColorUpper = firstColor[1]

    secondColorLower = secondColor[0]
    secondColorUpper = secondColor[1]

    findSpotResult = _findSpot(firstColorLower, firstColorUpper, direction)
    if(findSpotResult != None):
        (mask, cnts, c, ((x, y), radius), center) = findSpotResult
        yellowCenterArr.append(center)

    findSpotResult = _findSpot(secondColorLower, secondColorUpper, direction)
    if (findSpotResult != None):
        (mask, cnts, c, ((x, y), radius), center) = findSpotResult
        orangeCenterArr.append(center)

        if len(yellowCenterArr) >= 2  and len(orangeCenterArr) >= 2:  # 만약 2개의 점을 모두 검출하면 정렬,
            _makeCenterBeetweenColor(yellowCenterArr, orangeCenterArr) # 골대 외곽선 그리기


def findBallColor(lower, upper):
    findSpotResult = _findSpot(lower, upper)
    if(findSpotResult is not None):
        (mask, cnts, c, ((x, y), radius), center) = _findSpot(lower, upper)
        print 'ball size: ' +str(BALL_MIN_RADIUS) + ' ~ ' + str(BALL_MAX_RADIUS)

        if BALL_MIN_RADIUS < radius and radius < BALL_MAX_RADIUS:
            # print 'ball founded'
            cv2.circle(img, (int(x), int(y)), int(radius), (0, 0, 0), 2)
            ballCenter = list(center)
            _drawBallTrackLine(center) # 볼의 선을 그려서 경로 알아봄
            npBallCenter = np.array(ballCenter)
            emptyCenter = np.array([])

        return radius

def _findSpot(lower, upper, direction):
    hsvCopy = np.array(hsv) # copy


    # Todo: Deep Copy화 화면을 Binary로 채우는 것에 대한 Performance문제 해결

    # direction 왼쪽 => 오른쪽 half만큼 가려버림
    if direction == DIRECTION_LEFT:
        for x in range(X_DIMENSION/2, X_DIMENSION):
            for y in range(0, Y_DIMENSION):
                maskBlack = np.uint8([0, 0, 0])
                hsvCopy[y][x] = maskBlack # Todo: 왜 x,y 반대?, ref: http://docs.opencv.org/3.0-beta/doc/user_guide/ug_mat.html


    if direction == DIRECTION_RIGHT:
        for x in range(0, X_DIMENSION/2):
            for y in range(0, Y_DIMENSION):
                maskBlack = np.uint8([0, 0, 0])
                hsvCopy[y][x] = maskBlack # Todo: 왜 x,y 반대?, ref: http://docs.opencv.org/3.0-beta/doc/user_guide/ug_mat.html

    cv2.imshow('test'+direction, hsvCopy)

    mask = cv2.inRange(hsvCopy, hsvConverter(lower), hsvConverter(upper))
    # cv2.imshow('mask ' + direction + str(lower) + str(upper), mask)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        # print( 'contour', 'lower',lower, 'upper', upper,cnts[0][0][0][0])
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        if M["m00"] != 0 and M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            return (mask, cnts, c, ((x, y), radius), center)
        else: return None
    else: return None


def show_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONUP and img is not None:
        copyImg = originImg.copy()
        height, width, channels = img.shape

        point = copyImg[y, x] # (192, 192, 192)
        r = float(point[2])/255
        g = float(point[1])/255
        b = float(point[0])/255

        opencv_hsv = colorsys.rgb_to_hsv(r, g, b)
        opencv_h = opencv_hsv[0] * 360

        opencv_s = opencv_hsv[1] * 100
        opencv_v = opencv_hsv[2] * 100

        # print '======================'
        # print 'original: ', opencv_hsv
        print '색상: hsv: ', [opencv_h, opencv_s, opencv_v]
        print '클릭한 좌표: ',x, y
        # print 'RGB: ', [point[2], point[1], point[0]]
        # print '======================'


# Todo: 패턴 사이의 거리를 검증하는 코드가 필요함
# Todo 1. 노랑 사이의 거리
# Todo 2. 주황 사이의 거리
# Todo 3. 최종 중심점 사이의 거리가 일정 이상
def _makeCenterBeetweenColor(_yellowCenterArr, _orangeCenterArr): # arr의 length는 항상 4
    # _yellowCenterArr.sort() # _yellowCenterArr [(54, 165), (386, 159)]
    # _orangeCenterArr.sort() # _orangeCenterArr [(38, 172), (403, 157)]
    # Yellow와 Orange사이의 점들중 가장 작은 값을 찾아낸다.
    print '_yellowCenterArr', _yellowCenterArr # _yellowCenterArr [(54, 165), (386, 159)]

    # _orangeCenterArr [(38, 172), (403, 157)]
    print '_orangeCenterArr', _orangeCenterArr #

    # [(거리, y중심, x중심)]
    distanceEverySpot = []

    for yellowCenter in _yellowCenterArr:
        for orangeCenter in _orangeCenterArr:
            xPosYellow = yellowCenter[0]
            yPosYellow = yellowCenter[1]

            xPosOrange = orangeCenter[0]
            yPosOrange = orangeCenter[1]

            distance = (xPosOrange-xPosYellow)+(yPosOrange-yPosYellow)
            distanceEverySpot.append((distance,yellowCenter, orangeCenter))

    # 거리 중심 + x좌표 기준으로 sort
    distanceEverySpot = sorted(distanceEverySpot, key=lambda x : (abs(x[0]), x[1][0]) ) # x[1][0] => x좌표

    # print distanceEverySpot

    # 앞에서 2개의 중심을 구함
    # 좌상 (Marker)
    xPosFirstCenter = (distanceEverySpot[0][1][0] + distanceEverySpot[0][2][0]) / 2
    yPosFirstCenter = (distanceEverySpot[0][1][1] + distanceEverySpot[0][2][1]) / 2
    leftTopCenter = (xPosFirstCenter, yPosFirstCenter)

    # 우상 (Marker)
    xPosSecondCenter = (distanceEverySpot[1][1][0] + distanceEverySpot[1][2][0]) / 2
    yPosSecondCenter = (distanceEverySpot[1][1][1] + distanceEverySpot[1][2][1]) / 2
    rightTopCenter = (xPosSecondCenter, yPosSecondCenter)

    # 골대 길의, 높이 구하기 (골대 가로 * 2/3 = 골대 세로)
    # Todo 3. 최종 중심점 사이의 거리가 일정 이상일때만 processing
    goalpostWidth = (abs(leftTopCenter[0] - rightTopCenter[0]))
    goalpostHeight = int(goalpostWidth * float(2.0 / 3.0))

    # print 'goalpostWidth', goalpostWidth
    # print 'goalpostHeight', goalpostHeight

    # 좌하 (가상) 점 만듬
    leftBottomCenter = (leftTopCenter[0], leftTopCenter[1] + goalpostHeight)

    # 우하 (가상) 점 만듬
    rightBottomCenter = (rightTopCenter[0], rightTopCenter[1] + goalpostHeight)

    # 원본 골대 마커에 동그라미
    cv2.circle(img, leftTopCenter, 10, (0, 0, 255), 3)
    cv2.circle(img, rightTopCenter, 10, (0, 0, 255), 3)

    # 골대 range 값만큼 키움
    # 좌상 x-,y-
    leftTopCenter = (leftTopCenter[0] - GOAL_POST_RANGE, leftTopCenter[1] - GOAL_POST_RANGE)

    # 좌하 x-,y+
    leftBottomCenter = (leftBottomCenter[0] - GOAL_POST_RANGE, leftBottomCenter[1] + GOAL_POST_RANGE)

    # 우상 x+, y+
    rightTopCenter = (rightTopCenter[0] + GOAL_POST_RANGE, rightTopCenter[1] - GOAL_POST_RANGE)

    # 우하 x+, y-
    rightBottomCenter = (rightBottomCenter[0] + GOAL_POST_RANGE, rightBottomCenter[1] + GOAL_POST_RANGE)


    # 골대 외곽선 그리기

    # 좌상 우상 이음
    cv2.line(img, leftTopCenter, rightTopCenter, (0, 0, 0), 3)

    # 골대 아래 가로
    cv2.line(img, leftBottomCenter, rightBottomCenter, (0, 0, 0), 3)

    # 골대 세로
    # 왼쪽
    cv2.line(img, leftTopCenter, leftBottomCenter, (0, 0, 0), 3)

    # 오른쪽
    cv2.line(img, rightTopCenter, rightBottomCenter, (0, 0, 0), 3)




def _setBallSize(size):
    # print 'setted', size
    global BALL_SIZE
    BALL_SIZE = size


def _drawBallTrackLine(center):
    global prevCenter
    # 포인트 큐를 업데이트
    pts.appendleft(center)

    # 공의 속도를 추적
    # To find pixels/mm = Object size in pixels / object size in mm
    # (pixels/frame) / (pixels/mm) = mm/frame
    # mm/frame * frames/second = mm / second

    # Todo: Need To Refactor
    if center is not None and prevCenter is not None:
        if prevCenter[0] - center[0] > 0:
            print '왼쪽 속력 ', prevCenter[0] - center[0]
        elif center is not None:
            print '오른쪽 속력: ', center[0] - prevCenter[0]

    prevCenter = center

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


## 색공간: HSV (360 100, 100)
# 형광 노랑
yellowLower = (68, 66, 55)
yellowUpper = (75, 75, 80)


# 형광 주황
orangePostLower = (15, 68, 60)
orangePostUpper= (30, 80, 90)

# 공 (주황)
# 20, 74.9, 95.3
orangeLower = (0, 85, 0)
orangeUpper = (45, 110, 255)

# 무한루프
while True:
    # 현재 프레임을 잡아냄
    (grabbed, frame) = camera.read()

    yellowCenterArr = []
    orangeCenterArr = []

    # get current positions of four trackbars
    BALL_ERROR_RANGE = cv2.getTrackbarPos('BALL_ERROR_RANGE', 'result')
    BALL_MIN_RADIUS = BALL_SIZE - BALL_ERROR_RANGE
    BALL_MAX_RADIUS = BALL_SIZE + BALL_ERROR_RANGE
    GOAL_POST_RANGE = cv2.getTrackbarPos('GOAL_POST_RANGE', 'result')

    # 1. resize the frame
    frame = imutils.resize(frame, width=X_DIMENSION, height=Y_DIMENSION)
    cv2.imshow('frame', frame)

    # Q를 누르면 프로그램 종료
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


# 카메라 클리어하고 열려있는 창 닫음
camera.release()
cv2.destroyAllWindows()