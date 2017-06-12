# -*- coding: utf-8 -*-
# 위 코드로 한글주석 처리가 가능해짐
# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import the necessary packages
from collections import deque
import numpy as np
import argparse
import colorsys
import imutils
import cv2

# for fps measurement
import time


# Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
help="max buffer size")
args = vars(ap.parse_args())


WEBCAM_MODE = 'WEBCAM'
IMAGE_MODE = 'IMAGE'
VIDEO_MODE = 'VIDEO'

MODE = IMAGE_MODE


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
X_DIMENSION = 1280
Y_DIMENSION = 720
GAUSIAN_BLUR_SIZE = 11
CANNY_MIN_THRESH_HOLD = 100
CANNY_MAX_THRESH_HOLD = 200
BALL_SIZE = 15 # Todo: 공의 크기가 골대에 의해 계산되도록 재정의 (공식: 골대 x`가로길이 * 0.07)
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
def findGoalPostByColorAndDirection(yellowColor, orangeColor, direction):
    global frame
    # for yellow
    _yellowColorLower = yellowColor[0]
    _yellowColorUpper = yellowColor[1]

    # for orange
    _orangeColorLower = orangeColor[0]
    _orangeColorUpper = orangeColor[1]

    findSpotResult = _findSpot(_yellowColorLower, _yellowColorUpper, direction)
    if(findSpotResult != None):
        (mask, cnts, c, ((x, y), radius), center) = findSpotResult
        # cv2.circle(frame, center, 30, (255,0,0), 10)
        # cv2.putText(frame, 'Yellow Center: ' +str(center), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow('mask'+direction, mask) # Just Yellow
        yellowCenterArr.append(center)


    # Todo: Need to debug
    # _yellowCenterArr[(519, 321), (560, 504)]
    # _orangeCenterArr[(528, 320), (78, 323)]

    findSpotResult = _findSpot(_orangeColorLower, _orangeColorUpper, direction)
    if (findSpotResult != None):
        (mask, cnts, c, ((x, y), radius), center) = findSpotResult
        _center = center
        _center = (_center[0] + X_DIMENSION/2, _center[1])
        # cv2.putText(frame, 'Orange Center: ' + str(center), _center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        # cv2.circle(frame, _center, 30, (0, 255, 0), 10)  # Just for Orage
        cv2.imshow('mask' + direction, mask)
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
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 0), 2)
            ballCenter = list(center)
            _drawBallTrackLine(center) # 볼의 선을 그려서 경로 알아봄
            npBallCenter = np.array(ballCenter)
            emptyCenter = np.array([])

        return radius

def _findSpot(lower, upper, direction):
    hsvCopy = np.array(hsv) # copy


    # Todo: Deep Copy화 화면을 Binary로 채우는 것에 대한 Performance문제 해결

    # [
    #   [0],[],[],
    #   [],[],[],
    #   [],[],[],
    # ]

    # direction 왼쪽 => 오른쪽 half만큼 가려버림
    if direction == DIRECTION_LEFT:
        hsvCopy = hsvCopy[0 : Y_DIMENSION, 0 : X_DIMENSION/2]
        # for x in range(X_DIMENSION/2, X_DIMENSION):
        #     for y in range(0, Y_DIMENSION):
        #         maskBlack = np.uint8([0, 0, 0])
        #         hsvCopy[y][x] = maskBlack # Todo: 왜 x,y 반대?, ref: http://docs.opencv.org/3.0-beta/doc/user_guide/ug_mat.html


    if direction == DIRECTION_RIGHT:
        hsvCopy = hsvCopy[0 : Y_DIMENSION, X_DIMENSION/2 : X_DIMENSION]
        # for x in range(0, X_DIMENSION/2):
        #     for y in range(0, Y_DIMENSION):
        #         maskBlack = np.uint8([0, 0, 0])
        #         hsvCopy[y][x] = maskBlack # Todo: 왜 x,y 반대?, ref: http://docs.opencv.org/3.0-beta/doc/user_guide/ug_mat.html

    # cv2.imshow('test'+direction, hsvCopy)

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
    if event == cv2.EVENT_LBUTTONUP and frame is not None:
        copyImg = originImg.copy()
        height, width, channels = frame.shape

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

    # print '_yellowCenterArr', _yellowCenterArr # _yellowCenterArr [(54, 165), (386, 159)]

    # _orangeCenterArr [(38, 172), (403, 157)]
    # print '_orangeCenterArr', _orangeCenterArr #

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
    cv2.circle(frame, leftTopCenter, 10, (0, 0, 255), 3)
    cv2.circle(frame, rightTopCenter, 10, (0, 0, 255), 3)

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
    cv2.line(frame, leftTopCenter, rightTopCenter, (0, 0, 0), 3)

    # 골대 아래 가로
    cv2.line(frame, leftBottomCenter, rightBottomCenter, (0, 0, 0), 3)

    # 골대 세로
    # 왼쪽
    cv2.line(frame, leftTopCenter, leftBottomCenter, (0, 0, 0), 3)

    # 오른쪽
    cv2.line(frame, rightTopCenter, rightBottomCenter, (0, 0, 0), 3)




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
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

# 공이 중심을 찾고, 공이 들어왔는지 검사


## 색공간: HSV (360 100, 100)
# 형광 노랑
# 70.27624309392266, 71.25984251968505, 99.6078431372549
yellowLower = (60, 45, 45)
yellowUpper = (80, 100, 100)


# 형광 주황
# 26.424870466321234, 75.68627450980392, 100.0
orangePostLower = (20, 68, 80)
orangePostUpper= (40, 85, 110)

# 공 (주황)
orangeLower = (0, 85, 0)
orangeUpper = (45, 110, 255)

##################################################
############### Main Function ####################
##################################################
cap = cv2.VideoCapture('Futsal_Manager.mp4')

# For Output Config
fps = 30
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'MPEG')  # note the lower case
vout = cv2.VideoWriter()
success = vout.open('Futsal_Manager_Out.mp4', fourcc, fps, size, False)

if MODE == WEBCAM_MODE:
    camera = cv2.VideoCapture(0)

# 무한루프
while True:
    # Start time
    start = time.time()
    global grabbed
    global frame

    # 현재 프레임을 잡아냄
    if MODE == WEBCAM_MODE:
        (grabbed, frame) = camera.read()
    elif MODE == IMAGE_MODE:
        imgPath = 'futsalsta.jpeg'  # ball_fieldsample.jpg
        frame = cv2.imread(imgPath, 1)
    elif MODE == VIDEO_MODE:
        ret, img = cap.read()

    yellowCenterArr = []
    orangeCenterArr = []

    # get current positions of four trackbars
    BALL_ERROR_RANGE = cv2.getTrackbarPos('BALL_ERROR_RANGE', 'result')
    BALL_MIN_RADIUS = BALL_SIZE - BALL_ERROR_RANGE
    BALL_MAX_RADIUS = BALL_SIZE + BALL_ERROR_RANGE
    GOAL_POST_RANGE = cv2.getTrackbarPos('GOAL_POST_RANGE', 'result')

    if frame is None: # or not grabbed
        print'frame is non or not grabbed'
        break

    # 1. resize the frame
    # frame = imutils.resize(frame, width=X_DIMENSION, height=Y_DIMENSION)
    frame = cv2.resize(frame, (X_DIMENSION, Y_DIMENSION))

    cv2.putText(frame, 'Range: ' + str(BALL_MIN_RADIUS) + ' ~ ' + str(BALL_MAX_RADIUS), (0, 480),
                cv2.FONT_HERSHEY_SIMPLEX, 1, 20, 2)
    # 원본 이미지 복사
    originImg = frame.copy()

    # 3. 블러처리 (노이즈 제거)
    blur = cv2.GaussianBlur(frame, (GAUSIAN_BLUR_SIZE, GAUSIAN_BLUR_SIZE), 0)

    # 3. 노이즈 제거2 (부식 팽창)
    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)

    # 3. 색 공간 변경
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # 4. 왼쪽 마커 찾기
    findGoalPostByColorAndDirection([yellowLower, yellowUpper], [orangePostLower, orangePostUpper],DIRECTION_LEFT)

    # 5. 오른쪽 마커 찾기
    findGoalPostByColorAndDirection([yellowLower, yellowUpper], [orangePostLower, orangePostUpper],DIRECTION_RIGHT)

    # 주황(공) 위한 mask
    # radius = findBallColor(orangeLower, orangeUpper)

    # 6. Canny Edge Detect
    # cv2.putText(frame, 'real ball pixel' + str(radius), (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, 20, 2)
    cv2.putText(frame, '0.07 convert: ' + str(BALL_SIZE), (0, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, 20, 2)

    # 'BALL_SIZE'
    cv2.imshow('frame', frame)

    cv2.setMouseCallback('frame', show_color)

    # End time
    end = time.time()

    # Time elapsed
    seconds = 1 / (end - start)

# Q를 누르면 프로그램 종료
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


# 카메라 클리어
if MODE == WEBCAM_MODE:
    camera.release()

# 열려있는 창 닫음
cv2.destroyAllWindows()