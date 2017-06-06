# -*- coding: utf-8 -*-
# 위 코드로 한글주석 처리가
# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import the necessary packages
import argparse
from collections import deque

import cv2
import numpy as np

# Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

# 최소, 최대 경계값 설정 (Note HSV Color Code)
# Todo: 공의 색깔을 미리 입력하거나 탐색하는 과정이 사전에 필요함.
orangeLower = (0, 150, 150)
orangeUpper = (25, 255, 255)
ballDetectCutlineY = 240

pts = deque(maxlen=args["buffer"])

frame = cv2.imread("sample/OrangeObjects.jpg")

# 2. blur it
blurred = cv2.GaussianBlur(frame, (11, 11), 0)

# 3. and convert it to the HSV color space
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# 'color'을 위한 마스크를 설정하고 실행
# 마스크에 남아있는 얼룩을 일련의 팽창(dilations) and 부식(erosions)을 통해 없앰
mask = cv2.inRange(hsv, orangeLower, orangeUpper)
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)

# 마스크에서 외곽(contours)을 찾고 현재 볼의 x,y좌표 중앙점을 초기화
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)[-2]
center = None

# 최소한 하나의 외곽(contours)를 찾았을때만 진행
if len(cnts) > 0:
    # 1. 마스크에서 가장 큰 외곽을 찾고,
    # 2. 가장 원의 최소를 둘러싸는 원을 찾고(노란 외곽선)
    # 3. 중심에 둔다
    c = max(cnts, key=cv2.contourArea)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:

        # print c
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        print center

        # 최소 반지름 이상일때만 진행
        if radius > 7 and radius < 60 and y >= ballDetectCutlineY: ## Todo: 상수로 선언이 필요함, 최소 반지름
            # 노란 외곽선을 그리고 중심점 표시
            # 추적할 좌표를 업데이트
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

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
    thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
    cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

# 프레임을 우리 화면에 보여줌
cv2.imshow("Frame", frame)
cv2.waitKey(0)

# 카메라 클리어하고 열려있는 창 닫음
cv2.destroyAllWindows()