import cv2 as cv
import time
import numpy as np
# from scipy.signal import argrelmax

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'));
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv.CAP_PROP_FPS, 30)

cv.namedWindow("cam", cv.WINDOW_NORMAL)
net = cv.dnn.readNet('human-pose-estimation-0001.bin',
                     'human-pose-estimation-0001.xml')
# net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

#
# Warm up
#
# net.setInput(np.random.standard_normal([1, 3, 128, 344]).astype(np.float32))
# net.forward()

def isLeftSwipe(xs):
    # Just to make things easier
    if xs[0] is None or xs[-1] is None:
        return False

    numNones = 0
    numMoves = 0
    lastX = xs[0]
    for x in xs:
        if x is None:
            numNones += 1
        else:
            if x > lastX:
                numMoves += 1
            lastX = x
    return numMoves >= len(xs) / 3


RWRIST = 4
LWRIST = 7

rWristHistory = []

processedFrames = 0
startTime = time.time()
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        break

    blob = cv.dnn.blobFromImage(frame, 1.0, (320, 180), ddepth=cv.CV_8U)
    net.setInput(blob)
    out = net.forward()

    # Get heat maps for boths wrists.
    heatMap = out[0, [RWRIST, LWRIST], :, :]

    _, rWristConf, _, rWristPoint = cv.minMaxLoc(out[0, RWRIST, :, :])
    _, lWristConf, _, lWristPoint = cv.minMaxLoc(out[0, LWRIST, :, :])

    if abs(rWristPoint[0] - lWristPoint[0]) <= 2 and \
       abs(rWristPoint[1] - lWristPoint[1]) <= 2:
       if lWristConf > rWristConf:
           out[0, RWRIST, rWristPoint[1]-2:rWristPoint[1]+3, rWristPoint[0]-2:rWristPoint[0]+3] = 0
           _, rWristConf, _, rWristPoint = cv.minMaxLoc(out[0, RWRIST, :, :])
       else:
           out[0, LWRIST, lWristPoint[1]-2:lWristPoint[1]+3, lWristPoint[0]-2:lWristPoint[0]+3] = 0
           _, lWristConf, _, lWristPoint = cv.minMaxLoc(out[0, LWRIST, :, :])

    colors = [(0, 255, 0), (0, 0, 255)]
    points = [rWristPoint, lWristPoint]
    confidences = [rWristConf, lWristConf]
    for i in range(2):
        point = points[i]
        color = colors[i]
        conf = confidences[i]
        x = (frame.shape[1] * point[0]) / out.shape[3]
        y = (frame.shape[0] * point[1]) / out.shape[2]
        if conf > 0.1:
            cv.circle(frame, (x, y), 5, color, cv.FILLED)
        if i == 0:
            rWristHistory.append(point[0] if conf > 0.1 else None)

    # print len(rWristHistory)
    if len(rWristHistory) == 31:
        del rWristHistory[0]

        if isLeftSwipe(rWristHistory):
            print 'left'
            rWristHistory = []
        # isRightSwipe = True
        # isLeftSwipe = True
        # for i in range(1, len(rWristHistory)):
        #     isRightSwipe = isRightSwipe and rWristHistory[i] >= rWristHistory[i - 1]
        #     isLeftSwipe = isLeftSwipe and rWristHistory[i] <= rWristHistory[i - 1]
        #
        # if isRightSwipe:
        #     print 'right'
        # if isLeftSwipe:
        #     print 'left'

    processedFrames += 1
    cv.imshow("cam", frame)

    # print processedFrames / (time.time() - startTime)
