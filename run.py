import cv2 as cv
import numpy as np

def getBoardPoints():
    pts = []
    for x in range(0, 9):
        for y in range(0, 6):
            pts.append([x, y, 0])
    return np.array(pts, dtype=np.float32)


PATTERNS_SIZE = (9, 6)
cameraMatrix = np.array([[6.4227919864848855e+02, 0., 3.1273499298151864e+02],
                         [0., 6.4227919864848855e+02, 2.4291734893727576e+02],
                         [0., 0., 1.]], dtype=np.float32)
distCoeffs = np.array([0., 6.0115531735378080e-01,
                       2.7148752303135076e-01, 3.1928492846617351e-01], dtype=np.float32)

pts = getBoardPoints()

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        break

    frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(frameGray, PATTERNS_SIZE)
    if found:
        cv.drawChessboardCorners(frame, PATTERNS_SIZE, corners, True);
        _, rvec, tvec = cv.solvePnP(pts, corners.reshape(-1, 2), cameraMatrix, distCoeffs, useExtrinsicGuess=False)
        print rvec
        print ''
        print tvec
        print '-----'

    cv.imshow('res', frame)


# F = 631.893 +- 3.52409
# Cx = 309.484 +- 1.71287 	Cy = 251.918 +- 1.93963
# K1 = 0.0386118 +- 0.0546481
# K2 = -0.136619 +- 0.633714
# K3 = -0.354965 +- 2.09671
# TD1 = 0 +- 0
# TD2 = 0 +- 0
