import cv2 as cv
import numpy as np

PATTERN_SIZE = (9, 6)

PATTERN_WORLD_COORDS = []
# for y in reversed(range(0, PATTERN_SIZE[1])):
#     for x in range(0, PATTERN_SIZE[0]):
#         PATTERN_WORLD_COORDS.append([x, y, 0])
for z in reversed(range(0, PATTERN_SIZE[1])):
    for y in range(0, PATTERN_SIZE[0]):
        PATTERN_WORLD_COORDS.append([0, y, z])

PATTERN_WORLD_COORDS = np.array(PATTERN_WORLD_COORDS, dtype=np.float32)


class Camera:
    def __init__(self, cameraId):
        self.cap = cv.VideoCapture(cameraId)
        print '~~~~'
        print self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        print self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        print self.cap.get(cv.CAP_PROP_FPS)
        print self.cap.set(cv.CAP_PROP_FPS, 60)
        print self.cap.get(cv.CAP_PROP_FPS)
        self.cameraId = cameraId
        self.projMat = np.zeros((3, 4), dtype=np.float32)

        if cameraId == 0:
            self.cameraMatrix = np.array([[6.3659729074438474e+02, 0., 3.2094087540493729e+02],
                                          [0., 6.3659729074438474e+02, 2.4770099666051286e+02],
                                          [0., 0., 1.]], dtype=np.float32)
            self.distCoeffs = np.array([4.9396232740344444e-02, -3.1529222986493832e-01,
                                        -1.2330251486459315e-03, 1.0059487941726385e-02,
                                        -1.6321508465040835e-01], dtype=np.float32)

            # self.rvec = np.array([[-3.07532   ], [-0.02907131], [-0.2813097 ]], dtype=np.float32)
            # self.tvec = np.array([[-2.9641984], [ 3.9068174], [15.240325 ]], dtype=np.float32)
        else:
            self.cameraMatrix = np.array([[1.0571639461413613e+03, 0., 2.5225017081309610e+02],
                                          [0., 1.0578721556424782e+03, 2.5022501623381595e+02],
                                          [0., 0., 1.]], dtype=np.float32)
            self.distCoeffs = np.array([-1.1556418090418831e-01, 3.2658233173231233e-01,
                                        -2.9060310641188985e-03, 1.7311180451643903e-04,
                                        9.3374215572179953e-01], dtype=np.float32)

            # self.rvec = np.array([[ 2.6440668 ], [-0.01542352], [-0.59542775]], dtype=np.float32)
            # self.tvec = np.array([[-7.1999574], [ 1.5072976], [37.020607 ]], dtype=np.float32)

        # self.projMat = computeProjMat(self.cameraMatrix, self.rvec, self.tvec)


    def captureFrame(self):
        _, frame = self.cap.read()
        return frame


    def estimateCameraPos(self, frame):
        frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(frameGray, PATTERN_SIZE)
        if found:
            cv.drawChessboardCorners(frame, PATTERN_SIZE, corners, True);

            _, rvec, tvec = cv.solvePnP(PATTERN_WORLD_COORDS, corners.reshape(-1, 2),
                                        self.cameraMatrix, self.distCoeffs,
                                        useExtrinsicGuess=False)

            rm, _ = cv.Rodrigues(rvec)
            self.projMat = np.dot(self.cameraMatrix, np.hstack((rm, tvec)))
        return found

    # def captureFrame(self):
    #     hasFrame, frame = self.cap.read()
    #     pos = None
    #     if hasFrame:
    #         frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #         found, corners = cv.findChessboardCorners(frameGray, PATTERN_SIZE)
    #         if found:
    #             cv.drawChessboardCorners(frame, PATTERN_SIZE, corners, True);
    #
    #             _, rvec, tvec = cv.solvePnP(PATTERN_WORLD_COORDS, corners.reshape(-1, 2),
    #                                         self.cameraMatrix, self.distCoeffs,
    #                                         useExtrinsicGuess=False)
    #             rm, _ = cv.Rodrigues(rvec)
    #             pos = np.dot(-rm.transpose(), tvec)
    #
    #         cv.imshow("camera %d" % self.cameraId, frame)
    #     return pos
