import cv2 as cv
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from threading import Thread, Lock

from camera import Camera

cameras = [Camera(0), Camera(1)]
#
# Estimate cameras world positions
#

while cv.waitKey(1) < 0:
    frames = [camera.captureFrame() for camera in cameras]
    for i in range(len(cameras)):
        cameras[i].estimateCameraPos(frames[i])
        cv.imshow('camera %d' % i, frames[i])

# def projMat(cm, rvec, tvec):
#     rm, _ = cv.Rodrigues(rvec)
#     return np.dot(cm, np.hstack((rm, tvec)))
#
# # 0 5 0
# rvec0 = np.array([[-3.07532   ], [-0.02907131], [-0.2813097 ]], dtype=np.float32)
# tvec0 = np.array([[-2.9641984], [ 3.9068174], [15.240325 ]], dtype=np.float32)
# cm0 = np.array([[6.3659729074438474e+02, 0., 3.2094087540493729e+02],
#                 [0., 6.3659729074438474e+02, 2.4770099666051286e+02],
#                 [0., 0., 1.]], dtype=np.float32)
# pts0 = np.array([[203.63197], [202.2079] ], dtype=np.float32)
#
# rvec1 = np.array([[ 2.6440668 ], [-0.01542352], [-0.59542775]], dtype=np.float32)
# tvec1 = np.array([[-7.1999574], [ 1.5072976], [37.020607 ]], dtype=np.float32)
# cm1 = np.array([[1.0571639461413613e+03, 0., 2.5225017081309610e+02],
#                 [0., 1.0578721556424782e+03, 2.5022501623381595e+02],
#                 [0., 0., 1.]], dtype=np.float32)
# pts1 = np.array([[69.019295], [167.40955] ], dtype=np.float32)
#
#
# print cv.triangulatePoints(projMat(cm0, rvec0, tvec0), projMat(cm1, rvec1, tvec1), pts0, pts1)
#

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3,
               "LShoulder": 5, "LElbow": 6,
               "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17}


POSE_PAIRS = [ ["Neck", "Nose"], ["Nose", "REye"], ["Neck", "RShoulder"], ["Neck", "LShoulder"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"], ["RShoulder", "RElbow"], ["LShoulder", "LElbow"] ]


net = cv.dnn.readNet('human-pose-estimation-0001.bin', 'human-pose-estimation-0001.xml')

poses = []
mutex = Lock()
def detectionThread():
    global poses, mutex, cameras

    while cv.waitKey(1) < 0:
        frames = []
        for camera in cameras:
            frames.append(camera.captureFrame())

        blob = cv.dnn.blobFromImages(frames, 1.0, (456, 256))
        net.setInput(blob)
        out = net.forward()

        mutex.acquire()
        poses = []
        for part in BODY_PARTS.values():
            pts = []
            # color = (0, 255, 0) if part == NOSE else (0, 0, 255)
            color = (0, 255, 0)
            for i in range(2):
                _, conf, _, point = cv.minMaxLoc(out[i, part, :, :])
                x = (frames[i].shape[1] * point[0]) / out.shape[3]
                y = (frames[i].shape[0] * point[1]) / out.shape[2]
                pts.append(np.array([[x], [y]], dtype=np.float32))
                cv.circle(frames[i], (x, y), 5, color, cv.FILLED)
                cv.imshow("camera %d" % i, frames[i])

            p = cv.triangulatePoints(cameras[0].projMat, cameras[1].projMat, pts[0], pts[1])
            p[0] /= p[3]
            p[1] /= p[3]
            p[2] /= p[3]
            poses.append((p[0], p[1], p[2]))
        mutex.release()


thread = Thread(target=detectionThread)
thread.start()

def render():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBegin(GL_QUADS)
    glColor3ub(255, 255, 255)
    glVertex3f(0, 0, 0)
    glVertex3f(8, 0, 0)
    glVertex3f(8, 6, 0)
    glVertex3f(0, 6, 0)
    glEnd()

    glLineWidth(5)
    glBegin(GL_LINES)

    glColor3ub(255, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(100, 0, 0)

    glColor3ub(0, 255, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 100, 0)

    glColor3ub(0, 0, 255)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, 100)

    # rvec, tvec
    # [[-0.01792515]
    #  [ 2.6178017 ]
    #  [-2.1431963 ]]
    #
    # [[ 4.731706 ]
    #  [-1.5379528]
    #  [20.450714 ]]
    #

    glEnd()

    glColor3ub(255, 0, 255)
    glMatrixMode(GL_MODELVIEW)
    mutex.acquire()
    for pos in poses:
        glPushMatrix()
        glTranslatef(pos[0], pos[1], pos[2])
        glutSolidSphere(0.5, 10, 10)
        glPopMatrix()

    if len(poses) == len(BODY_PARTS):
        for edge in POSE_PAIRS:
            idFrom = BODY_PARTS.keys().index(edge[0])
            idTo = BODY_PARTS.keys().index(edge[1])

            pointFrom = poses[idFrom]
            pointTo = poses[idTo]

            glBegin(GL_LINES)
            glColor3ub(255, 255, 0)
            glVertex3f(pointFrom[0], pointFrom[1], pointFrom[2])
            glVertex3f(pointTo[0], pointTo[1], pointTo[2])
            glEnd()


    mutex.release()

    glutSwapBuffers()

def reshape(w, h):
    fovy = 30
    zNear = 0.1
    zFar = 1000
    aspect = float(w) / h

    glViewport(0, 0, w, h)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(fovy, aspect, zNear, zFar)
    gluLookAt(20, -9, 70, 0, 0, 0, 0, 0, 1)


glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
glutInitWindowSize(0, 0)
glutInitWindowPosition(0, 0)
glutInit(sys.argv)
glutCreateWindow("My Window")
glutReshapeFunc(reshape)

glClearColor(0, 0, 0, 1)

glutIdleFunc(render)
glutMainLoop()
