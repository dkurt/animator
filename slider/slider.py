import cv2 as cv
import time
import numpy as np
import os
import argparse

from threading import Thread
import Queue

import win32api
import win32con

from helper import checkOrDownload

parser = argparse.ArgumentParser('Slider: a magical slides switcher using Intel OpenVINO')
parser.add_argument('-m', default='', dest='models', help='Path models folder. If not found, download it')
parser.add_argument('--cpu', default=0, help='Number of infer requests for CPU (0 to disable)')
parser.add_argument('--gpu', default=0, help='Number of infer requests for GPU, fp32 (0 to disable)')
parser.add_argument('--gpu_fp16', default=0, help='Number of infer requests for GPU, fp16 (0 to disable)')
parser.add_argument('--vpu', default=0, help='Number of infer requests for VPU (0 to disable)')
args = parser.parse_args()

class NetWrapper:
    def __init__(self, target, maxNumRequests, targetId):
        # Setup paths and download necessary files if needed
        binPath = 'human-pose-estimation-0001.bin'
        xmlPath = 'human-pose-estimation-0001.xml'
        baseURL = 'https://download.01.org/opencv/2019/open_model_zoo/R1/models_bin/human-pose-estimation-0001/'
        if target == cv.dnn.DNN_TARGET_MYRIAD or target == cv.dnn.DNN_TARGET_OPENCL_FP16:
            precision = 'FP16'
            binHash = '6640f79764d47a059e1240e12a294563197ccba6'
            xmlHash = '5f66315be602a3c517df92fddb1c1bed4d0cdb38'
        else:
            precision = 'FP32'
            binHash = '8a34fea398b1c91834a4dfb0386ea07d19a4810f'
            xmlHash = '3821247086aee45daee7f38e8f7293130f357c99'

        binURL = baseURL + '/' + precision + '/' + binPath
        xmlURL = baseURL + '/' + precision + '/' + xmlPath
        binPath = os.path.join(args.models, precision, binPath)
        xmlPath = os.path.join(args.models, precision, xmlPath)
        checkOrDownload(binPath, binURL, binHash)
        checkOrDownload(xmlPath, xmlURL, xmlHash)

        # Initialize network
        self.net = cv.dnn.readNet(binPath, xmlPath)
        self.net.setPreferableTarget(target)
        self.numRequests = 0
        self.processedFrames = 0
        self.maxNumRequests = maxNumRequests
        self.targetId = targetId

        #
        # Warm up
        #
        self.net.setInput(np.random.standard_normal([1, 3, 256, 256]).astype(np.float32))
        self.net.forward()


    def process(self, frame):
        self.numRequests += 1

        blob = cv.dnn.blobFromImage(frame, 1.0, (256, 256))
        self.net.setInput(blob)
        return self.net.forwardAsync()


class Slider:
    def __init__(self):
        self.rWristHistory = []
        self.lWristHistory = []

    def process(self, frame, out):
        RWRIST = 4
        LWRIST = 7

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
                self.rWristHistory.append(point[0] if conf > 0.1 else None)
            else:
                self.lWristHistory.append(point[0] if conf > 0.1 else None)

        if len(self.rWristHistory) == 31:
            del self.rWristHistory[0]
            if self.isSwipe(self.rWristHistory, 'left'):
                win32api.keybd_event(win32con.VK_RIGHT, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0) #press
                win32api.Sleep(50)
                win32api.keybd_event(win32con.VK_RIGHT, 0, win32con.KEYEVENTF_EXTENDEDKEY | win32con.KEYEVENTF_KEYUP, 0) #release
                self.rWristHistory = []
                print 'left'

        if len(self.lWristHistory) == 31:
            del self.lWristHistory[0]
            if self.isSwipe(self.lWristHistory, 'right'):
                win32api.keybd_event(win32con.VK_LEFT, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0) #press
                win32api.Sleep(50)
                win32api.keybd_event(win32con.VK_LEFT, 0, win32con.KEYEVENTF_EXTENDEDKEY | win32con.KEYEVENTF_KEYUP, 0) #release
                self.lWristHistory = []
                print 'right'


    def isSwipe(self, xs, type):
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
                if type == 'left' and x > lastX or type == 'right' and x < lastX:
                    numMoves += 1
                lastX = x
        return numMoves >= 7


framesQueue = Queue.Queue()
process = True
def framesCaptureThread():
    global framesQueue, process

    cap = cv.VideoCapture(1)

    # print (cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G')))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv.CAP_PROP_FPS, 30)
    # print("Camera FPS: %d" % cap.get(cv.CAP_PROP_FPS))

    while process:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        framesQueue.put(frame)


nets = []
if args.cpu:
    nets.append(NetWrapper(cv.dnn.DNN_TARGET_CPU, args.cpu, 'CPU'))
if args.vpu:
    nets.append(NetWrapper(cv.dnn.DNN_TARGET_MYRIAD, args.vpu, 'MYRIAD'))
if args.gpu:
    nets.append(NetWrapper(cv.dnn.DNN_TARGET_OPENCL, args.gpu, 'GPU FP32'))
if args.gpu_fp16:
    nets.append(NetWrapper(cv.dnn.DNN_TARGET_OPENCL_FP16, args.gpu_fp16, 'GPU FP16'))
if not nets:
    print('Please select one of the targets (see --help)')
    exit(0)

thread = Thread(target=framesCaptureThread)
thread.start()

cv.namedWindow("cam", cv.WINDOW_NORMAL)

# Skip the first frame to wait camera readiness
framesQueue.get()

# Main processing loop
startTime = time.time()
lastLogTime = startTime
framesToRender = []
futureMats = []
slider = Slider()
while True:
    try:
        for i in range(len(nets)):
            if nets[i].numRequests < nets[i].maxNumRequests:
                frame = framesQueue.get()
                futureMats.append((i, nets[i].process(frame)))
                framesToRender.append(frame)
    except Queue.Empty:
        pass

    # Check for finished requests.
    a = len(futureMats)
    while futureMats and futureMats[0][1].wait_for(0) == 0:
        netId = futureMats[0][0]
        out = futureMats[0][1].get()
        nets[netId].numRequests -= 1
        nets[netId].processedFrames += 1
        del futureMats[0]

        slider.process(framesToRender[0], out)

        cv.imshow("cam", framesToRender[0])
        cv.waitKey(1)
        del framesToRender[0]

    currTime = time.time()
    if currTime - lastLogTime > 1:
        for net in nets:
            print("%s: %.2f FPS" % (net.targetId, net.processedFrames / (currTime - startTime)))
        print('')
        lastLogTime = currTime

process = False
thread.join()
