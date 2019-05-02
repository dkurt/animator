import cv2 as cv
import time
import numpy as np

from threading import Thread
import Queue

framesQueue = Queue.Queue()
process = True
# help(framesQueue)
def framesCaptureThread():
    global framesQueue, process

    cap = cv.VideoCapture(1)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'));
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv.CAP_PROP_FPS, 30)
    print("Camera FPS: %d" % cap.get(cv.CAP_PROP_FPS))

    while process:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        framesQueue.put(frame)


maxNumRequests = 10
futureMats = []
framesToRender = []
# net = cv.dnn.readNet('human-pose-estimation-0001_fp16.bin',
#                      'human-pose-estimation-0001_fp16.xml')
# net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

# export LD_LIBRARY_PATH=/opt/intel/openvino/inference_engine/external/tbb/lib/:$LD_LIBRARY_PATH
net = cv.dnn.readNet('human-pose-estimation-0001.bin',
                     'human-pose-estimation-0001.xml')

#
# Warm up
#
net.setInput(np.random.standard_normal([1, 3, 128, 344]).astype(np.float32))
net.forward()

thread = Thread(target=framesCaptureThread)
thread.start()

cv.namedWindow("cam", cv.WINDOW_NORMAL)

# Main processing loop
processedFrames = 0
startTime = time.time()
while cv.waitKey(1) < 0:
    try:
        # while len(futureMats) < maxNumRequests:
        frame = framesQueue.get_nowait()
        framesToRender.append(frame)

        blob = cv.dnn.blobFromImage(frame, 1.0, (344, 128))
        net.setInput(blob)
        futureMats.append(net.forwardAsync())
    except Queue.Empty:
        pass

    a = len(futureMats)
    while futureMats and (futureMats[0].wait_for(0) == 0 or len(futureMats) >= maxNumRequests):
    # while futureMats and futureMats[0].wait_for(0) == 0:
        out = futureMats[0].get()
        del futureMats[0]
        processedFrames += 1
        cv.imshow("cam", framesToRender[0])
        del framesToRender[0]

    print processedFrames / (time.time() - startTime), a, len(futureMats), framesQueue.qsize()


process = False
thread.join()
print 1

# cv.namedWindow("cam", cv.WINDOW_NORMAL)
#
# numFrames = 0
# startTime = time.time()
# while cv.waitKey(1) < 0:
#     hasFrame, frame = cap.read()
#     if not hasFrame:
#         break
#     numFrames += 1
#     print (numFrames / (time.time() - startTime))
#     cv.imshow("cam", frame)
