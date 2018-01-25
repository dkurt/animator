import cv2 as cv
import numpy as np
import argparse
import json
import os

from math import cos, sin, pi, atan2, acos, sqrt

parser = argparse.ArgumentParser(
        description='This script is used to demonstrate OpenPose human pose estimation network '
                    'from https://github.com/CMU-Perceptual-Computing-Lab/openpose project using OpenCV. '
                    'The sample and model are simplified and could be used for a single person on the frame.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--proto', required=True, help='Path to .prototxt')
parser.add_argument('--model', required=True, help='Path to .caffemodel')
parser.add_argument('--config', required=True, help='Path to .json file with character description')
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--output', help='Path to output directory to write frames. Display output on the screen if not specified.')
parser.add_argument('--width', default=368, type=int, help='Resize network input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize network input to specific height.')
parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')
args = parser.parse_args()

# Names of body joints. NOTE: do not shuffle them because an order coresponds to
# heatmaps are predicted by deep learning network.
JOINTS = ["Head", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
          "LWrist", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "Chest"]

# Skeleton as a hierarchy of bones where an every bone is a pair of connected joints.
BODY = ["Chest", "Neck",
            ["Neck", "Head"],
            ["Neck", "RShoulder", ["RShoulder", "RElbow", ["RElbow", "RWrist"] ] ],
            ["Neck", "LShoulder", ["LShoulder", "LElbow", ["LElbow", "LWrist"] ] ],
            ["Chest", "RHip", ["RHip", "RKnee", ["RKnee", "RAnkle"] ] ],
            ["Chest", "LHip", ["LHip", "LKnee", ["LKnee", "LAnkle"] ] ] ]

# Bone object connected to some parent Bone.
class Bone(object):
    # Length in pixels, angle in degrees.
    def __init__(self, jointFrom, jointTo, length, angle, parent=None):
        self.img = None
        self.order = 0
        self.jointFromPoint = None
        self.jointToPoint = None
        self.length = length
        self.parent = parent
        self.angle = angle
        self.jointFrom = jointFrom
        self.jointTo = jointTo

    def startPoint(self):
        if self.parent:
            if self.jointFrom == self.parent.jointTo:
                return self.parent.endPoint()
            else:
                return self.parent.startPoint()
        else:
            return (500, 500)

    def computeAngle(self):
        return self.angle + (self.parent.angle if self.parent else 0)

    # Returns end point of a line segment with a corresponding [length] starts at [start] point.
    # A line segment has a specific [angle] with an abscissa.
    def endPoint(self):
        # NOTE: Abscissa is from left to right and ordinate is from top to bottom.
        #       Angle is computed in clock-wise direction
        #    +-----> x    0
        #    |           .
        #    |         .
        # y  v      .
        #   90  .
        start = self.startPoint()
        angle = self.computeAngle()
        return (start[0] + self.length * cos(angle * pi / 180),
                start[1] + self.length * sin(angle * pi / 180))

    def draw(self, img):
        start = self.startPoint()
        end = self.endPoint()

        # cv.line(img, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (0, 255, 0), thickness=3)
        # cv.ellipse(img, (int(start[0]), int(start[1])), (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
        # cv.ellipse(img, (int(end[0]), int(end[1])), (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        if self.img is None:
            return

        angle = atan2(end[1] - start[1], end[0] - start[0])
        angle -= atan2(self.jointToPoint[1] - self.jointFromPoint[1],
                       self.jointToPoint[0] - self.jointFromPoint[0])

        mat = np.array([[1.0, 0.0, -self.jointFromPoint[0]],
                        [0.0, 1.0, -self.jointFromPoint[1]],
                        [0.0, 0.0, 1.0]])
        mat = np.dot(np.array([[cos(angle), -sin(angle), start[0]],
                        [sin(angle), cos(angle), start[1]]]), mat)
        wrappedImg = cv.warpAffine(self.img, mat, (img.shape[1], img.shape[0]))
        np.copyto(img, wrappedImg[:,:,:3], where=(wrappedImg[:,:,3:] != 0))

        # cv.line(img, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (0, 255, 0), thickness=3)
        # cv.ellipse(img, (int(start[0]), int(start[1])), (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
        # cv.ellipse(img, (int(end[0]), int(end[1])), (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)


class Skeleton(object):
    def __init__(self):
        self.bones = []

        def build(part, parentBone=None):
            jointFrom = part[0]
            jointTo = part[1]
            bone = Bone(jointFrom, jointTo, 20, 0, parentBone)
            self.bones.append(bone)
            for i in range(2, len(part)):
                build(part[i], bone)

        build(BODY)

    def draw(self, img):
        for bone in sorted(self.bones, key=lambda b: b.order):
            bone.draw(img)


class Character(Skeleton):
    def __init__(self, configPath):
        Skeleton.__init__(self)

        with open(configPath, 'rt') as f:
            config = json.load(f)

        img = cv.imread(config["image"], cv.IMREAD_UNCHANGED)

        for i, bone in enumerate(config["bones"]):
            if "skin" in bone:
                x = int(bone["skin"]["x"])
                y = int(bone["skin"]["y"])
                width = int(bone["skin"]["width"])
                height = int(bone["skin"]["height"])
                self.bones[i].img = img[y:y+height, x:x+width]
            self.bones[i].jointFromPoint = (int(bone["joint_from"]["x"]) - x,
                                            int(bone["joint_from"]["y"]) - y)
            self.bones[i].jointToPoint = (int(bone["joint_to"]["x"]) - x,
                                          int(bone["joint_to"]["y"]) - y)
            self.bones[i].length = sqrt((int(bone["joint_to"]["x"]) - int(bone["joint_from"]["x"])) ** 2 +
                                        (int(bone["joint_to"]["y"]) - int(bone["joint_from"]["y"])) ** 2)
            angle = atan2(int(bone["joint_to"]["y"]) - int(bone["joint_from"]["y"]),
                          int(bone["joint_to"]["x"]) - int(bone["joint_from"]["x"])) * 180 / pi
            self.bones[i].angle = angle - (self.bones[i].parent.angle if self.bones[i].parent else 0)
            self.bones[i].order = int(bone["order"]) if "order" in bone else 0


character = Character(args.config)

inWidth = args.width
inHeight = args.height

net = cv.dnn.readNet(args.proto, args.model)

cap = cv.VideoCapture(args.input if args.input else 0)

frameId = 0
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    inp = cv.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()

    # Extract predicted joints
    points = []
    for i in range(len(JOINTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        # Add a point if it's confidence is higher than threshold.
        points.append((x, y) if conf > args.thr else None)

    # Draw detected bones on the frame.
    for bone in character.bones:
        idFrom = JOINTS.index(bone.jointFrom)
        idTo = JOINTS.index(bone.jointTo)

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))


    # Update character's bones corresponding to person's ones.
    for bone in character.bones:
        idFrom = JOINTS.index(bone.jointFrom)
        idTo = JOINTS.index(bone.jointTo)

        if points[idFrom] and points[idTo]:
            angle = atan2(points[idTo][1] - points[idFrom][1], points[idTo][0] - points[idFrom][0])
            if not bone.img is None:
                bone.angle = angle * 180 / pi - (bone.parent.angle if bone.parent else 0)


    bonesImg = np.zeros((1000, 1000, 3), dtype=np.uint8)
    character.draw(bonesImg)

    cv.imshow('OpenPose using OpenCV', frame)
    cv.imshow('bones', bonesImg)
    if args.output:
        cv.imwrite(os.path.join(args.output, '%06d.png' % frameId), bonesImg)
        frameId += 1
