# Slider: a magical slides switcher using Intel OpenVINO

This is a computer vision and deep learning application which is used for
intelligent slides switching with help of Intel OpenVINO toolkit.

It detects human's wrists positions and recognizes left/right hands swipes using
[Human Pose Estimation model](https://github.com/opencv/open_model_zoo/tree/master/intel_models/human-pose-estimation-0001).
Depends on the gesture, it imitates keyboard's left or right arrows pressing.
So you can navigate forward and backward among your slides.

By default, this script works with a camera @ 30fps so to achieve robust repsonds
it's required to process frames with speed more than camera framerate. During
testing I tried the following configurations:

| hardware | FPS with Number asynchronous requests == 2 |
|---|---|
| CPU (Intel&reg; Core&trade; i5-6300U) | 12.3 |
| GPU, fp32 (Intel&reg; HD Graphics 520) | 16.5 |
| GPU, fp16 (Intel&reg; HD Graphics 520) | 21.5 |
| Myriad X (Intel&reg; Neural Compute Stick) | 21.1 |

As you may see none of hardware targets couldn't get desired efficiency to be able to process all the frames without skips.

But! You may easy use any combination of them so all the frames from camera will
be processed in time! So for 30fps camera input you can have 30 processed frames per second.
In example, to use both GPU and VPU targets:

```
python.exe slider.py --vpu 2 --gpu_fp16 2
```

All the devices perform with 15FPS and we have 30fps in total and no skipped frames.

```
MYRIAD: 14.84 FPS
GPU FP16: 14.84 FPS

MYRIAD: 14.83 FPS
GPU FP16: 14.83 FPS

MYRIAD: 14.81 FPS
GPU FP16: 14.81 FPS

MYRIAD: 14.82 FPS
GPU FP16: 14.82 FPS
```

**NOTE**: The script is tested only with Windows and Python 2.
