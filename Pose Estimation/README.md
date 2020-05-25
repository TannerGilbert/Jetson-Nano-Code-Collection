# Real-time Pose Estimation with TensorRT

![](doc/pose_estimation.gif)

The [NVIDIA-AI-IOT Github account](https://github.com/NVIDIA-AI-IOT) provides a [human pose detection project](https://github.com/NVIDIA-AI-IOT/trt_pose) that can run with up to 22FPS on the Jetson Nano. It contains both inference code as well as a training script that allows you to train on any keypoint task data in MSCOCO format. The repository makes use of PyTorch and [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt).

For more information check out [the repository](https://github.com/NVIDIA-AI-IOT/trt_pose).

## Installation

Requirements:
* PyTorch
* Torchvision
* TensorRT

```bash
sudo pip3 install tqdm cython pycocotools
sudo apt-get install python3-matplotlib
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
sudo python3 setup.py install
```

Installing torch2trt (needed to convert the PyTorch model to TensorRT):
```bash
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo python setup.py install
```

Installing Jetcam (used to access camera in the inference script):
```bash
git clone https://github.com/NVIDIA-AI-IOT/jetcam
cd jetcam
sudo python3 setup.py install
```

## Running demo application

To run a demo on a real-time camera input you need to work through the following steps.

1. Download model weights and place them into the tasks/human_pose directory.
    | Model | Jetson Nano | Jetson Xavier | Weights |
    |-------|-------------|---------------|---------|
    | resnet18_baseline_att_224x224_A | 22 | 251 | [download (81MB)](https://drive.google.com/open?id=1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd) |
    | densenet121_baseline_att_256x256_B | 12 | 101 | [download (84MB)](https://drive.google.com/open?id=13FkJkx7evQ1WwP54UmdiDXWyFMY1OxDU) |

2. Run the live_demo.ipynb notebook, which is located inside the tasks/human_pose directory.
