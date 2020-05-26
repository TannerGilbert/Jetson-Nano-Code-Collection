# Detectron2 on the Jetson Nano

![](https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png)

Detectron2 can be run on the Jetson Nano the same way as on any Linux device.

## Installation

See the official [installation guide](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

## Install using Docker

Another great way to install Detectron2 is by using Docker. Docker is great because you don't need to install anything locally, which allows you to keep your machine nice and clean.

If you want to run Detectron2 with Docker you can find a Dockerfile and docker-compose.yml file in the [docker directory of the repository](https://github.com/facebookresearch/detectron2/tree/master/docker).

For those of you who also want to use Jupyter notebooks inside their container, I created a custom Docker configuration, which automatically starts Jupyter after running the container. If you're interested you can find the files in the [docker directory of my Detectron2 repository](https://github.com/TannerGilbert/Object-Detection-and-Image-Segmentation-with-Detectron2/tree/master/docker).

## Inference with pre-trained model

Using a pre-trained model for inference is as easy as loading in the cofiguration and weights and creating a predictor object. For a example check out [Detectron2_inference_with_pre_trained_model.ipynb](Detectron2_inference_with_pre_trained_model.ipynb).