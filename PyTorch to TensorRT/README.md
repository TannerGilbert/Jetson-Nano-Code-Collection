# PyTorch to TensorRT

PyTorch models can be converted to TensorRT using the [torch2trt converter](https://github.com/NVIDIA-AI-IOT/torch2trt).

torch2trt is a PyTorch to TensorRT converter which utilizes the TensorRT Python API. The converter is
* Easy to use - Convert modules with a single function call torch2trt
* Easy to extend - Write your own layer converter in Python and register it with @tensorrt_converter

## Installation

```bash
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo python setup.py install
```

## Conversion Example

```python
import torch
from torch2trt import torch2trt
from torchvision.models.alexnet import alexnet

# create some regular pytorch model...
model = alexnet(pretrained=True).eval().cuda()

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])
```

## Make predictions

The TRTModule can be used just like the original PyTorch model.

```python
y = model(x)
y_trt = model_trt(x)

# check the output against PyTorch
print(torch.max(torch.abs(y - y_trt)))
```

## Save and load model

We can save the model as a ``state_dict``.

```python
torch.save(model_trt.state_dict(), 'alexnet_trt.pth')
```

We can load the saved model into a ``TRTModule``

```python
from torch2trt import TRTModule

model_trt = TRTModule()

model_trt.load_state_dict(torch.load('alexnet_trt.pth'))
```
## Further information

For more information check out the [torch2trt repository](https://github.com/NVIDIA-AI-IOT/torch2trt).