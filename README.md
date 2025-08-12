# Dual Batch Size Learning

## Environment
- python 3.11
- tensorflow 2.13
- pytorch 2.1
- pytorch-cuda 12.1

## Installation and Activation
```
conda create -n dbsl -c pytorch -c nvidia python=3.11 tensorflow=2.13 pytorch=2.1 pytorch-cuda=12.1
conda activate dbsl
```

Optional:
`conda install matplotlib scikit-learn keras-cv torchvision`

## Check Whether CUDA Available
```
import torch
import tensorflow as tf
torch.cuda.device_count()
tf.config.list_physical_devices('GPU')
```