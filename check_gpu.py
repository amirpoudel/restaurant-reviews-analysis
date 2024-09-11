import torch
print("PyTorch CUDA Available:", torch.cuda.is_available())

import tensorflow as tf
print("TensorFlow GPU Devices:", tf.config.list_physical_devices('GPU'))
