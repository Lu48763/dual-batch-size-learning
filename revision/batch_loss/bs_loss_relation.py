#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf

from time import perf_counter
from tensorflow import keras

import tf_data_model


# In[ ]:


print(tf.__version__)
for gpu in tf.config.list_physical_devices("GPU"):
    print(tf.config.experimental.get_device_details(gpu))


# In[ ]:


# Optimization Setting
device_index = 1
MIXED_PRECISION_FLAG = True
JIT_COMPILE_FLAG = False # cause error

# Data && Model Setting
## Dataloader Setting
batch_size = 1024 # [32, 64, 128, 256, 512, 1024]

# Training Setting
epochs = 140 # 140
## loss function
learning_rate = 1e-1 # usually (1e-1 * (batch_size / 128))
momentum = 0.9 # 0.9
weight_decay = 1e-4 # 1e-4
## lr scheduler
milestones = [80, 120] # [80, 120]
gamma = 0.1


# In[ ]:


physical_devices = tf.config.list_physical_devices('GPU')
print(f'Numbers of Physical Devices: {len(physical_devices)}')
tf.config.set_visible_devices(physical_devices[device_index], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[device_index], True)
print(f'Using device: {physical_devices[device_index]}')
print(tf.config.experimental.get_device_details(physical_devices[device_index]))


# In[ ]:


# only TPUs support 'mixed_bfloat16'
# if using NVIDIA GPUs, choose 'mixed_float16'
if MIXED_PRECISION_FLAG:
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)
    print(f'Policy: {policy.name}')
    print(f'Compute dtype: {policy.compute_dtype}')
    print(f'Variable dtype: {policy.variable_dtype}')
    #keras.mixed_precision.set_dtype_policy('mixed_float16')
    #print(f'{keras.mixed_precision.dtype_policy()}')


# In[ ]:


def lr_schedule(epoch, lr, milestones, gamma: float = 0.1):
    if epoch in milestones:
        lr *= gamma
    return lr

class TimeCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.history = []
    def on_epoch_begin(self, epoch, logs=None):
        self.time_epoch_begin = perf_counter()
    def on_epoch_end(self, epoch, logs=None):
        self.history.append(perf_counter() - self.time_epoch_begin)

lr_scheduler_callback = keras.callbacks.LearningRateScheduler(
    lambda x, y: lr_schedule(x, y, milestones=milestones, gamma=gamma)
)
time_callback = TimeCallback()


# In[ ]:


dataloader = tf_data_model.load_data(32, batch_size, "cifar100", None, 1024)


# In[ ]:


model = tf_data_model.modify_resnet("cifar100", 18, 0.2, 32)
#model.summary()


# In[ ]:


model.compile(
    optimizer=keras.optimizers.experimental.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
    jit_compile=JIT_COMPILE_FLAG,
)


# In[ ]:


logs = model.fit(
    dataloader['train'],
    epochs=epochs,
    verbose=2,
    callbacks=[lr_scheduler_callback, time_callback],
    validation_data=dataloader['val'],
)
logs.history['t'] = time_callback.history


# In[ ]:


# logs.history -> type : dict
np.save(f"logs_e_{epochs}_bs_{batch_size}.npy", logs.history)


# In[ ]:


temp = np.load(f"logs_e_{epochs}_bs_{batch_size}.npy", allow_pickle=True).item()
print("----")
print(f"save_file: logs_e_{epochs}_bs_{batch_size}.npy")
print(f"epochs: {epochs}")
print(f"batch_size: {batch_size}")
print(temp.keys())


# In[ ]:




