#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf

from time import perf_counter
from tensorflow import keras

import tf_data_model_unified as tf_data_model


# In[ ]:


print(tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
print(f'Numbers of Physical Devices: {len(physical_devices)}')
#for gpu in tf.config.list_physical_devices("GPU"):
#    print(tf.config.experimental.get_device_details(gpu))


# In[ ]:


# Random Seed, useless
seed = 48763
keras.utils.set_random_seed(seed)

# Optimization Setting
device_index = 0 #!!!! select GPU index for the test
MIXED_PRECISION_FLAG = True
JIT_COMPILE_FLAG = False # cause error

# Data && Model Setting
## dataloader setting
dataset = "imagenet" # ["cifar10", "cifar100", "imagenet"]
batch_size = 500 # (5, 500+1, 5) [32, 64, 128, 256, 512, 1024]
depth = 18 # [18, 34]
steps_per_epoch = 100
start_bs = 5 # 5
step_size = start_bs # 5
## auto setting
dataset_size = 1_281_167 if dataset == "imagenet" else 50_000 if "cifar" in dataset else None
resolution = 32 if "cifar" in dataset else 224
dir_path = None if "cifar" in dataset else "/data"

# Training Setting
## loss function
learning_rate = 1e-1 # usually (1e-1 * (batch_size / 128))
momentum = 0.9 # 0.9
weight_decay = 1e-4 # 1e-4


# In[ ]:


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


class TimeCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.history = []
    def on_epoch_begin(self, epoch, logs=None):
        self.time_epoch_begin = perf_counter()
    def on_epoch_end(self, epoch, logs=None):
        self.history.append(perf_counter() - self.time_epoch_begin)


# In[ ]:


record = []
time_callback = TimeCallback()
base_dataloader = tf_data_model.load_data(resolution, dataset, dir_path)

for bs in range(start_bs, batch_size+1, step_size):
    # 0. show progress
    print(f"batch size = {bs}")
    # 1. model and data
    dataloader = tf_data_model.batch_dataloader(base_dataloader, bs)
    model = tf_data_model.modify_resnet(dataset, depth, 0.2, resolution)
    # 2. model compile
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
    # 3. model fit
    logs = model.fit(
        dataloader['train'].repeat(),
        epochs=2,
        verbose=2,
        callbacks=[time_callback],
        steps_per_epoch=steps_per_epoch,
    )
    # 4. record the result of "training time per batch"
    record.append(time_callback.history[-1] / steps_per_epoch)
    # 5. divider
    print("====")


# In[ ]:


np_record = np.array(record)


# In[ ]:


np.save(f"timePerBatch_{dataset}_d{depth}_{start_bs}_{batch_size+1}_{step_size}.npy", np_record)
temp = np.load(f"timePerBatch_{dataset}_d{depth}_{start_bs}_{batch_size+1}_{step_size}.npy")


# In[ ]:




