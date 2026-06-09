#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
import tensorflow as tf

from time import perf_counter
from tensorflow import keras

import tf_data_model


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
device_index = 1
MIXED_PRECISION_FLAG = True
JIT_COMPILE_FLAG = False # cause error

# Data && Model Setting
## dataloader setting
dataset = "cifar100" # ["cifar10", "cifar100", "imagenet"]
batch_size = 1024 # [32, 64, 128, 256, 512, 1024]
## auto setting
dataset_size = 1_281_167 if dataset == "imagenet" else 50_000 if "cifar" in dataset else None

# Training Setting
## loss function
learning_rate = 1e-1 # usually (1e-1 * (batch_size / 128))
momentum = 0.9 # 0.9
weight_decay = 1e-4 # 1e-4
## epochs && decay lr_decay_multiplier
epochs = 90 # 90
warmup_epochs = 5 # 5
lr_decay_multiplier = 1e-2
## lr scheduler
### if using warmup, [(warmup_target <- initial_learning_rate) && (initial_learning_rate <- final_learning_rate)]
warmup_target = learning_rate
alpha = lr_decay_multiplier
initial_learning_rate = learning_rate * lr_decay_multiplier
steps_per_epoch = math.ceil(dataset_size / batch_size)
decay_steps = epochs * steps_per_epoch
warmup_steps = warmup_epochs * steps_per_epoch


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


dataloader = tf_data_model.load_data(32, batch_size, dataset, None, 1024)
model = tf_data_model.modify_resnet(dataset, 18, 0.2, 32)
#model.summary()


# In[ ]:


lr_cosine_decay_scheduler = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    alpha=alpha,
    warmup_target=warmup_target,
    warmup_steps=warmup_steps,
)

time_callback = TimeCallback()


# In[ ]:


model.compile(
    optimizer=keras.optimizers.experimental.SGD(
        learning_rate=lr_cosine_decay_scheduler,
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
    epochs=(warmup_epochs + epochs),
    verbose=2,
    callbacks=[time_callback],
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




