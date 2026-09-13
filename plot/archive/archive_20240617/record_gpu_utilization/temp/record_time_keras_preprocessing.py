#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision


# In[ ]:


parser = argparse.ArgumentParser(
        description='Too Simple! Sometimes Naive!'
)
parser.add_argument(
    '-m', '--mixed',
    type=bool,
    default=False,
    help='MIXED_PRECISION'
)
parser.add_argument(
    '-j', '--jit',
    type=bool,
    default=False,
    help='JIT_COMPILE'
)
parser.add_argument(
    '-s0', '--start',
    type=int,
    default=1,
    help='start_batch_size'
)
parser.add_argument(
    '-s1', '--stop',
    type=int,
    default=500,
    help='stop_batch_size'
)
parser.add_argument(
    '-s2', '--step',
    type=int,
    default=1,
    help='step'
)
# parser.parse_args() used in .py
# parser.parse_args('') used in .ipynb
args = parser.parse_args()


# In[ ]:


# Optimization Setting
device_index = -1
MIXED_PRECISION_FLAG = args.mixed
JIT_COMPILE_FLAG = args.jit

# Dataloader Setting
start_batch_size = args.start
stop_batch_size = args.stop
step = args.step

# Training Setting
learning_rate = 1e-2
momentum = 0.9
epochs = 1

# record training time
ls = []


# In[ ]:


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[device_index], 'GPU')
#tf.config.experimental.set_memory_growth(physical_devices[device_index], True)


# In[ ]:


if MIXED_PRECISION_FLAG:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)


# In[ ]:


def load_cifar100():
    def data_preprocessing(x, y):
        mean = tf.constant([129.3, 124.1, 112.4]) / 255
        std = tf.constant([68.2, 65.4, 70.4]) / 255
        pre = keras.Sequential([
            layers.Rescaling(1/255),
            layers.Normalization(mean=mean, variance=tf.math.square(std))
        ])
        return pre(x), keras.utils.to_categorical(y)
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    x_train, y_train = data_preprocessing(x_train, y_train)
    x_test, y_test = data_preprocessing(x_test, y_test)
    return (x_train, y_train), (x_test, y_test)


# In[ ]:


def make_resnet18(inputs: keras.Input = keras.Input(shape=(32, 32, 3)),
                  num_classes: int = 100
                 ) -> keras.Model:
    def basicblock(inputs: keras.Input, filters: int, bottleneck: bool):
        if bottleneck:
            identity = layers.Conv2D(filters, 1, strides=2, padding='valid',
                                     kernel_initializer='he_normal'
                                    )(inputs)
            identity = layers.BatchNormalization()(identity)
            x = layers.Conv2D(filters, 3, strides=2, padding='same',
                              kernel_initializer='he_normal'
                             )(inputs)
        else:
            identity = inputs
            x = layers.Conv2D(filters, 3, strides=1, padding='same',
                              kernel_initializer='he_normal',
                             )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, strides=1, padding='same',
                          kernel_initializer='he_normal',
                         )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, identity])
        x = layers.Activation('relu')(x)
        return x
    
    x = layers.Conv2D(64, 3, strides=1, padding='same',
                      kernel_initializer='he_normal',
                     )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = basicblock(x, 64, False)
    x = basicblock(x, 64, False)
    x = basicblock(x, 128, True)
    x = basicblock(x, 128, False)
    x = basicblock(x, 256, True)
    x = basicblock(x, 256, False)
    x = basicblock(x, 512, True)
    x = basicblock(x, 512, False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=outputs)


# In[ ]:


(x_train, y_train), (x_test, y_test) = load_cifar100()


# In[ ]:


datagen = keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)


# In[ ]:


for batch_size in range(start_batch_size, stop_batch_size+1, step):
    # set model
    model = make_resnet18()
    model.compile(
        optimizer=(
            keras.mixed_precision.LossScaleOptimizer(
                keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
            ) if MIXED_PRECISION_FLAG
            else keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        ),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
        jit_compile=JIT_COMPILE_FLAG
    )
    # warmup
    model.fit(
        datagen.flow(x_train[:batch_size], y_train[:batch_size], batch_size=batch_size),
        epochs=1,
        workers=tf.data.AUTOTUNE
    )
    # record training time
    t = time.monotonic()
    logs = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_test, y_test),
        validation_batch_size=batch_size,
        workers=tf.data.AUTOTUNE
    )
    t = time.monotonic() - t
    # print results
    print('----')
    print(f'BATCH_SIZE: {batch_size}')
    print(f'MIXED_PRECISION: {MIXED_PRECISION_FLAG}')
    print(f'JIT_COMPILE: {JIT_COMPILE_FLAG}')
    print(f'TIME: {t}')
    print(f'LOGS: {logs.history}')
    print('----')
    # append to list
    ls.append(t)


# In[ ]:


np.save(
    'record_time_keras_preprocessing_'
    f'{1 if MIXED_PRECISION_FLAG else 0}{1 if JIT_COMPILE_FLAG else 0}_'
    f'{start_batch_size}_{stop_batch_size}.npy',
    np.array(ls)
)


# In[ ]:




