#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import time
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
    '-b', '--batch',
    type=int,
    default=500,
    help='batch_size'
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
batch_size = args.batch

# Training Setting
learning_rate = 1e-2
momentum = 0.9
epochs = 1


# In[ ]:


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[device_index], 'GPU')
#tf.config.experimental.set_memory_growth(physical_devices[device_index], True)


# In[ ]:


if MIXED_PRECISION_FLAG:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)


# In[ ]:


def load_cifar100(
    batch_size: int,
    seed: int = None,
    num_parallel_calls: int = tf.data.AUTOTUNE):
    
    def map_func(image, label, TRAIN_FLAG):
        mean = [0.50705886, 0.48666665, 0.4407843 ]
        variance = [0.07153001, 0.06577717, 0.0762193 ]
        transform = keras.Sequential([
            layers.Rescaling(1/255),
            layers.Normalization(mean=mean, variance=variance)
        ])
        if TRAIN_FLAG:
            transform = keras.Sequential([
                transform,
                layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='constant'),
                layers.RandomFlip('horizontal')
            ])
        return transform(image), label
    
    def dataloader(image, label, TRAIN_FLAG, batch_size, seed, num_parallel_calls):
        return (
            tf.data.Dataset.from_tensor_slices((image, label))
            .shuffle(buffer_size=len(label), seed=seed)
            .map(lambda x, y: (map_func(x, y, TRAIN_FLAG)), num_parallel_calls=num_parallel_calls)
            .batch(batch_size=batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    train_dataset = dataloader(x_train, y_train, True, batch_size, seed, num_parallel_calls)
    test_dataset = dataloader(x_test, y_test, False, batch_size, seed, num_parallel_calls)
    return train_dataset, test_dataset


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


train_dataset, test_dataset = load_cifar100(batch_size=batch_size)


# In[ ]:


model = make_resnet18()


# In[ ]:


model.compile(
    optimizer=(
        keras.mixed_precision.LossScaleOptimizer(
            keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        ) if MIXED_PRECISION_FLAG
        else keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
    jit_compile=JIT_COMPILE_FLAG
)


# In[ ]:


t = time.monotonic()
logs = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=test_dataset
)
t = time.monotonic() - t


# In[ ]:


print('----')
print(f'BATCH_SIZE: {batch_size}')
print(f'MIXED_PRECISION: {MIXED_PRECISION_FLAG}')
print(f'JIT_COMPILE: {JIT_COMPILE_FLAG}')
print(f'TIME: {t}')
print(f'LOGS: {logs.history}')
print('----')


# In[ ]:




