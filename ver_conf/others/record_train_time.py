#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


# In[ ]:


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')


# In[ ]:


def load_cifar10():
    def data_preprocessing(x, y):
        mean = tf.constant([125.3, 123.0, 113.9]) / 255
        std = tf.constant([63.0, 62.1, 66.7]) / 255
        pre = keras.Sequential([
            layers.Rescaling(1/255),
            layers.Normalization(mean=mean, variance=tf.math.square(std))
        ])
        return pre(x), keras.utils.to_categorical(y)
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train, y_train = data_preprocessing(x_train, y_train)
    x_test, y_test = data_preprocessing(x_test, y_test)
    return (x_train, y_train), (x_test, y_test)

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


model = make_resnet18()
(x_train, y_train), (x_test, y_test) = load_cifar100()


# In[ ]:


model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=1e-1, momentum=0.9),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)


# In[ ]:


datagen = keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)


# In[ ]:


workers = 2
epochs = 5
validation_batch_size = 1000

batch_size = 500
data_size = 13125


# In[ ]:


x_train, y_train = x_train[:data_size], y_train[:data_size]


# In[ ]:


logs = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    epochs=1,
    #validation_data=(x_test, y_test),
    #validation_batch_size=validation_batch_size,
    workers=workers
)


# In[ ]:


times = time.time()

logs = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    epochs=epochs,
    #validation_data=(x_test, y_test),
    #validation_batch_size=validation_batch_size,
    workers=workers
)

times = time.time() - times
t_e = times / epochs
print(times, t_e)


# In[ ]:


np.save(f'{batch_size}_{data_size}_{t_e}.npy', times)


# In[ ]:




