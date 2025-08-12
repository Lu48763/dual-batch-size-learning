#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import time

import torch
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


# In[ ]:


workers = None
epochs = 1


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Different numbers of threads use in data preprocessing.')
    parser.add_argument(
        '-w', '--worker',
        type=int,
        help='Number of threads.')
    args = parser.parse_args()
    assert args.worker is not None, 'Must provide worker argument.'
    
    workers = args.worker


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


start_batch = 1
end_batch = 1000 + 1
step = 1
record_train_time = np.zeros((end_batch - start_batch) // step + 1)


# In[ ]:


counter = 0
for batch_size in range(start_batch, end_batch, step):
    print(f'Batch Size = {batch_size}')
    # warming
    model.fit(
        datagen.flow(x_train[:batch_size], y_train[:batch_size], batch_size=batch_size),
        epochs=epochs,
        workers=workers
    )
    # start
    record_train_time[counter] = time.time()
    # main
    model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        workers=workers
    )
    # end
    record_train_time[counter] = time.time() - record_train_time[counter]
    counter += 1


# In[ ]:


np.save(f'staticMem_trainTime_{start_batch}_{end_batch}_{step}_w{workers}.npy', record_train_time)


# In[ ]:


record_train_time


# In[ ]:




