import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def load_cifar10():
    def data_preprocessing(x, y):
        mean = np.array([125.3, 123.0, 113.9]) / 255
        std = np.array([63.0, 62.1, 66.7]) / 255
        pre = keras.Sequential([
            layers.Rescaling(1/255),
            layers.Normalization(mean=mean, variance=np.square(std))
        ])
        return pre(x), keras.utils.to_categorical(y)
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train, y_train = data_preprocessing(x_train, y_train)
    x_test, y_test = data_preprocessing(x_test, y_test)
    return (x_train, y_train), (x_test, y_test)

def load_cifar100():
    def data_preprocessing(x, y):
        mean = np.array([129.3, 124.1, 112.4]) / 255
        std = np.array([68.2, 65.4, 70.4]) / 255
        pre = keras.Sequential([
            layers.Rescaling(1/255),
            layers.Normalization(mean=mean, variance=np.square(std))
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
