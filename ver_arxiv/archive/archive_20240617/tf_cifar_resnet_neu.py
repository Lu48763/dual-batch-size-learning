import tensorflow as tf
from tensorflow import keras
# should be modified

def load_cifar100(
    batch_size: int,
    validation_batch_size: int,
    seed: int = None,
    num_parallel_calls: int = tf.data.AUTOTUNE
):
    def map_preprocessing(image):
        # for cifar-10
        #mean = [0.49137255, 0.48235294, 0.44666667]
        #variance = [0.06103806, 0.05930657, 0.06841815]
        # for cifar-100
        mean = [0.50705882, 0.48666667, 0.44078431]
        variance = [0.07153003, 0.06577716, 0.0762193 ]
        transform = keras.Sequential([
            keras.layers.Rescaling(1/255),
            keras.layers.Normalization(mean=mean, variance=variance)
        ])
        return transform(image)
    
    def map_augmentation(image):
        transform = keras.Sequential([
            keras.layers.RandomTranslation(
                height_factor=0.1,
                width_factor=0.1,
                fill_mode='constant'
            ),
            keras.layers.RandomFlip('horizontal')
        ])
        return transform(image)
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    dataloader = {
        'train': (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                  .map(lambda x, y: (map_preprocessing(x), y),
                       num_parallel_calls=num_parallel_calls)
                  .cache()
                  .shuffle(buffer_size=len(x_train), seed=seed)
                  .map(lambda x, y: (map_augmentation(x), y),
                       num_parallel_calls=num_parallel_calls)
                  .batch(batch_size=batch_size)
                  .prefetch(buffer_size=tf.data.AUTOTUNE)),
        'test': (tf.data.Dataset.from_tensor_slices((x_test, y_test))
                 .map(lambda x, y: (map_preprocessing(x), y),
                      num_parallel_calls=num_parallel_calls)
                 .cache()
                 .batch(batch_size=validation_batch_size)
                 .prefetch(buffer_size=tf.data.AUTOTUNE))
    }
    return dataloader


def make_resnet18(
    inputs: keras.Input = keras.Input(shape=(32, 32, 3)),
    classes: int = 100
) -> keras.Model:
    def basicblock(x: keras.Input, filters: int, conv_shortcut: bool = False):
        if conv_shortcut:
            shortcut = keras.layers.Conv2D(filters, 1, strides=2)(x)
            shortcut = keras.layers.BatchNormalization(epsilon=1.001e-5)(shortcut)
            x = keras.layers.Conv2D(filters, 3, strides=2, padding='same')(x)
        else:
            shortcut = x
            x = keras.layers.Conv2D(filters, 3, padding='same')(x)
        x = keras.layers.BatchNormalization(epsilon=1.001e-5)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(filters, 3, padding='same')(x)
        x = keras.layers.BatchNormalization(epsilon=1.001e-5)(x)
        x = keras.layers.Add()([shortcut, x])
        x = keras.layers.Activation('relu')(x)
        return x
    
    filters = 16
    x = keras.layers.Conv2D(filters, 3, padding='same')(inputs)
    x = keras.layers.BatchNormalization(epsilon=1.001e-5)(x)
    x = keras.layers.Activation('relu')(x)
    x = basicblock(x, filters)
    x = basicblock(x, filters)
    x = basicblock(x, filters)
    filters *= 2
    x = basicblock(x, filters, True)
    x = basicblock(x, filters)
    x = basicblock(x, filters)
    filters *= 2
    x = basicblock(x, filters, True)
    x = basicblock(x, filters)
    x = basicblock(x, filters)
    x = keras.layers.Dropout(rate=0.2)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(classes, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=outputs)
