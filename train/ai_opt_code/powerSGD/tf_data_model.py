from typing import Optional

import tensorflow as tf
from tensorflow import keras


# Maintain WEIGHT_DECAY of different TF versions through OLD_VERSION
## "True" for TF 2.6 and other older versions, "False" for newer versions
## Support weight_dacay via keras.model instead of keras.optimizers
### the accuracy is so suck without kernel_regularizer, just use it
OLD_VERSION = False
weight_decay = 1e-4

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
var = [0.052441, 0.050176, 0.050625] # tf.math.square(std)
cifar_dataset_list = ['cifar10', 'cifar100']
imagenet_dataset_list = ['imagenet']
dataset_list = cifar_dataset_list + imagenet_dataset_list
depth_list = [18, 34]
cifar_resolution_list = [16, 24, 32]
imagenet_resolution_list = [160, 224, 288]


def get_prefetch_buffer_size(prefetch_buffer_size: int):
    if prefetch_buffer_size == -1:
        return tf.data.AUTOTUNE
    if prefetch_buffer_size < 1:
        raise ValueError('"prefetch_buffer_size" must be -1 or greater than or equal to 1')
    return prefetch_buffer_size


def load_cifar(resolution: int, batch_size: int, dataset: str, val_batch_size: int, num_shards: int = 1, shard_rank: int = 0, prefetch_buffer_size: int = -1):
    if resolution not in cifar_resolution_list:
        raise ValueError(f'Invalid resolution "{resolution}", it should be in {cifar_resolution_list}.')
    if dataset not in cifar_dataset_list:
        raise ValueError(f'Invalid resolution "{dataset}", it should be in {cifar_dataset_list}.')
    prefetch_size = get_prefetch_buffer_size(prefetch_buffer_size)
    
    def preprocessing_map(image):
        transform = keras.Sequential([
            keras.layers.Resizing(resolution, resolution)
        ])
        return transform(image)
    
    if dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    elif dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    
    dataloader = {
        'train': (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shard(num_shards, shard_rank)
            .repeat() # Ensure enough data for additional-time-ratio
            .map(
                lambda x, y: (preprocessing_map(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            .cache()
            .shuffle(buffer_size=50000)
            .batch(batch_size=batch_size)
            .prefetch(buffer_size=prefetch_size)
        ),
        'val': (
            tf.data.Dataset.from_tensor_slices((x_test, y_test))
            # Removed .shard() to ensure global evaluation on every worker
            .map(
                lambda x, y: (preprocessing_map(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(batch_size=val_batch_size)
            .cache()
            .prefetch(buffer_size=prefetch_size)
        )
    }
    
    return dataloader


def load_imagenet(resolution: int, batch_size: int, dir_path: str, val_batch_size: int, num_shards: int = 1, shard_rank: int = 0, prefetch_buffer_size: int = -1):
    if resolution not in imagenet_resolution_list:
        raise ValueError(f'Invalid resolution "{resolution}", it should be in {imagenet_resolution_list}.')
    prefetch_size = get_prefetch_buffer_size(prefetch_buffer_size)
    
    dataloader = {
        'train': (
            keras.utils.image_dataset_from_directory(
                directory=f'{dir_path}/imagenet/train',
                label_mode='int', # for keras.losses.SparseCategoricalCrossentropy()
                batch_size=batch_size,
                image_size=(resolution, resolution),
                shuffle=True,
                seed=48763 # Fixed seed ensures unique shards across nodes with identical datasets
            )
            .shard(num_shards, shard_rank)
            .repeat() # Ensure enough data for additional-time-ratio
            .prefetch(buffer_size=prefetch_size)
        ),
        'val': (
            keras.utils.image_dataset_from_directory(
                directory=f'{dir_path}/imagenet/val',
                label_mode='int', # for keras.losses.SparseCategoricalCrossentropy()
                batch_size=val_batch_size,
                image_size=(resolution, resolution),
                shuffle=False
            )
            # Removed .shard() to ensure global evaluation on every worker
            .prefetch(buffer_size=prefetch_size)
        )
    }
    
    return dataloader


def build_resnet(
    dataset: str,
    depth: int,
    dropout_rate: float,
    resolution: int,
) -> keras.Model:
    if dataset not in dataset_list:
        raise ValueError(f'Invalid dataset "{dataset}", it should be in {dataset_list}.')
    if depth not in depth_list:
        raise ValueError(f'Invalid depth "{depth}", it should be in {depth_list}.')
    
    # Kernel regularizer setup
    kernel_regularizer = keras.regularizers.L2(weight_decay) if OLD_VERSION else None
    
    if dataset == 'cifar10':
        classes = 10
    elif dataset == 'cifar100':
        classes = 100
    elif dataset == 'imagenet':
        classes = 1000
    
    stack_list = [2, 2, 2, 2] if depth == 18 else [3, 4, 6, 3]
    
    def basic_block(x: keras.Input, filters: int, conv_shortcut: bool = False):
        if conv_shortcut:
            shortcut = keras.layers.Conv2D(
                filters, 1, strides=2, use_bias=False,
                kernel_initializer='he_normal',
                kernel_regularizer=kernel_regularizer
            )(x)
            shortcut = keras.layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5)(shortcut)
            x = keras.layers.Conv2D(
                filters, 3, strides=2, padding='same', use_bias=False,
                kernel_initializer='he_normal',
                kernel_regularizer=kernel_regularizer
            )(x)
        else:
            shortcut = x
            x = keras.layers.Conv2D(
                filters, 3, padding='same', use_bias=False,
                kernel_initializer='he_normal',
                kernel_regularizer=kernel_regularizer
            )(x)
        x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(
            filters, 3, padding='same', use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer
        )(x)
        x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5)(x)
        x = keras.layers.Add()([shortcut, x])
        x = keras.layers.Activation('relu')(x)
        return x
    
    def basic_stack(x: keras.Input, filters: int, stack: int, conv_shortcut: bool = False):
        for i in range(stack):
            if i == 0 and conv_shortcut:
                filters *= 2
                x = basic_block(x, filters, True)
            else:
                x = basic_block(x, filters)
        return x, filters
    
    inputs = keras.Input(shape=(resolution, resolution, 3))
    
    # Data augmentation and normalization
    x = keras.layers.RandomFlip('horizontal')(inputs)
    x = keras.layers.RandomRotation(0.05)(x)
    x = keras.layers.RandomZoom(0.25)(x)
    x = keras.layers.Rescaling(1/255)(x)
    x = keras.layers.Normalization(mean=mean, variance=var)(x)
    
    filters = 64
    if 'cifar' in dataset:
        x = keras.layers.Conv2D(
            filters, 3, padding='same', use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer
        )(x)
        x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5)(x)
        x = keras.layers.Activation('relu')(x)
    elif dataset == 'imagenet':
        x = keras.layers.Conv2D(
            filters, 7, strides=2, padding='same', use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer
        )(x)
        x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    for i, stack in enumerate(stack_list):
        x, filters = basic_stack(x, filters, stack, conv_shortcut=(i > 0))
    
    x = keras.layers.GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(
        classes, activation='softmax',
        kernel_regularizer=kernel_regularizer
    )(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)



def load_data(
    resolution: int,
    batch_size: int,
    dataset: str,
    dir_path: Optional[str] = None,
    val_batch_size: Optional[int] = None,
    num_shards: int = 1,
    shard_rank: int = 0,
    prefetch_buffer_size: int = -1
):
    if val_batch_size == None:
        val_batch_size = batch_size
    if 'cifar' in dataset:
        return load_cifar(resolution=resolution, batch_size=batch_size, dataset=dataset, val_batch_size=val_batch_size, num_shards=num_shards, shard_rank=shard_rank, prefetch_buffer_size=prefetch_buffer_size)
    elif dataset == 'imagenet':
        if dir_path == None:
            raise ValueError(f'Invalid directory path "{dir_path}".')
        return load_imagenet(resolution=resolution, batch_size=batch_size, dir_path=dir_path, val_batch_size=val_batch_size, num_shards=num_shards, shard_rank=shard_rank, prefetch_buffer_size=prefetch_buffer_size)
    else:
        raise ValueError(f'Invalid dataset "{dataset}", it should be in {dataset_list}.')


def modify_resnet(
    dataset: str,
    depth: int,
    dropout_rate: float,
    resolution: int,
    old_model: Optional[keras.Model] = None
) -> keras.Model:
    # Retrieve weights before clearing the session to prevent reference invalidation
    weights = old_model.get_weights() if old_model else None
    
    # Clear session to prevent memory leaks when switching resolutions/models
    keras.backend.clear_session()
    
    new_model = build_resnet(
        dataset=dataset,
        depth=depth,
        dropout_rate=dropout_rate,
        resolution=resolution,
    )
    if weights:
        new_model.set_weights(weights)
    
    return new_model
