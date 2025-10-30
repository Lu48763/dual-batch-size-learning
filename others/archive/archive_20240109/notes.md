# Note
## Test for Using `weight_decay`
### resolution = 32 x 32
- model weight_decay "ON"
    - BS = 100
        - loss 2.0965
        - acc 0.6914
    - BS = 500
        - loss 2.1999
        - acc 0.6547
- model weight_decay "OFF"
    - BS = 100
        - loss 1.7303
        - acc 0.6675
    - BS = 500
        - loss 1.5586
        - acc 0.6471
## Test for Different Training Setting
### cifar-100
- same BS
    - iter RES
        - loss: 0.9121 - accuracy: 0.9657 - val_loss: 2.3022 - val_accuracy: 0.6487
    - cycle RES
        - loss: 0.7401 - accuracy: 0.9932 - val_loss: 2.0872 - val_accuracy: 0.6771
- diff BS
    - iter RES
        - loss: 0.9030 - accuracy: 0.9740 - val_loss: 2.4487 - val_accuracy: 0.6259
    - cycle RES
        - loss: 0.8260 - accuracy: 0.9905 - val_loss: 2.1457 - val_accuracy: 0.6765
### imagenet
- diff BS
    - iter RES
        - 
    - cycle RES
        - loss: 1.2527 - accuracy: 0.7801 - val_loss: 1.7567 - val_accuracy: 0.6752
