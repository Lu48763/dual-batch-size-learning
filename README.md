# Hybrid Dual-Batch and Cyclic Progressive Learning for Efficient Distributed Training
- K. -W. Lu, P. Liu, D. -Y. Hong and J. -J. Wu, "Efficient Dual Batch Size Deep Learning for Distributed Parameter Server Systems," 2022 IEEE 46th Annual Computers, Software, and Applications Conference (COMPSAC), Los Alamitos, CA, USA, 2022, pp. 630-639, doi: 10.1109/COMPSAC54236.2022.00110.
- Lu, K., Hong, D., Liu, P., & Wu, J. (2025). Efficient Distributed Training via Dual Batch Sizes and Cyclic Progressive Learning. ArXiv, abs/2509.26092.

## Environment
- python 3.11
- tensorflow 2.13
- pytorch 2.1
- pytorch-cuda 12.1

## Installation
1. Create a conda virtual environment and install required packages:
    ```
    conda create -n hdbl -c pytorch -c nvidia python=3.11 tensorflow=2.13 pytorch=2.1 pytorch-cuda=12.1 
    ```
2. (Optional) Install additional packages for data analysis:
    ```
    conda install -n hdbl matplotlib scikit-learn jupyterlab
    ```
3. Activate the virtual environment:
    ```
    conda activate hdbl
    ```
4. Notes:
    - Avoid running `conda update --all`, as it may cause dependency conflicts or package errors.
    - Comment out the line `127.0.1.1 ${SERVER_NAME}` in `/etc/hosts` to work around a known `torch.distributed` bug.

## Citation
```
@misc{lu2025efficientdistributedtrainingdual,
      title={Efficient Distributed Training via Dual Batch Sizes and Cyclic Progressive Learning}, 
      author={Kuan-Wei Lu and Ding-Yong Hong and Pangfeng Liu and Jan-Jan Wu},
      year={2025},
      eprint={2509.26092},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2509.26092}, 
}
```

## Miscellaneous
### Check Whether CUDA Is Available
```
import torch
import tensorflow as tf
torch.cuda.device_count()
tf.config.list_physical_devices('GPU')
```

### Jupyter Remote Configuration
1. python
    - from jupyter_server.auth import passwd
    - passwd()
        [argon2:xxx]
2. jupyter
    - `jupyter notebook`
    - jupyter server --generate-config
    - vim ~/.jupyter/jupyter_server_config.py
        - c.ServerApp.ip = '*'
        - c.ServerApp.open_browser = False
        - c.ServerApp.password = u'argon2:xxxx'
        - c.ServerApp.port = 8763

### Settings
- For tensorflow old version, try to use `imgaug` instead of `keras-cv` for doing image augmentation.
- Dataset
    - CIFAR
        - objects classes: 10 / 100
        - training images: 50000
        - validation images: 10000
    - ImageNet
        - objects classes: 1000
        - training images: 1281167
        - validation images: 50000
- Maximum Batch Size for GTX-1080
    - CIFAR
        - resolution_ls = [24, 32]
        - batch_size_ls = [600, 560]
        - batch_size_ls = [430, 580], for `--xla`
    - ImageNet
        - resolution_ls = [160, 224, 288]
        - batch_size_ls = [340, 170, 100], for `--amp`
        - batch_size_ls = [550, 160, 100], for `--amp --xla`
- Test Maximum Batch Size
    - `python record_batchSize_trainTime.py -r=${RES} -d=${DATA} -p=/ssd --start=${TEST_BS} --stop=5001 --step=10000 -t=10 ${--amp --xla --no-save}`
- Record Training Time
    - `--start=5 --step=5 --take=50`
    - `python record_batchSize_trainTime.py -r=${RESOLUTION} -d=${DATASET} -p=/ssd --start= --stop= --step= -t=50 ${--amp --xla}`
    - ex. `python record_batchSize_trainTime.py -r=32 -d=cifar100 -p=/ssd --start=5 --stop=561 --step=10 -t=50`
- Training
    - `python main.py -r= -w= -s= -a= -d= -p=/ssd -t=1.05 ${--amp --xla}`
    - ex.
        - for server, `python main.py -r=0 -w=2 -s=0 -a=$(IP_ADDRESS) -d=cifar100 -p=/ssd -t=1.05`
        - for worker, `python main.py -r=1 -w=2 -s=0 -a=$(IP_ADDRESS) -d=cifar100 -p=/ssd -t=1.05`
- Experiments
    - GTX-1080 for cifar100
        - `python exp1080/main.py -r= -w=5 -s= -a=140.109.23.106 -d=cifar100 -t=1.05`
        - file = `main.py` / `main_conf.py`
        - t = `1.05` / `1.1`
    - RTX-3090 for imagent with `--amp`
        - `python exp3090/main_3090.py -r= -w=5 -s= -a=140.109.23.231 -d=imagenet -p=/data -t=1.05 --amp`
        - file = `main_3090.py` / `main_conf_3090.py`
        - t = `1.05` / `1.1`
