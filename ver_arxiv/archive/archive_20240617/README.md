# Experimental of Efficient Dual Batch Size Deep Learning for Distributed Parameter Server Systems
<!--
K. -W. Lu, P. Liu, D. -Y. Hong and J. -J. Wu, "Efficient Dual Batch Size Deep Learning for Distributed Parameter Server Systems," 2022 IEEE 46th Annual Computers, Software, and Applications Conference (COMPSAC), 2022, pp. 630-639, doi: [10.1109/COMPSAC54236.2022.00110](https://doi.org/10.1109/COMPSAC54236.2022.00110).
-->

## Dataset and Model
- The CIFAR dataset
    - A. Krizhevsky, G. Hinton, et al., [“Learning multiple layers of features from tiny images,”](https://www.cs.toronto.edu/~kriz/) Citeseer, 2009.
- The ResNet model
    - K. He, X. Zhang, S. Ren, and J. Sun, [“Deep residual learning for image recognition,”](https://doi.org/10.1109/CVPR.2016.90) in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.
## Create Folders
`mkdir DBSL_npy DBSL_model`
If you do not do this, the results can not be saved.

## Environment (experimental)
- python 3.11
- pytorch 2.0
- torchvision 0.15
- pytorch-cuda 11.8
- tensorflow 2.12
- keras-core 0.1
- keras-cv 0.6
- cudatoolkit 11.8
- nvidia-cudnn-cu11 8.9

## Installation (experimental)
### method 1, mix conda and pypi, nice but hard to manage packages
1. Menage conda environment:
    - if analyzing results:
        ```
        conda update --all -c pytorch -c conda-forge -c nvidia python cudatoolkit notebook matplotlib scikit-learn pytorch torchvision pytorch-cuda
        ```
    - else:
        ```
        conda update --all -c pytorch -c conda-forge -c nvidia python cudatoolkit pytorch torchvision pytorch-cuda
        ```
2. Menage pypi packages, install TensorFlow:
    - tensorflow 2.13 has some problem, try to use 2.12 temporarily.
    ```
    pip install -U tensorflow==2.12 keras-cv keras-core nvidia-cudnn-cu11
    ```
3. ~~Set the PATH environment variable: (depricated)~~
    ```
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    ```
### method 2, only conda, failed
The conda package conflicts, resulting in an error.
```
conda update --all -c pytorch -c conda-forge -c nvidia python=3.11 notebook matplotlib scikit-learn pytorch torchvision pytorch-cuda tensorflow=*=gpu* keras-cv cudatoolkit cudnn
```

## Environment (deprecated)
- python 3.10
- pytorch 2.0
- torchvision 0.15
- pytorch-cuda 11.8
- tensorflow 2.11

## Installation (deprecated)
1. Upgrade CUDA at https://developer.nvidia.com/cuda-downloads.
2. Create a new virtual environment. Recommend using `conda` to control it.
    ```
    conda create -n dbsl
    ```
3. Activate the virtual environment.
    ```
    conda activate dbsl
    ```
4. Install **essential** conda packages.
    ```
    conda update --all -c pytorch -c nvidia -c conda-forge python scikit-learn pytorch torchvision tensorflow
    ```
    - You could also appoint the package's version, e.g., `python=3.11`.
5. Install **optional** conda packages.
    ```
    conda update --all -c pytorch -c nvidia -c conda-forge matplotlib notebook pandas
    ```

<!--
## DBSL
Run `DBSL.py` by:
```
python DBSL.py -a='$(serverIP)' -w=$(wordSize) -r=$(rank)
```
- You should check ufw first
  - need the permission to access any `port` of the devices
  - `ufw allow from $(deviceIP)`
  - maybe you also need to modify `/etc/hosts` and comment `127.0.0.1 localhost`
  - suck PyTorch RPC zzz...
- addres: Server IP
- world: numbers of machines on parameter server
- rank: 1~(w-1) if worker, 0 if server
- hyperparameters in code:
    - a, b: device information, get from linear regression
    - num_GPU, num_small
    - base_BS, base_LR
    - extra_time_ratio
    - rounds, threshold, gamma

## Plot Figure
Please use `Makefile` under the directory `plot`.
1. gnuplot: `make gnuplot`
2. pyplot: `make pyplot`
3. both: `make`
4. clean: `make clean`
-->