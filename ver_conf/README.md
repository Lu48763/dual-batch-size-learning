# Efficient Dual Batch Size Deep Learning for Distributed Parameter Server Systems

## Environment
- python 3.9
- cudatoolkit 11.3
- pytorch 1.10.1
- tensorflow 2.6.2
- torchvision 0.11.2

## Install Command
1. Please use conda and create a new environment by:
    ```
    conda create -n $(env_name)
    ```
2. Install packages:
    1. vim `~/.condarc` (Optional)
    ```
    channels:
      - pytorch
      - nvidia
      - conda-forge
      - defaults
    ```

    2. install all packages
    ```
    conda install matplotlib notebook scikit-learn python=3.9 \
    cudatoolkit=11.3 pytorch=1.10 torchvision=0.11 tensorflow=2.6 \
    -n $(env_name) -c pytorch -c conda-forge
    ```

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
