# Efficient Dual Batch Size Deep Learning for Distributed Parameter Server Systems
- K. -W. Lu, P. Liu, D. -Y. Hong and J. -J. Wu, "Efficient Dual Batch Size Deep Learning for Distributed Parameter Server Systems," 2022 IEEE 46th Annual Computers, Software, and Applications Conference (COMPSAC), Los Alamitos, CA, USA, 2022, pp. 630-639, doi: 10.1109/COMPSAC54236.2022.00110.

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

## Jupyter Remote Setting
- packages: `jupyterlab`, `notebook`, `nbclassic`
1. python
    - from jupyter_server.auth import passwd
    - passwd()
        [argon2:xxx]
2. jupyter
    - for `jupyterlab`:
        - `jupyter lab`
        - the setting is same as `notebook`
        - the optional themes are `jupyterlab_legos_ui` and `jupyterlab_darkside_ui`
    - for `notebook`:
        - `jupyter notebook`
        - jupyter server --generate-config
        - vim ~/.jupyter/jupyter_server_config.py
            - c.ServerApp.ip = '*'
            - c.ServerApp.open_browser = False
            - c.ServerApp.password = u'argon2:xxxx'
            - c.ServerApp.port = 8763
    - for `nbclassic`:
        - `jupyter nbclassic`
        - jupyter notebook --generate-config
        - the optional theme is `jupyterthemes`
            - `jt -t oceans16`
        - vim ~/.jupyter/jupyter_notebook_config.py
            - c.ExtensionApp.open_browser = False
            - c.ServerApp.ip = '*'
            - c.ServerApp.password = u'argon2:xxx'
            - c.ServerApp.port = 8763

## Citation
```
@INPROCEEDINGS{lu2022efficient,
  author={Lu, Kuan-Wei and Liu, Pangfeng and Hong, Ding-Yong and Wu, Jan-Jan},
  booktitle={2022 IEEE 46th Annual Computers, Software, and Applications Conference (COMPSAC)}, 
  title={Efficient Dual Batch Size Deep Learning for Distributed Parameter Server Systems}, 
  year={2022},
  volume={},
  number={},
  pages={630-639},
  keywords={Training;Deep learning;Computational modeling;Neural networks;Predictive models;Data models;Hardware;deep neural networks;batch size;distributed learning;parameter server},
  doi={10.1109/COMPSAC54236.2022.00110}}
```