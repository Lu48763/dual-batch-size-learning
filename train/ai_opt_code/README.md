# Hybrid Dual-Batch and Cyclic Progressive Learning for ImageNet (RTX 3090 Optimized)

This directory contains the implementation of the Progressive Dual Batch Size Deep Learning system, specifically optimized for training **ImageNet** on **NVIDIA RTX 3090** distributed parameter server systems. It leverages TensorFlow for model training and PyTorch RPC for distributed coordination.

## Environment
- **Python:** 3.11
- **TensorFlow:** 2.13
- **PyTorch:** 2.1 (for `torch.distributed.rpc`)
- **CUDA:** 12.1 (Cuda-nvcc 11.8)

## Installation
1. **Create the Conda Environment:**
    ```bash
    conda create -n hdbl -c pytorch -c nvidia python=3.11 tensorflow=2.13 pytorch=2.1 pytorch-cuda=12.1 cuda-nvcc=11.8
    ```
2. **(Optional) Install Data Analysis Tools:**
    ```bash
    conda install -n hdbl matplotlib scikit-learn jupyterlab typing_extensions
    ```
3. **Activate the Environment:**
    ```bash
    conda activate hdbl
    ```
4. **System Note:** 
    To avoid `torch.distributed` connectivity issues, it is recommended to comment out the line `127.0.1.1 ${SERVER_NAME}` in `/etc/hosts`.

## Usage
The training process requires one Parameter Server (Rank 0) and one or more Workers (Rank 1+). The system automatically handles **distributed data sharding**, ensuring each worker trains on a unique portion of the dataset.

### Core Arguments
- `-r`, `--rank`: Global rank (0 for Server, 1+ for Workers).
- `-w`, `--world-size`: Total number of participants (Server + Workers).
- `-s`, `--small`: Number of workers assigned to use small batch sizes.
- `-a`, `--addr`: IP address of the Parameter Server.
- `-d`, `--dataset`: Only `imagenet` is supported in this version.
- `-p`, `--path`: Path to the dataset directory.
    - **Important:** The code expects a structure like `${path}/imagenet/train/[class_id]/*.jpg` and `${path}/imagenet/val/[class_id]/*.jpg`.
- `-t`, `--ratio`: Permitted additional training time ratio (e.g., 1.05 for 5% extra time). This parameter influences the balancing of dual batch sizes.
- `--amp`: Enable Mixed Precision training (Highly recommended for RTX 3090).
- `--xla`: Enable JIT compilation with XLA.

### Execution Examples

**Start the Parameter Server (Rank 0):**
```bash
python main_3090.py -r=0 -w=5 -s=2 -a=$(SERVER_IP) -d=imagenet -p=/data -t=1.05 --amp
```

**Start a Worker (e.g., Rank 1):**
```bash
python main_3090.py -r=1 -w=5 -s=2 -a=$(SERVER_IP) -d=imagenet -p=/data -t=1.05 --amp
```

## System Architecture
- **`main_3090.py`**: Entry point that initializes RPC and orchestrates the server/worker roles.
- **`parameter_server_3090.py`**: Contains the `Server` and `Worker` class logic, including the dual-batch calculations and milestone management.
- **`tf_data_model.py`**: Handles ResNet model construction and ImageNet data loading pipelines using TensorFlow.

## Citation
If you use this code in your research, please cite our work:

```bibtex
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
