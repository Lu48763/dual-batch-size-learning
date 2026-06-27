# Hybrid Dual-Batch and Cyclic Progressive Learning for ImageNet

This directory contains the current `ai_opt_code` training flow for the paper implementation. It uses TensorFlow/Keras for model training and PyTorch RPC for the distributed parameter-server control path.

The current command-line interface is intentionally limited to the implementation that has matching timing coefficients and schedule logic:

- Dataset: `imagenet`
- Model: ResNet-18
- Hardware profile: RTX 3090 timing model
- Training mode: mixed precision AMP is required for ImageNet

`tf_data_model.py` still contains lower-level CIFAR and ResNet-34 helpers, but the distributed training entrypoint rejects those options because the parameter-server batch-size and timing schedule are not implemented for them.

## Environment

- Python: 3.11
- TensorFlow: 2.13
- PyTorch: 2.1, used for `torch.distributed.rpc`
- CUDA: 12.1 runtime with `cuda-nvcc` 11.8 in the original environment

Example conda setup:

```bash
conda create -n hdbl -c pytorch -c nvidia python=3.11 tensorflow=2.13 pytorch=2.1 pytorch-cuda=12.1 cuda-nvcc=11.8
conda activate hdbl
```

Optional analysis tools:

```bash
conda install -n hdbl matplotlib scikit-learn jupyterlab typing_extensions
```

For some machines, PyTorch RPC may bind to the wrong local hostname if `/etc/hosts` maps `127.0.1.1` to the server name. Check that setting if RPC workers cannot connect to rank 0.

## Files

- `main_3090.py`: CLI parser, argument validation, TensorFlow AMP/XLA setup, GPU selection, and PyTorch RPC initialization.
- `parameter_server_3090.py`: `Server` and `Worker` logic, dual-batch calculation, progressive resolution schedule, learning-rate milestones, validation logging, and output saving.
- `tf_data_model.py`: TensorFlow data loaders and ResNet construction.
- `tests/`: Lightweight regression tests for parser behavior and `--no-cycle` schedule behavior.

## Dataset Layout

`--dir-path` should point to the directory that contains the `imagenet` folder:

```text
${dir_path}/imagenet/train/<class_id>/*.jpg
${dir_path}/imagenet/val/<class_id>/*.jpg
```

The training dataset is sharded across workers with `Dataset.shard(num_shards=world_size - 1, shard_rank=rank - 1)`. Validation is not sharded; every worker evaluates on the full validation dataset after each local training commit.

## Usage

The distributed job has one parameter server and one or more workers:

- Rank 0: parameter server.
- Rank 1 and above: workers.
- `--world-size` is the total process count, including rank 0.
- The number of training workers is `world_size - 1`.

Start rank 0 first:

```bash
python main_3090.py \
  -r 0 \
  -w 5 \
  -s 2 \
  -a "$SERVER_IP" \
  -d imagenet \
  -p /data \
  -t 1.05 \
  --amp
```

Start each worker with its own rank:

```bash
python main_3090.py -r 1 -w 5 -a "$SERVER_IP"
python main_3090.py -r 2 -w 5 -a "$SERVER_IP"
python main_3090.py -r 3 -w 5 -a "$SERVER_IP"
python main_3090.py -r 4 -w 5 -a "$SERVER_IP"
```

Rank 0 validates and sends the training configuration to workers through RPC. Passing the same training arguments to workers is harmless, but the current validator only requires dataset/path/AMP on rank 0.

## Parser Options

Required distributed settings:

- `-r`, `--rank`: Global rank. Use `0` for the parameter server and `1..world_size-1` for workers.
- `-w`, `--world-size`: Total number of RPC processes. Must be at least `2`.
- `-a`, `--server-addr`, `--addr`: Rank 0 address used by RPC.

Rank 0 training settings:

- `-d`, `--dataset`, `--data`: Dataset. Only `imagenet` is accepted.
- `-p`, `--dir-path`, `--path`: Dataset root path.
- `--mixed-precision`, `--amp`: Required for the current ImageNet training path.

Optional settings:

- `-s`, `--num-small`, `--small`: Number of workers assigned to the small-batch group. Default: `0`. Must be between `0` and `world_size - 1`.
- `--server-port`, `--port`: RPC port. Default: `48763`.
- `--device-index`: Index into TensorFlow's visible GPU list. Default: `0`.
- `--depth`: ResNet depth. Only `18` is accepted by the current parser.
- `-t`, `--additional-time-ratio`, `--time-ratio`, `--ratio`: Permitted additional training time ratio. Default: `1`. Must be greater than `0`.
- `--jit-compile`, `--xla`: Enables the existing TensorFlow JIT/XLA setup path. Actual runtime success still depends on the CUDA/XLA environment.
- `-c`, `--comments`: Adds a suffix to saved output filenames.
- `--no-cycle`: Disables cyclic progressive resolution changes.
- `--temp`: Saves temporary model/log files at learning-rate milestones, then removes those temporary files when training completes.
- `--no-save`: Disables final model and `.npy` history output.

## Current Schedule

Default cyclic mode follows three ImageNet stages:

| Stage | Resolution | Dropout | Large batch size, AMP | Large batch size, AMP+XLA |
| --- | ---: | ---: | ---: | ---: |
| 0 | 160 | 0.1 | 2330 | 2800 |
| 1 | 224 | 0.2 | 1110 | 1400 |
| 2 | 288 | 0.3 | 740 | 900 |

Small-batch sizes are calculated at startup from:

- `world_size`
- `--num-small`
- `--additional-time-ratio`
- the timing intercept/coefficient table in `parameter_server_3090.py`

Training length is fixed at 105 epochs in distributed mini-epoch units:

- `mini_epochs = 105 * (world_size - 1)`
- learning-rate decay milestones: epochs 60, 90, 105
- cyclic stage milestones: epochs 20, 40, 60, 70, 80, 90, 95, 100, 105

With `--no-cycle`, the run starts directly from the final DBL stage and stays there:

- resolution: 288
- dropout: 0.3
- large batch size: stage 2 value
- learning-rate milestones are still applied at epochs 60, 90, and 105
- total epoch count and validation frequency are unchanged

## Output

By default, rank 0 saves:

- `<outfile>_model`
- `<outfile>.npy`

The output name includes dataset, model depth, epoch count, time ratio, world-size/small-worker count, AMP/XLA flags, `noCycle` when applicable, and optional comments.

Use `--no-save` for dry runs that should not write final model/history files. Use `--temp` only when temporary milestone files are needed; they are cleaned up at the end of a normal run.

## Validation

Run the lightweight checks from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s train/ai_opt_code/tests -v
python3 -B -m py_compile train/ai_opt_code/main_3090.py train/ai_opt_code/parameter_server_3090.py train/ai_opt_code/tf_data_model.py
```

The parser tests verify default values, aliases, unsupported option rejection, and required argument validation. The no-cycle tests verify that `--no-cycle` starts at the final DBL stage while keeping learning-rate milestone behavior.

## Citation

If you use this code in your research, please cite:

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
