# ai_opt_code ImageNet Training

This folder contains the current ImageNet/ResNet-18 training flow for the paper implementation. It uses TensorFlow/Keras for training and PyTorch RPC for parameter-server coordination.

Current supported scope:

- Dataset: `imagenet`
- Model: ResNet-18 only
- Hardware profile: RTX 3090 timing model
- Required training mode: AMP mixed precision
- Schedules: `cyclic`, `no-cycle`, `uniform`

`tf_data_model.py` contains lower-level CIFAR and ResNet-34 helpers, but `main_3090.py` rejects those options because the distributed timing and batch-size schedule are implemented only for ImageNet/ResNet-18.

## Files

- `main_3090.py`: CLI parser, validation, AMP/XLA setup, GPU selection, and RPC startup.
- `parameter_server_3090.py`: server/worker training loop, batch-size calculation, schedule milestones, logging, and output saving.
- `tf_data_model.py`: TensorFlow data loading and ResNet construction.
- `tests/`: regression tests for parser and schedule behavior.
- `2509.26092v2.pdf`: referenced paper.

## Environment

Original environment:

```bash
conda create -n hdbl -c pytorch -c nvidia python=3.11 tensorflow=2.13 pytorch=2.1 pytorch-cuda=12.1 cuda-nvcc=11.8
conda activate hdbl
```

If PyTorch RPC cannot connect to rank 0, check whether `/etc/hosts` maps `127.0.1.1` to the machine hostname.

## Dataset

`--dir-path` should point to the directory that contains the `imagenet` folder:

```text
${dir_path}/imagenet/train/<class_id>/*.jpg
${dir_path}/imagenet/val/<class_id>/*.jpg
```

Training data is sharded across workers. Validation is not sharded; every worker evaluates on the full validation set after each local commit.

## Quick Start

Rank 0 is the parameter server. Rank 1 and above are workers. `--world-size` includes rank 0, so `--world-size 5` means one parameter server plus four workers.

Set common variables:

```bash
cd /home/r08944044/dual-batch-size-learning/train/ai_opt_code
export SERVER_IP=192.168.0.1
export DATA_ROOT=/data
```

Start rank 0 first. This is the compact command for the default `cyclic` schedule:

```bash
python main_3090.py -r 0 -w 5 -s 2 -a "$SERVER_IP" -d imagenet -p "$DATA_ROOT" -t 1.05 --amp
```

Start workers:

```bash
python main_3090.py -r 1 -w 5 -a "$SERVER_IP"
python main_3090.py -r 2 -w 5 -a "$SERVER_IP"
python main_3090.py -r 3 -w 5 -a "$SERVER_IP"
python main_3090.py -r 4 -w 5 -a "$SERVER_IP"
```

If rank 0 uses a non-default `--server-port`, pass the same port to every worker.

## Schedule Commands

Use one of these on rank 0:

```bash
# Default paper hybrid cyclic schedule
python main_3090.py -r 0 -w 5 -s 2 -a "$SERVER_IP" -d imagenet -p "$DATA_ROOT" -t 1.05 --amp

# Same as default, explicit form
python main_3090.py -r 0 -w 5 -s 2 -a "$SERVER_IP" -d imagenet -p "$DATA_ROOT" -t 1.05 --amp --schedule cyclic

# Non-cyclic schedule using original LR-stage boundaries
python main_3090.py -r 0 -w 5 -s 2 -a "$SERVER_IP" -d imagenet -p "$DATA_ROOT" -t 1.05 --amp --schedule no-cycle

# Non-cyclic schedule split into three equal 35-epoch stages
python main_3090.py -r 0 -w 5 -s 2 -a "$SERVER_IP" -d imagenet -p "$DATA_ROOT" -t 1.05 --amp --schedule uniform

# Legacy alias for --schedule no-cycle
python main_3090.py -r 0 -w 5 -s 2 -a "$SERVER_IP" -d imagenet -p "$DATA_ROOT" -t 1.05 --amp --no-cycle
```

Do not combine `--schedule ...` with `--no-cycle`; argparse rejects that combination.

## Arguments

Required RPC settings:

| Flag | Default | Required on | Meaning |
| --- | --- | --- | --- |
| `--rank`, `-r` | `None` | all ranks | Global rank. Use `0` for parameter server and `1..world_size-1` for workers. |
| `--world-size`, `-w` | `None` | all ranks | Total RPC process count. Must be at least `2`. |
| `--server-addr`, `--addr`, `-a` | `None` | all ranks | Rank 0 address for PyTorch RPC. |

Rank 0 training settings:

| Flag | Default | Required on | Meaning |
| --- | --- | --- | --- |
| `--dataset`, `--data`, `-d` | `None` | rank 0 | Dataset. Only `imagenet` is accepted. |
| `--dir-path`, `--path`, `-p` | `None` | rank 0 | Dataset root containing `imagenet/train` and `imagenet/val`. |
| `--mixed-precision`, `--amp` | `False` | rank 0 | Required for the current ImageNet path. |

Optional settings:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--num-small`, `--small`, `-s` | `0` | Number of small-batch workers. Must be between `0` and `world_size - 1`. |
| `--server-port`, `--port` | `'48763'` | RPC port. Use the same value on all ranks. |
| `--device-index` | `0` | TensorFlow visible GPU index. |
| `--depth` | `18` | ResNet depth. Only `18` is accepted. |
| `--additional-time-ratio`, `--time-ratio`, `--ratio`, `-t` | `1` | Permitted extra training time ratio. Must be greater than `0`. |
| `--jit-compile`, `--xla` | `False` | Enables the existing TensorFlow JIT/XLA setup path. Runtime success depends on CUDA/XLA environment setup. |
| `--schedule` | `'cyclic'` | Training schedule. Choices: `cyclic`, `no-cycle`, `uniform`. |
| `--no-cycle` | not set | Legacy alias for `--schedule no-cycle`. Mutually exclusive with `--schedule`. |
| `--comments`, `-c` | `None` | Extra suffix for output filenames. |
| `--temp` | `False` | Save temporary milestone files, then remove them when training completes. |
| `--no-save` | not set | Disable final model and `.npy` history output. Final saving is enabled by default. |

Internal derived field:

| Field | Value |
| --- | --- |
| `cycle` | `True` when `schedule == 'cyclic'`, otherwise `False`. This is not a CLI flag. |

## Schedule Summary

Common stage settings:

| Stage | Resolution | Dropout | BL AMP | BL AMP+XLA |
| --- | ---: | ---: | ---: | ---: |
| 0 | 160 | 0.1 | 2330 | 2800 |
| 1 | 224 | 0.2 | 1110 | 1400 |
| 2 | 288 | 0.3 | 740 | 900 |

Milestones:

| Schedule | Stage behavior | LR/stage boundaries |
| --- | --- | --- |
| `cyclic` | Repeats stages `0 -> 1 -> 2` within each LR phase. | stage: `20,40,60,70,80,90,95,100,105`; LR: `60,90,105` |
| `no-cycle` | One pass through stages `0 -> 1 -> 2`, aligned with original LR stages. | `60,90,105` |
| `uniform` | One pass through stages `0 -> 1 -> 2`, equally split. | `35,70,105` |

Training length is fixed at `105 * (world_size - 1)` global commits. Final milestone `105` marks the end of training; non-cyclic schedules do not wrap back to stage 0.

Small-batch sizes are calculated at startup from `world_size`, `--num-small`, `--additional-time-ratio`, and timing coefficients in `parameter_server_3090.py`.

## Output

By default, rank 0 saves:

- `<outfile>_model`
- `<outfile>.npy`

The output filename includes dataset, model depth, epoch count, time ratio, world-size/small-worker count, AMP/XLA flags, schedule, and optional comments.

## Validation

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s train/ai_opt_code/tests -v
python3 -B -m py_compile train/ai_opt_code/main_3090.py train/ai_opt_code/parameter_server_3090.py train/ai_opt_code/tf_data_model.py
```

## Citation

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
