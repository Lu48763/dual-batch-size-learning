# command: python main.py -r= -w=5 -s= -a=140.109.23.106 -d=imagenet -p=/data -t=1.05 --amp

import argparse
import os

import tensorflow as tf
from tensorflow import keras
import torch
from torch.distributed import rpc

import parameter_server_3090 as ps

# parser
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
    pass
parser = argparse.ArgumentParser(
    description='Progressive Dual Batch Size Deep Learning for Distributed Parameter Server Systems',
    epilog=(
        'The parser only supports high-level control options. '
        'If the user wants to adjust low-level control options, modify the code. '
        'Required settings [--rank, --world-size, --server-addr] or [-r, -w, -a]. '
        'Rank 0 also requires [--dataset, --dir-path, --amp] or [-d, -p, --amp]. '
        'Optional settings [--num-small, --time-ratio, --xla, --depth, --server-port, --schedule, --no-cycle, --temp, --no-save].'
    ),
    formatter_class=CustomFormatter,
)
## RPC setting
parser.add_argument(
    '--rank', '-r',
    type=int,
    help='global ranking of the process, pass in 0 for master, and pass in others for workers',
)
parser.add_argument(
    '--world-size', '-w',
    type=int,
    help='total number of servers participating in the process',
)
parser.add_argument(
    '--num-small', '--small', '-s',
    dest='small',
    default=0,
    type=int,
    help='number of small-batch workers in the process, default is "0"',
)
parser.add_argument(
    '--server-addr', '--addr', '-a',
    dest='addr',
    type=str,
    help='the master address of the parameter server',
)
parser.add_argument(
    '--server-port', '--port',
    dest='port',
    default='48763',
    type=str,
    help='the port that the master listens to, default is "48763"',
)
parser.add_argument(
    '--device-index',
    type=int,
    default=0,
    help='the index of the GPU used to run the program, "0" or "-1" is a good choice',
)
## data and model
parser.add_argument(
    '--dataset', '--data', '-d',
    type=str,
    choices=['imagenet'],
    help='dataset to train; the current RTX 3090 training flow supports only "imagenet"',
)
parser.add_argument(
    '--dir-path', '--path', '-p',
    type=str,
    help='path to the dataset directory',
)
parser.add_argument(
    '--depth',
    type=int,
    default=18,
    choices=[18],
    help='resnet depth; current timing coefficients support ResNet-18 only',
)
## experimant: intercept, coefficient, and permitted additional training time
parser.add_argument(
    '--additional-time-ratio', '--time-ratio', '--ratio', '-t',
    dest='time_ratio',
    type=float,
    default=1,
    help='ratio of additional training time allowed to original training time',
)
## optimization options
parser.add_argument(
    '--mixed-precision', '--amp',
    dest='amp',
    action='store_true',
    help='train with mixed precision (amp)',
)
parser.add_argument(
    '--jit-compile', '--xla',
    dest='xla',
    action='store_true',
    help='train with jit compile (xla)',
)
## training schedule
schedule_group = parser.add_mutually_exclusive_group()
schedule_group.add_argument(
    '--schedule',
    choices=['cyclic', 'no-cycle', 'uniform'],
    default='cyclic',
    help='training schedule: "cyclic" for paper hybrid CPL, "no-cycle" for LR-stage progressive schedule, "uniform" for three equal 35-epoch stages',
)
schedule_group.add_argument(
    '--no-cycle',
    dest='schedule',
    action='store_const',
    const='no-cycle',
    help='legacy alias for "--schedule no-cycle"',
)
## output files setting
parser.add_argument(
    '--comments', '-c',
    type=str,
    help='add additional comments on filename',
)
parser.add_argument(
    '--temp',
    dest='temp',
    action='store_true',
    help='save checkpoints/temporary files at milestones during training, and delete them when training completes',
)
parser.add_argument(
    '--no-save',
    dest='save',
    action='store_false',
    help='do not save the training results, including "_model" and ".npy"',
)


# running server && worker
def run_server(args):
    ps_rref = rpc.RRef(ps.Server(args))
    future_list = []
    for i in range(1, args.world_size):
        future_list.append(
            rpc.rpc_async(
                f'worker_{i}',
                run_worker,
                args=(ps_rref, args, i, i <= args.small),
            )
        )
    torch.futures.wait_all(future_list)
    ps_rref.rpc_sync().save_outfile()
    print('Complete, End Program')

def run_worker(ps_rref, args, rank, is_small_batch):
    worker = ps.Worker(ps_rref, args, rank, is_small_batch)
    worker.train()
    print(f'Worker {rank} Training Complete')


def normalize_schedule_args(args):
    if not hasattr(args, 'schedule'):
        args.schedule = 'cyclic' if getattr(args, 'cycle', True) else 'no-cycle'
    args.cycle = args.schedule == 'cyclic'


def validate_args(args):
    normalize_schedule_args(args)
    if args.rank == None:
        raise ValueError('"rank" argument is required')
    if args.world_size == None:
        raise ValueError('"world_size" argument is required')
    if args.addr == None:
        raise ValueError('"master_addr" argument is required')
    if args.world_size < 2:
        raise ValueError('"world_size" must be greater than or equal to 2')
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError('"rank" must be between 0 and world_size - 1')
    if args.small < 0 or args.small > args.world_size - 1:
        raise ValueError('"small" must be between 0 and world_size - 1')
    if args.time_ratio <= 0:
        raise ValueError('"time_ratio" must be greater than 0')
    if args.rank == 0:
        if args.dataset == None:
            raise ValueError('"dataset" argument is required on rank 0')
        if args.dir_path == None:
            raise ValueError('"dir_path" argument is required on rank 0')
        if not args.amp:
            raise ValueError('"amp" is required for ImageNet training')


# main
def main():
    # parse args
    args = parser.parse_args()
    validate_args(args)
    print('----')
    print(args)
    print('----')

    # amp, xla
    if args.xla:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
        tf.config.optimizer.set_jit('autoclustering')
        print(f'Optimizer set_jit: "{tf.config.optimizer.get_jit()}"')
    if args.amp:
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        print(f'Policy: {policy.name}')
        print(f'Compute dtype: {policy.compute_dtype}')
        print(f'Variable dtype: {policy.variable_dtype}')
    print('----')
    print(f'MIXED_PRECISION: {args.amp}')
    print(f'JIT_COMPILE: {args.xla}')
    print('----')

    # GPU initialization
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[args.device_index], 'GPU')
    for device in tf.config.get_visible_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    print('----')
    print(f'The Number of Available Physical Devices: {len(physical_devices)}')
    print(f'Using Devices: {tf.config.get_visible_devices("GPU")}')
    print('----')

    # RPC
    backend_options = rpc.TensorPipeRpcBackendOptions(
        init_method=f'tcp://{args.addr}:{args.port}',
        rpc_timeout=60*60*24*7, # important, the maximum exist time of the program, set to be 1 week
    )
    if args.rank == 0:  # server
        print(f'Server {args.rank} initializing RPC')
        rpc.init_rpc(
            name=f'server_{args.rank}',
            rank=args.rank,
            world_size=args.world_size,
            rpc_backend_options=backend_options,
        )
        run_server(args)
    else:               # worker
        print(f'Worker {args.rank} initializing RPC')
        rpc.init_rpc(
            name=f'worker_{args.rank}',
            rank=args.rank,
            world_size=args.world_size,
            rpc_backend_options=backend_options,
        )
    rpc.shutdown()


if __name__ == '__main__':
    # args:
    # [rank, world_size, small, addr, port, time_ratio,
    #  dataset, dir_path, amp, xla, comments,
    #  device_index, depth, schedule, cycle, temp, save]
    main()
