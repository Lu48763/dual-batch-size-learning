# clear; python DBSL.py -a='140.109.23.236' -w=5 -r= &
# server at gpu08
import argparse
import os
import threading
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import torch
import torch.distributed.rpc as rpc

import tf_cifar_resnet

#### hyperparameter ####
# GPU setting
num_GPU = 4
num_small = 1
num_large = num_GPU - num_small
# batch size and learning rate and extra time rate
base_BS = 500
base_LR = 1e-1
extra_time_ratio = 1.1
# get small_BS, base_data, small_data
## should modify num_small && extra_time_ratio
def count_small_BS_data_size():
    # cifar10/100, resnet18, GTX1080
    # ax+b
    a, b = 0.00056607, 0.014041204118283607
    num_train_data = 50000
    # t = (a+b/x1)*d1 = (a+b/x2)*d2
    time_base = (a + b/base_BS) * num_train_data / num_GPU
    time_new = extra_time_ratio * time_base
    # data
    base_data = int(extra_time_ratio * num_train_data / num_GPU)
    small_data = int((num_train_data - base_data * num_large) / num_small
                    ) if num_small else 0
    # batch size
    small_BS = int(b / (time_new / small_data - a)) if small_data else 0
    return small_BS, base_data, small_data
small_BS, base_data, small_data = count_small_BS_data_size()
# create BS, LR list
BS_list = [small_BS] * num_small + [base_BS] * num_large
LR_list = [base_LR] * num_GPU
# scheduler
rounds = 140
threshold = [80, 120]
gamma = 0.2

#### static ####
# parameter setting
momentum = 0.9
decay = 0
workers = 2
eval_batch_size = 1000
# max training times reserve for each worker
record = int(2*rounds)
# set to use GPU
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')

################################################################
######## Server ########
class ParameterServer(object):
    def __init__(self, world_size):
        # others
        self.start_time = time.time()
        self.world_size = world_size
        self.total_rounds = rounds * (world_size - 1)
        self.stage_threshold = np.array(threshold) * (world_size - 1)
        self.mission_complete = False
        # model
        self.model_lock = threading.Lock()
        self.model = tf_cifar_resnet.make_resnet18(num_classes=100)
        # epoch counter
        self.epoch_lock = threading.Lock()
        self.epoch_counter = 0
        # learning rate and batch_size
        self.batch_size = [base_BS] + BS_list
        self.learning_rate = [base_LR] + LR_list
        # record information
        self.push_time_history = np.zeros((world_size, record), dtype=float)
        self.train_loss_history = np.zeros((world_size, record))
        self.train_acc_history = np.zeros((world_size, record))
        self.test_loss_history = np.zeros((world_size, record))
        self.test_acc_history = np.zeros((world_size, record))
    
    def count_epoch(self):
        with self.epoch_lock:
            if self.epoch_counter == self.total_rounds:
                self.mission_complete = True
            if self.epoch_counter in self.stage_threshold:
                for i in range(1, len(self.batch_size)):
                    self.learning_rate[i] *= gamma
            self.epoch_counter += 1

    def get_mission_model_bs_lr(ps_rref, worker_rank):
        self = ps_rref.local_value()
        self.count_epoch()
        return (self.mission_complete, self.model.get_weights(),
                self.batch_size[worker_rank], self.learning_rate[worker_rank])

    def push_and_pull_model(ps_rref, worker_weights, worker_batch_size,
                            worker_rank, epoch):
        print(f'Worker {worker_rank} epoch {epoch} now pushing model...')
        self = ps_rref.local_value()
        self.push_time_history[worker_rank, epoch] = time.time()
        # average server and worker models' weights
        with self.model_lock:
            server_weights = self.model.get_weights()
            update_factor = 1
            if worker_batch_size == small_BS:
                update_factor = small_data / base_data
            for i in range(len(server_weights)):
                server_weights[i] = ((2 - update_factor) * server_weights[i]
                                     + update_factor * worker_weights[i]
                                    ) / 2
            self.model.set_weights(server_weights)
            return self.model.get_weights()

    def record_loss_acc(ps_rref, worker_rank, epoch,
                        train_loss, train_acc, test_loss, test_acc):
        self = ps_rref.local_value()
        self.train_loss_history[worker_rank, epoch] = train_loss
        self.train_acc_history[worker_rank, epoch] = train_acc
        self.test_loss_history[worker_rank, epoch] = test_loss
        self.test_acc_history[worker_rank, epoch] = test_acc

    def save_history(self):
        content = {
            'world_size': self.world_size,
            'start_time': self.start_time,
            'push_time': self.push_time_history,
            'train_loss': self.train_loss_history, 'train_acc': self.train_acc_history,
            'test_loss': self.test_loss_history, 'test_acc': self.test_acc_history}
        fname = f'tf{num_GPU}_extra{extra_time_ratio}_{num_small}s_{small_BS}_{base_BS}'
        # load: npy = np.load('filename.npy', allow_pickle=True)
        # read: npy.item()['xxx']
        np.save(f'tf_npy/{fname}.npy', content)
        # load: model = keras.models.load_model('directory_path')
        # The warning causes since there is no {model.compile()} in the server code.
        self.model.save(f'tf_model/{fname}')

################################################################
######## Worker ########
class Worker(object):
    def __init__(self, ps_rref, rank, world_size):
        # others
        self.ps_rref = ps_rref
        self.rank = rank
        self.world_size = world_size
        # data
        ((self.x_train, self.y_train), (self.x_test, self.y_test)
        ) = tf_cifar_resnet.load_cifar100()
        # model
        self.model = tf_cifar_resnet.make_resnet18(num_classes=100)

    def shuffle_data(self, x, y):
        index = np.random.permutation(y.shape[0])
        x = tf.gather(x, index)
        y = tf.gather(y, index)
        return x, y

    def print_loss_acc(self, epoch, batch_size, lr,
                       train_loss, train_acc, test_loss, test_acc):
        print('--------')
        print(f'Epoch {epoch} BS {batch_size} LR {lr}')
        print(f'train_loss: {train_loss}, train_acc: {train_acc}')
        print(f'test_loss: {test_loss}, test_acc: {test_acc}')
        print('--------')

    def train(self):
        for epoch in range(record):
            # get model, bs, lr
            mission_complete, weights, batch_size, learning_rate = rpc.rpc_sync(
                self.ps_rref.owner(),
                ParameterServer.get_mission_model_bs_lr,
                args=(self.ps_rref, self.rank))
            self.model.set_weights(weights)
            # check mission complete
            if mission_complete:
                return
            # model compile
            self.model.compile(
                optimizer=keras.optimizers.SGD(learning_rate=learning_rate,
                                               momentum=momentum, decay=decay),
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
            # data gen
            datagen = keras.preprocessing.image.ImageDataGenerator(
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True)
            # fetch data
            self.x_train, self.y_train = self.shuffle_data(self.x_train, self.y_train)
            data_size = base_data if batch_size==base_BS else small_data
            # train
            print(f'Epoch {epoch} Train Stage')
            train_logs = self.model.fit(
                datagen.flow(
                    self.x_train[:data_size],
                    self.y_train[:data_size],
                    batch_size=batch_size),
                epochs=1,
                verbose=0,
                workers=workers)
            train_loss = train_logs.history['loss'][0]
            train_acc = train_logs.history['accuracy'][0]
            # push and pull model
            print(f'Epoch {epoch} Push & Pull Stage')
            weights = rpc.rpc_sync(
                self.ps_rref.owner(),
                ParameterServer.push_and_pull_model,
                args=(self.ps_rref, self.model.get_weights(),
                      batch_size, self.rank, epoch))
            self.model.set_weights(weights)
            # eval
            print(f'Epoch {epoch} Eval Stage')
            test_logs = self.model.evaluate(
                self.x_test, self.y_test,
                batch_size=eval_batch_size,
                verbose=0,
                workers=workers)
            test_loss = test_logs[0]
            test_acc = test_logs[1]
            ######## push loss & acc, wait server get next epoch batch size ########
            print(f'Epoch {epoch} Upload Loss & Acc Stage')
            rpc.rpc_sync(
                self.ps_rref.owner(),
                ParameterServer.record_loss_acc,
                args=(self.ps_rref, self.rank, epoch,
                      train_loss, train_acc, test_loss, test_acc))
            self.print_loss_acc(epoch, batch_size, learning_rate,
                                train_loss, train_acc, test_loss, test_acc)
            print(f'Worker {self.rank} epoch {epoch} complete!')

################################################################
def run_worker(ps_rref, rank, world_size):
    worker = Worker(ps_rref, rank, world_size)
    worker.train()
    print('Finish training.')

################################################################
def run_parameter_server(world_size):
    ps_rref = rpc.RRef(ParameterServer(world_size))
    future_list = []
    for i in range(1, world_size):
        future_list.append(
            rpc.rpc_async(f'worker_{i}', run_worker, args=(ps_rref, i, world_size)))
    torch.futures.wait_all(future_list)
    ps_rref.rpc_sync().save_history()
    print('Finish, save the history.')

################################################################
def run_program(rank, world_size, addr, port):
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = port
    backend_options = rpc.TensorPipeRpcBackendOptions(
        init_method=f'tcp://{addr}:{port}',
        rpc_timeout=0)
    if rank:           # worker
        print(f'Worker {rank} initializing RPC')
        rpc.init_rpc(
            name=f'worker_{rank}',
            rank=rank,
            world_size=world_size,
            rpc_backend_options=backend_options)
    else:               # server
        print(f'Server {rank} initializing RPC')
        rpc.init_rpc(
            name=f'server_{rank}',
            rank=rank,
            world_size=world_size,
            rpc_backend_options=backend_options)
        run_parameter_server(world_size)
    rpc.shutdown()

################################################################
# clear; python thesis/chimera.py -a='140.109.23.236' -w=5 -r=
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multiple Batch Sizes Asynchronous Learning via Parameter Server')
    parser.add_argument(
        '-r', '--rank',
        type=int,
        default=None,
        help='Global rank of the process, pass in 0 for master and others for workers.')
    parser.add_argument(
        '-w', '--world_size',
        type=int,
        help='Total number of participating processes.')
    parser.add_argument(
        '-a', '--master_addr',
        type=str,
        help='Address of the master.')
    parser.add_argument(
        '-p', '--master_port',
        type=str,
        default='48763',
        help='Port that master is listening on, default "48763".')
    
    args = parser.parse_args()
    assert args.rank is not None, 'Must provide rank argument.'
    assert args.world_size is not None, 'Must provide world_size argument.'
    assert args.world_size > 1, 'There must be at least 1 server and 1 worker'
    assert args.master_addr is not None, 'Must provide master_addr argument.'

    run_program(args.rank, args.world_size, args.master_addr, args.master_port)
