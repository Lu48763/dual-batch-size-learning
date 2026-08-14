import itertools
import os
import random
import shutil
import threading
import time

import numpy as np
from tensorflow import keras
from torch.distributed import rpc

import tf_data_model

# Server
class Server(object):
    def __init__(self, args):
        # raise error if not using 3090 setting with "-d=imagenet"
        if args.dataset != 'imagenet':
            ValueError(f'the dataset only allows "-d=imagenet"')
        # global setting
        self.args = args
        self.start_time = time.perf_counter()
        self.mission_complete = False
        # training parameters 1
        self.parameter_lock = threading.Lock()
        self.epochs = 140 if 'cifar' in args.dataset else 105
        self.steps = 3
        self.mini_epochs = self.epochs * (args.world_size - 1) * args.sync_freq
        self.global_commit_ID = 0
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            self.total_data_amount = round(50000 / args.sync_freq)
            self.intercept_ls = (
                [0.010718227125969579, 0.012992225216649333] if args.xla
                else [0.014788833468238893, 0.015562114074817612]
            )
            self.coef_ls = (
                [0.0003139986390761775, 0.0004375558540115719] if args.xla
                else [0.00035268184915307385, 0.0005056358858583448]
            )
            self.large_batch_size_ls = [430, 580] if args.xla else [600, 560]
            self.resolution_ls = [24, 32]
            self.dropout_rate_ls = [0.1, 0.2]
        elif args.dataset == 'imagenet':
            self.total_data_amount = round(1281167 / args.sync_freq)
            if args.amp:
                self.intercept_ls = (
                    [5.683134939754586e-05, 0.01073572758256125, 0.010680175780982126] if args.xla
                    else [0.012036769987567586, 0.01705088976279029, 0.016659967832795497]
                )
                self.coef_ls = (
                    [0.00025895413494090355, 0.00045862692849402506, 0.0007576280737663753] if args.xla
                    else [0.0003013537830804257, 0.0005683767808602767, 0.0009387165655225843]
                )
                self.large_batch_size_ls = [2800, 1400, 900] if args.xla else [2330, 1110, 740]
            else:
                raise ValueError('The ImageNet training process only supports "--amp"')
            self.resolution_ls = [160, 224, 288]
            self.dropout_rate_ls = [0.1, 0.2, 0.3]
        else:
            raise ValueError(f'Invalid dataset "{args.dataset}".')
        self.large_data_amount, self.small_data_amount, self.small_batch_size_ls = self.get_large_small_dataAmount_batchSize()
        # iter and cycles milestones
        ## milestones
        '''
        self.iter_milestones = list(
            self.mini_epochs // self.steps * i
            for i in range(1, self.steps + 1)
        )
        self.cycle_milestones = list(
            self.mini_epochs // (self.steps * len(self.resolution_ls)) * i
            for i in range(1, self.steps * len(self.resolution_ls) + 1)
        )
        '''
        self.iter_milestones = np.array(
            [80, 120, 140] if 'cifar' in args.dataset
            else [60, 90, 105]
        ) * (args.world_size - 1) * args.sync_freq
        self.cycle_milestones = np.array(
            [40, 80, 100, 120, 130, 140] if 'cifar' in args.dataset
            else [20, 40, 60, 70, 80, 90, 95, 100, 105]
        ) * (args.world_size - 1) * args.sync_freq
        ####'''
        ## modified by iter_milestones [30, 60, 90]
        ### (+1) for preventing iter overflow
        self.global_step_ID_iter = iter(list(range(0, self.steps + 1)))
        self.learning_rate_iter = iter(list(2e-1 * 0.1 ** i for i in range(0, self.steps + 1)))
        print('================')
        print(f'iter_milestones: {self.iter_milestones}')
        print(f'cycle_milestones: {self.cycle_milestones}')
        print('================')
        ## modified by cycle_milestones [10, 20, ..., 90]
        self.global_stage_ID_iter = itertools.cycle(list(range(0, len(self.resolution_ls))))
        self.large_batch_size_iter = itertools.cycle(self.large_batch_size_ls)
        self.small_batch_size_iter = itertools.cycle(self.small_batch_size_ls)
        self.resolution_iter = itertools.cycle(self.resolution_ls)
        self.dropout_rate_iter = itertools.cycle(self.dropout_rate_ls)
        # training parameters 2
        self.parameter = {
            # iter values
            'global_step_ID': next(self.global_step_ID_iter),
            'learning_rate': next(self.learning_rate_iter),
            # cycle values
            'global_stage_ID': next(self.global_stage_ID_iter),
            'large_batch_size': next(self.large_batch_size_iter),
            'small_batch_size': next(self.small_batch_size_iter),
            'resolution': next(self.resolution_iter),
            'dropout_rate': next(self.dropout_rate_iter),
            # fixed values
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'total_data_amount': self.total_data_amount,
            'large_data_amount': self.large_data_amount,
            'small_data_amount': self.small_data_amount,
        }
        # global model
        self.global_model_lock = threading.Lock()
        self.global_model = tf_data_model.modify_resnet(
            dataset=args.dataset,
            depth=args.depth,
            dropout_rate=self.dropout_rate_ls[0],
            resolution=self.resolution_ls[0],
        )
        # record
        self.history_lock = threading.Lock()
        self.history = {
            # ID
            'worker_ID': [],
            'global_commit_ID': [], # count by server
            'local_commit_ID': [],  # count by worker
            'step_ID': [],
            'stage_ID': [],
            # train
            'train_loss': [],
            'train_acc': [],
            'train_time': [],   # count from model.fit start
            # val
            'val_loss': [],
            'val_acc': [],
            'commit_time': [],  # count from program start
        }
        self.outfile = (
            f'{args.dataset}_resnet{args.depth}_e{self.epochs}' # dataset, model, epochs
            f'_t{("%.2f" % args.time_ratio).replace(".", "")}'  # additional time ratio
            f'_w{args.world_size}s{args.small}'                 # world size, num of small workers
            f'_f{args.sync_freq}'                               # sync freq
            f'{"_amp" if args.amp else ""}'                     # amp
            f'{"_xla" if args.xla else ""}'                     # xla
            f'{"" if args.cycle else "_noCycle"}'               # cyclic learning
            f'{"_" + args.comments if args.comments else ""}'   # comments
        )
        self.tempfile = f'temp_{self.outfile}'
    
    def get_large_small_dataAmount_batchSize(self):
        # numbers of GPUs
        num_total = self.args.world_size - 1
        num_small = self.args.small
        num_large = num_total - num_small
        # calculation
        large_data_amount = round(self.args.time_ratio * self.total_data_amount / num_total) if num_small else round(self.total_data_amount / num_total)
        small_data_amount = round((self.total_data_amount - large_data_amount * num_large) / num_small) if num_small else 0
        small_batch_size_ls = []
        for large_batch_size, intercept, coef in zip(self.large_batch_size_ls, self.intercept_ls, self.coef_ls):
            time_origin = (coef + intercept / large_batch_size) * self.total_data_amount / num_total
            time_new = self.args.time_ratio * time_origin
            small_batch_size_ls.append(round(intercept / (time_new / small_data_amount - coef)) if num_small else 0)
        return large_data_amount, small_data_amount, small_batch_size_ls

    def get_parameter(ps_rref):
        self = ps_rref.local_value()
        with self.parameter_lock:
            return self.parameter

    def get_global_model_weights(ps_rref):
        self = ps_rref.local_value()
        with self.global_model_lock:
            return self.global_model.get_weights()

    #### iter_milestones is [30, 60, 90] * # of GPUs
    #### cycle_milestones is [10, 20, ..., 90] * # of GPUs
    #### record global_commit_ID from '0' to '(89 * # of GPUs)'
    #### thus the last global_commit_ID is '(90 * # of GPUs - 1)'
    def push_and_pull_model(ps_rref, worker_weights, rank, is_small_batch, parameter):
        self = ps_rref.local_value()
        with self.parameter_lock and self.global_model_lock:
            self.global_commit_ID += 1
            # update global model and parameters
            if not self.mission_complete: # make sure mission is not complete yet
                ## update global model
                server_weights = self.global_model.get_weights()
                update_factor = parameter['small_data_amount'] / parameter['large_data_amount'] if is_small_batch else 1
                for i in range(len(server_weights)):
                    server_weights[i] = (
                        server_weights[i] + update_factor * worker_weights[i]
                    ) / (1 + update_factor)
                self.global_model.set_weights(server_weights)
                # update parameters
                if self.global_commit_ID in self.iter_milestones:
                    self.parameter['global_step_ID'] = next(self.global_step_ID_iter)
                    self.parameter['learning_rate'] = next(self.learning_rate_iter)
                if self.global_commit_ID in self.cycle_milestones:
                    self.parameter['global_stage_ID'] = next(self.global_stage_ID_iter)
                    self.parameter['large_batch_size'] = next(self.large_batch_size_iter)
                    self.parameter['small_batch_size'] = next(self.small_batch_size_iter)
                    self.parameter['resolution'] = next(self.resolution_iter)
                    self.parameter['dropout_rate'] = next(self.dropout_rate_iter)
                # print messages
                ## (-1) for shifting to record value
                print(f'Update Global Model from Worker {rank} at Global Mini-Epoch {self.global_commit_ID - 1}')
            # check and update mission_complete
            if self.global_commit_ID == self.mini_epochs:
                self.mission_complete = True
            # return results
            ## (-1) for shifting to record value
            return self.global_model.get_weights(), self.global_commit_ID - 1, self.mission_complete

    def update_history(ps_rref, record):
        self = ps_rref.local_value()
        with self.history_lock:
            # record commit time
            record['commit_time'] = time.perf_counter() - self.start_time
            # check global commit ID and update record
            if record['global_commit_ID'] < self.mini_epochs:
                for key, value in record.items():
                    self.history[key].append(value)
                print(
                    f'worker {record["worker_ID"]} commit,',
                    f'ID: {record["global_commit_ID"]},',
                    f'time: {round(record["commit_time"], 3)},',
                    f'loss: {round(record["val_loss"], 3)},',
                    f'acc: {round(record["val_acc"] * 100, 1)}%'
                )
        # save tempfile
        if self.args.temp and record['global_commit_ID'] + 1 in self.iter_milestones:
            with self.global_model_lock:
                self.save_tempfile(record['global_commit_ID'])

    def save_tempfile(self, temp_commit_ID):
        # lock is held by update_history()
        keras.models.save_model(self.global_model, f'{self.tempfile}_model')
        np.save(f'{self.tempfile}.npy', self.history)
        print(f'Save The Temporary Files at Global Mini-Epoch {temp_commit_ID}')

    def save_outfile(self):
        # No lock is needed because training is complete
        if self.args.save:
            keras.models.save_model(self.global_model, f'{self.outfile}_model')
            np.save(f'{self.outfile}.npy', self.history)
            print('----')
            print(f'Save Model: {self.outfile}_model')
            print(f'Save Logs: {self.outfile}.npy')
            print('----')
        if self.args.temp:
            shutil.rmtree(f'{self.tempfile}_model', ignore_errors=True)
            if os.path.isfile(f'{self.tempfile}.npy'):
                os.remove(f'{self.tempfile}.npy')
            print('Clean The Temporary Files')

# Worker
class Worker(object):
    def __init__(self, ps_rref, args, rank, is_small_batch):
        # settings
        self.args = args
        self.ps_rref = ps_rref
        self.rank = rank
        self.is_small_batch = is_small_batch
        self.local_commit_ID = 0
        self.step_ID = None
        self.stage_ID = None
        self.mission_complete = False
        # parameters, data, and model
        self.parameter = None
        self.dataloader = None
        self.model = None
        self.verbose = 2 # [2, 'auto']
        # callback
        class TimeCallback(keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                self.history = []
            def on_epoch_begin(self, epoch, logs=None):
                self.time_epoch_begin = time.perf_counter()
            def on_epoch_end(self, epoch, logs=None):
                self.history.append(time.perf_counter() - self.time_epoch_begin)
        self.time_callback = TimeCallback()

    def train(self):
        while not self.mission_complete:
            # get parameter
            self.parameter = rpc.rpc_sync(self.ps_rref.owner(), Server.get_parameter, kwargs={'ps_rref': self.ps_rref})
            # check if parameter is modified
            if self.step_ID != self.parameter['global_step_ID'] or self.stage_ID != self.parameter['global_stage_ID']:
                self.step_ID = self.parameter['global_step_ID']
                self.stage_ID = self.parameter['global_stage_ID']
                # get data
                self.dataloader = tf_data_model.load_data(
                    resolution=self.parameter['resolution'],
                    batch_size=self.parameter['small_batch_size'] if self.is_small_batch else self.parameter['large_batch_size'],
                    dataset=self.args.dataset,
                    dir_path=self.args.dir_path,
                    val_batch_size=self.parameter['large_batch_size'],
                )
                # get model
                self.model = tf_data_model.modify_resnet(
                    dataset=self.args.dataset,
                    depth=self.args.depth,
                    dropout_rate=self.parameter['dropout_rate'],
                    resolution=self.parameter['resolution'],
                    old_model=self.model if self.local_commit_ID != 0 else None,
                )
                # compile model
                self.model.compile(
                    optimizer=keras.optimizers.experimental.SGD(
                        learning_rate=self.parameter['learning_rate'],
                        momentum=self.parameter['momentum'],
                        weight_decay=self.parameter['weight_decay'],
                    ),
                    loss=keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'],
                )
            # set model weights
            self.model.set_weights(
                rpc.rpc_sync(self.ps_rref.owner(), Server.get_global_model_weights, kwargs={'ps_rref': self.ps_rref})
            )
            # print training summary message
            print('----')
            print(f'Local Mini-Epoch {self.local_commit_ID}, Step {self.step_ID}, Stage {self.stage_ID}')
            print(
                f'Resolution {self.parameter["resolution"]},',
                f'LR {self.parameter["learning_rate"]: g},',
                f'BS {self.parameter["small_batch_size"] if self.is_small_batch else self.parameter["large_batch_size"]}'
            )
            # train
            num_take_batch = round(
                self.parameter['small_data_amount'] / self.parameter['small_batch_size'] if self.is_small_batch
                else self.parameter['large_data_amount'] / self.parameter['large_batch_size']
            )
            num_total_batch = ((self.parameter['total_data_amount'] * self.args.sync_freq) //
                               (self.parameter['small_batch_size'] if self.is_small_batch
                               else self.parameter['large_batch_size']))
            ## resolve the shuffle buffer issue for imagent dataset, default buffer_size = batch_size * 8
            ### plan A: let it go
            num_skip_batch = 0
            #'''
            ### plan B: bin iteration, reasonable, since each worker's data is close at the same epoch
            #num_batch_bin = (num_total_batch // num_take_batch) - 1 # -1: ensure at least 1 bin can be taken
            #num_skip_batch = num_take_batch * (self.local_commit_ID % num_batch_bin)
            #'''
            #'''
            ### plan C: random skip, each worker's data might be different at the same epoch
            #num_skip_batch = random.randint(0, num_total_batch - num_take_batch)
            #'''
            train_logs = self.model.fit(
                (self.dataloader['train']
                 .skip(0 if 'cifar' in self.args.dataset else num_skip_batch)
                 .take(num_take_batch)
                ),
                verbose=self.verbose,
                callbacks=[self.time_callback],
            )
            train_logs.history['t'] = self.time_callback.history
            # push and pull model
            global_model_weights, global_commit_ID, self.mission_complete = rpc.rpc_sync(
                self.ps_rref.owner(),
                Server.push_and_pull_model,
                kwargs={
                    'ps_rref': self.ps_rref,
                    'worker_weights': self.model.get_weights(),
                    'rank': self.rank,
                    'is_small_batch': self.is_small_batch,
                    'parameter': self.parameter,
                },
            )
            self.model.set_weights(global_model_weights)
            # val
            val_logs = self.model.evaluate(
                self.dataloader['val'],
                verbose=self.verbose,
                return_dict=True,
            )
            # record
            record = {
                # ID
                'worker_ID': self.rank,
                'global_commit_ID': global_commit_ID,       # count by server
                'local_commit_ID': self.local_commit_ID,    # count by worker
                'step_ID': self.step_ID,
                'stage_ID': self.stage_ID,
                # train
                'train_loss': train_logs.history['loss'][0],
                'train_acc': train_logs.history['accuracy'][0],
                'train_time': train_logs.history['t'][0],   # count from model.fit start
                # val
                'val_loss': val_logs['loss'],
                'val_acc': val_logs['accuracy'],
                'commit_time': None,    # count from program start
            }
            rpc.rpc_sync(self.ps_rref.owner(), Server.update_history, kwargs={'ps_rref': self.ps_rref, 'record': record})
            print(f'Worker {self.rank} Local Mini-Epoch {self.local_commit_ID} Complete')
            self.local_commit_ID += 1
