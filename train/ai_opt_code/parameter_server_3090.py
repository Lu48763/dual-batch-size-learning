import itertools
import os
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
        if args.dataset != 'imagenet':
            raise ValueError('The dataset only allows "-d=imagenet"')
        
        self.args = args
        self.start_time = time.perf_counter()
        self.mission_complete = False
        self.parameter_lock = threading.Lock()
        self.global_model_lock = threading.Lock()
        
        self.epochs = 105
        self.steps = 3
        self.mini_epochs = self.epochs * (args.world_size - 1)
        self.global_commit_ID = 0
        self.total_data_amount = 1281167

        if args.amp:
            self.intercept_ls = (
                [5.683134939754586e-05, 0.01073572758256125, 0.010680175780982126] if args.xla
                else [0.012036769987567586, 0.01705088976279029, 0.016659967832795497]
            )
            self.coef_ls = (
                [0.00025895413494090355, 0.00045862692849402506, 0.0007576280737663753] if args.xla
                else [0.0003013537830804257, 0.0005683767808602767, 0.0009387165655225843]
            )
            self.large_batch_size_ls = [2800, 1400, 900] if args.xla else [2130, 960, 640]
        else:
            raise ValueError('The ImageNet training process only supports "--amp"')
            
        self.resolution_ls = [160, 224, 288]
        self.dropout_rate_ls = [0.1, 0.2, 0.3]
        
        # Calculate data amounts and small batch sizes
        self.large_data_amount, self.small_data_amount, self.small_batch_size_ls = self.get_large_small_dataAmount_batchSize()
        
        # Milestones
        self.iter_milestones = np.array([60, 90, 105]) * (args.world_size - 1)
        self.cycle_milestones = np.array([20, 40, 60, 70, 80, 90, 95, 100, 105]) * (args.world_size - 1)
        
        # Parameter state
        self.step_idx = 0
        self.stage_idx = 0
        self.learning_rates = [2e-1 * 0.1 ** i for i in range(self.steps + 1)]
        
        self.parameter = self._update_parameter_dict()
        
        # Global model
        self.global_model = tf_data_model.modify_resnet(
            dataset=args.dataset,
            depth=args.depth,
            dropout_rate=self.dropout_rate_ls[0],
            resolution=self.resolution_ls[0],
        )
        
        # Record
        self.history_lock = threading.Lock()
        self.history = {
            'worker_ID': [], 'global_commit_ID': [], 'local_commit_ID': [],
            'step_ID': [], 'stage_ID': [], 'train_loss': [], 'train_acc': [],
            'train_time': [], 'val_loss': [], 'val_acc': [], 'commit_time': [],
        }
        
        self.outfile = (
            f'{args.dataset}_resnet{args.depth}_e{self.epochs}'
            f'_t{("%.2f" % args.time_ratio).replace(".", "")}'
            f'_w{args.world_size}s{args.small}'
            f'{"_amp" if args.amp else ""}{"_xla" if args.xla else ""}'
            f'{"" if args.cycle else "_noCycle"}'
            f'{"_" + args.comments if args.comments else ""}'
        )
        self.tempfile = f'temp_{self.outfile}'

    def _update_parameter_dict(self):
        return {
            'global_step_ID': self.step_idx,
            'learning_rate': self.learning_rates[min(self.step_idx, len(self.learning_rates)-1)],
            'global_stage_ID': self.stage_idx,
            'large_batch_size': self.large_batch_size_ls[self.stage_idx % len(self.large_batch_size_ls)],
            'small_batch_size': self.small_batch_size_ls[self.stage_idx % len(self.small_batch_size_ls)],
            'resolution': self.resolution_ls[self.stage_idx % len(self.resolution_ls)],
            'dropout_rate': self.dropout_rate_ls[self.stage_idx % len(self.dropout_rate_ls)],
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'large_data_amount': self.large_data_amount,
            'small_data_amount': self.small_data_amount,
        }

    def get_large_small_dataAmount_batchSize(self):
        num_total = self.args.world_size - 1
        num_small = self.args.small
        num_large = num_total - num_small
        
        large_data_amount = round(self.args.time_ratio * self.total_data_amount / num_total) if num_small else round(self.total_data_amount / num_total)
        small_data_amount = max(0, round((self.total_data_amount - large_data_amount * num_large) / num_small)) if num_small else 0
        
        small_batch_size_ls = []
        for large_batch_size, intercept, coef in zip(self.large_batch_size_ls, self.intercept_ls, self.coef_ls):
            time_origin = (coef + intercept / large_batch_size) * self.total_data_amount / num_total
            time_new = self.args.time_ratio * time_origin
            if num_small and small_data_amount > 0:
                # Calculate required BS, ensuring denominator is not too close to zero
                denominator = (time_new / small_data_amount - coef)
                if denominator > 1e-6:
                    small_batch_size = round(intercept / denominator)
                else:
                    # Fallback to a safe small value if time constraints are too tight
                    small_batch_size = 1
                small_batch_size_ls.append(max(1, small_batch_size))
            else:
                small_batch_size_ls.append(0)
        return large_data_amount, small_data_amount, small_batch_size_ls

    def sync_step(ps_rref, worker_weights, rank, is_small_batch):
        self = ps_rref.local_value()
        with self.parameter_lock:
            with self.global_model_lock:
                if not self.mission_complete:
                    # Update global model ONLY if weights are provided (avoid crash during initial sync)
                    if len(worker_weights) > 1:
                        server_weights = self.global_model.get_weights()
                        update_factor = self.small_data_amount / self.large_data_amount if is_small_batch else 1.0
                        
                        for i in range(len(server_weights)):
                            server_weights[i] = (server_weights[i] + update_factor * worker_weights[i]) / (1 + update_factor)
                        self.global_model.set_weights(server_weights)
                        
                        self.global_commit_ID += 1
                        
                        # Update parameters based on milestones
                        if self.global_commit_ID in self.iter_milestones:
                            self.step_idx += 1
                        if self.global_commit_ID in self.cycle_milestones:
                            self.stage_idx += 1
                        
                        self.parameter = self._update_parameter_dict()
                        
                        if self.global_commit_ID >= self.mini_epochs:
                            self.mission_complete = True
                        
                        print(f'Sync from Worker {rank} at Global Commit {self.global_commit_ID - 1}')
                    else:
                        print(f'Initial sync from Worker {rank}, fetching parameters...')
                
                return self.global_model.get_weights(), self.parameter, self.global_commit_ID - 1, self.mission_complete

    def update_history(ps_rref, record):
        self = ps_rref.local_value()
        with self.history_lock:
            record['commit_time'] = time.perf_counter() - self.start_time
            if record['global_commit_ID'] < self.mini_epochs:
                for key, value in record.items():
                    self.history[key].append(value)
                print(f'Worker {record["worker_ID"]} commit ID: {record["global_commit_ID"]}, '
                      f'time: {record["commit_time"]:.3f}, loss: {record["val_loss"]:.3f}, '
                      f'acc: {record["val_acc"]*100:.1f}%')
        
        if self.args.temp and (record['global_commit_ID'] + 1) in self.iter_milestones:
            self.save_tempfile(record['global_commit_ID'])

    def save_tempfile(self, temp_commit_ID):
        with self.global_model_lock:
            self.global_model.save(f'{self.tempfile}_model')
            np.save(f'{self.tempfile}.npy', self.history)
            print(f'Saved temporary files at Global Commit {temp_commit_ID}')

    def save_outfile(self):
        if self.args.save:
            self.global_model.save(f'{self.outfile}_model')
            np.save(f'{self.outfile}.npy', self.history)
            print(f'Saved Model: {self.outfile}_model\nSaved Logs: {self.outfile}.npy')
        if self.args.temp:
            shutil.rmtree(f'{self.tempfile}_model', ignore_errors=True)
            if os.path.isfile(f'{self.tempfile}.npy'):
                os.remove(f'{self.tempfile}.npy')


# Worker
class Worker(object):
    def __init__(self, ps_rref, args, rank, is_small_batch):
        self.args = args
        self.ps_rref = ps_rref
        self.rank = rank
        self.is_small_batch = is_small_batch
        
        self.local_commit_ID = 0
        self.step_ID = -1
        self.stage_ID = -1
        self.mission_complete = False
        
        self.parameter = None
        self.dataloader = None
        self.model = None
        self.verbose = 2
        
        class TimeCallback(keras.callbacks.Callback):
            def on_train_begin(self, logs=None): self.history = []
            def on_epoch_begin(self, epoch, logs=None): self.time_epoch_begin = time.perf_counter()
            def on_epoch_end(self, epoch, logs=None): self.history.append(time.perf_counter() - self.time_epoch_begin)
        self.time_callback = TimeCallback()

    def train(self):
        # Initial sync to get parameters and global weights
        global_weights, self.parameter, _, self.mission_complete = rpc.rpc_sync(
            self.ps_rref.owner(),
            Server.sync_step,
            kwargs={
                'ps_rref': self.ps_rref,
                'worker_weights': [0], # Dummy weights for initial sync
                'rank': self.rank,
                'is_small_batch': self.is_small_batch
            }
        )
        
        while not self.mission_complete:
            # Check if model or data needs to be re-initialized
            if self.step_ID != self.parameter['global_step_ID'] or self.stage_ID != self.parameter['global_stage_ID']:
                self.step_ID = self.parameter['global_step_ID']
                self.stage_ID = self.parameter['global_stage_ID']
                
                # Update dataloader with sharding
                self.dataloader = tf_data_model.load_data(
                    resolution=self.parameter['resolution'],
                    batch_size=self.parameter['small_batch_size'] if self.is_small_batch else self.parameter['large_batch_size'],
                    dataset=self.args.dataset,
                    dir_path=self.args.dir_path,
                    val_batch_size=self.parameter['large_batch_size'],
                    num_shards=self.args.world_size - 1,
                    shard_rank=self.rank - 1
                )
                
                # Update model structure and transfer weights
                self.model = tf_data_model.modify_resnet(
                    dataset=self.args.dataset,
                    depth=self.args.depth,
                    dropout_rate=self.parameter['dropout_rate'],
                    resolution=self.parameter['resolution'],
                    old_model=self.model,
                )
                
                # Re-compile only when structural parameters change
                self.model.compile(
                    optimizer=keras.optimizers.experimental.SGD(
                        learning_rate=self.parameter['learning_rate'],
                        momentum=self.parameter['momentum'],
                        weight_decay=self.parameter['weight_decay'],
                    ),
                    loss=keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'],
                )

            # Set latest global weights
            self.model.set_weights(global_weights)
            
            print(f'---- Worker {self.rank} ----\nLocal Commit {self.local_commit_ID}, Step {self.step_ID}, Stage {self.stage_ID}')
            print(f'Res {self.parameter["resolution"]}, LR {self.parameter["learning_rate"]:g}, '
                  f'BS {self.parameter["small_batch_size"] if self.is_small_batch else self.parameter["large_batch_size"]}')
            
            # Train for one mini-epoch
            steps_per_epoch = round(
                (self.parameter['small_data_amount'] if self.is_small_batch else self.parameter['large_data_amount']) /
                (self.parameter['small_batch_size'] if self.is_small_batch else self.parameter['large_batch_size'])
            )
            
            train_logs = self.model.fit(
                self.dataloader['train'].take(steps_per_epoch),
                verbose=self.verbose,
                callbacks=[self.time_callback],
            )
            
            # Combined Push weights & Pull new weights/parameters
            global_weights, self.parameter, global_commit_ID, self.mission_complete = rpc.rpc_sync(
                self.ps_rref.owner(),
                Server.sync_step,
                kwargs={
                    'ps_rref': self.ps_rref,
                    'worker_weights': self.model.get_weights(),
                    'rank': self.rank,
                    'is_small_batch': self.is_small_batch
                },
            )
            
            # Evaluate after sync
            val_logs = self.model.evaluate(self.dataloader['val'], verbose=self.verbose, return_dict=True)
            
            # Update history on server
            record = {
                'worker_ID': self.rank, 'global_commit_ID': global_commit_ID,
                'local_commit_ID': self.local_commit_ID, 'step_ID': self.step_ID,
                'stage_ID': self.stage_ID, 'train_loss': train_logs.history['loss'][0],
                'train_acc': train_logs.history['accuracy'][0], 'train_time': self.time_callback.history[-1],
                'val_loss': val_logs['loss'], 'val_acc': val_logs['accuracy'], 'commit_time': None,
            }
            rpc.rpc_sync(self.ps_rref.owner(), Server.update_history, kwargs={'ps_rref': self.ps_rref, 'record': record})
            
            print(f'Worker {self.rank} Local Commit {self.local_commit_ID} Complete')
            self.local_commit_ID += 1

