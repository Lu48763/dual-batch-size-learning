import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "parameter_server_3090.py"
MAIN_PATH = ROOT / "main_3090.py"

FAKE_TRAIN_DATASETS = []
FAKE_FIT_INPUTS = []
FAKE_FIT_STEPS = []


class FakeTrainIterator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.steps_consumed = 0

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration


class FakeTrainDataset:
    def __init__(self):
        self.iterator = FakeTrainIterator(self)
        self.iter_calls = 0
        self.take_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        return self.iterator

    def take(self, _steps):
        self.take_calls += 1
        raise AssertionError("split sync training must not restart the dataset with take()")


class FakeModel:
    def __init__(self):
        self._weights = [np.array([1.0]), np.array([2.0])]

    def compile(self, **_kwargs):
        return None

    def fit(self, data, steps_per_epoch=None, verbose=None, callbacks=None):
        if steps_per_epoch is None:
            raise AssertionError("streaming iterator training requires explicit steps_per_epoch")
        FAKE_FIT_INPUTS.append(data)
        FAKE_FIT_STEPS.append(steps_per_epoch)
        if hasattr(data, "steps_consumed"):
            data.steps_consumed += steps_per_epoch
        for callback in callbacks or []:
            callback.history = [0.25]
        return SimpleNamespace(history={"loss": [1.0], "accuracy": [0.5]})

    def evaluate(self, *_args, **_kwargs):
        return {"loss": 1.2, "accuracy": 0.6}

    def get_weights(self):
        return [weight.copy() for weight in self._weights]

    def set_weights(self, weights):
        self._weights = [np.array(weight).copy() for weight in weights]

    def save(self, _path):
        return None


class FakeRRef:
    def __init__(self, value):
        self._value = value

    def local_value(self):
        return self._value

    def owner(self):
        return "server_0"


def install_stubs():
    FAKE_TRAIN_DATASETS.clear()
    FAKE_FIT_INPUTS.clear()
    FAKE_FIT_STEPS.clear()

    keras_mod = types.ModuleType("tensorflow.keras")
    keras_mod.callbacks = SimpleNamespace(Callback=object)
    keras_mod.optimizers = SimpleNamespace(
        experimental=SimpleNamespace(SGD=lambda **_kwargs: object())
    )
    keras_mod.losses = SimpleNamespace(SparseCategoricalCrossentropy=lambda: object())
    keras_mod.models = SimpleNamespace(save_model=lambda *_args, **_kwargs: None)

    tensorflow_mod = types.ModuleType("tensorflow")
    tensorflow_mod.keras = keras_mod

    rpc_mod = types.ModuleType("torch.distributed.rpc")
    distributed_mod = types.ModuleType("torch.distributed")
    distributed_mod.rpc = rpc_mod
    torch_mod = types.ModuleType("torch")
    torch_mod.distributed = distributed_mod

    tf_data_model_mod = types.ModuleType("tf_data_model")

    def fake_load_data(**_kwargs):
        train_dataset = FakeTrainDataset()
        FAKE_TRAIN_DATASETS.append(train_dataset)
        return {"train": train_dataset, "val": object()}

    tf_data_model_mod.load_data = fake_load_data
    tf_data_model_mod.modify_resnet = lambda **_kwargs: FakeModel()

    parameter_server_mod = types.ModuleType("parameter_server_3090")
    parameter_server_mod.Server = object
    parameter_server_mod.Worker = object

    sys.modules["tensorflow"] = tensorflow_mod
    sys.modules["tensorflow.keras"] = keras_mod
    sys.modules["torch"] = torch_mod
    sys.modules["torch.distributed"] = distributed_mod
    sys.modules["torch.distributed.rpc"] = rpc_mod
    sys.modules["tf_data_model"] = tf_data_model_mod
    sys.modules["parameter_server_3090"] = parameter_server_mod


def load_module(module_name, module_path):
    install_stubs()
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_parameter_server_module():
    return load_module("ai_opt_code_4x_parameter_server_3090", MODULE_PATH)


def load_main_module():
    return load_module("ai_opt_code_4x_main_3090", MAIN_PATH)


def make_args(**overrides):
    args = {
        "dataset": "imagenet",
        "dir_path": "/data",
        "amp": True,
        "xla": False,
        "world_size": 5,
        "small": 2,
        "time_ratio": 1.05,
        "depth": 18,
        "cycle": True,
        "comments": None,
        "temp": False,
        "save": False,
        "sync_multiplier": 4,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def make_record(global_commit_ID):
    return {
        "worker_ID": 1,
        "global_commit_ID": global_commit_ID,
        "local_commit_ID": global_commit_ID,
        "step_ID": 0,
        "stage_ID": 0,
        "train_loss": 1.0,
        "train_acc": 0.5,
        "train_time": 3.0,
        "val_loss": 1.2,
        "val_acc": 0.6,
        "commit_time": None,
    }


class FourXScheduleTest(unittest.TestCase):
    def test_server_scales_internal_sync_schedule_but_keeps_validation_epochs(self):
        module = load_parameter_server_module()

        server = module.Server(make_args())

        self.assertEqual(server.sync_frequency_multiplier, 4)
        self.assertEqual(server.validation_interval, 4)
        self.assertEqual(server.validation_mini_epochs, 105 * 4)
        self.assertEqual(server.mini_epochs, 105 * 4 * 4)
        self.assertEqual(server.epoch_data_amount, 1281167)
        self.assertEqual(server.total_data_amount, 1281167)
        self.assertEqual(server.iter_milestones.tolist(), [60 * 4 * 4, 90 * 4 * 4, 105 * 4 * 4])
        self.assertEqual(
            server.cycle_milestones.tolist(),
            [epoch * 4 * 4 for epoch in [20, 40, 60, 70, 80, 90, 95, 100, 105]],
        )
        self.assertEqual(server.validation_iter_milestones.tolist(), [60 * 4, 90 * 4, 105 * 4])
        self.assertEqual(server.parameter["sync_frequency_multiplier"], 4)
        self.assertIn("_f4", server.outfile)

    def test_server_accepts_non_default_sync_multiplier(self):
        module = load_parameter_server_module()

        server = module.Server(make_args(sync_multiplier=3))

        self.assertEqual(server.sync_frequency_multiplier, 3)
        self.assertEqual(server.validation_interval, 3)
        self.assertEqual(server.mini_epochs, 105 * 4 * 3)
        self.assertEqual(server.iter_milestones.tolist(), [60 * 4 * 3, 90 * 4 * 3, 105 * 4 * 3])
        self.assertEqual(server.parameter["sync_frequency_multiplier"], 3)
        self.assertIn("_f3", server.outfile)

    def test_main_parser_defaults_to_four_and_accepts_sync_multiplier(self):
        module = load_main_module()

        default_args = module.parser.parse_args([
            "-r", "0", "-w", "5", "-a", "127.0.0.1",
            "-d", "imagenet", "-p", "/data", "-t", "1.05", "--amp",
        ])
        custom_args = module.parser.parse_args([
            "-r", "0", "-w", "5", "-a", "127.0.0.1",
            "-d", "imagenet", "-p", "/data", "-t", "1.05", "--amp",
            "--sync-multiplier", "3",
        ])
        alias_args = module.parser.parse_args([
            "-r", "0", "-w", "5", "-a", "127.0.0.1",
            "-d", "imagenet", "-p", "/data", "-t", "1.05", "--amp",
            "-f", "2",
        ])

        self.assertEqual(default_args.sync_multiplier, 4)
        self.assertEqual(custom_args.sync_multiplier, 3)
        self.assertEqual(alias_args.sync_multiplier, 2)

    def test_validation_helpers_fire_once_per_four_internal_syncs(self):
        module = load_parameter_server_module()

        validation_commits = [
            commit_ID
            for commit_ID in range(16)
            if module.should_validate_commit(commit_ID, 4)
        ]
        visible_commit_IDs = [
            module.validation_commit_ID(commit_ID, 4)
            for commit_ID in validation_commits
        ]

        self.assertEqual(validation_commits, [3, 7, 11, 15])
        self.assertEqual(visible_commit_IDs, [0, 1, 2, 3])

    def test_steps_for_sync_preserves_original_local_step_count(self):
        module = load_parameter_server_module()

        original_steps = round(320292 / 2330)
        split_steps = [
            module.steps_for_sync(
                data_amount=320292,
                batch_size=2330,
                sync_frequency_multiplier=4,
                local_commit_ID=local_commit_ID,
            )
            for local_commit_ID in range(4)
        ]

        self.assertEqual(sum(split_steps), original_steps)
        self.assertLessEqual(max(split_steps) - min(split_steps), 1)

    def test_history_uses_validation_epoch_upper_bound_not_internal_sync_bound(self):
        module = load_parameter_server_module()
        server = module.Server(make_args())

        module.Server.update_history(
            FakeRRef(server),
            make_record(global_commit_ID=server.validation_mini_epochs),
        )

        self.assertEqual(server.history["worker_ID"], [])

    def test_worker_consumes_continuous_train_iterator_for_split_syncs(self):
        module = load_parameter_server_module()
        args = make_args(small=0)
        weights = [np.array([1.0]), np.array([2.0])]
        parameter = {
            "global_step_ID": 0,
            "learning_rate": 0.1,
            "global_stage_ID": 0,
            "large_batch_size": 2,
            "small_batch_size": 1,
            "resolution": 160,
            "dropout_rate": 0.1,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "sync_frequency_multiplier": 4,
            "large_data_amount": 8,
            "small_data_amount": 4,
        }
        calls = {"sync": 0, "history": 0}

        def fake_rpc_sync(_owner, _func, kwargs):
            if "record" in kwargs:
                calls["history"] += 1
                return None
            if kwargs["worker_weights"] == [0]:
                return weights, parameter, -1, False

            calls["sync"] += 1
            return weights, parameter, calls["sync"] - 1, calls["sync"] >= 4

        module.rpc.rpc_sync = fake_rpc_sync

        worker = module.Worker(FakeRRef(None), args, rank=1, is_small_batch=False)
        worker.train()

        self.assertEqual(calls["sync"], 4)
        self.assertEqual(calls["history"], 1)
        self.assertEqual(FAKE_FIT_STEPS, [1, 1, 1, 1])
        self.assertEqual(len(FAKE_TRAIN_DATASETS), 1)
        self.assertEqual(FAKE_TRAIN_DATASETS[0].iter_calls, 1)
        self.assertEqual(FAKE_TRAIN_DATASETS[0].take_calls, 0)
        self.assertEqual(FAKE_TRAIN_DATASETS[0].iterator.steps_consumed, 4)
        self.assertEqual(len({id(data) for data in FAKE_FIT_INPUTS}), 1)
        self.assertIs(FAKE_FIT_INPUTS[0], FAKE_TRAIN_DATASETS[0].iterator)


if __name__ == "__main__":
    unittest.main()
