import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "parameter_server_3090.py"


class FakeModel:
    def __init__(self):
        self._weights = [np.array([0.0]), np.array([0.0])]

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


def install_stubs():
    keras_mod = types.ModuleType("tensorflow.keras")
    keras_mod.callbacks = SimpleNamespace(Callback=object)
    keras_mod.optimizers = SimpleNamespace(
        experimental=SimpleNamespace(SGD=lambda **_kwargs: object())
    )
    keras_mod.losses = SimpleNamespace(SparseCategoricalCrossentropy=lambda: object())

    tensorflow_mod = types.ModuleType("tensorflow")
    tensorflow_mod.keras = keras_mod

    rpc_mod = types.ModuleType("torch.distributed.rpc")
    distributed_mod = types.ModuleType("torch.distributed")
    distributed_mod.rpc = rpc_mod
    torch_mod = types.ModuleType("torch")
    torch_mod.distributed = distributed_mod

    tf_data_model_mod = types.ModuleType("tf_data_model")
    tf_data_model_mod.modify_resnet = lambda **_kwargs: FakeModel()

    sys.modules["tensorflow"] = tensorflow_mod
    sys.modules["tensorflow.keras"] = keras_mod
    sys.modules["torch"] = torch_mod
    sys.modules["torch.distributed"] = distributed_mod
    sys.modules["torch.distributed.rpc"] = rpc_mod
    sys.modules["tf_data_model"] = tf_data_model_mod


def load_parameter_server_module():
    install_stubs()
    module_name = "ai_opt_code_parameter_server_3090"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def make_args(**overrides):
    args = {
        "dataset": "imagenet",
        "amp": True,
        "xla": False,
        "world_size": 5,
        "small": 2,
        "time_ratio": 1.05,
        "depth": 18,
        "schedule": "cyclic",
        "cycle": True,
        "comments": None,
        "temp": False,
        "save": False,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


class NoCycleScheduleTest(unittest.TestCase):
    def test_no_cycle_starts_from_first_progressive_stage(self):
        module = load_parameter_server_module()

        server = module.Server(make_args(schedule="no-cycle", cycle=False))

        self.assertEqual(server.parameter["global_stage_ID"], 0)
        self.assertEqual(server.parameter["large_batch_size"], 2330)
        self.assertEqual(server.parameter["resolution"], 160)
        self.assertEqual(server.parameter["dropout_rate"], 0.1)
        self.assertEqual(server.parameter["learning_rate"], 0.2)

    def test_no_cycle_uses_learning_rate_milestones_as_stage_boundaries(self):
        module = load_parameter_server_module()
        server = module.Server(make_args(schedule="no-cycle", cycle=False))
        rref = FakeRRef(server)

        server.global_commit_ID = 20 * (server.args.world_size - 1) - 1
        module.Server.sync_step(
            rref,
            [np.array([1.0]), np.array([1.0])],
            rank=1,
            is_small_batch=False,
        )

        self.assertEqual(server.parameter["global_stage_ID"], 0)
        self.assertEqual(server.parameter["resolution"], 160)
        self.assertEqual(server.parameter["large_batch_size"], 2330)
        self.assertEqual(server.parameter["learning_rate"], 0.2)

        server.global_commit_ID = 60 * (server.args.world_size - 1) - 1
        module.Server.sync_step(
            rref,
            [np.array([1.0]), np.array([1.0])],
            rank=1,
            is_small_batch=False,
        )

        self.assertEqual(server.parameter["global_stage_ID"], 1)
        self.assertEqual(server.parameter["resolution"], 224)
        self.assertEqual(server.parameter["large_batch_size"], 1110)
        self.assertAlmostEqual(server.parameter["learning_rate"], 0.02)

        server.global_commit_ID = 90 * (server.args.world_size - 1) - 1
        module.Server.sync_step(
            rref,
            [np.array([1.0]), np.array([1.0])],
            rank=1,
            is_small_batch=False,
        )

        self.assertEqual(server.parameter["global_stage_ID"], 2)
        self.assertEqual(server.parameter["resolution"], 288)
        self.assertEqual(server.parameter["large_batch_size"], 740)
        self.assertAlmostEqual(server.parameter["learning_rate"], 0.002)

    def test_uniform_schedule_uses_even_stage_boundaries(self):
        module = load_parameter_server_module()
        server = module.Server(make_args(schedule="uniform", cycle=False))
        rref = FakeRRef(server)

        server.global_commit_ID = 20 * (server.args.world_size - 1) - 1
        module.Server.sync_step(
            rref,
            [np.array([1.0]), np.array([1.0])],
            rank=1,
            is_small_batch=False,
        )

        self.assertEqual(server.parameter["global_stage_ID"], 0)
        self.assertEqual(server.parameter["resolution"], 160)
        self.assertEqual(server.parameter["large_batch_size"], 2330)
        self.assertEqual(server.parameter["learning_rate"], 0.2)

        server.global_commit_ID = 35 * (server.args.world_size - 1) - 1
        module.Server.sync_step(
            rref,
            [np.array([1.0]), np.array([1.0])],
            rank=1,
            is_small_batch=False,
        )

        self.assertEqual(server.parameter["global_stage_ID"], 1)
        self.assertEqual(server.parameter["resolution"], 224)
        self.assertEqual(server.parameter["large_batch_size"], 1110)
        self.assertAlmostEqual(server.parameter["learning_rate"], 0.02)

        server.global_commit_ID = 70 * (server.args.world_size - 1) - 1
        module.Server.sync_step(
            rref,
            [np.array([1.0]), np.array([1.0])],
            rank=1,
            is_small_batch=False,
        )

        self.assertEqual(server.parameter["global_stage_ID"], 2)
        self.assertEqual(server.parameter["resolution"], 288)
        self.assertEqual(server.parameter["large_batch_size"], 740)
        self.assertAlmostEqual(server.parameter["learning_rate"], 0.002)


if __name__ == "__main__":
    unittest.main()
