import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "main_3090.py"
PARAMETER_SERVER_PATH = ROOT / "parameter_server_3090.py"


class FakeModel:
    def __init__(self):
        self._weights = [
            np.zeros((4, 4), dtype=np.float32),
            np.zeros((4,), dtype=np.float32),
        ]

    def get_weights(self):
        return [weight.copy() for weight in self._weights]

    def set_weights(self, weights):
        self._weights = [np.asarray(weight).copy() for weight in weights]

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
    keras_mod.models = SimpleNamespace(save_model=lambda *_args, **_kwargs: None)

    tensorflow_mod = types.ModuleType("tensorflow")
    tensorflow_mod.keras = keras_mod

    rpc_mod = types.ModuleType("torch.distributed.rpc")
    distributed_mod = types.ModuleType("torch.distributed")
    distributed_mod.rpc = rpc_mod
    torch_mod = types.ModuleType("torch")
    torch_mod.distributed = distributed_mod
    torch_mod.futures = SimpleNamespace(wait_all=lambda _futures: None)

    tf_data_model_mod = types.ModuleType("tf_data_model")
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
    return load_module("ai_opt_code_powerSGD_parameter_server_3090", PARAMETER_SERVER_PATH)


def load_main_module():
    return load_module("ai_opt_code_powerSGD_main_3090", MAIN_PATH)


def make_args(**overrides):
    args = {
        "dataset": "imagenet",
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
        "trace_dir": None,
        "powersgd": True,
        "powersgd_rank": 1,
        "powersgd_start_iter": 0,
        "powersgd_min_compression_size": 4,
        "powersgd_error_feedback": True,
        "powersgd_seed": 48763,
        "prefetch_buffer_size": -1,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


class PowerSGDRpcTest(unittest.TestCase):
    def test_main_parser_exposes_powersgd_options(self):
        module = load_main_module()

        default_args = module.parser.parse_args([
            "-r", "0", "-w", "5", "-a", "127.0.0.1",
            "-d", "imagenet", "-p", "/data", "-t", "1.05", "--amp",
        ])
        custom_args = module.parser.parse_args([
            "-r", "0", "-w", "5", "-a", "127.0.0.1",
            "-d", "imagenet", "-p", "/data", "-t", "1.05", "--amp",
            "--powersgd-rank", "2",
            "--powersgd-start-iter", "3",
            "--powersgd-min-compression-size", "16",
            "--no-powersgd-error-feedback",
            "--powersgd-seed", "7",
            "--prefetch-buffer-size", "2",
            "--trace-dir", "/tmp/powersgd-traces",
        ])
        disabled_args = module.parser.parse_args([
            "-r", "0", "-w", "5", "-a", "127.0.0.1",
            "-d", "imagenet", "-p", "/data", "-t", "1.05", "--amp",
            "--no-powersgd",
        ])

        self.assertTrue(default_args.powersgd)
        self.assertEqual(default_args.powersgd_rank, 1)
        self.assertEqual(default_args.powersgd_start_iter, 2)
        self.assertEqual(default_args.powersgd_min_compression_size, 4096)
        self.assertTrue(default_args.powersgd_error_feedback)
        self.assertEqual(default_args.prefetch_buffer_size, -1)
        self.assertFalse(disabled_args.powersgd)
        self.assertEqual(custom_args.powersgd_rank, 2)
        self.assertEqual(custom_args.powersgd_start_iter, 3)
        self.assertEqual(custom_args.powersgd_min_compression_size, 16)
        self.assertFalse(custom_args.powersgd_error_feedback)
        self.assertEqual(custom_args.powersgd_seed, 7)
        self.assertEqual(custom_args.prefetch_buffer_size, 2)
        self.assertEqual(custom_args.trace_dir, "/tmp/powersgd-traces")

    def test_server_defaults_trace_dir_to_method_directory_and_honors_override(self):
        module = load_parameter_server_module()

        default_server = module.Server(make_args())
        custom_server = module.Server(make_args(trace_dir="/tmp/custom-powersgd-traces"))

        self.assertEqual(Path(default_server.trace_dir), ROOT / "traces")
        self.assertEqual(Path(default_server.outfile).parent, ROOT / "traces")
        self.assertEqual(Path(default_server.tempfile).parent, ROOT / "traces")
        self.assertEqual(Path(custom_server.trace_dir), Path("/tmp/custom-powersgd-traces"))
        self.assertEqual(Path(custom_server.outfile).parent, Path("/tmp/custom-powersgd-traces"))

    def test_low_rank_payload_roundtrip_preserves_shape_and_reduces_error(self):
        module = load_parameter_server_module()
        config = module.PowerSGDConfig(
            enabled=True,
            rank=2,
            start_iter=0,
            min_compression_size=4,
            use_error_feedback=False,
            seed=123,
        )
        weight = np.arange(36, dtype=np.float32).reshape(6, 6)

        payloads, residuals = module.build_powerSGD_payload(
            [weight],
            config=config,
            iteration=0,
            error_feedback=None,
        )
        restored = module.decompress_powerSGD_payload(payloads)[0]

        self.assertIsNone(residuals)
        self.assertEqual(payloads[0]["kind"], "low_rank")
        self.assertEqual(restored.shape, weight.shape)
        self.assertLess(np.linalg.norm(weight - restored), np.linalg.norm(weight))

    def test_warmup_sends_raw_worker_weights(self):
        module = load_parameter_server_module()
        config = module.PowerSGDConfig(
            enabled=True,
            rank=1,
            start_iter=3,
            min_compression_size=4,
            use_error_feedback=True,
            seed=123,
        )
        global_weights = [np.zeros((4, 4), dtype=np.float32)]
        worker_weights = [np.ones((4, 4), dtype=np.float32)]

        payload, residuals = module.build_worker_payload(
            worker_weights,
            global_weights,
            config=config,
            iteration=2,
            error_feedback=[],
        )

        self.assertIs(payload, worker_weights)
        self.assertEqual(residuals, [])

    def test_server_applies_decompressed_powersgd_weights(self):
        module = load_parameter_server_module()
        config = module.PowerSGDConfig(
            enabled=True,
            rank=4,
            start_iter=0,
            min_compression_size=4,
            use_error_feedback=False,
            seed=123,
        )
        server = module.Server(make_args(small=0))
        server.global_model.set_weights([
            np.zeros((4, 4), dtype=np.float32),
            np.zeros((4,), dtype=np.float32),
        ])
        worker_weights = [
            np.ones((4, 4), dtype=np.float32),
            np.ones((4,), dtype=np.float32),
        ]
        global_weights = [
            np.zeros((4, 4), dtype=np.float32),
            np.zeros((4,), dtype=np.float32),
        ]
        payload, _ = module.build_worker_payload(
            worker_weights,
            global_weights,
            config=config,
            iteration=0,
            error_feedback=None,
        )

        self.assertEqual(payload["payload_type"], "powersgd_weights")

        module.Server.sync_step(
            FakeRRef(server),
            payload,
            rank=1,
            is_small_batch=False,
        )

        updated_weights = server.global_model.get_weights()
        np.testing.assert_allclose(updated_weights[0], np.full((4, 4), 0.5, dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(updated_weights[1], np.full((4,), 0.5, dtype=np.float32), atol=1e-5)

    def test_compressed_update_preserves_original_weight_averaging_with_stale_server(self):
        module = load_parameter_server_module()
        config = module.PowerSGDConfig(
            enabled=True,
            rank=4,
            start_iter=0,
            min_compression_size=4,
            use_error_feedback=False,
            seed=123,
        )
        server = module.Server(make_args(small=0))
        server.global_model.set_weights([
            np.full((4, 4), 10.0, dtype=np.float32),
            np.full((4,), 10.0, dtype=np.float32),
        ])
        pulled_global_weights = [
            np.zeros((4, 4), dtype=np.float32),
            np.zeros((4,), dtype=np.float32),
        ]
        worker_weights = [
            np.ones((4, 4), dtype=np.float32),
            np.ones((4,), dtype=np.float32),
        ]
        payload, _ = module.build_worker_payload(
            worker_weights,
            pulled_global_weights,
            config=config,
            iteration=0,
            error_feedback=None,
        )

        module.Server.sync_step(
            FakeRRef(server),
            payload,
            rank=1,
            is_small_batch=False,
        )

        updated_weights = server.global_model.get_weights()
        np.testing.assert_allclose(updated_weights[0], np.full((4, 4), 5.5, dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(updated_weights[1], np.full((4,), 5.5, dtype=np.float32), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
