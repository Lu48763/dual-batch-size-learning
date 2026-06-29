import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "main_3090.py"


def install_stubs():
    keras_mod = types.ModuleType("tensorflow.keras")
    keras_mod.mixed_precision = SimpleNamespace(
        Policy=lambda name: SimpleNamespace(name=name, compute_dtype="float16", variable_dtype="float32"),
        set_global_policy=lambda _policy: None,
    )

    tensorflow_mod = types.ModuleType("tensorflow")
    tensorflow_mod.keras = keras_mod

    rpc_mod = types.ModuleType("torch.distributed.rpc")
    distributed_mod = types.ModuleType("torch.distributed")
    distributed_mod.rpc = rpc_mod
    torch_mod = types.ModuleType("torch")
    torch_mod.distributed = distributed_mod
    torch_mod.futures = SimpleNamespace(wait_all=lambda _futures: None)

    parameter_server_mod = types.ModuleType("parameter_server_3090")
    parameter_server_mod.Server = object
    parameter_server_mod.Worker = object

    sys.modules["tensorflow"] = tensorflow_mod
    sys.modules["tensorflow.keras"] = keras_mod
    sys.modules["torch"] = torch_mod
    sys.modules["torch.distributed"] = distributed_mod
    sys.modules["torch.distributed.rpc"] = rpc_mod
    sys.modules["parameter_server_3090"] = parameter_server_mod


def load_main_module():
    install_stubs()
    module_name = "ai_opt_code_main_3090"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def base_args(**overrides):
    args = {
        "rank": 0,
        "world_size": 5,
        "addr": "127.0.0.1",
        "small": 2,
        "dataset": "imagenet",
        "dir_path": "/data",
        "time_ratio": 1.05,
        "amp": True,
        "schedule": "cyclic",
    }
    args.update(overrides)
    return SimpleNamespace(**args)


class ParserImplementationTest(unittest.TestCase):

    def test_parser_defaults_and_aliases_cover_all_cli_options(self):
        module = load_main_module()

        defaults = module.parser.parse_args([
            "-r", "1", "-w", "5", "-a", "127.0.0.1",
        ])

        self.assertEqual(defaults.rank, 1)
        self.assertEqual(defaults.world_size, 5)
        self.assertEqual(defaults.addr, "127.0.0.1")
        self.assertEqual(defaults.small, 0)
        self.assertEqual(defaults.port, "48763")
        self.assertEqual(defaults.device_index, 0)
        self.assertIsNone(defaults.dataset)
        self.assertIsNone(defaults.dir_path)
        self.assertEqual(defaults.depth, 18)
        self.assertEqual(defaults.time_ratio, 1)
        self.assertFalse(defaults.amp)
        self.assertFalse(defaults.xla)
        self.assertEqual(defaults.schedule, "cyclic")
        self.assertIsNone(defaults.comments)
        self.assertFalse(defaults.temp)
        self.assertTrue(defaults.save)

        custom = module.parser.parse_args([
            "--rank", "0",
            "--world-size", "5",
            "--server-addr", "127.0.0.1",
            "--small", "2",
            "--port", "12345",
            "--device-index", "-1",
            "--data", "imagenet",
            "--path", "/data",
            "--ratio", "1.05",
            "--mixed-precision",
            "--jit-compile",
            "--schedule", "uniform",
            "-c", "exp",
            "--temp",
            "--no-save",
        ])

        self.assertEqual(custom.rank, 0)
        self.assertEqual(custom.world_size, 5)
        self.assertEqual(custom.addr, "127.0.0.1")
        self.assertEqual(custom.small, 2)
        self.assertEqual(custom.port, "12345")
        self.assertEqual(custom.device_index, -1)
        self.assertEqual(custom.dataset, "imagenet")
        self.assertEqual(custom.dir_path, "/data")
        self.assertEqual(custom.depth, 18)
        self.assertEqual(custom.time_ratio, 1.05)
        self.assertTrue(custom.amp)
        self.assertTrue(custom.xla)
        self.assertEqual(custom.schedule, "uniform")
        self.assertEqual(custom.comments, "exp")
        self.assertTrue(custom.temp)
        self.assertFalse(custom.save)

        legacy_no_cycle = module.parser.parse_args([
            "-r", "0", "-w", "5", "-a", "127.0.0.1", "--no-cycle",
        ])
        self.assertEqual(legacy_no_cycle.schedule, "no-cycle")

        with self.assertRaises(SystemExit):
            module.parser.parse_args([
                "-r", "0", "-w", "5", "-a", "127.0.0.1",
                "--schedule", "uniform", "--no-cycle",
            ])

    def test_parser_rejects_unimplemented_dataset_and_depth(self):
        module = load_main_module()

        valid = module.parser.parse_args([
            "-r", "0", "-w", "5", "-a", "127.0.0.1",
            "-d", "imagenet", "-p", "/data", "-t", "1.05", "--amp",
        ])

        self.assertEqual(valid.dataset, "imagenet")
        with self.assertRaises(SystemExit):
            module.parser.parse_args([
                "-r", "0", "-w", "5", "-a", "127.0.0.1",
                "-d", "cifar100", "-p", "/data", "-t", "1.05", "--amp",
            ])
        with self.assertRaises(SystemExit):
            module.parser.parse_args([
                "-r", "0", "-w", "5", "-a", "127.0.0.1",
                "-d", "imagenet", "-p", "/data", "-t", "1.05", "--amp",
                "--depth", "34",
            ])

    def test_validate_args_rejects_invalid_distributed_settings(self):
        module = load_main_module()

        invalid_cases = [
            base_args(rank=-1),
            base_args(rank=5),
            base_args(world_size=1),
            base_args(small=-1),
            base_args(small=5),
            base_args(time_ratio=0),
        ]

        for args in invalid_cases:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    module.validate_args(args)

    def test_validate_args_enforces_rank_zero_training_requirements(self):
        module = load_main_module()

        for args in [
            base_args(dataset=None),
            base_args(dir_path=None),
            base_args(amp=False),
        ]:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    module.validate_args(args)

        module.validate_args(base_args())
        module.validate_args(base_args(rank=1, dataset=None, dir_path=None, amp=False))


if __name__ == "__main__":
    unittest.main()
