import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tf_data_model.py"


class FakeDataset:
    def __init__(self):
        self.operations = []

    def shard(self, num_shards, shard_rank):
        self.operations.append(("shard", num_shards, shard_rank))
        return self

    def repeat(self):
        self.operations.append(("repeat",))
        return self

    def prefetch(self, buffer_size):
        self.operations.append(("prefetch", buffer_size))
        return self


def load_tf_data_model_module():
    datasets = []
    directory_calls = []

    def image_dataset_from_directory(**kwargs):
        dataset = FakeDataset()
        datasets.append(dataset)
        directory_calls.append(kwargs)
        return dataset

    keras_mod = types.ModuleType("tensorflow.keras")
    keras_mod.Model = object
    keras_mod.utils = SimpleNamespace(
        image_dataset_from_directory=image_dataset_from_directory,
    )

    tensorflow_mod = types.ModuleType("tensorflow")
    tensorflow_mod.keras = keras_mod
    tensorflow_mod.data = SimpleNamespace(AUTOTUNE="autotune")

    sys.modules["tensorflow"] = tensorflow_mod
    sys.modules["tensorflow.keras"] = keras_mod

    module_name = "ai_opt_code_tf_data_model"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, datasets, directory_calls


class ImageNetInputPipelineTest(unittest.TestCase):
    def test_training_reads_only_batches_requested_by_steps_per_epoch(self):
        module, datasets, directory_calls = load_tf_data_model_module()

        module.load_imagenet(
            resolution=160,
            batch_size=2330,
            dir_path="/data",
            val_batch_size=2330,
        )

        self.assertEqual(
            datasets[0].operations,
            [("repeat",), ("prefetch", "autotune")],
        )
        self.assertNotIn("seed", directory_calls[0])


if __name__ == "__main__":
    unittest.main()
