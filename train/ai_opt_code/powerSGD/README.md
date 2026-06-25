# PowerSGD RPC Communication Variant

This folder is a comparison-method variant of `train/ai_opt_code`.
It keeps the existing TensorFlow/Keras training loop and PyTorch RPC parameter-server architecture, then applies PowerSGD-style low-rank compression to the worker-to-server communication path.

## Scope

This is not a PyTorch DDP rewrite. PyTorch's built-in `powerSGD_hook` only applies to DDP gradient communication, while this code trains Keras models and uses RPC to push worker model updates. The implementation here keeps the original RPC flow and compresses worker model weights before they are sent to the server.

## Current Status

- [x] Keep the original `main_3090.py`, `parameter_server_3090.py`, and `tf_data_model.py` structure.
- [x] Add CLI controls for enabling/disabling PowerSGD-style compression.
- [x] Compress worker model weights with low-rank factors after a configurable warmup.
- [x] Decompress approximate worker weights on the server and apply the original weighted-average update.
- [x] Keep full-weight communication available through `--no-powersgd`.
- [x] Add unit tests for CLI parsing, payload compression, warmup fallback, and server update behavior.
- [ ] Run a multi-process or multi-node training smoke test on the target machines.
- [ ] Compare communication overhead and accuracy against the original `ai_opt_code` baseline.
- [ ] Evaluate combining this variant with the `4x` synchronization-frequency variant.

## How It Works

Each worker keeps the same training loop:

1. Pull global weights and training parameters from the RPC server.
2. Train locally for one mini-epoch.
3. If PowerSGD is active for the current local commit, compress eligible worker weight tensors into low-rank factors.
4. Send the payload to the server.
5. The server decompresses approximate worker weights and applies the same weighted-average update used by the original parameter-server code.

Small tensors, vectors, warmup iterations, and disabled PowerSGD runs use the original full-weight communication path.

## CLI Options

PowerSGD-style compression is enabled by default in this folder.

```bash
python main_3090.py -r=0 -w=5 -s=2 -a=${SERVER_IP} -d=imagenet -p=/data -t=1.05 --amp
python main_3090.py -r=1 -w=5 -s=2 -a=${SERVER_IP} -d=imagenet -p=/data -t=1.05 --amp
```

Useful options:

- `--no-powersgd`: disable compression and send full worker weights.
- `--powersgd-rank N`: set low-rank approximation rank. Default: `1`.
- `--powersgd-start-iter N`: send full weights before local commit `N`. Default: `2`.
- `--powersgd-min-compression-size N`: only compress tensors with at least `N` elements. Default: `4096`.
- `--no-powersgd-error-feedback`: disable worker-side residual error feedback.
- `--powersgd-seed N`: deterministic sketch seed. Default: `48763`.
- `--prefetch-buffer-size N`: set `tf.data` prefetch buffer size. Default: `-1`, which keeps `tf.data.AUTOTUNE`; use a positive integer to cap RAM pressure.
- `--trace-dir PATH`, `--output-dir PATH`: redirect saved `.npy` history traces and Keras model files.

Outputs default to `train/ai_opt_code/powerSGD/traces/`, independent of the current working directory. Filenames include `_psgdR<N>` when compression is enabled, or `_noPowerSGD` when disabled.

## Tests

Run from the repository root:

```bash
python3 -B train/ai_opt_code/powerSGD/tests/test_powersgd_rpc.py -v
python3 -m py_compile train/ai_opt_code/powerSGD/main_3090.py train/ai_opt_code/powerSGD/parameter_server_3090.py train/ai_opt_code/powerSGD/tf_data_model.py train/ai_opt_code/powerSGD/tests/test_powersgd_rpc.py
```

## Notes

- The compressed path preserves the original server weighted-average formula; only the communicated worker tensors are approximated.
- Rank and compression threshold should be tuned. Low ranks reduce communication but increase approximation error.
- Multi-node performance and accuracy still need real training validation.
