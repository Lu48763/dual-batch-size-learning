# AI Opt Code 4x Sync Implementation Tracking

**Goal:** Increase synchronization frequency in `train/ai_opt_code/4x`, defaulting to four times the paper schedule while keeping validation count and final epoch count unchanged.

**Current status:** Implemented and tested. The code keeps the original ImageNet hybrid schedule as the validation/epoch schedule, adds a configurable internal sync multiplier that defaults to 4, scales internal commit milestones by the configured multiplier, splits each worker's original mini-epoch steps across that many syncs, and validates/logs only once per original mini-epoch.

**Important correction:** Split syncs now consume a persistent training iterator. Each partial sync continues from the previous batch instead of repeatedly calling `.take(...)` on the dataset and rereading the dataset prefix.

**Tech stack:** Python 3, TensorFlow/Keras training code, PyTorch RPC coordination, Python standard-library `unittest` for focused schedule tests.

## Files

- `train/ai_opt_code/4x/main_3090.py`: exposes `--sync-multiplier` / `-f`, defaulting to 4.
- `train/ai_opt_code/4x/parameter_server_3090.py`: implements the internal sync multiplier, validation-visible commit mapping, split-step schedule, persistent train iterator, and once-per-original-mini-epoch validation/history logging.
- `train/ai_opt_code/4x/tests/test_ai_opt_code_4x_schedule.py`: verifies schedule scaling, parser options, validation helpers, split-step preservation, history bounds, and persistent iterator behavior.
- `train/ai_opt_code/4x/README.md`: summarizes usage and sync multiplier behavior.

## Implemented Checklist

- [x] Add configurable `DEFAULT_SYNC_FREQUENCY_MULTIPLIER = 4`.
- [x] Add validation helpers: `should_validate_commit(...)` and `validation_commit_ID(...)`.
- [x] Add `steps_for_sync(...)` so groups of partial syncs sum to the original local step count.
- [x] Scale internal server `mini_epochs`, LR milestones, and cycle milestones by the sync multiplier.
- [x] Keep validation-visible epochs at the original 105-epoch schedule.
- [x] Accumulate training metrics across partial syncs and record validation/history once per original mini-epoch.
- [x] Use a persistent train iterator for split syncs to avoid rereading the dataset prefix.
- [x] Keep the sync multiplier configurable instead of hard-coding 4 throughout the implementation.

## Verification

Run from repository root:

```bash
python3 -B train/ai_opt_code/4x/tests/test_ai_opt_code_4x_schedule.py -v
python3 -m py_compile train/ai_opt_code/4x/main_3090.py train/ai_opt_code/4x/parameter_server_3090.py train/ai_opt_code/4x/tf_data_model.py train/ai_opt_code/4x/tests/test_ai_opt_code_4x_schedule.py
```

Expected result: all unit tests pass and `py_compile` exits with code 0.
