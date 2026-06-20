# AI Opt Code 4x Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase synchronization frequency in `train/ai_opt_code/4x`, defaulting to four times the paper schedule while keeping validation count and final epoch count unchanged.

**Architecture:** Keep the original ImageNet hybrid schedule as the validation/epoch schedule. Add a configurable internal sync multiplier that defaults to 4 to the parameter server, scale internal commit milestones by the configured multiplier, split each worker's original mini-epoch steps across that many syncs, and validate/log only once per original mini-epoch.

**Tech Stack:** Python 3, TensorFlow/Keras training code, PyTorch RPC coordination, Python standard-library `unittest` for focused schedule tests.

---

### Task 1: Add Schedule Tests

**Files:**
- Create: `train/ai_opt_code/tests/test_ai_opt_code_4x_schedule.py`

- [ ] **Step 1: Write the failing tests**

Add tests that stub TensorFlow, PyTorch RPC, and `tf_data_model`, import `train/ai_opt_code/4x/parameter_server_3090.py`, and verify:
- `Server.mini_epochs` is `105 * (world_size - 1) * 4`.
- `Server.validation_mini_epochs` remains `105 * (world_size - 1)`.
- `Server.epoch_data_amount` remains the full ImageNet training size.
- iteration and cycle milestones are scaled for internal sync.
- validation helper functions fire once per configured sync interval.
- `update_history` does not append records at the validation commit upper bound.

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
python3 -B train/ai_opt_code/tests/test_ai_opt_code_4x_schedule.py -v
```

Expected: failure because the 4x multiplier and validation helpers do not exist yet.

### Task 2: Implement 4x Internal Sync

**Files:**
- Modify: `train/ai_opt_code/4x/parameter_server_3090.py`

- [ ] **Step 1: Add configurable schedule helpers**

Add:
```python
DEFAULT_SYNC_FREQUENCY_MULTIPLIER = 4

def should_validate_commit(global_commit_ID, validation_interval):
    return global_commit_ID >= 0 and (global_commit_ID + 1) % validation_interval == 0

def validation_commit_ID(global_commit_ID, validation_interval):
    return global_commit_ID // validation_interval

def steps_for_sync(data_amount, batch_size, sync_frequency_multiplier, local_commit_ID):
    total_steps = round(data_amount / batch_size)
    base_steps = total_steps // sync_frequency_multiplier
    extra_steps = total_steps % sync_frequency_multiplier
    return base_steps + (1 if local_commit_ID % sync_frequency_multiplier < extra_steps else 0)
```

- [ ] **Step 2: Scale server schedule**

Set `self.sync_frequency_multiplier` from `args.sync_multiplier`, defaulting to `DEFAULT_SYNC_FREQUENCY_MULTIPLIER`, `self.validation_interval = self.sync_frequency_multiplier`, `self.validation_mini_epochs = self.epochs * (args.world_size - 1)`, `self.mini_epochs = self.validation_mini_epochs * self.sync_frequency_multiplier`, and scale `iter_milestones` and `cycle_milestones` by `self.sync_frequency_multiplier`.

- [ ] **Step 3: Split each sync's train steps**

Keep the full ImageNet size as `self.epoch_data_amount = 1281167`, leave `large_data_amount` and `small_data_amount` as original per-mini-epoch amounts, and use `steps_for_sync(...)` in `Worker.train` so every group of local partial syncs sums to the original `round(data_amount / batch_size)` step count.

- [ ] **Step 4: Preserve validation history count**

Add the sync multiplier to `self.parameter`. In `Worker.train`, accumulate train loss, train accuracy, and train time across partial syncs. Run `model.evaluate()` and `Server.update_history()` only when `should_validate_commit(global_commit_ID, self.parameter['sync_frequency_multiplier'])` is true. Record `global_commit_ID` as `validation_commit_ID(...)`.

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
python3 -B train/ai_opt_code/tests/test_ai_opt_code_4x_schedule.py -v
```

Expected: all tests pass.

### Task 3: Verify Syntax and Diff

**Files:**
- Modify: `train/ai_opt_code/4x/parameter_server_3090.py`
- Create: `train/ai_opt_code/tests/test_ai_opt_code_4x_schedule.py`
- Create: `train/ai_opt_code/docs/superpowers/plans/2026-06-20-ai-opt-code-4x-sync.md`

- [ ] **Step 1: Compile changed Python files**

Run:
```bash
python3 -m py_compile train/ai_opt_code/4x/parameter_server_3090.py train/ai_opt_code/tests/test_ai_opt_code_4x_schedule.py
```

Expected: no output and exit code 0.

- [ ] **Step 2: Review diff**

Run:
```bash
git diff -- train/ai_opt_code/4x/parameter_server_3090.py train/ai_opt_code/tests/test_ai_opt_code_4x_schedule.py train/ai_opt_code/docs/superpowers/plans/2026-06-20-ai-opt-code-4x-sync.md
```

Expected: only the 4x sync implementation, focused tests, and this plan are changed.
