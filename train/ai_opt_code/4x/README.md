# AI Opt Code Sync Multiplier Variant

This folder is a configurable synchronization-frequency variant of `train/ai_opt_code`.
It follows the paper's ImageNet hybrid dual-batch and cyclic progressive learning schedule,
but lets workers synchronize more often inside each original mini-epoch.

## Current Experiment Setting

The default setting is a 4x synchronization multiplier:

```bash
python main_3090.py -r=0 -w=5 -s=2 -a=${SERVER_IP} -d=imagenet -p=/data -t=1.05 --amp -f=4
python main_3090.py -r=1 -w=5 -s=2 -a=${SERVER_IP} -d=imagenet -p=/data -t=1.05 --amp -f=4
```

`-f=4` is also the default, so omitting it gives the same behavior in this folder.

## What Changed

- The internal server synchronization schedule is multiplied by the sync multiplier.
- The paper's validation-visible schedule stays unchanged: ImageNet still runs for 105 epochs.
- Learning-rate milestones remain at paper epochs 60, 90, and 105.
- Resolution/dropout cycle milestones remain at paper epochs 20, 40, 60, 70, 80, 90, 95, 100, and 105.
- Each worker splits the original mini-epoch training steps across the configured number of syncs.
- Validation and history logging run only once per original mini-epoch, not once per internal sync.
- Output filenames include `_fN`, where `N` is the sync multiplier.

## Changing the Sync Multiplier

Use the same multiplier on every server and worker process:

```bash
python main_3090.py ... -f=2
python main_3090.py ... --sync-multiplier=8
```

Accepted aliases:

- `-f`
- `--freq`
- `--sync-freq`
- `--sync-multiplier`
- `--sync-frequency-multiplier`

The multiplier must be a positive integer. If you set `-f=N`, the code will:

- run `N` internal syncs per original mini-epoch,
- scale internal commit milestones by `N`,
- split each original mini-epoch's local training steps across `N` syncs,
- keep validation/history count aligned with the original 105-epoch schedule.

For example, changing from `-f=4` to `-f=8` doubles the number of internal syncs again, but does not double validation runs or final epochs.
