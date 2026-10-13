# Usage

Commands for data, training and evaluation. Run them from the repository root. [RESULTS.md](RESULTS.md) gives the
numbers that these commands produce.

## Setup

```bash
git clone https://github.com/gzee3184/senku_wm.git && cd senku_wm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu128
export SENKU_HOME=/path/to/storage    # datasets in $SENKU_HOME/data, checkpoints in $SENKU_HOME/checkpoints
python -m pytest tests                # CPU tests
```

The planner needs an NVIDIA GPU. `requirements.txt` pins the versions that produced the numbers in
[RESULTS.md](RESULTS.md); install it first (`pip install -r requirements.txt --extra-index-url
https://download.pytorch.org/whl/cu128`) to reproduce them exactly. Newer versions work, but a few episodes can end
differently: on Cube seed 42, torch 2.14 and MuJoCo 3.15 give 84.0 instead of 86.0.

## Data

Put the files in `$SENKU_HOME/data`.

| Benchmark | Files | Source |
|---|---|---|
| OGBench Cube | `cube_single_expert.h5` | [quentinll/lewm-cube](https://huggingface.co/datasets/quentinll/lewm-cube) (LeWM) |
| PushT | `pusht_expert_train.h5` | [quentinll/lewm-pusht](https://huggingface.co/datasets/quentinll/lewm-pusht) (LeWM) |
| Visual AntMaze | `visual_antmaze_medium_explore_stitch_train.h5`, `visual_antmaze_medium_stitch_val.h5` | [jepa-world-models/h-jepa](https://huggingface.co/datasets/jepa-world-models/h-jepa) (H-JEPA) |
| H-JEPA's tasks | `expert_grid_d3_n50.pt` (AntMaze), `goal_offset_20_pickup_val.pt` (Cube) | [jepa-world-models/h-jepa](https://huggingface.co/datasets/jepa-world-models/h-jepa), `eval_trajs/` |
| Cube mix (optional) | `cube_branched_pairs_v{1,2}__NOMARKER.h5`, `cube_noisyoracle_p0.3_v2__NOMARKER.h5` | [abrar008/Senku_WM](https://huggingface.co/datasets/abrar008/Senku_WM) (dataset), or `scripts/data/gen_*_cube.py` |
| OGBench Scene (optional) | `scene_play_224px_official_policy.h5` | `scripts/data/gen_scene_play.py` |

```bash
hf download quentinll/lewm-cube --repo-type dataset --local-dir $SENKU_HOME/data --include "*.zst"
tar -I unzstd -xf $SENKU_HOME/data/cube_single_expert.tar.zst -C $SENKU_HOME/data
hf download quentinll/lewm-pusht --repo-type dataset --local-dir $SENKU_HOME/data --include "*.zst"
unzstd $SENKU_HOME/data/pusht_expert_train.h5.zst
hf download jepa-world-models/h-jepa --repo-type dataset --local-dir $SENKU_HOME/data \
  --include "visual_antmaze_medium_explore_stitch_train.h5" --include "visual_antmaze_medium_stitch_val.h5"
hf download jepa-world-models/h-jepa --repo-type dataset --local-dir $SENKU_HOME/data --include "eval_trajs/*"
mv $SENKU_HOME/data/eval_trajs/ant/expert_grid_d3_n50.pt \
   $SENKU_HOME/data/eval_trajs/ogbench/goal_offset_20_pickup_val.pt $SENKU_HOME/data/
```

LeWM baselines (`eval.py --lewm`) read `$SENKU_HOME/checkpoints/lewm-<env>/`:

```bash
hf download quentinll/lewm-cube --local-dir $SENKU_HOME/checkpoints/lewm-cube
hf download quentinll/lewm-pusht --local-dir $SENKU_HOME/checkpoints/lewm-pusht
```

```bash
hf download abrar008/Senku_WM --repo-type dataset --local-dir $SENKU_HOME/data --include "*.h5"   # Cube mix
```

Generated data must pass `python scripts/data/check_corpus_parity.py`: it must render like the evaluation.

## Training

`--sources` takes `file.h5:weight[:holdout|holdoutN],...`. `holdout` keeps the episodes with id % 10 == 0 out of
training. `holdout9800` also keeps the episodes 9800 and later out (H-JEPA's Cube validation split). `--compile` gives
about 30% more speed.

```bash
# Cube, expert data (Senku (expert))
python train.py --checkpoint-dir runs/cube --compile
# Cube mix (Senku (Cube mix))
python train.py --checkpoint-dir runs/cube_mix --compile --sources cube_single_expert.h5:0.6:holdout9800,\
cube_branched_pairs_v1__NOMARKER.h5:0.1,cube_branched_pairs_v2__NOMARKER.h5:0.15,cube_noisyoracle_p0.3_v2__NOMARKER.h5:0.15
# PushT: 150k steps at peak lr 2e-4, then a new warmup and cosine at peak 1e-4 to 250k, then again to 400k
python train.py --checkpoint-dir runs/pusht --compile --sources pusht_expert_train.h5:1.0:holdout --lr 2e-4 \
  --total-steps 150000
python train.py --checkpoint-dir runs/pusht --compile --sources pusht_expert_train.h5:1.0:holdout --lr 1e-4 \
  --total-steps 250000 --resume runs/pusht/step_150000.pt --reschedule
python train.py --checkpoint-dir runs/pusht --compile --sources pusht_expert_train.h5:1.0:holdout --lr 1e-4 \
  --total-steps 400000 --resume runs/pusht/step_250000.pt --reschedule
# AntMaze with joint readings (drop --proprio for the image-only model)
python train.py --checkpoint-dir runs/antmaze --compile --sources visual_antmaze_medium_explore_stitch_train.h5:1.0 \
  --proprio --total-steps 300000
```

Each run writes `step_<N>.pt` every 10k steps and `training_log.json`.

## Checkpoints

[abrar008/Senku_WM](https://huggingface.co/abrar008/Senku_WM) has every Senku checkpoint in
[RESULTS.md](RESULTS.md) (`cube/`, `pusht/`, `antmaze/`, `ablations_40k/`); the model card lists what each file is.

```bash
hf download abrar008/Senku_WM --local-dir $SENKU_HOME/checkpoints/senku
```

## Evaluation

Every command prints a `RESULT` line. `--out-json FILE` also saves the per-episode results. `--seed` selects the
starts (LeWM's sampler) and seeds the planner.

### Cube

```bash
# LeWM's protocol, held-out starts (HeldCEM is the default planner)
python eval.py --ckpt CKPT --env cube --seed 42 --holdout-mod 10
# starts that need the cube: pick, hard_far (+25) and place_away (+50)
python eval.py --ckpt CKPT --env cube --seed 7 --holdout-mod 10 --start-filter pick
python eval.py --ckpt CKPT --env cube --seed 7 --holdout-mod 10 --start-filter place_away --goal-offset 50 --eval-budget 100
# longer goals
python eval.py --ckpt CKPT --env cube --seed 42 --holdout-mod 10 --goal-offset 75 --eval-budget 150
# H-JEPA's 50 Cube tasks
python eval.py --ckpt CKPT --env cube --seed 42 --task-file $SENKU_HOME/data/goal_offset_20_pickup_val.pt \
  --goal-offset 20 --eval-budget 30
# LeWM with the same planner
python eval.py --lewm --env cube --seed 42 --holdout-mod 10
```

### PushT

```bash
# gradient planner (H-JEPA's settings: 800 candidates, lr 0.05)
python eval.py --ckpt CKPT --env pusht --seed 42 --holdout-mod 10 --solver gd --receding-horizon 2
# LeWM's CEM
python eval.py --ckpt CKPT --env pusht --seed 42 --holdout-mod 10 --solver cem
```

### Visual AntMaze

```bash
# H-JEPA's 50 tasks with the gradient planner (--solver cem for CEM)
python eval_antmaze.py --ckpt CKPT --seed 42 --solver gd
# short goals (+25 or +50) on H-JEPA's validation episodes
python eval_antmaze_short.py --ckpt CKPT --seed 42 --goal-offset 25 --solver gd
```

### Planner flags

| Flag | Effect |
|---|---|
| `--solver held` / `cem` / `gd` | HeldCEM (default), LeWM's CEM, the gradient planner |
| `--horizon`, `--receding-horizon` | plan length and blocks executed before a re-plan, in 5-step blocks |
| `--num-samples`, `--n-steps` | CEM samples and iterations |
| `--gd-samples`, `--gd-lr` | gradient planner candidates and learning rate |
| `--cost-mode` | `both` (default), `g` or `tokens` |

### Long horizon (optional)

`eval.py --imagined-subgoals 25` plans with imagined subgoals. `--lh-search rep` (action repertoire) and
`--lh-search upper` (upper levels) need modules trained with `python -m senku.long_horizon.repertoire`,
`python -m senku.long_horizon.td_head` and `python -m senku.long_horizon.upper_levels`. Each module has `--help`.

## Load a model

```python
from senku.model import load_model
model = load_model("step_100000.pt", device="cuda")
g, patches = model.encode(frames)          # frames: [B, 3, 224, 224], ImageNet-normalized
s = model.state_tokens(patches)            # [B, 49, 192]
```
