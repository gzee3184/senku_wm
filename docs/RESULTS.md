# Results

This file gives the closed-loop success rates of Senku and LeWM on four benchmarks. Probes and other analyses are in
a separate section at the end. [PLANNING.md](PLANNING.md) describes the planners. [USAGE.md](USAGE.md) gives the
commands.

## Conventions

- **Success** is a percentage of the episodes, pooled over the seeds in the row. One seed has 50 episodes. The
  standard error of one seed is about 7 points.
- **Held-out starts** come from episodes with id % 10 == 0. Senku does not train on these episodes. LeWM's released
  checkpoints train on a random 90% of all training clips, so LeWM has seen most of these episodes.
- **Goal +K** is the recorded frame K env steps after the start.
- **Planners:** HeldCEM (the default), LeWM's CEM and the gradient planner. See [PLANNING.md](PLANNING.md).
- **Joint readings** are the ant's joint positions and velocities (`proprio` in the code). Only AntMaze uses them.
- **Paired tests:** two methods on the same seeds see the same starts. We give the difference with a bootstrap 95%
  interval over episodes and an exact McNemar test (`scripts/paired_stats.py`).
- **Per-episode results** for every row are in [`results/`](../results/README.md).

## Models

| Model | Benchmark | Training data | Steps |
|---|---|---|---|
| Senku (expert) | Cube | LeWM's Cube expert data | 100k |
| Senku (Cube mix) | Cube | expert data 0.6, branched grasp pairs 0.25, noisy oracle 0.15 | 100k |
| Senku | PushT | LeWM's PushT expert data | 400k (chosen on validation) |
| Senku (image) | AntMaze | H-JEPA's AntMaze data, image only | 200k |
| Senku (joints) | AntMaze | H-JEPA's AntMaze data, image and joint readings | 300k |
| LeWM | all | the authors' released checkpoints (AntMaze: H-JEPA's release) | |

Senku (expert) and LeWM train on the same data. The Cube mix adds generated episodes. A scripted controller with
access to the simulator state makes them; Senku trains only on their pixels and actions. A branched pair starts from
one state and shows two outcomes: a close with the cube between the fingers, and a close away from the cube. All
Senku models train from scratch with the same loss ([ARCHITECTURE.md](ARCHITECTURE.md)).

## Planner choice

The rule, fixed before the runs: for each benchmark, use the planner with the highest mean success on validation
starts. Validation starts come from training episodes with id % 10 == 1, so they differ from the test starts. Seeds
42, 1, 2 (Cube pick: 7, 8, 9).

| Validation test | LeWM's CEM | HeldCEM | Gradient | Gradient + hold |
|---|---|---|---|---|
| Cube, LeWM's protocol (Senku, Cube mix) | 77.3 | **89.3** | 80.0 | 88.0 |
| Cube, pick (Senku, Cube mix) | 12.7 | 37.3 | 22.7 | **44.0** |
| PushT, LeWM's protocol (Senku) | 79.3 | 84.0 | **92.0** | 88.7 |
| AntMaze, +50 in H-JEPA's training file (Senku, joints) | 41.3 | | **70.0** | |

The rule picks the gradient planner with the hold on Cube (mean 66.0 vs HeldCEM 63.3, within the noise), the gradient
planner on PushT and on AntMaze. The Cube test runs with the gradient planner and the hold, for both models, are
running. Until they finish, the Cube tables below use HeldCEM for both models.

## OGBench Cube

### LeWM's protocol

Goal +25, budget 50 steps, held-out starts (LeWM's `config/eval/cube.yaml`).

| Method | Planner | Seeds | Success |
|---|---|---|---|
| Stand still | none | 10 | 44.2 |
| Replay the recorded actions | none | 10 | 100.0 |
| LeWM | LeWM's CEM | 10 | 67.0 |
| LeWM | HeldCEM | 10 | 74.2 |
| Senku (expert) | LeWM's CEM | 10 | 82.4 |
| Senku (expert) | HeldCEM | 4 | 86.0 |
| Senku (Cube mix) | LeWM's CEM | 10 | 75.6 |
| Senku (Cube mix) | HeldCEM | 10 | **88.8** |
| Senku (Cube mix) | gradient planner | 10 | 82.4 |

Standing still solves 44.2% of these starts, so the scores are close to the top. LeWM's paper reports 74 on one
draw of starts. Paired with LeWM (HeldCEM): Senku (Cube mix) +14.6 [+10.6, +18.6], p = 2e-12; Senku (expert, 4
seeds) +12.0 [+6.5, +18.0], p = 1e-4. With LeWM's CEM for both models: Senku (expert) +15.4
[+12.0, +19.0], p = 7e-18; Senku (Cube mix) +8.6 [+4.6, +12.4], p = 2e-5.

### Starts that need the cube

We keep only the starts where the cube must move. Positions come from the simulator state. Seeds 7 to 12, 300
episodes per cell.

| Filter | Start | Goal |
|---|---|---|
| pick (+25) | the gripper 3 cm or more from a resting cube | the cube lifted in the gripper |
| hard_far (+25) | the gripper 3 cm or more from a resting cube | the cube 5 cm or more from its start |
| place_away (+50) | the cube in the gripper | the cube on the table, 5 cm or more away; the gripper 10 cm or more from it |

| Method | Planner | pick | hard_far | place_away |
|---|---|---|---|---|
| LeWM | LeWM's CEM | 0.3 | 0.3 | 33.0 |
| LeWM | HeldCEM | 9.0 | 7.7 | **57.0** |
| Senku (expert) | HeldCEM | 28.0 | 28.0 | 29.7 |
| Senku (Cube mix) | HeldCEM | **37.3** | **35.7** | 11.0 |
| Senku (Cube mix) | gradient planner | 21.3 | 22.0 | |

With HeldCEM, Senku solves more pick and hard_far starts than LeWM. LeWM solves more place_away starts. Paired
differences with LeWM (HeldCEM), 95% interval, all p < 1e-8:

| Model | pick | hard_far | place_away |
|---|---|---|---|
| Senku (expert) | +19.0 [+13.3, +25.0] | +20.3 [+14.7, +26.0] | -27.3 [-34.0, -20.7] |
| Senku (Cube mix) | +28.3 [+22.3, +34.3] | +28.0 [+22.3, +34.0] | -46.0 [-52.3, -39.7] |

In the recorded place_away failures, Senku opens the gripper early and moves the arm to the goal arm pose (see "Cube vs arm
weight" in the analysis section).

### Longer goals

LeWM's protocol with goals +50, +75 and +100, budget 2 x the goal offset, seeds 42, 1, 2. The table counts only the
starts where the cube must move (120, 95 and 99 of 150 episodes).

| Method | Planner | +50 | +75 | +100 |
|---|---|---|---|---|
| LeWM | LeWM's CEM | 37.5 | 34.7 | 29.3 |
| LeWM | HeldCEM | **49.2** | 45.3 | 43.4 |
| Senku (Cube mix) | HeldCEM | 36.7 | **58.9** | **58.6** |

### H-JEPA's 50 Cube tasks

H-JEPA's task file `goal_offset_20_pickup_val.pt`: 20-step windows around the grasp, budget 30 steps. Each task
starts from its recorded simulator state. Success: the cube reaches its goal position.

| Method | Planner | Seeds | Success |
|---|---|---|---|
| LeWM (H-JEPA paper) | gradient planner | 3 | 32.0 |
| H-JEPA, 2 / 3 / 4 levels (H-JEPA paper) | hierarchical gradient planner | 3 | 47.3 / 60.0 / 55.3 |
| LeWM | HeldCEM | 3 | 20.7 |
| Senku (expert) | HeldCEM | 1 | **68.0** |
| Senku (Cube mix) | HeldCEM | 3 | 63.3 |
| Senku (Cube mix) | gradient planner | 3 | 49.3 |

Our goal is the frame 20 steps after the start. H-JEPA's evaluation uses the frame 19 steps after the start. Paired
with LeWM (HeldCEM): Senku (Cube mix) +42.7 [+33.3, +52.0], p = 5e-14.

With H-JEPA's goal (+19), the same 50 tasks:

| Method | Planner | Seeds | Success |
|---|---|---|---|
| Stand still | none | 1 | 0.0 |
| Replay the recorded actions | none | 1 | 100.0 |
| LeWM | gradient planner | 3 | 20.7 |
| LeWM | HeldCEM | 3 | 20.7 |
| Senku (expert) | HeldCEM | 3 | 64.7 |
| Senku (Cube mix) | HeldCEM | 3 | **68.7** |
| Senku (Cube mix) | gradient + hold | 3 | 64.7 |

Paired with LeWM (HeldCEM): Senku (Cube mix) +48.0 [+38.0, +57.3], p = 5e-16; Senku (expert) +44.0 [+34.0, +54.0],
p = 1e-13. With our gradient planner, LeWM's released checkpoint gives 20.7, below the 32.0 in H-JEPA's paper. We have
not reproduced the paper's LeWM number with H-JEPA's own code on Cube.

## PushT

LeWM's protocol: goal +25, budget 50 steps, held-out starts, 10 seeds. Senku is the 400k checkpoint, chosen on the
validation starts with the gradient planner (300k / 350k / 400k: 86.0 / 88.7 / 92.7).

| Method | Planner | Success |
|---|---|---|
| Stand still | none | 0.2 |
| Replay the recorded actions | none | 99.6 |
| LeWM | LeWM's CEM | **85.4** |
| LeWM | HeldCEM | 84.8 |
| LeWM | gradient planner | 90.2 |
| Senku | LeWM's CEM | 80.0 |
| Senku | HeldCEM | **86.2** |
| Senku | gradient planner | **92.0** |

Paired differences, Senku minus LeWM with the same planner:

| Planner | Difference | p |
|---|---|---|
| LeWM's CEM | -5.4 [-9.4, -1.4] | 0.009 |
| HeldCEM | +1.4 [-2.6, +5.4] | 0.55 |
| gradient planner | +1.8 [-1.4, +5.0] | 0.32 |

With LeWM's CEM, Senku is 5 points below LeWM. With HeldCEM or the gradient planner (the validation choice), the two
models do not differ.

### Training length

One LeWM epoch on PushT is 13,933 steps at batch 128 (1.78M training clips), so 10 epochs is 139k steps. LeWM's
config caps training at 100 epochs; the released checkpoint does not record how long it trained. Senku at 140k steps
(10 LeWM epochs), 250k (18 epochs) and 400k (29 epochs), test starts, 10 seeds:

| Planner | LeWM | Senku, 140k | Senku, 250k | Senku, 400k |
|---|---|---|---|---|
| LeWM's CEM | **85.4** | 77.8 | 80.8 | 80.0 |
| HeldCEM | 84.8 | 82.0 | 81.0 | **86.2** |
| gradient planner | 90.2 | 90.8 | 91.8 | **92.0** |

At 10 LeWM epochs, Senku with the gradient planner already matches LeWM (+0.6 [-2.8, +3.8], p = 0.81); with LeWM's
CEM it is 7.6 points below (p = 2e-4). Longer training changes the gradient planner result by about one point and
leaves Senku below LeWM with LeWM's CEM at every length.

### H-JEPA's PushT tasks

H-JEPA's task file `goal_offset_75_val.pt`: 50 tasks from H-JEPA's PushT validation episodes, goal +75, budget 90
steps. The scalers are fit on the training data.

| Method | Planner | Seeds | Success |
|---|---|---|---|
| Stand still | none | 1 | 0.0 |
| Replay the recorded actions | none | 1 | 100.0 |
| LeWM (H-JEPA paper) | gradient planner | 3 | 40.0 |
| H-JEPA, 2 levels (H-JEPA paper) | hierarchical gradient planner | 3 | **45.3** |
| LeWM | gradient planner | 3 | 32.0 |
| Senku | LeWM's CEM | 3 | 2.0 |
| Senku | gradient planner | 3 | 28.7 |

With the same planner, Senku and LeWM do not differ (-3.3 [-12.7, +6.0], p = 0.58). Both are below the paper's
numbers, and far below the replay bound. With LeWM's CEM, Senku solves almost nothing at this horizon (plans of 15
blocks, 150 numbers, from 300 samples).

## Visual AntMaze

H-JEPA's protocol: 50 tasks (`expert_grid_d3_n50.pt`), start and goal 3 maze cells apart, budget 350 steps. Each
task starts from its recorded state. The goal is the task's goal image, plus its goal joint readings for models that
use them. Success: the ant's xy within 1.0 of the goal xy.

**Exception to "pixels and actions only":** H-JEPA's AntMaze models (LeWM and H-JEPA) also get the ant's joint
readings. Senku (joints) gets the same input; Senku (image) does not. Each action is used for one env step (no hold):
replaying the task experts' actions held per 5-step block solves 0 of 50 tasks.

| Method | Planner | Seeds | Success | Closest distance to the goal (start 9.0) |
|---|---|---|---|---|
| LeWM (H-JEPA paper) | gradient planner | 3 | 18.0 | |
| H-JEPA, 3 levels (H-JEPA paper) | hierarchical gradient planner | 3 | **73.3** | |
| LeWM, H-JEPA's evaluation code, rerun | gradient planner | 1 | 14.0 | 5.8 |
| Senku (image) | CEM | 3 | 0.0 | 7.0 |
| Senku (image) | gradient planner | 3 | 12.7 | 6.1 |
| Senku (joints) | CEM | 3 | 1.3 | 7.2 |
| Senku (joints) | gradient planner | 3 | 25.3 | 5.1 |

Short goals (`eval_antmaze_short.py`): starts from H-JEPA's held-out validation episodes, goal +25 or +50, budget 2 x
the goal offset. Only windows where the expert moves 2.0 or more. Seeds 42, 1, 2. Replaying the expert gives 100%;
standing still gives 0%.

| Method | Planner | +25 | +50 |
|---|---|---|---|
| Senku (image) | CEM | 23.3 | 17.3 |
| Senku (image) | gradient planner | 24.7 | 22.0 |
| Senku (joints) | CEM | **61.3** | 20.7 |
| Senku (joints) | gradient planner | 51.3 | **41.3** |

On the long tasks, the ant stops at walls and corners on the straight line to the goal (see "AntMaze cost geometry").

## Ablations

### Model parts (leave one out)

Senku (expert) and five variants, each trained once to 40k steps on LeWM's Cube expert data, HeldCEM. LeWM's protocol:
seeds 42, 1, 2, 3. pick, hard_far and place_away: seeds 7 to 12 (300 episodes per cell). H-JEPA's tasks (+20): seed
42. The probe columns come from fixed held-out frames, without planning.

| Variant | LeWM's protocol | pick | hard_far | place_away | H-JEPA's tasks | Arm-only frame cheaper than cube-only frame (lower is better) | Rollout error after 10 blocks, g / tokens |
|---|---|---|---|---|---|---|---|
| Senku (all parts) | 88.5 | 20.0 | 17.0 | 22.7 | 54 | 0.12 | 0.136 / 0.214 |
| no auxiliary predictor | 86.0 | 23.7 | 25.0 | 16.7 | 74 | 0.35 | 0.168 / 0.267 |
| no rollout regularizers | 84.0 | 21.7 | 23.3 | 12.3 | 56 | 0.72 | 0.089 / 0.165 |
| no motion weights | 88.0 | 21.7 | 20.3 | 26.3 | 52 | 0.11 | 0.169 / 0.280 |
| plain predictor (no READ/WRITE) | 89.0 | 22.7 | 25.0 | 28.7 | 72 | 0.14 | 0.215 / 0.359 |
| plain predictor, no auxiliary predictor | 89.5 | **36.7** | **40.7** | 12.3 | **84** | 0.51 | 0.258 / 0.421 |

The probe column: re-rendered frames where only the arm reaches its goal pose (a failure) or only the cube reaches its
goal (a success). It gives the share of cases where the state-token cost prefers the failure.

No variant is best on all tests. At 40k, the simplest variant (plain predictor, no auxiliary predictor) is best on
pick, hard_far and H-JEPA's tasks, and among the worst on place_away. Its token cost prefers the arm-only failure in
51% of the cases (Senku: 12%), so its gains on the +25 goals, where the goal arm pose is close to the cube, may come
from following the arm. Its rollout error is also the highest of all variants, about twice Senku's. Without the rollout regularizers the rollout error is lower, but the cube weight is the worst of
all variants (0.72). We kept all parts because of the probes and
place_away: without the auxiliary predictor the token cost prefers the arm-only failure about 3 times as often, and without READ/WRITE the
open-loop rollout error grows by about 60%. These are single training runs at 40k, not the released 100k models, so
the closed-loop differences between variants are not conclusive.

At 60k / 100k the simplest variant shows the same pattern: pick 40.7 / 48.0 vs 18.3 / 28.0, place_away 13.0 / 10.7
vs 31.3 / 29.7 (Senku), arm-only frame cheaper 0.49 / 0.51 vs 0.14 / 0.13.

### Other parts

| Change | Without it | With it |
|---|---|---|
| State tokens next to g | g-only dynamics do not keep the cube | the planner sees the cube |
| Sparse g (ReLU + RDMReg), hard starts (+25), best checkpoint | 21% | 31% |
| HeldCEM instead of LeWM's CEM, LeWM, 10 seeds | 67.0 | 74.2 |
| Inverse-dynamics loss (optional, off by default), Cube mix, place_away at 40k | 14.0 | 18.7 |

With the inverse-dynamics loss, the encoder and the predicted trajectories match the run without it (within 1 cm),
so we do not use it.

## Long horizon (optional modules)

Full Cube episodes: start at step 0 of a held-out episode, goal +200, budget 400 steps, seeds 7, 8, 9. Senku (Cube
mix) with HeldCEM.

| Subgoals | Success |
|---|---|
| none | 6.7 |
| action repertoire (goal-blind, learned from the data) | **17.3** |
| upper levels (H-JEPA style, on the frozen model) | 8.7 |
| the model's latents along the expert's actions (oracle) | 48.0 |

With the oracle subgoals, success is 48.0. The gap between the oracle row and the other rows comes from the choice
of subgoals and actions, not from what the model can represent.

## Analysis (not closed loop)

### Cube vs arm weight (place_away)

The latent change for an 8 cm cube move, divided by the latent change for an 8 cm arm move:

| Model | Ratio | place_away success |
|---|---|---|
| LeWM | 15.5 | 57.0 |
| Senku (Cube mix + inverse dynamics, 100k) | 14.0 | 34.3 |
| Senku (Cube mix) | 3.3 | 11.0 |

The three models rank the same way in both columns (only 3 models). With a weak cube term, the goal arm pose is the
cheapest state. Prediction does not explain the gap: in a probe, Senku predicts the drop after the gripper opens as
well as LeWM (score 0.69 to 0.77 vs 0.60).

### AntMaze cost geometry

Spearman correlation of the latent goal distance with the geodesic distance (around the walls). 3000 held-out
states, 50 goals; Senku (joints) at 100k.

| Model | vs geodesic | geodesic beyond the straight-line distance |
|---|---|---|
| LeWM | 0.37 | 0.08 |
| Senku (joints) | 0.23 | 0.04 |
| Senku (joints), all states given the same joint readings | 0.41 | 0.09 |
| Senku (image) | 0.38 | 0.09 |

For no model does the latent distance follow the walls: the part of the geodesic beyond the straight-line distance
correlates at 0.09 or less. This agrees with H-JEPA (Fig. 7) for level-1 costs. In Senku (joints), the leg pose
dominates the latent distance. The goal's leg pose is a standing pose, which a walking ant does not have.

## Planning time

Planner time in seconds for 10 episodes solved together (seed 42, held-out starts), one run at a time per GPU
(RTX 6000 Blackwell). Rollouts are predictor rollouts per episode. Cube uses Senku (Cube mix), PushT Senku (PushT,
250k checkpoint; the architecture and so the cost are the same at 400k), AntMaze Senku (joints) on 10 of H-JEPA's
tasks. Files: `results/compute/`.

| Benchmark | Model | HeldCEM | LeWM's CEM | Gradient planner | Gradient planner + hold |
|---|---|---|---|---|---|
| Cube (+25, 50 steps) | Senku | 5.7 s | 6.1 s | 55.0 s | 54.1 s |
| Cube (+25, 50 steps) | LeWM | 5.7 s | 4.5 s | 49.6 s | 48.3 s |
| PushT (+25, 50 steps) | Senku | 5.6 s | 6.5 s | 59.8 s | 51.5 s |
| AntMaze (100 steps) | Senku | | 121.9 s | 147.0 s | |
| Rollouts per episode (Cube, Senku) | | 18,000 | 18,000 | 106,800 | 93,200 |

HeldCEM costs the same as LeWM's CEM: both evaluate 300 plans for 30 iterations. With LeWM's CEM, Senku takes
1.4 times LeWM's time, from the state tokens. The gradient planner takes about 9 times longer than CEM on Cube and
PushT, because it runs 800 candidates and re-plans every 2 blocks instead of every 5.

## Limitations

| Area | Result |
|---|---|
| Cube place_away | LeWM 57.0, Senku (Cube mix) 11.0 |
| Full Cube episodes (+200) | best 17.3 (action repertoire) |
| OGBench Scene, official 5 tasks | 1 of 50 (Latent Particle WM: 40%) |
| AntMaze, H-JEPA's tasks | Senku (joints) 25.3, below H-JEPA with 3 levels (73.3) |
| Planning time | with LeWM's CEM, Senku takes 1.4 times LeWM's time (6.1 s vs 4.5 s for 10 Cube episodes); the gradient planner, chosen on PushT and AntMaze, takes about 9 times longer than CEM |
| PushT with LeWM's CEM | Senku 80.0 vs LeWM 85.4 (paired test p = 0.009) |
| PushT with the gradient planner | Senku 92.0 vs LeWM 90.2, no significant difference (p = 0.32) |
| Ablations | single training runs at 40k; no variant is best on every test |
| Training seeds | one training run per model; the spread over seeds covers the planner and the starts only |
| Model selection | the architecture and the loss were chosen during development with closed-loop tests on the held-out starts; only the planner choice uses a separate validation split |
