# Planning

All planners use the same world model and the same goal image. They differ in how they search for actions.
[RESULTS.md](RESULTS.md) has the full results.

## The cost

The planner encodes the last three frames and the goal image, rolls a candidate plan through the predictor and
scores the predicted state against the goal:

    cost = z(|g - g_goal|^2) + z(mean over tokens of |s - s_goal|^2)

z standardizes each term over the candidates, so neither term needs a weight (`--cost-mode both`, the default). The
gradient planner cannot use z: a z-score summed over the candidates has zero gradient. It divides each term by its
spread over the initial candidates instead. As in H-JEPA, the gradient planner scores the last two predicted states;
CEM scores the last one.

## The three planners

| | LeWM's CEM | HeldCEM (ours) | Gradient planner |
|---|---|---|---|
| Source | LeWM, stable-worldmodel `CEMSolver` | this repo, `senku/planning.py` | H-JEPA's `GradientSolver`, reimplemented in `senku/planning.py` |
| Search | sample 300 plans, keep the 30 best, refit a Gaussian, 30 iterations | as LeWM's CEM, with three changes (below) | AdamW on the actions through the model, from N(0, 1) candidates |
| Action per 5-step block | 5 actions | 1 action, held for 5 steps | 5 actions |
| Re-plan (AntMaze: every 2 blocks for all) | every 5 blocks | every 5 blocks | every 2 blocks |
| Horizon | fixed | fixed | shrinks over the episode |
| Warm start | yes | yes | no |
| Flag | `--solver cem` | `--solver held` (default) | `--solver gd` |

HeldCEM differs from LeWM's CEM in three task-free ways:

1. Each action holds one value for its 5-step block. The search space is 5 times smaller and the plans are smooth.
2. 30% of the elites stay for the next iteration (as iCEM).
3. The planner executes the best elite, not the elite mean.

The gradient planner uses H-JEPA's settings: 800 candidates and lr 0.05 on PushT and Cube, 100 candidates and lr
0.3 on AntMaze; at most 90 steps, with an early stop after 30 when the best cost stalls.

## Which planner wins

The planner for each benchmark is chosen on validation starts (training episodes with id % 10 == 1), not on the test
starts. See [RESULTS.md](RESULTS.md), "Planner choice".

| Benchmark | Chosen on validation | Validation success (chosen vs next) | Test success |
|---|---|---|---|
| Cube | gradient + hold | mean of LeWM's protocol and pick: 66.0 vs 63.3 (HeldCEM) | running; HeldCEM 88.8 / 37.3 |
| PushT | gradient planner | 92.0 vs 88.7 (gradient + hold), 250k checkpoint | 92.0 vs 86.2 (HeldCEM), 400k checkpoint |
| AntMaze | gradient planner | 70.0 vs 41.3 (CEM) | 25.3 vs 2.0 (CEM) |

On Cube, HeldCEM and the gradient planner with the hold are within the noise of each other.

## Why the planners differ

Same seeds in each row. Cube: Senku (Cube mix). PushT: Senku.

| Test | HeldCEM | HeldCEM, no hold | LeWM's CEM | LeWM's CEM, 800 plans x 90 iterations | Gradient | Gradient + hold |
|---|---|---|---|---|---|---|
| Cube, LeWM's protocol (seeds 42, 1, 2) | 88.7 | 77.3 | | | 81.3 | **89.3** |
| Cube, hard_far (seeds 7, 8, 9) | **32.7** | | | | 21.3 | 29.3 |
| Cube, H-JEPA's tasks (seeds 42, 43, 44) | 63.3 | | | | 49.3 | **66.0** |
| PushT (seeds 42, 1, 2) | | | 79.3 | 81.3 | **95.3** | **95.3** |

- **Cube:** the hold explains the gap. With the hold, the gradient planner matches HeldCEM. Without the hold,
  HeldCEM loses 11 points.
- **PushT:** the gradient search explains the gap. LeWM's CEM with the gradient planner's
  compute gains only 2 points. The hold has no effect on PushT.
- **AntMaze:** no planner uses the hold. The task experts' actions, held per 5-step block, solve 0 of 50 tasks.
