# Per-episode results

One file per seed for every row of the tables in [docs/RESULTS.md](../docs/RESULTS.md):
`<benchmark>/<test>/<model>_<planner>/s<seed>.json`. Each file has the per-episode successes, the starts (dataset
episode and step, or the task index), the planner settings and, for newer runs, the planner compute.

Two runs with the same test and seed use the same starts, so they can be compared episode by episode:

```bash
python scripts/paired_stats.py --a 'results/cube/pick/senku_mix_heldcem/s*.json' \
  --b 'results/cube/pick/lewm_heldcem/s*.json'
```

`validation/` holds the planner-choice runs on training episodes (id % 10 == 1). `ablations_40k/` holds the Cube
ablations (expert data, 40k steps, HeldCEM). `compute/` holds the planning-time runs (10 episodes, seed 42), with the
planner seconds and rollouts in `planner_compute`.
