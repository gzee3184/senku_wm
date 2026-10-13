"""OGBench Visual AntMaze env, copied from H-JEPA (github.com/kevinghst/H-JEPA, MIT, commit 840e76b,
stable_worldmodel/envs/ogbench/{pointmaze,locomaze,antmaze}_env.py), so our harness evaluates
on their exact env and tasks.
Only the import paths were changed."""
from senku.envs.ogb_antmaze.antmaze_env import AntMazeEnv  # noqa: F401
