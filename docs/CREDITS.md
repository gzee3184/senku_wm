# Credits

The papers below give techniques that the released Senku model and its planners use. "Used" means we use the
method as published. "Adapted" means we changed it. The new parts of Senku are listed at the end. The README credits
the benchmarks, data and environment code.

## Model and training

| Part of Senku | Source | How |
|---|---|---|
| Joint-embedding world model from pixels, trained end to end; history of 3 latents, 5-step frame skip | LeWM: Maes et al., 2026, [arXiv 2603.19312](https://arxiv.org/abs/2603.19312) | used |
| SIGReg (isotropic Gaussian test on random projections) | LeJEPA: Balestriero and LeCun, 2025, [arXiv 2511.08544](https://arxiv.org/abs/2511.08544); code from LeWM | used; we also apply it to the state tokens and to patch residuals |
| Projector (Linear, BatchNorm, GELU, Linear) | LeWM | used |
| Sparse global latent: ReLU with GELU gradient, RDMReg toward a rectified Laplace | LpWM: Kuang et al., 2026, [arXiv 2608.22764](https://arxiv.org/abs/2608.22764) | used |
| AdaLN-zero action conditioning | DiT: Peebles and Xie, 2022, [arXiv 2212.09748](https://arxiv.org/abs/2212.09748); used for actions in LeWM | used in the READ blocks |
| ViT encoder with a convolutional stem | ViT: Dosovitskiy et al., 2020, [arXiv 2010.11929](https://arxiv.org/abs/2010.11929); Xiao et al., 2021, [arXiv 2106.14881](https://arxiv.org/abs/2106.14881) | adapted (small sizes) |
| Joint readings on AntMaze (27-D, added to the image encoding) | H-JEPA: Zhang et al., 2026, [arXiv 2610.06805](https://arxiv.org/abs/2610.06805) | used |

## Planning

| Part of Senku | Source | How |
|---|---|---|
| CEM planner and its settings (300 samples, 30 elites, 30 iterations) | LeWM; `CEMSolver` in [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) | used |
| Keeping a share of the elites between iterations (HeldCEM) | iCEM: Pinneri et al., 2020, [arXiv 2008.06389](https://arxiv.org/abs/2008.06389) | used |
| Gradient planner: AdamW on the actions, early stop, shrinking horizon, cost on the last two states | H-JEPA; `GradientSolver` in stable-worldmodel | reimplemented, batched over environments |

## New in Senku

- A global latent and 49 state tokens, with a READ/WRITE predictor: the global latent reads the tokens, and the
  tokens are rewritten from the predicted global latent. The action reaches the tokens only through the global latent.
- An auxiliary token-conditioned predictor that trains the encoder, with 0.3 of the main predictor's gradient.
- A prediction loss with a fixed share for the global latent and per-token motion weights.
- SIGReg on patch residuals (each patch minus its frame mean).
- A planning cost that z-scores the global and the token distances over the candidates, with no tuned weight.
- HeldCEM: one action per 5-step block, held, plus elite retention and executing the best elite.
- Branched grasp pairs for the Cube mix: one start, two outcomes (a close on the cube and a close beside it).
