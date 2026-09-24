# Torch-PC-ALM: educational PyTorch port of PC-ALM

This repository is an **unofficial PyTorch port** of the PC-ALM algorithm for
students and educational projects, created with the help of **Codex
(GPT-6-Sol)** and **Claude Code (Opus 5.5)**. Full credit for the algorithm,
paper, and original JAX implementation belongs to **Jeffrey Seely and Julian
Gould at Sakana AI**. See the [original PC-ALM repository](https://github.com/SakanaAI/pc-alm)
and [paper](https://arxiv.org/abs/2605.31022). This port is not maintained or
endorsed by the original authors.

![PC-ALM Figure 1](assets/fig1.png)

*Figure from the original PC-ALM repository.*

> **PC-ALM** aligns local predictive-coding updates with backpropagation by
> accumulating layer-local constraint errors in Lagrange multipliers.

[![arXiv](https://img.shields.io/badge/arXiv-2605.31022-b31b1b?style=flat-square)](https://arxiv.org/abs/2605.31022)
[![Blog](https://img.shields.io/badge/Blog-Sakana%20AI-1f6feb?style=flat-square)](https://pub.sakana.ai/pc-alm/)

This port covers the paper's residual MLP width/depth grid on MNIST and
Fashion-MNIST, with BP, PC, and PC-ALM on the same architecture. It uses the
paper's fixed `gamma0=1` parameterization. See
[How this port differs from the original JAX code](#how-this-port-differs-from-the-original-jax-code).

## Installation

Run from a source checkout. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync
```

To run the test suite:

```bash
uv sync --extra test
uv run pytest
```

The project pins PyTorch to the CUDA 13.0 wheel index in `pyproject.toml`.
Use an NVIDIA driver compatible with that wheel. The training commands use
`device: cuda` by default and report an error if CUDA is unavailable. Add
`--device cpu` to either training command for CPU validation.

PC and PC-ALM inference steps are compiled with `torch.compile`, the PyTorch
counterpart of the original `jax.jit`. The first steps of each run take a few
seconds to compile. Compilation needs Triton on CUDA and a C++ compiler on CPU.
To run eagerly instead, for debugging or on a machine without those, set
`TORCH_COMPILE_DISABLE=1`. Other differences between this port and the original
JAX code are described in 
[this section](#how-this-port-differs-from-the-original-jax-code).

## Data

The synthetic smoke test does not require downloaded data:

```bash
uv run python train.py --config configs/smoke.yaml
```

Download MNIST and [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
from the repository root:

```bash
mkdir -p data/MNIST/raw data/FashionMNIST/raw
for file in train-images-idx3-ubyte train-labels-idx1-ubyte \
            t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -fL "https://storage.googleapis.com/cvdf-datasets/mnist/${file}.gz" \
    -o "data/MNIST/raw/${file}.gz" || break
  curl -fL "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/${file}.gz" \
    -o "data/FashionMNIST/raw/${file}.gz" || break
done
```

Both uncompressed IDX files and `.gz` IDX files are supported. Use `--data-dir`
to point to another directory containing `MNIST/raw/` and `FashionMNIST/raw/`.

## Training

Reproduce one Fashion-MNIST cell: width `N=32`, depth `L=32`, ReLU, seed 0,
one epoch on the full dataset, and inference budget `T=2L`:

```bash
uv run python scripts/run_headline_grid.py --config configs/headline_fashion.yaml \
  --widths 32 --depths 32 --activations relu --seeds 0 --methods bp,pc,pcalm \
  --budget-rule 2L --state-lr-table configs/eta_best_by_cell.csv \
  --output-dir results/repro_fashion_n32_l32 --data-dir data
```

Earlier JAX CPU reference results (PyTorch uses a different weight RNG, so exact
seed-matched values are not expected):

| Method | Test accuracy | Gradient cosine to BP |
|---|---:|---:|
| BP | 78.66% | 1.000 |
| PC | 68.13% | 0.604 |
| PC-ALM | 77.75% | 0.909 |

`configs/eta_best_by_cell.csv` contains the paper's frozen activity step sizes
(`eta_h = 1/lambda_max`, median over seeds) for each dataset/activation/width/depth.
For custom runs, edit a YAML config or use `uv run python train.py --help` for
single-run options.

<details>
<summary>Full Fashion-MNIST grid (675 runs)</summary>

```bash
uv run python scripts/run_headline_grid.py --config configs/headline_fashion.yaml \
  --widths 8,16,32,64,128 --depths 8,16,32,64,128 \
  --activations linear,tanh,relu --seeds 0,1,2 --methods bp,pc,pcalm \
  --budget-rule 2L --state-lr-table configs/eta_best_by_cell.csv \
  --output-dir results/headline_fashion --data-dir data
```

</details>

Use `configs/headline_mnist.yaml` for MNIST.

## Evaluation

Each run writes:

```text
results/path-to-run/
  config.json
  metrics.csv
  summary.json
```

For grid runs, `cells.csv` summarizes all methods and seeds. A compact heatmap
can be generated with:

```bash
uv run python scripts/plot_headline_grid.py \
  --input results/repro_fashion_n32_l32/cells.csv \
  --output results/repro_fashion_n32_l32/gain_pcalm_minus_pc.png \
  --activation relu
```

Use the full grid's `cells.csv` for a heatmap across widths and depths.

## How this port differs from the original JAX code

The algorithm is unchanged. For the same starting weights and data, this port
computes the same predictions, inference results and weight updates as the
original (see [Validation](#validation)). What changed is how the code is
organized, so that it reads like ordinary PyTorch code and runs fast on a GPU.

Two terms used below:

- **Hidden states** are the values held by the hidden layers. PC and PC-ALM
  treat them as free variables, which is why the original code calls them
  `free`. This port calls them `z`, the paper's notation. Before each weight
  update, they adjust these values step by step to lower an energy; this phase
  is called *inference*.
- **Duals** (Lagrange multipliers) are extra values, one per hidden unit and
  per example, that PC-ALM accumulates during inference. They record how far
  each layer still is from matching the prediction of the layer below. In PC
  they stay at zero.

### Summary

| Topic | Original JAX code | This port | Why |
|---|---|---|---|
| Weights | A plain list with one matrix per layer, built by `init_params` | A `ResidualMLP` class (`pcalm/model.py`) holding three weight arrays: `w_in` (first layer), `w_mid` (all middle layers stacked into one 3-D array) and `w_out` (output layer) | All middle layers have the same size, so one operation can process all of them at once. This is much faster on a GPU than one small operation per layer. |
| Model computations | Separate functions (`forward`, `block_pred`, `constraint_residuals`) that receive the weights, layer scales, skip-connection flags and activation function as arguments | Methods of the model: `feedforward_hidden` (hidden states from a plain forward pass, used as the starting point of inference, like the original `free_init`), `readout` (output layer), `constraint_residuals` (per-layer mismatches, same name as before) and `forward` | The model keeps its own scales and activation function, so they are no longer passed to every function. Skip connections follow from the layer's position (only middle layers have one), so the flags are gone. |
| Hidden states and duals | `free` and `duals`: a list with one array per hidden layer | `z` and `lam`: one array each, of shape `[layers, batch, width]` | Same reason as for the weights: every layer is updated in one operation. |
| Inference | `run_pc`, `run_pcalm`, `infer_for_schedule` and `_solve_inner`, looping with `jax.lax.scan` | One function, `infer` (`pcalm/inference.py`), with an ordinary Python loop | PC is the special case of PC-ALM in which the duals never change, so one function covers both. |
| Gradient of the energy with respect to the hidden states | `jax.grad` | `torch.func.grad`, the PyTorch equivalent | Both compute the derivative automatically; the code states only the energy formula. |
| Weight gradients | `method_grad` returns a list of gradients, which the training step passes to the optimizer | `method_loss` returns a single number (the BP loss, or the PC/PC-ALM energy at the end of inference). Calling `.backward()` on it fills in the gradients | This is the standard PyTorch training loop: `zero_grad()`, `loss.backward()`, `optimizer.step()`. |
| Method settings | A `Schedule` class, filled from the config | The `MethodConfig` config class is used directly and checks its values when it is created | Removes a class that duplicated the config, and puts all input checks in one place. |
| Optimizer | Adam written by hand (`pcalm/optim.py`) | PyTorch's built-in `torch.optim.Adam` | Same formula and same settings, tested against the original. |
| Metrics | Hand-written loss, cross-entropy and gradient cosine (`tree_l2`, `tree_dot`, `tree_cos`) | PyTorch's built-in functions (`F.mse_loss`, `F.cross_entropy`, `F.cosine_similarity`), and `grad_cosine` replaces the three `tree_*` helpers | Less code, same values. |
| Compilation | `jax.jit` compiles the whole training step | `torch.compile` compiles one inference iteration (`_outer_step`) | See below. |
| Hardware | Runs wherever JAX runs by default | A `device` option (`cuda` by default, or `cpu`) in the config and on the command line | Explicit choice, with a clear error if the GPU is not available. |
| Random starting weights | Drawn with JAX's random generator | Drawn with PyTorch's random generator, always on the CPU, then moved to the chosen device | The same seed therefore gives different weights than in JAX, but the same weights on CPU and GPU. |

Unchanged: dataset loading and subsets, batch order, the YAML configs and
command-line options (apart from `--device`), the learning-rate rule, the
output files, and the grid and plotting scripts.

### Compilation

Both versions compile the code: before running it, they turn the Python
functions into optimized machine code, which is much faster than running each
operation separately. JAX compiles the whole training step, and its
`lax.scan` loop is compiled once and reused for every inference step. PyTorch
has no direct equivalent of `lax.scan`: compiling the full inference loop would
unroll it into 64 to 256 copies and take a long time. This port therefore
compiles a single inference iteration (one or more gradient steps on the
hidden states, then the dual update) and calls it from a plain Python loop.

Compilation happens on the first call and takes a few seconds per run.
`train_one` clears the compiled code at the start of each run. Without that, a
grid that runs many model sizes in one process would reach PyTorch's limit on
compiled versions and fall back to slower, uncompiled code with only a warning.
To skip compilation, for debugging or on a machine without the required
compilers, set `TORCH_COMPILE_DISABLE=1`; the results are the same.

### Validation

The tests include fixed-array parity checks against the original JAX
implementation, including three identical minibatch/Adam updates for each
method. Run these checks with:

```bash
uv run pytest -q tests/test_parity.py
```

These tests use the same saved starting weights, inputs, labels, and batch order
for JAX and PyTorch. The JAX results they compare against are stored in two
files, tracked in git:

- `tests/jax_reference.json`: a small fixed case (3 examples, input size 2,
  width 3, depth 4) with its weights, inputs and labels. For each activation
  (linear, tanh, ReLU) it holds the JAX feed-forward values, constraint
  residuals, starting energy, hidden states and duals after PC and PC-ALM
  inference, BP/PC/PC-ALM weight gradients, and the weights after one Adam
  step.
- `tests/jax_training_reference.json`: three consecutive training steps for
  each method (batch indices, gradients and weights after each step).

These files were produced once by running the original JAX code; the script
that generated them is not part of this repository.

The full headline command in [Training](#training) starts from each
framework's own random weights; compare its accuracy and gradient-cosine
trends with the JAX reference, not exact seed-matched values.

To run the CUDA smoke checks where the GPU is visible:

```bash
uv run pytest -q tests/test_training.py -k cuda
```

For a representative CUDA step-time and peak-memory check before a large grid:

```bash
uv run python scripts/benchmark_train_step.py --method pcalm \
  --width 32 --depth 32 --budget 64
```

## Citation

If you use this port, please cite the original work:

```bibtex
@misc{seely2026pc-alm,
  title         = {Augmented Lagrangian Predictive Coding},
  author        = {Jeffrey Seely and Julian Gould},
  year          = {2026},
  eprint        = {2605.31022},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2605.31022},
}
```

## License

This port retains the original repository's MIT license and copyright notice
in [LICENSE](LICENSE).
