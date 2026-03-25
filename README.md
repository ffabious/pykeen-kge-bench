# PyKEEN KGE Case Study

This repository contains a reproducible PyKEEN benchmark for comparing four knowledge graph embedding models:

- `TransE`
- `PairRE`
- `DistMult`
- `ConvE`

The case study keeps the split and evaluation protocol fixed inside each dataset, then compares:

- `MRR`
- `Hits@1`, `Hits@3`, `Hits@10`
- training time
- parameter count

## Benchmark Modes

There are two benchmark presets:

- `minimal`: a small, fast benchmark on `Nations`
- `complete`: a broader benchmark on `Kinships` and `UMLS` with much longer training

The notebook exposes this as a single variable:

```python
BENCHMARK_MODE = "minimal"
```

Switch it to `"complete"` if you want the larger benchmark.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run From Python

Minimal benchmark:

```bash
. .venv/bin/activate
python benchmark_case_study.py --mode minimal
```

Complete benchmark:

```bash
. .venv/bin/activate
python benchmark_case_study.py --mode complete
```

Artifacts are saved separately:

- `results/minimal/`
- `results/complete/`

Each folder contains:

- `benchmark_results.csv`
- `benchmark_metadata.json`
- `training_losses.json`

## Notebook

The main deliverable is [case_study.ipynb](/Users/m3/Documents/uni/s26/nlp/pykeen-kge-bench/case_study.ipynb).

It:

- explains the setup
- lets you choose `minimal` or `complete`
- runs the benchmark
- shows result tables and plots
- gives a model selection rationale

To re-execute the notebook:

```bash
. .venv/bin/activate
python -m jupyter nbconvert --to notebook --execute --inplace case_study.ipynb
```

## Experimental Choices

Minimal mode:

- Dataset: `Nations`
- Epochs: `30`
- Embedding dimension: `64`
- Batch size: `128`

Complete mode:

- Datasets: `Kinships`, `UMLS`
- Epochs: `100`
- Embedding dimension: `64`
- Batch size: `128`

Shared settings:

- Random seed: `42`
- Optimizer: `Adam`
- Learning rate: `1e-3`
- Inverse triples: enabled
- Evaluation: filtered ranking metrics on fixed test splits

## Current Minimal Snapshot

The current checked-in notebook is intended to run in `minimal` mode by default. A recent run produced:

| Dataset | Model | Train Seconds | Parameters | MRR | Hits@1 | Hits@3 | Hits@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nations | PairRE | 2.57 | 14976 | 0.6989 | 0.5448 | 0.8308 | 0.9851 |
| Nations | ConvE | 9.13 | 94352 | 0.6876 | 0.5423 | 0.7736 | 0.9851 |
| Nations | DistMult | 2.57 | 7936 | 0.6712 | 0.5249 | 0.7687 | 0.9677 |
| Nations | TransE | 2.53 | 7936 | 0.3402 | 0.0000 | 0.5622 | 0.9726 |

In the minimal benchmark, `PairRE` is the best choice because it has the highest MRR while staying far smaller and faster than `ConvE`.
