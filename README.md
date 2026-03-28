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
- stability across random seeds

## Benchmark Modes

There are two benchmark presets:

- `minimal`: a small, fast benchmark on `Nations`
- `complete`: a broader 7-dataset benchmark on `Countries`, `Nations`, `Kinships`, `UMLS`, `CoDExSmall`, `DBpedia50`, and `FB15k237`

The presets now differ in seed count as well:

- `minimal` runs `3` seeds by default: `42`, `43`, `44`
- `complete` runs `1` seed by default: `42`

You can override the seeds from the CLI:

```bash
. .venv/bin/activate
python benchmark_case_study.py --mode minimal --seeds 42 43 44
```

The notebook exposes this as a single variable:

```python
BENCHMARK_MODE = "minimal"
```

Switch it to `"complete"` if you want the larger benchmark.

## Setup

```bash
python3 -m venv .venv
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

- `benchmark_runs.csv`
- `benchmark_results.csv`
- `benchmark_metadata.json`
- `training_losses.json`

`benchmark_runs.csv` stores one row per dataset-model-seed run. `benchmark_results.csv` stores the aggregated summary used by the notebook, including mean metrics and standard deviations across seeds.

## Notebook

The main deliverable is [case_study.ipynb](./case_study.ipynb).

It:

- explains the setup
- lets you choose `minimal` or `complete`
- runs the benchmark
- shows the per-seed raw runs
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
- Seeds: `42`, `43`, `44`
- Embedding dimension: `64`
- Batch size: `128`

Complete mode:

- Datasets: `Countries`, `Nations`, `Kinships`, `UMLS`, `CoDExSmall`, `DBpedia50`, `FB15k237`
- Epochs: `50`
- Seeds: `42`
- Embedding dimension: `64`
- Batch size: `128`

Shared settings:

- Base seed constant: `42`
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

In the minimal benchmark, `PairRE` is the current best default because it has the highest average MRR while staying far smaller and faster than `ConvE`.

## Current Complete Snapshot

The current checked-in complete benchmark covers `7` datasets and keeps `PairRE` as the best overall default model, even though `DistMult` wins more individual datasets by `MRR`.

Per-dataset `MRR` winners:

| Dataset | Best Model | Best MRR | Second Best |
| --- | --- | ---: | ---: |
| CoDExSmall | DistMult | **0.2335** | PairRE `0.2325` |
| Countries | DistMult | **0.7622** | PairRE `0.7595` |
| DBpedia50 | DistMult | **0.3376** | PairRE `0.2809` |
| FB15k237 | DistMult | **0.1825** | PairRE `0.1780` |
| Kinships | PairRE | **0.6094** | ConvE `0.5517` |
| Nations | PairRE | **0.7245** | ConvE `0.7170` |
| UMLS | PairRE | **0.7718** | ConvE `0.6985` |

Average across the complete benchmark:

| Model | Avg Train Seconds | Avg Parameters | Avg MRR | Avg Hits@1 | Avg Hits@3 | Avg Hits@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PairRE | 155.69 | 408,859 | **0.5081** | **0.3964** | **0.5768** | **0.7139** |
| DistMult | 164.86 | 394,999 | 0.4199 | 0.3024 | 0.4796 | 0.6514 |
| ConvE | 509.80 | 487,356 | 0.3355 | 0.2416 | 0.3854 | 0.5188 |
| TransE | 150.05 | 394,999 | 0.2568 | 0.0811 | 0.3600 | 0.5729 |

That is why the broader benchmark still supports choosing `PairRE` as the main model: it leads every average ranking metric while training slightly faster than `DistMult` on average and far faster than `ConvE`. Multi-seed runs in the minimal preset now make that kind of recommendation more defensible than a single-run result.
