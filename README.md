# PyKEEN KGE Case Study

This project is a minimal reproducible benchmark for comparing three knowledge graph embedding models in PyKEEN:

- `TransE`
- `PairRE`
- `DistMult`

The case study keeps the dataset split and evaluation protocol fixed, then compares:

- `MRR`
- `Hits@1`, `Hits@3`, `Hits@10`
- training time
- parameter count

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run The Benchmark

```bash
. .venv/bin/activate
python benchmark_case_study.py
```

This saves the main artifacts in `results/`:

- `benchmark_results.csv`
- `benchmark_metadata.json`
- `training_losses.json`

## Open The Notebook

The main deliverable is `case_study.ipynb`.

It:

- explains the controlled setup
- runs the benchmark
- shows tables and plots
- gives a short model selection rationale based on the observed results

If you want to re-execute the notebook:

```bash
. .venv/bin/activate
python -m jupyter nbconvert --to notebook --execute --inplace case_study.ipynb
```

## Experimental Choices

- Dataset: `Nations` from PyKEEN
- Random seed: `42`
- Embedding dimension: `64`
- Epochs: `30`
- Optimizer: `Adam` with learning rate `1e-3`
- Evaluation: filtered ranking metrics on the fixed test split

The setup is intentionally simple and student-sized: enough to compare models fairly without turning the case study into a hyperparameter tuning project.

## Current Result Snapshot

From the executed notebook in this repository:

| Model | Train Seconds | Parameters | MRR | Hits@1 | Hits@3 | Hits@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PairRE | 1.90 | 7936 | 0.6447 | 0.4627 | 0.7711 | 0.9876 |
| DistMult | 1.92 | 4416 | 0.4776 | 0.2836 | 0.5622 | 0.9552 |
| TransE | 2.12 | 4416 | 0.3232 | 0.0000 | 0.5498 | 0.9826 |

In this controlled setup, `PairRE` is the recommended final model because it gives the best ranking quality with almost the same training time as the other two models.
