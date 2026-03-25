from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import pandas as pd
from pykeen.datasets import Nations
from pykeen.pipeline import pipeline

RESULTS_DIR = Path("results")
RESULTS_CSV_PATH = RESULTS_DIR / "benchmark_results.csv"
LOSSES_JSON_PATH = RESULTS_DIR / "training_losses.json"
METADATA_JSON_PATH = RESULTS_DIR / "benchmark_metadata.json"

DATASET_NAME = "Nations"
RANDOM_SEED = 42
EMBEDDING_DIM = 64
NUM_EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

MODEL_CONFIGS = [
    {"model": "TransE", "model_kwargs": {"embedding_dim": EMBEDDING_DIM}},
    {"model": "PairRE", "model_kwargs": {"embedding_dim": EMBEDDING_DIM}},
    {"model": "DistMult", "model_kwargs": {"embedding_dim": EMBEDDING_DIM}},
]


def _configure_runtime() -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger("pykeen").setLevel(logging.ERROR)
    logging.getLogger("torch_max_mem").setLevel(logging.ERROR)


def _dataset_summary(dataset: Nations) -> dict[str, int | str]:
    return {
        "dataset": DATASET_NAME,
        "training_triples": dataset.training.num_triples,
        "validation_triples": dataset.validation.num_triples,
        "testing_triples": dataset.testing.num_triples,
        "entities": dataset.training.num_entities,
        "relations": dataset.training.num_relations,
    }


def _build_pipeline_kwargs(dataset: Nations) -> dict:
    return {
        "dataset": dataset,
        "random_seed": RANDOM_SEED,
        "optimizer": "Adam",
        "optimizer_kwargs": {"lr": LEARNING_RATE},
        "negative_sampler": "basic",
        "negative_sampler_kwargs": {"num_negs_per_pos": 1},
        "training_kwargs": {
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "use_tqdm": False,
        },
        "evaluator_kwargs": {
            "filtered": True,
            "batch_size": BATCH_SIZE,
        },
        "evaluation_kwargs": {"use_tqdm": False},
        "device": "cpu",
    }


def run_case_study() -> tuple[pd.DataFrame, dict[str, list[float]], dict[str, int | str]]:
    _configure_runtime()
    dataset = Nations()
    base_kwargs = _build_pipeline_kwargs(dataset)

    rows: list[dict[str, float | int | str]] = []
    losses: dict[str, list[float]] = {}

    for config in MODEL_CONFIGS:
        model_name = config["model"]
        result = pipeline(**base_kwargs, **config)
        losses[model_name] = [float(loss) for loss in result.losses]
        rows.append(
            {
                "model": model_name,
                "train_seconds": float(result.train_seconds),
                "parameter_count": int(result.model.num_parameters),
                "mrr": float(
                    result.metric_results.get_metric(
                        "both.realistic.inverse_harmonic_mean_rank"
                    )
                ),
                "hits@1": float(
                    result.metric_results.get_metric("both.realistic.hits_at_1")
                ),
                "hits@3": float(
                    result.metric_results.get_metric("both.realistic.hits_at_3")
                ),
                "hits@10": float(
                    result.metric_results.get_metric("both.realistic.hits_at_10")
                ),
            }
        )

    results = (
        pd.DataFrame(rows)
        .sort_values(["mrr", "hits@10"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return results, losses, _dataset_summary(dataset)


def save_case_study_artifacts(
    results: pd.DataFrame,
    losses: dict[str, list[float]],
    dataset_info: dict[str, int | str],
    output_dir: Path = RESULTS_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / RESULTS_CSV_PATH.name, index=False)
    (output_dir / LOSSES_JSON_PATH.name).write_text(
        json.dumps(losses, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "dataset": dataset_info,
        "random_seed": RANDOM_SEED,
        "embedding_dim": EMBEDDING_DIM,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "models": MODEL_CONFIGS,
        "evaluation_protocol": "Filtered ranking evaluation on the fixed Nations test split.",
    }
    (output_dir / METADATA_JSON_PATH.name).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    results, losses, dataset_info = run_case_study()
    save_case_study_artifacts(results, losses, dataset_info)
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
