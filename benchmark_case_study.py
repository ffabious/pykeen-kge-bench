from __future__ import annotations

import argparse
import json
import logging
import traceback
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from pykeen.datasets import (
    CoDExSmall,
    Countries,
    DBpedia50,
    FB15k237,
    Kinships,
    Nations,
    UMLS,
)
from pykeen.pipeline import pipeline

RESULTS_DIR = Path("results")
RANDOM_SEED = 42

DATASET_REGISTRY = {
    "CoDExSmall": CoDExSmall,
    "Countries": Countries,
    "DBpedia50": DBpedia50,
    "FB15k237": FB15k237,
    "Nations": Nations,
    "Kinships": Kinships,
    "UMLS": UMLS,
}

BENCHMARK_CONFIGS: dict[str, dict[str, Any]] = {
    "minimal": {
        "description": "Small, fast benchmark on Nations for a compact student submission.",
        "datasets": ["Nations"],
        "embedding_dim": 64,
        "num_epochs": 30,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "create_inverse_triples": True,
        "models": [
            {"model": "TransE", "model_kwargs": {"embedding_dim": 64}},
            {"model": "PairRE", "model_kwargs": {"embedding_dim": 64}},
            {"model": "DistMult", "model_kwargs": {"embedding_dim": 64}},
            {
                "model": "ConvE",
                "model_kwargs": {
                    "embedding_dim": 64,
                    "embedding_height": 8,
                    "output_channels": 16,
                },
            },
        ],
    },
    "complete": {
        "description": "Broader 7-dataset benchmark spanning Countries, Nations, Kinships, UMLS, CoDExSmall, DBpedia50, and FB15k237 for a stronger comparison without the duplicate and heaviest datasets.",
        "datasets": [
            "Countries",
            "Nations",
            "Kinships",
            "UMLS",
            "CoDExSmall",
            "DBpedia50",
            "FB15k237",
        ],
        "embedding_dim": 64,
        "num_epochs": 50,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "create_inverse_triples": True,
        "models": [
            {"model": "TransE", "model_kwargs": {"embedding_dim": 64}},
            {"model": "PairRE", "model_kwargs": {"embedding_dim": 64}},
            {"model": "DistMult", "model_kwargs": {"embedding_dim": 64}},
            {
                "model": "ConvE",
                "model_kwargs": {
                    "embedding_dim": 64,
                    "embedding_height": 8,
                    "output_channels": 16,
                },
            },
        ],
    },
}


def _configure_runtime() -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger("pykeen").setLevel(logging.ERROR)
    logging.getLogger("torch_max_mem").setLevel(logging.ERROR)
    logging.getLogger("pykeen.triples").setLevel(logging.ERROR)
    logging.getLogger("pykeen.training").setLevel(logging.ERROR)


def get_benchmark_config(mode: str) -> dict[str, Any]:
    try:
        return BENCHMARK_CONFIGS[mode]
    except KeyError as error:
        allowed = ", ".join(sorted(BENCHMARK_CONFIGS))
        raise ValueError(f"Unknown benchmark mode {mode!r}. Expected one of: {allowed}.") from error


def get_dataset_summaries(mode: str) -> list[dict[str, int | str]]:
    config = get_benchmark_config(mode)
    summaries: list[dict[str, int | str]] = []
    for dataset_name in config["datasets"]:
        dataset = DATASET_REGISTRY[dataset_name](
            create_inverse_triples=config["create_inverse_triples"]
        )
        summaries.append(
            {
                "dataset": dataset_name,
                "training_triples": dataset.training.num_triples,
                "validation_triples": dataset.validation.num_triples,
                "testing_triples": dataset.testing.num_triples,
                "entities": dataset.training.num_entities,
                "relations": dataset.training.num_relations,
            }
        )
    return summaries


def _build_pipeline_kwargs(dataset: Any, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "random_seed": RANDOM_SEED,
        "optimizer": "Adam",
        "optimizer_kwargs": {"lr": config["learning_rate"]},
        "negative_sampler": "basic",
        "negative_sampler_kwargs": {"num_negs_per_pos": 1},
        "training_kwargs": {
            "num_epochs": config["num_epochs"],
            "batch_size": config["batch_size"],
            "use_tqdm": False,
        },
        "evaluator_kwargs": {
            "filtered": True,
            "batch_size": config["batch_size"],
        },
        "evaluation_kwargs": {"use_tqdm": False},
        "device": "cpu",
    }


def run_case_study(mode: str = "minimal") -> tuple[pd.DataFrame, dict[str, dict[str, list[float]]], list[dict[str, int | str]], dict[str, Any]]:
    _configure_runtime()
    config = get_benchmark_config(mode)
    total_runs = len(config["datasets"]) * len(config["models"])
    completed_runs = 0

    print(
        f"Starting {mode} benchmark: {len(config['datasets'])} datasets x "
        f"{len(config['models'])} models, {config['num_epochs']} epochs each.",
        flush=True,
    )

    rows: list[dict[str, float | int | str]] = []
    losses: dict[str, dict[str, list[float]]] = {}

    for dataset_name in config["datasets"]:
        dataset = DATASET_REGISTRY[dataset_name](
            create_inverse_triples=config["create_inverse_triples"]
        )
        base_kwargs = _build_pipeline_kwargs(dataset, config)
        losses[dataset_name] = {}

        for model_config in config["models"]:
            model_name = model_config["model"]
            run_index = completed_runs + 1
            print(
                f"[{run_index}/{total_runs}] Running {model_name} on {dataset_name}...",
                flush=True,
            )
            started_at = perf_counter()
            try:
                result = pipeline(**base_kwargs, **model_config)
            except Exception as error:
                elapsed = perf_counter() - started_at
                print(
                    f"[{run_index}/{total_runs}] ERROR while running {model_name} on "
                    f"{dataset_name} after {elapsed:.1f}s: {error}",
                    flush=True,
                )
                traceback.print_exc()
                raise
            completed_runs += 1
            elapsed = perf_counter() - started_at
            losses[dataset_name][model_name] = [float(loss) for loss in result.losses]
            rows.append(
                {
                    "dataset": dataset_name,
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
            print(
                f"[{completed_runs}/{total_runs}] Finished {model_name} on {dataset_name} "
                f"in {elapsed:.1f}s, MRR={rows[-1]['mrr']:.4f}.",
                flush=True,
            )

    results = (
        pd.DataFrame(rows)
        .sort_values(["dataset", "mrr", "hits@10"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    return results, losses, get_dataset_summaries(mode), config


def save_case_study_artifacts(
    results: pd.DataFrame,
    losses: dict[str, dict[str, list[float]]],
    dataset_summaries: list[dict[str, int | str]],
    config: dict[str, Any],
    mode: str,
    output_root: Path = RESULTS_DIR,
) -> Path:
    output_dir = output_root / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "benchmark_results.csv").write_text(
        results.to_csv(index=False),
        encoding="utf-8",
    )
    (output_dir / "training_losses.json").write_text(
        json.dumps(losses, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "benchmark_mode": mode,
        "description": config["description"],
        "datasets": dataset_summaries,
        "random_seed": RANDOM_SEED,
        "embedding_dim": config["embedding_dim"],
        "num_epochs": config["num_epochs"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "create_inverse_triples": config["create_inverse_triples"],
        "models": config["models"],
        "evaluation_protocol": "Filtered ranking evaluation on the fixed test split for each dataset.",
    }
    (output_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PyKEEN KGE case study benchmark.")
    parser.add_argument(
        "--mode",
        default="minimal",
        choices=sorted(BENCHMARK_CONFIGS),
        help="Benchmark preset to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        results, losses, dataset_summaries, config = run_case_study(mode=args.mode)
        output_dir = save_case_study_artifacts(
            results=results,
            losses=losses,
            dataset_summaries=dataset_summaries,
            config=config,
            mode=args.mode,
        )
        print(f"Saved artifacts to {output_dir}")
        print(results.round(4).to_string(index=False))
    except Exception as error:
        print(f"Benchmark failed: {error}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
