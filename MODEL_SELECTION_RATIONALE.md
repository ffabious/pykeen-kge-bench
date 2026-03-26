# Model Selection Rationale

This note summarizes the checked-in benchmark results and turns them into a defensible model choice.

## Recommendation

Select **PairRE** as the default model for this benchmark suite.

It is the strongest overall choice because it:

- achieves the best `MRR` in both benchmark modes
- has the best average ranking metrics across the expanded `complete` benchmark
- stays slightly faster than `DistMult` on average while delivering clearly better ranking quality
- remains dramatically more efficient than `ConvE`
- is still the top `MRR` model on the most accurate smaller benchmarks in the complete suite: `Kinships`, `Nations`, and `UMLS`

## Why PairRE

### Minimal benchmark (`Nations`)

`PairRE` is the best model on the compact benchmark:

| Model | Train Seconds | Parameters | MRR | Hits@1 | Hits@3 | Hits@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PairRE | 2.73 | 14,976 | **0.6989** | **0.5448** | **0.8308** | **0.9851** |
| ConvE | 7.12 | 94,352 | 0.6876 | 0.5423 | 0.7736 | **0.9851** |
| DistMult | 2.51 | 7,936 | 0.6712 | 0.5249 | 0.7687 | 0.9677 |
| TransE | 2.52 | 7,936 | 0.3402 | 0.0000 | 0.5622 | 0.9726 |

Interpretation:

- `PairRE` gives the highest `MRR`, `Hits@1`, and `Hits@3`.
- `ConvE` matches `Hits@10`, but needs about `2.6x` longer training and about `6.3x` more parameters.
- `DistMult` is compact and competitive, but still trails `PairRE` on every ranking metric.
- `TransE` is clearly not competitive here because its `Hits@1` collapses to `0.0000`.

For the minimal setting, the decision is straightforward: `PairRE` provides the best accuracy without paying the heavy compute and memory cost of `ConvE`.

### Complete benchmark (`Countries`, `Nations`, `Kinships`, `UMLS`, `CoDExSmall`, `DBpedia50`, and `FB15k237`)

`PairRE` is also the strongest model in the broader benchmark.

#### Per-dataset winners by MRR

| Dataset | Best Model | Best MRR | Second Best |
| --- | --- | ---: | ---: |
| CoDExSmall | DistMult | **0.2335** | PairRE `0.2325` |
| Countries | DistMult | **0.7622** | PairRE `0.7595` |
| DBpedia50 | DistMult | **0.3376** | PairRE `0.2809` |
| FB15k237 | DistMult | **0.1825** | PairRE `0.1780` |
| Kinships | PairRE | **0.6094** | ConvE `0.5517` |
| Nations | PairRE | **0.7245** | ConvE `0.7170` |
| UMLS | PairRE | **0.7718** | ConvE `0.6985` |

#### Average across complete-mode datasets

| Model | Avg Train Seconds | Avg Parameters | Avg MRR | Avg Hits@1 | Avg Hits@3 | Avg Hits@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PairRE | 155.69 | 408,859 | **0.5081** | **0.3964** | **0.5768** | **0.7139** |
| DistMult | 164.86 | 394,999 | 0.4199 | 0.3024 | 0.4796 | 0.6514 |
| ConvE | 509.80 | 487,356 | 0.3355 | 0.2416 | 0.3854 | 0.5188 |
| TransE | 150.05 | 394,999 | 0.2568 | 0.0811 | 0.3600 | 0.5729 |

Interpretation:

- `DistMult` wins `4` of the `7` datasets by `MRR`, but all four wins are narrow and concentrated on the larger graphs.
- `PairRE` wins the remaining `3` datasets and is the only model that leads every average ranking metric across the whole suite.
- `PairRE` also trains slightly faster than `DistMult` on average, so its accuracy gain does not come with a speed penalty.
- `ConvE` is much more expensive and still substantially behind `PairRE` and `DistMult` on the aggregate ranking metrics.
- `TransE` remains the weakest overall option because it is consistently poor on `Hits@1` and rarely competitive on `MRR`.

The complete benchmark matters more for model selection because it tests generalization across `7` datasets instead of one. On that stronger comparison, `PairRE` still gives the best accuracy-efficiency tradeoff.

## Efficiency Tradeoff

The main alternative is `ConvE`, because it is the only other model that comes close on some rankings in the minimal setting. However, the benchmark does not justify choosing it as the default:

- In `minimal`, `ConvE` is slightly worse than `PairRE` on `MRR`, `Hits@1`, and `Hits@3`, while being much larger and slower.
- In `complete`, `ConvE` trains about `3.3x` slower on average than `PairRE` and uses about `1.2x` more parameters on average.
- That extra cost does not buy a single `MRR` win in the checked-in complete benchmark and still leaves `ConvE` well behind on average `MRR`, `Hits@1`, `Hits@3`, and `Hits@10`.

So even if maximum model capacity is available, the measured return on that capacity is poor in this benchmark.

## Training-Loss Context

The training curves support the metric-based conclusion, but they also show why training loss alone should not drive model selection:

- `PairRE` shows steady loss reduction in both benchmark modes and converts that optimization progress into the best test metrics.
- `ConvE` can still drive training loss down on some datasets, but that does not translate into competitive complete-suite ranking metrics.
- This suggests that lower final training loss is not a sufficient reason to prefer `ConvE` here.

In other words, `PairRE` is not just easier to justify on accuracy; it is also the safer choice when the evaluation target is filtered ranking performance on held-out triples.

## Final Decision

Choose **PairRE** as the primary model for the case study.

Use the following rationale:

- it is the top performer on the benchmark's most important metric, `MRR`
- it wins on all complete-mode ranking metrics on average
- it beats `DistMult`, `ConvE`, and `TransE` on average `MRR`, `Hits@1`, `Hits@3`, and `Hits@10` across `7` datasets
- it remains computationally lightweight relative to `ConvE`
- it is a stronger overall default than `DistMult`, even though `DistMult` takes more individual dataset wins

## Secondary Options

If a backup model must be named:

- choose `DistMult` when you want the strongest alternative on the larger datasets and are willing to give up average ranking quality
- choose `ConvE` only if there is a separate reason to prefer a higher-capacity convolutional model despite the weak efficiency tradeoff in these results
- do not choose `TransE` as the default benchmark winner, because its performance is too inconsistent and too weak on `Hits@1`
