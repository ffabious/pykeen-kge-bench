# Model Selection Rationale

This note summarizes the checked-in benchmark results and turns them into a defensible model choice.

## Recommendation

Select **PairRE** as the default model for this benchmark suite.

It is the strongest overall choice because it:

- achieves the best `MRR` in both benchmark modes
- ranks first on all reported metrics in the `complete` benchmark
- stays close to the fastest models in training time
- uses far fewer parameters than `ConvE` while outperforming it

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

### Complete benchmark (`Kinships` and `UMLS`)

`PairRE` is also the strongest model in the broader benchmark.

#### Per-dataset winners by MRR

| Dataset | Best Model | Best MRR | Second Best |
| --- | --- | ---: | ---: |
| Kinships | PairRE | **0.6590** | ConvE `0.6091` |
| UMLS | PairRE | **0.8212** | TransE `0.6645` |

#### Average across complete-mode datasets

| Model | Avg Train Seconds | Avg Parameters | Avg MRR | Avg Hits@1 | Avg Hits@3 | Avg Hits@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PairRE | 18.20 | 16,736 | **0.7401** | **0.6041** | **0.8497** | **0.9678** |
| ConvE | 113.45 | 98,713.5 | 0.5814 | 0.4192 | 0.6817 | 0.9000 |
| DistMult | 15.17 | 12,192 | 0.4764 | 0.3299 | 0.5287 | 0.7953 |
| TransE | 13.31 | 12,192 | 0.4457 | 0.2329 | 0.5889 | 0.8064 |

Interpretation:

- `PairRE` is the only model that is consistently strong across both datasets.
- `ConvE` is much more expensive, but still loses by a clear margin on every average ranking metric.
- `TransE` is fast and does reasonably well on `UMLS`, but it is too weak on `Kinships` to be the default choice.
- `DistMult` is efficient, yet its overall accuracy is not close to `PairRE`.

The complete benchmark matters more for model selection because it tests generalization across more than one dataset. On that stronger comparison, `PairRE` dominates the accuracy-efficiency tradeoff.

## Efficiency Tradeoff

The main alternative is `ConvE`, because it is the only other model that comes close on some rankings in the minimal setting. However, the benchmark does not justify choosing it as the default:

- In `minimal`, `ConvE` is slightly worse than `PairRE` on `MRR`, `Hits@1`, and `Hits@3`, while being much larger and slower.
- In `complete`, `ConvE` trains about `6.2x` slower on average than `PairRE` and uses about `5.9x` more parameters.
- That extra cost does not translate into better ranking quality on either `Kinships` or `UMLS`.

So even if maximum model capacity is available, the measured return on that capacity is poor in this benchmark.

## Training-Loss Context

The training curves support the metric-based conclusion, but they also show why training loss alone should not drive model selection:

- `PairRE` shows steady loss reduction in both benchmark modes and converts that optimization progress into the best test metrics.
- `ConvE` often drives training loss very low, especially on `UMLS`, but this does not produce the best test ranking metrics.
- This suggests that lower final training loss is not a sufficient reason to prefer `ConvE` here.

In other words, `PairRE` is not just easier to justify on accuracy; it is also the safer choice when the evaluation target is filtered ranking performance on held-out triples.

## Final Decision

Choose **PairRE** as the primary model for the case study.

Use the following rationale:

- it is the top performer on the benchmark's most important metric, `MRR`
- it wins on all complete-mode ranking metrics on average
- it remains computationally lightweight relative to `ConvE`
- it is more robust across datasets than `TransE` and `DistMult`

## Secondary Options

If a backup model must be named:

- choose `DistMult` when parameter count and speed matter more than peak accuracy
- choose `ConvE` only if there is a separate reason to prefer a higher-capacity convolutional model despite the weak efficiency tradeoff in these results
- do not choose `TransE` as the default benchmark winner, because its performance is too inconsistent and too weak on `Hits@1`
