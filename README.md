# CSE-754: Evaluating Geometric Naming in DN-CBM

This course project extends **Discover-then-Name (DN-CBM)** to examine whether names assigned to sparse autoencoder (SAE) features are supported by their behavior on images. It uses SUN attribute annotations with a frozen CLIP encoder and SAE.

## Research questions

- Does decoder–text cosine similarity predict how well an individual feature detects its assigned attribute?
- Can a supervised combination of features sharing a name recover more attribute information than a single feature or maximum aggregation?

## Approach

Two naming settings are evaluated: original DN-CBM names that exactly match SUN attributes, and nearest-name assignment within the restricted 102-attribute SUN vocabulary.

Individual features are evaluated using AUROC and average precision. Group comparisons use a validation-selected single feature, maximum aggregation, and an L1-regularized logistic readout. Additional analyses include matched random groups, annotation sensitivity, attribute co-occurrence, and a separate comparison with direct CLIP scores.

## Main results

For SUN-restricted naming, the individual-feature cosine–AUROC Spearman correlation is **0.021**.

| Method | Macro AUROC across 93 attributes |
| --- | ---: |
| Validation-selected single feature | 0.682 |
| Maximum aggregation | 0.700 |
| Sparse logistic readout | **0.801** |

Sparse readouts exceed readouts trained on random groups matched for size and activation frequency by **0.091 AUROC** on average. Gains are less consistent for groups based on original DN-CBM names. In a separate evaluation, filtered groups using maximum aggregation underperform direct CLIP scores.

**Interpretation:** Same-name groups can contain useful attribute information even when naming similarity has little association with individual detection quality. Successful supervised prediction does not establish that every feature in a group has the assigned meaning.

## Repository guide

- [scripts/project/](scripts/project/): project experiments, from data preparation and naming to evaluation and controls.
- [experiment-results/](experiment-results/): saved experiment summaries and result tables.
- [dncbm/](dncbm/): DN-CBM implementation.
- [sparse_autoencoder/](sparse_autoencoder/): SAE implementation.

Reproducing the experiments requires the SUN images and annotations, a compatible pretrained SAE checkpoint, and the feature caches expected by the scripts. Check script paths and dependencies before execution.

## Scope and attribution

This project evaluates one frozen representation on SUN. It does not introduce a new naming algorithm or demonstrate causal concept control.

The work builds on Rao et al., *Discover-then-Name: Task-Agnostic Concept Bottlenecks via Automated Concept Discovery* (ECCV 2024). See the [original paper](https://arxiv.org/abs/2407.14499) for the underlying method. Original code retains its existing attribution and license terms.

