# ICDM Dataset Description

> **Anonymous review note:** author and maintainer metadata has been redacted
> in this artifact for double-blind review. Required third-party dataset license
> notices are retained, and full attribution will be restored in the
> de-anonymized public release.

This project evaluates continual visual anomaly detection on two established
industrial inspection benchmarks: MVTec AD and VisA. The repository provides
local readers, continual-learning scenario construction, optional step-schedule
grouping, and experiment runners for image-level and pixel-level evaluation.

The repository does **not** redistribute image files. Download the source
datasets from their original providers and point the local resolver to the
extracted folders as described below.

---

## Overview

The experiments treat each object or texture category as a continual-learning
concept. Training concepts contain normal images for the current category.
Test concepts contain normal and anomalous images, with binary image labels
and optional pixel-level ground-truth masks for anomalous samples.

| Dataset  | Categories | Train samples | Test samples | Normal test | Anomalous test | License         |
|----------|:----------:|:-------------:|:------------:|:-----------:|:--------------:|-----------------|
| MVTec AD | 15         | 3,629         | 1,725        | 467         | 1,258          | CC BY-NC-SA 4.0 |
| VisA     | 12         | 8,659         | 2,162        | 962         | 1,200          | CC BY 4.0       |

Sample counts refer to the original dataset releases. The local readers derive
the actual samples from the files present in the configured dataset root.

---

## Dataset Checklist Details

### Splits and validation

| Dataset | Training split | Validation/model-selection split | Test split |
|---------|----------------|----------------------------------|------------|
| MVTec AD | `train/good/` only; all training samples are nominal. | No separate official validation split is used for the main baselines. Assignment-based NSR temporarily holds out `--assignment-val-fraction` of current-step nominal images to choose the expert family, then retrains the selected family on the full step. | All images under `test/`; `test/good/` is normal and all other defect directories are anomalous. |
| VisA | Rows with `split=train` in `split_csv/1cls.csv`; the reader follows the official split file. | Same protocol as MVTec AD; no additional persistent validation set is created. | Rows with `split=test` in `split_csv/1cls.csv`, with labels and mask paths read from the CSV. |

The main continual-learning stream uses the official category order provided by
the reader. Step schedules such as `10-1x5` group consecutive training
categories into multi-category training steps but keep per-category test
concepts for evaluation.

### Excluded data

- No dataset category is excluded by default.
- Anomalous images are excluded from training by design, because the continual
  anomaly-detection setting assumes only nominal images are available at each
  training step.
- The optional `--categories`, `--max-train`, and `--max-test` arguments can
  restrict data for debugging or ablation runs; they should be reported
  explicitly if used for a paper result.
- Pixel metrics skip anomalous test images without masks by default
  (`--pixel-skip-missing-masks`). Normal test images without masks receive an
  all-zero mask.
- The repository ignores local dataset folders and does not redistribute images
  or masks.

### Preprocessing

- Images are loaded from the local benchmark root using the official folder or
  CSV layout.
- Images are converted to RGB by default and resized to the requested square
  resolution. The paper experiments resize images to `256x256` before feature
  extraction and scoring.
- Pixel values are converted to `float32` and scaled to `[0, 1]` when needed.
- Model-specific normalization, backbone selection, thresholds, and training
  epochs are defined by the model config files and can be overridden from the
  CLI or via `--config-file` / `--config-json`.
- Ground-truth masks are converted to binary arrays; bitmap masks are resized
  with nearest-neighbor interpolation for pixel-level metrics.

### Data collection

No new dataset was collected for this work. Both benchmarks are existing public
industrial anomaly-detection datasets; data collection, annotation, and quality
control are governed by their original dataset releases.

---

## Continual Protocol

### Standard per-category stream

By default, each category becomes one training concept and one corresponding
test concept. The resulting `ConceptsDataset` can be passed to pyCLAD
scenarios and strategies directly.

- **Training data:** normal training images only.
- **Test data:** normal and anomalous images with labels (`0` = normal,
  `1` = anomalous).
- **Image-level evaluation:** uses anomaly scores and binary image labels.
- **Pixel-level evaluation:** uses anomaly maps and ground-truth masks when
  masks are available for anomalous images.

The preset readers currently support:

- `mvtec` / `mvtec_ad`
- `visa`

### Step-scheduled stream

For ICDM-style continual experiments, consecutive categories can be grouped
into multi-category training steps using a CDAD-style schedule. Examples:

- `14-1` -> first step has 14 categories, second step has 1 category
- `10-5` -> two steps with 10 and 5 categories
- `3x5` -> five steps with 3 categories each
- `10-1x5` -> one 10-category step followed by five 1-category steps

Training concepts are grouped according to the schedule. Test concepts remain
per-category by default, so result matrices still expose per-category
performance after each training step.

---

## Benchmark Details

### MVTec AD

**Download:** https://www.mvtec.com/research-teaching/datasets/mvtec-ad  
**License:** CC BY-NC-SA 4.0

MVTec AD contains 15 categories of industrial objects and textures. Each
category provides normal training images and a test split containing both
normal and defective images. Pixel-level ground-truth masks are provided for
defective test samples.

Categories: `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`,
`leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`,
`wood`, `zipper`

Expected directory structure after download:

```text
mvtec_ad/
  bottle/
    train/good/
    test/good/
    test/broken_large/
    test/broken_small/
    test/contamination/
    ground_truth/broken_large/
    ground_truth/broken_small/
    ground_truth/contamination/
  cable/
  ...
  zipper/
```

The reader scans `train/good/` for normal training samples and each directory
under `test/` for test samples. Any `test/good/` image receives label `0`;
images in other test subdirectories receive label `1`. Masks are resolved from
the matching `ground_truth/<defect_type>/` directory.

---

### VisA

**Download:** https://registry.opendata.aws/visa/  
**License:** CC BY 4.0

VisA contains 12 categories spanning printed circuit boards and food products.
The official split file is used to identify train/test samples, image labels,
and mask paths.

Categories: `candle`, `capsules`, `cashew`, `chewinggum`, `fryum`,
`macaroni1`, `macaroni2`, `pcb1`, `pcb2`, `pcb3`, `pcb4`, `pipe_fryum`

Expected directory structure after download:

```text
VisA/
  split_csv/
    1cls.csv
  candle/
    Data/Images/Normal/
    Data/Images/Anomaly/
    Data/Masks/Anomaly/
  capsules/
  ...
  pipe_fryum/
```

The preset reader expects `split_csv/1cls.csv` and resolves image and mask
paths relative to the VisA root directory.

---

## Paper Scenarios

The paper evaluates both grouped and one-category-per-step continual streams:

| Dataset | Scenario | Step schedule |
|---------|----------|---------------|
| MVTec AD | One-by-one stress test | default per-category stream, equivalent to `1x15` |
| MVTec AD | Large deployment plus one update | `14-1` |
| MVTec AD | Balanced two-stage update | `10-5` |
| MVTec AD | Repeated small groups | `3x5` |
| MVTec AD | Large deployment plus five updates | `10-1x5` |
| VisA | One-by-one stress test | default per-category stream, equivalent to `1x12` |
| VisA | Large deployment plus one update | `11-1` |
| VisA | Balanced two-stage update | `8-4` |
| VisA | Large deployment plus four updates | `8-1x4` |

---

## Setup

### Step 1 - Download the source images

Download each dataset from its original source, extract the archive, and keep
the directory layout shown above. Image files are intentionally outside the
anonymous artifact.

### Step 2 - Register local dataset locations

Choose one of the following methods. The resolver checks them in this order.

**Option A - explicit root path:**

```bash
python examples/clvad/run_continual_visual_ad.py \
    --model patchcore \
    --strategy replay \
    --benchmark mvtec \
    --root /data/mvtec_ad \
    --resize 256
```

**Option B - per-benchmark environment variable:**

```bash
export PYCLAD_MVTEC_ROOT=/data/mvtec_ad
export PYCLAD_VISA_ROOT=/data/VisA
```

**Option C - shared dataset root with auto-detection:**

```bash
export PYCLAD_VISUAL_DATASETS_ROOT=/data
```

The resolver looks for known subdirectory names under the shared root:

| Benchmark | Recognized subdirectory names |
|-----------|-------------------------------|
| `mvtec`   | `mvtec_ad`, `mvtec`, `mvtec_anomaly_detection` |
| `visa`    | `visa`, `VisA` |

**Option D - registry JSON:**

Create or edit `src/pyclad/data/datasets/visual_datasets/registry.json`:

```json
{
    "mvtec": "/data/mvtec_ad",
    "visa": "/data/VisA"
}
```

An alternative registry path can be passed with `--registry-path` or provided
through `PYCLAD_VISUAL_DATASETS_FILE`.

---

## Loading Datasets in Python

```python
from pyclad.data.readers.visual_dataset_registry import (
    read_registered_visual_benchmark_dataset,
)

dataset = read_registered_visual_benchmark_dataset(
    benchmark="mvtec",
    root="/data/mvtec_ad",
    resize_to=(256, 256),
    color_mode="rgb",
)

print(dataset.name())
print([concept.name for concept in dataset.train_concepts()])
```

For step-scheduled training streams:

```python
from pyclad.data.readers.visual_step_schedule import (
    read_step_scheduled_visual_benchmark_dataset,
)

dataset = read_step_scheduled_visual_benchmark_dataset(
    benchmark="mvtec",
    root="/data/mvtec_ad",
    schedule="10-1x5",
    resize_to=(256, 256),
    color_mode="rgb",
)
```

Readers also support `data_mode="paths"` when experiments should keep image
paths instead of materializing image arrays immediately.

---

## Running Experiments

### Single-model continual baseline

```bash
python examples/clvad/run_continual_visual_ad.py \
    --model patchcore \
    --strategy replay \
    --benchmark mvtec \
    --root /data/mvtec_ad \
    --resize 256
```

Supported baseline strategies are `naive`, `cumulative`, `replay`, and `ste`.

### Image-level and pixel-level evaluation

```bash
python examples/clvad/run_continual_visual_ad_levels.py \
    --model patchcore \
    --strategy replay \
    --benchmark mvtec \
    --root /data/mvtec_ad \
    --eval-level both \
    --resize 256 \
    --backbone resnet18 \
    --pretrained \
    --pixel-threshold-mode train-quantile \
    --pixel-threshold-quantile 0.95
```

The runner supports `--eval-level image`, `--eval-level pixel`, and
`--eval-level both`. Pixel-level metrics require models that expose anomaly
maps and samples with available masks.

### Routing Normality Experts

```bash
# NSR (5): assignment-based Normality-Score Routing over all expert families
python examples/clvad/run_multi_expert_v2.py \
    --expert-types padim,patchcore,cfa,fastflow,stfpm \
    --expert-assignment best \
    --router min_score \
    --benchmark mvtec \
    --root /data/mvtec_ad \
    --step-schedule 10-1x5 \
    --eval-level both \
    --resize 256 \
    --backbone resnet18 \
    --pretrained \
    --pixel-threshold-mode train-quantile \
    --pixel-threshold-quantile 0.95
```

The paper's practical router is Normality-Score Routing (NSR), exposed as
`--router min_score`. It scores every active expert, z-score normalizes each
expert's anomaly scores using nominal calibration data, and selects the expert
with the lowest normalized anomaly score. Supported routers are `min_score`,
`max_conf`, and `oracle`; the `oracle` router uses ground-truth task identity
and should be interpreted as a diagnostic upper bound.

Assignment-based variants select the detector family per task before fitting
the final expert. The paper reports:

```bash
# NSR (3): assignment over PaDiM, PatchCore, and CFA
python examples/clvad/run_multi_expert_v2.py \
    --expert-types padim,patchcore,cfa \
    --expert-assignment best \
    --router min_score \
    --benchmark mvtec \
    --root /data/mvtec_ad \
    --step-schedule 10-1x5 \
    --eval-level both \
    --resize 256 \
    --backbone resnet18 \
    --pretrained \
    --pixel-threshold-mode train-quantile \
    --pixel-threshold-quantile 0.95
```

Round-robin NSR can be run by omitting `--expert-assignment best`. Cumulative
and dynamic-growth expert-bank variants are available through `--cumulative-cl`
and `--dynamic-growth --growth-threshold ...`.

### Batch sweeps

```bash
python examples/visual_models/run_all_experiments.py \
    --datasets-root /data \
    --benchmarks mvtec visa \
    --models padim patchcore cfa fastflow stfpm \
    --strategies naive replay cumulative \
    --eval-level both \
    --resize 256 \
    --backbone resnet18 \
    --pretrained \
    --pixel-threshold-mode train-quantile \
    --pixel-threshold-quantile 0.95
```

The batch runner discovers dataset directories under `--datasets-root` using
the same recognized subdirectory names listed above.
Run CL expert baselines for the supported model families separately with
`--models padim patchcore cfa --strategies cl`.

---

## Experimental Reporting Details

### Hyperparameters

The runners do not perform an implicit hyperparameter search. The configuration
used for a run is determined by model defaults plus explicit CLI overrides, and
the resolved configuration is written to the output JSON through each model's
`info()` provider.

Common experiment-level defaults:

| Argument | Default | Notes |
|----------|---------|-------|
| `--n-runs` | `1` | Number of seeded repetitions. |
| `--master-seed` | `42` | Used to generate per-run seeds. |
| `--replay-buffer-fraction` | `0.2` | Replay budget as a fraction of average concept size. |
| `--resize` | model default unless set | Paper commands in this artifact use `--resize 256`. |
| `--threshold-quantile` | model default, usually `0.99` | Used when no fixed image-score threshold is supplied. |
| `--pixel-threshold` | `0.5` | Threshold for pixel F1, Dice, and IoU in fixed-threshold mode. |
| `--pixel-threshold-mode` | `fixed` | Alternative: `train-quantile`. |
| `--pixel-threshold-quantile` | `0.995` | Used only in `train-quantile` pixel-threshold mode. |
| `--assignment-val-fraction` | `0.2` | Held-out normal fraction for assignment-based NSR model-family selection. |
| `--growth-threshold` | `0.0` | Dynamic-growth update-vs-new-expert threshold. |

The paper commands override several defaults: they use `--resize 256`,
ImageNet-pretrained ResNet-18 feature extractors (`--backbone resnet18
--pretrained`), replay buffer fraction `0.2`, and train-normal quantile `0.95`
for pixel-thresholded quantities. Dynamic-growth experiments vary
`--growth-threshold` over `-0.5`, `0.0`, `0.5`, and `1.0`.

Model-specific defaults are defined in:

- `src/pyclad/models/padim/config.py`
- `src/pyclad/models/patchcore/config.py`
- `src/pyclad/models/cfa/config.py`
- `src/pyclad/models/stfpm/config.py`

The full hyperparameter configuration for a paper result should include the
exact command, `--n-runs`, `--master-seed`, and any CLI/config-file overrides.
The main paper evaluates 126 method-scenario configurations and reports one
aggregate result per configuration.

### Metrics and summary statistics

Image-level metrics use binary image labels and anomaly scores:

- ROC-AUC
- Average Precision
- F1-Score

Pixel-level metrics use ground-truth masks and anomaly maps:

- Pixel ROC-AUC
- Pixel Average Precision
- Pixel AUPRO
- Pixel F1-Score
- Pixel IoU
- Pixel Dice Score

Continual metrics are computed from the concept-by-concept result matrix:

- Continual Average
- Diagonal Average
- Backward Transfer
- Forward Transfer
- Rectangular variants for step-scheduled streams

In the paper tables, ROC-AUC/AP cells report final-step means and, when shown,
standard deviations across final-step category values. These deviations are not
multi-seed error bars. For additional multi-seed experiments, report the seed
list and aggregate those runs separately from the category-level variation.

### Runtime, memory, and infrastructure

Each runner writes:

- `run_status.elapsed_seconds` for wall-clock runtime,
- `time_evaluation_callback.train_time_total` and `eval_time_total`,
- per-concept timing in `time_evaluation_callback.time_by_concept`,
- `memory_usage_callback.current_memory_usage` and `peak_memory_usage`.

The selected compute device is controlled by `--device`; if omitted, the code
uses CUDA when available, then Apple MPS when available, otherwise CPU. The
timed paper executions were run on GPU compute nodes with 2 NVIDIA L40 and 2
NVIDIA H100 accelerators. Energy usage is not estimated by default.

Runtime summary from the paper, in minutes:

| Dataset | Setting | Train | Eval | Total |
|---------|---------|------:|-----:|------:|
| MVTec AD | Single-model | 7.9 +/- 8.8 | 17.8 +/- 18.1 | 25.7 +/- 22.7 |
| MVTec AD | Multi-expert | 21.0 +/- 24.0 | 73.3 +/- 88.4 | 94.4 +/- 102.6 |
| VisA | Single-model | 30.3 +/- 46.9 | 34.2 +/- 40.9 | 64.4 +/- 77.7 |
| VisA | Multi-expert | 54.8 +/- 64.8 | 125.0 +/- 124.3 | 179.8 +/- 145.7 |
| All | Single-model | 17.8 +/- 33.8 | 25.1 +/- 31.5 | 42.9 +/- 57.8 |
| All | Multi-expert | 36.2 +/- 49.8 | 96.5 +/- 109.0 | 132.6 +/- 130.9 |

### Pretrained and trained models

The artifact does not redistribute paper-specific trained checkpoints. The
paper experiments train anomaly experts from the public dataset splits and use
public ImageNet-pretrained Torch/Torchvision feature extractors.

---

## Local Sample Schema

Internally, each indexed image is represented as a `VisualSample`:

| Field | Description |
|-------|-------------|
| `category` | Object or texture category used as a continual concept |
| `split` | `train` or `test` |
| `image_path` | Absolute path to the source image |
| `image_label` | `0` for normal, `1` for anomalous |
| `mask_path` | Optional path to the pixel-level ground-truth mask |
| `defect_type` | Defect subtype for anomalous samples, when available |

The materialized pyCLAD object is a `ConceptsDataset` with one or more training
concepts and per-category test concepts.
