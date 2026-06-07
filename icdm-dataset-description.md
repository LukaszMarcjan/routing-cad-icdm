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

**Download:** https://www.mvtec.com/company/research/datasets/mvtec-ad  
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

**Download:** https://github.com/amazon-science/spot-diff  
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
    --resize 256
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
    --resize 256
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
    --resize 256
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
    --strategies naive replay cumulative cl
```

The batch runner discovers dataset directories under `--datasets-root` using
the same recognized subdirectory names listed above.

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
