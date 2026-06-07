# Routing Normality Experts for Continual Industrial Anomaly Detection

This repository contains the code for the ICDM submission on routing normality
experts for continual industrial anomaly detection. The paper studies inspection
settings where new products, components, or acquisition conditions arrive over
time. At each training step, only nominal images from the current task are
available, while evaluation must remain reliable on current and previously seen
categories.

The central idea is to maintain a bank of anomaly-detection experts and route
each test image to the expert whose learned normality model best matches the
sample. Instead of forcing one detector to absorb all task shifts, the system
combines frozen task experts, score calibration, and routing.

We leverage the pyCLAD library for continual scenario orchestration. Install it
directly:

```bash
pip install -e .
```

#### Optional dependencies

The core pyCLAD installation covers tabular and time-series anomaly detection.
For visual models, install the vision stack separately:

```bash
pip install torch torchvision pytorch-lightning
```

### Getting started

pyCLAD is built upon a few core concepts:

- **Scenario**: a continual scenario defines the data stream so that it reflects different real-life conditions and what
  are the challenges faced by continual strategy.
- **Strategy**: a strategy is a way to manage model updates. Continual strategy is responsible for how, when, and with
  which data models should be updated. Its aim is to introduce knowledge retention while keeping the ability to adapt.
- **Model**: a model is a machine learning model used for anomaly detection. Models are often leveraged by continual
  strategies that add an additional layer of managing model updates.
- **Dataset**: a dataset is a collection of data used for training and evaluation of the model.
- **Metrics**: a metric is a way to evaluate the performance of the model.
- **Callbacks**: a callback is a function that is called at specific points during the scenario. Callbacks are
  useful for monitoring the process, calculating metrics, and more.

## Visual Anomaly Detection

We implement visual anomaly detection methods in pyCLAD: image-based normality
experts, image-level and pixel-level metrics, continual-learning strategies for
vision, routing-based multi-expert methods, and local readers for visual
benchmarks.

### Normality Expert Families

| Model | CL variant | Description |
|-------|:----------:|-------------|
| PatchCore | PatchCore-CL | Memory-bank approach using deep features from a pretrained backbone |
| PaDiM | PaDiM-CL | Patch-level Gaussian distribution modelling |
| CFA | CFA-CL | Coupled hypersphere-based feature adaptation |
| FastFlow | — | Normalizing flow on deep features |
| STFPM | — | Student-teacher feature pyramid matching |

CL variants (e.g. PatchCore-CL, PaDiM-CL) extend the base model with memory management for continual learning and can be used with vision-specific continual strategies.

### Method Variants

The paper compares non-routed baselines with routed normality-expert variants:

| Variant | CLI mapping | Description |
|---------|-------------|-------------|
| Single-model baselines | `run_continual_visual_ad.py --strategy naive/cumulative/replay/ste` | Standard continual baselines without test-time expert routing |
| CL expert baselines | `--strategy cl` in batch sweeps for supported models | Native CL variants for PatchCore, PaDiM, and CFA |
| NSR | `run_multi_expert_v2.py --router min_score` | Normality-Score Routing: score every active expert, z-score normalize each expert's anomaly scores, and select the expert with the lowest normalized score |
| Oracle routing | `--router oracle` | Diagnostic upper bound that routes using ground-truth task-to-expert information |
| Cumulative expert banking | `--cumulative-cl` | Builds a new CL-aware expert at each step using the task history seen so far |
| Dynamic growth | `--dynamic-growth --growth-threshold ...` | Updates an existing expert when the new task looks sufficiently normal under that expert, otherwise grows the bank |
| Assignment-based NSR (3) | `--expert-assignment best --expert-types padim,patchcore,cfa` | Chooses the detector family per task from PaDiM, PatchCore, and CFA |
| Assignment-based NSR (5) | `--expert-assignment best --expert-types padim,patchcore,cfa,fastflow,stfpm` | Chooses the detector family per task from all five expert families |

Replay buffer budgeting supports three modes: `fixed` (fixed total size), `avg-concept-fraction` (fraction of average concept size), and `per-concept-budget` (fixed budget multiplied by number of concepts seen).

### Metrics

**Image-level:**

| Metric | Class |
|--------|-------|
| ROC-AUC | `RocAuc` |
| Average Precision | `AveragePrecision` |
| F1-Score | `F1Score` |

**Pixel-level** (require ground-truth anomaly masks):

| Metric | Class |
|--------|-------|
| Pixel ROC-AUC | `PixelRocAuc` |
| Pixel Average Precision | `PixelAveragePrecision` |
| Pixel AUPRO | `PixelAUPRO` |
| Pixel F1-Score | `PixelF1Score` |
| Pixel IoU | `PixelIoU` |
| Pixel Dice Score | `PixelDiceScore` |

**Continual metrics** (derived from the concept-level result matrix):

| Metric | Class | Description |
|--------|-------|-------------|
| Continual Average | `ContinualAverage` | Mean over all evaluated (concept, training-step) pairs |
| Diagonal Average | `DiagonalAverage` | Mean of diagonal entries — performance right after training each concept |
| Backward Transfer | `BackwardTransfer` | Change in performance on previously learned concepts |
| Forward Transfer | `ForwardTransfer` | Change in performance on future concepts before training on them |

Pixel-level evaluation is handled by `VisualPixelConceptMetricCallback` and its
rectangular step-schedule counterpart. These callbacks resolve ground-truth
masks from the registered dataset and support configurable thresholding modes
for thresholded pixel metrics.

## Routing Normality Experts

At training step `t`, the scenario provides a task `T_t`: either one category
or a group of categories from a step schedule. The multi-expert runner creates
or updates a normality expert for that step. Base expert banking trains a fresh
expert on the current nominal images and freezes it; cumulative and dynamic
growth variants use CL-aware experts when the experiment calls for updating or
reusing an expert.

The default expert pool is heterogeneous: PaDiM, PatchCore, CFA, FastFlow, and
STFPM encode different assumptions about normality. In round-robin mode, tasks
cycle through the supplied expert types. In assignment mode, each task trains
temporary candidate experts on a fit split, evaluates them on held-out normals
and previously seen task samples, then retrains the winning detector family on
the full current task.

At inference, every active expert scores each sample. Because different detector
families produce incompatible raw score scales, the router calibrates each
expert on nominal scores from its training step and uses z-score normalization
before routing.

- `min_score` implements the paper's Normality-Score Routing (NSR): choose the
  expert with the lowest calibrated anomaly score.
- `oracle` uses the ground-truth task-to-expert mapping and is intended only as
  a diagnostic comparison.
- `max_conf` is a naming alias for the same "lowest anomaly score means highest
  normality confidence" rule.

The implementation lives in `pyclad.models.vision.heterogeneous_expert_ensemble`
and `pyclad.models.vision.routers`, orchestrated by `MultiExpertStrategy`.

```bash
# NSR (5): assignment-based Normality-Score Routing over five expert families
python examples/clvad/run_multi_expert_v2.py \
    --expert-types padim,patchcore,cfa,fastflow,stfpm \
    --expert-assignment best \
    --router min_score \
    --benchmark mvtec --root /path/to/mvtec_ad \
    --step-schedule 10-1x5 --eval-level both \
    --resize 256
```

## Benchmarks

| Dataset  | Categories | License         |
|----------|:----------:|-----------------|
| MVTec AD | 15         | CC BY-NC-SA 4.0 |
| VisA     | 12         | CC BY 4.0       |

Datasets are read from local folders through the visual dataset registry. Each
category is treated as a continual-learning concept, and optional step
schedules can group consecutive categories into multi-category training steps.

For the full dataset and protocol description used by the ICDM artifact, see
[ICDM Dataset Description](./icdm-dataset-description.md). The repository does
not redistribute benchmark images; download MVTec AD and VisA from their
original providers and point the runners to the extracted folders.

### Quick start

Load a benchmark and run a single-model naive experiment:

```bash
python examples/clvad/run_continual_visual_ad.py \
    --model patchcore \
    --strategy naive \
    --benchmark mvtec \
    --root /path/to/mvtec_ad \
    --resize 256
```

### Smoke checks

For a fresh checkout without installing the package, expose the `src` layout
before invoking example runners:

```bash
PYTHONPATH=src python3 examples/clvad/run_continual_visual_ad.py --help
PYTHONPATH=src python3 examples/clvad/run_continual_visual_ad_levels.py --help
PYTHONPATH=src python3 examples/clvad/run_multi_expert_v2.py --help
```

To check that the documented CLI options still parse without loading datasets
or training models:

```bash
python3 -m pytest tests/examples/test_smoke_runners.py tests/output/test_visual_runner_checkpointing.py -q
```

Or drive a full sweep over models, strategies and benchmarks with the batch
orchestrator:

```bash
python examples/visual_models/run_all_experiments.py \
    --benchmarks mvtec visa --models padim patchcore --strategies naive cl
```
