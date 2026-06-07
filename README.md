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

### Reproducibility Checklist

| Checklist topic | Artifact coverage |
|-----------------|-------------------|
| Dependencies | `pyproject.toml` declares core dependencies; `requirements.txt` lists the full runtime stack used by the visual experiments. |
| Datasets and splits | `icdm-dataset-description.md` gives dataset statistics, download links, official train/test split handling, excluded data, and preprocessing. |
| Training code | `examples/clvad/run_continual_visual_ad.py`, `examples/clvad/run_continual_visual_ad_levels.py`, and `examples/clvad/run_multi_expert_v2.py`. |
| Evaluation code | Image, pixel, continual, timing, memory, and routing metrics are implemented under `src/pyclad/metrics/` and `src/pyclad/callbacks/evaluation/`. |
| Pretrained models | No paper-specific trained checkpoints are redistributed. Paper experiments train anomaly experts from public dataset splits and use public ImageNet-pretrained Torch/Torchvision feature extractors. |
| Runs and seeds | Runners accept `--n-runs` and `--master-seed`; the paper reports one aggregate per method-scenario configuration, and table standard deviations are across final-step category values rather than random seeds. |
| Hyperparameters | CLI overrides are listed in `--help`; model defaults live in `src/pyclad/models/*/config.py`; raw overrides can be supplied with `--config-file` or `--config-json`. |
| Runtime and memory | Output JSON files include `run_status.elapsed_seconds`, `time_evaluation_callback`, and `memory_usage_callback`. |

The following commands produce the JSON result files used to assemble the paper
tables. Replace `/data` with a directory containing `mvtec_ad` and `VisA`.
The paper uses `--resize 256`, ImageNet-pretrained ResNet-18 feature
extractors, replay buffer fraction `0.2`, and train-normal pixel threshold
quantile `0.95`.

| Result block | Command |
|--------------|---------|
| Single-model baselines | `python3 examples/visual_models/run_all_experiments.py --datasets-root /data --benchmarks mvtec visa --models padim patchcore cfa fastflow stfpm --strategies naive replay cumulative --eval-level both --resize 256 --backbone resnet18 --pretrained --pixel-threshold-mode train-quantile --pixel-threshold-quantile 0.95 --n-runs 1 --master-seed 123` |
| CL expert baselines | `python3 examples/visual_models/run_all_experiments.py --datasets-root /data --benchmarks mvtec visa --models padim patchcore cfa --strategies cl --eval-level both --resize 256 --backbone resnet18 --pretrained --pixel-threshold-mode train-quantile --pixel-threshold-quantile 0.95 --n-runs 1 --master-seed 123` |
| NSR (3) | `python3 examples/clvad/run_multi_expert_v2.py --expert-types padim,patchcore,cfa --expert-assignment best --router min_score --benchmark mvtec --root /data/mvtec_ad --step-schedule 10-1x5 --eval-level both --resize 256 --backbone resnet18 --pretrained --pixel-threshold-mode train-quantile --pixel-threshold-quantile 0.95 --n-runs 1 --master-seed 123` |
| NSR (5) | `python3 examples/clvad/run_multi_expert_v2.py --expert-types padim,patchcore,cfa,fastflow,stfpm --expert-assignment best --router min_score --benchmark mvtec --root /data/mvtec_ad --step-schedule 10-1x5 --eval-level both --resize 256 --backbone resnet18 --pretrained --pixel-threshold-mode train-quantile --pixel-threshold-quantile 0.95 --n-runs 1 --master-seed 123` |

The paper evaluates 126 method-scenario configurations. Runtime measurements
were collected on GPU compute nodes with 2 NVIDIA L40 and 2 NVIDIA H100
accelerators; see `icdm-dataset-description.md` for the runtime summary table.

Or drive a full sweep over models, strategies and benchmarks with the batch
orchestrator:

```bash
python examples/visual_models/run_all_experiments.py \
    --benchmarks mvtec visa --models padim patchcore --strategies naive cl
```

## Results

The tables below reproduce the main paper results, generated by the commands in
the previous section with a single master seed (`--master-seed 123`, which
derives the per-run seed `33158374`). Values are percentages. `ROC`/`AP` are
final-step means; the `±` term is the standard deviation **across final-step
category values** (not multi-seed error bars). `BWT`/`FWT` are schedule-aware
backward/forward transfer from the ROC-AUC result matrix (higher `BWT` = less
forgetting). `RankSum` sums per-column ranks over available numeric entries
(lower is better). `NCL` = non-continual baseline, `CL`/`Replay` = continual
variant, `NSR` = Normality-Score Routing (non-assignment), `NSR (3)`/`NSR (5)` =
assignment-based NSR over 3/5 expert families, `Oracle` = diagnostic upper bound.
`--` marks entries unavailable in the source tables. **Bold** marks the best
entry per column.

### Image-level results — MVTec AD

| Method | 1x15 ROC | 1x15 AP | 1x15 BWT | 1x15 FWT | 14-1 ROC | 14-1 AP | 14-1 BWT | 14-1 FWT | 10-5 ROC | 10-5 AP | 10-5 BWT | 10-5 FWT | 3x5 ROC | 3x5 AP | 3x5 BWT | 3x5 FWT | 10-1x5 ROC | 10-1x5 AP | 10-1x5 BWT | 10-1x5 FWT | RankSum↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PaDiM NCL | 55.2±13.6 | 77.2±13.3 | -37.2 | 54.2 | 55.2±13.6 | 77.2±13.3 | -26.9 | 57.0 | 68.5±21.4 | 86.5±10.9 | -20.7 | 61.0 | 63.6±18.9 | 83.7±10.1 | -28.3 | 58.6 | 55.2±13.6 | 77.2±13.3 | -29.7 | 60.5 | 208.0 |
| PaDiM CL | **89.8±8.8** | 95.4±4.2 | **-0.0** | 55.2 | 80.3±14.0 | 91.3±7.0 | -0.0 | 57.1 | 82.3±14.5 | 92.4±6.8 | -0.0 | 60.9 | 86.3±11.7 | 94.4±4.6 | -0.0 | 58.7 | 82.9±14.9 | 92.7±6.9 | -0.0 | 61.1 | 80.0 |
| PatchCore NCL | 63.1±20.8 | 81.9±16.0 | -35.3 | 57.8 | 63.1±20.8 | 81.9±16.0 | -34.6 | 41.2 | 70.8±24.0 | 87.4±11.3 | -37.1 | **66.1** | 64.7±22.5 | 84.6±10.7 | -38.4 | 65.3 | 63.1±20.8 | 81.9±16.0 | -34.9 | 61.2 | 183.0 |
| PatchCore CL | 67.0±23.6 | 84.0±14.5 | -31.2 | 59.8 | 69.8±17.7 | 87.2±8.1 | -27.6 | 37.6 | 76.9±19.8 | 91.1±7.6 | -28.1 | 65.9 | 72.8±18.5 | 89.4±6.9 | -28.1 | **65.7** | 66.7±19.9 | 83.7±13.5 | -31.1 | 60.9 | 149.0 |
| CFA NCL | 58.3±16.6 | 79.9±12.4 | -37.2 | 50.4 | 56.6±17.2 | 77.6±14.1 | -32.1 | 38.9 | 69.3±17.9 | 86.6±8.2 | -23.8 | 64.2 | 62.7±19.3 | 83.7±9.9 | -27.4 | 51.5 | 58.0±17.2 | 78.9±12.3 | -31.9 | 55.5 | 219.0 |
| CFA CL | 89.6±12.9 | **95.7±5.8** | -2.1 | 50.8 | 84.9±13.4 | 94.0±5.6 | **16.1** | 34.0 | 80.0±13.9 | 91.6±7.6 | **12.8** | 62.6 | 81.1±13.3 | 91.9±7.1 | -4.7 | 52.7 | 83.9±14.8 | 93.4±6.8 | -3.7 | 63.4 | 99.0 |
| FastFlow NCL | 47.2±13.0 | 72.0±13.0 | -41.0 | 48.2 | 51.9±15.0 | 74.9±13.2 | -17.1 | 42.6 | 56.8±17.9 | 78.4±12.8 | -28.5 | 51.5 | 54.3±18.4 | 77.0±9.9 | -31.4 | 45.0 | 49.9±14.1 | 73.8±12.0 | -32.1 | 49.6 | 264.0 |
| FastFlow Replay | 65.9±9.6 | 82.7±9.8 | -16.6 | 48.3 | 66.0±14.1 | 83.2±10.1 | 2.9 | **67.9** | 71.2±10.7 | 86.2±8.1 | -0.9 | 54.8 | 67.5±12.7 | 83.2±10.7 | -10.2 | 46.4 | 61.0±13.6 | 79.9±11.4 | -18.2 | 53.6 | 180.0 |
| STFPM NCL | 67.7±21.8 | 83.4±16.7 | -26.0 | 58.5 | 60.3±17.8 | 79.1±15.8 | -32.1 | 55.0 | 71.9±22.4 | 87.0±12.5 | -30.3 | 63.1 | 68.6±18.3 | 85.5±9.9 | -28.4 | 61.9 | 63.2±18.6 | 82.1±13.3 | -30.8 | 67.6 | 161.0 |
| STFPM Replay | 75.7±20.7 | 88.3±11.1 | -17.3 | **63.1** | 77.3±19.1 | 87.4±15.5 | -14.5 | 63.5 | 79.8±20.0 | 90.7±9.7 | -18.0 | 66.1 | 68.3±22.9 | 83.4±14.0 | -27.7 | 64.0 | 74.1±19.0 | 87.7±11.6 | -18.3 | **70.3** | 117.0 |
| NSR | 51.5±24.3 | 77.7±12.7 | -28.1 | 47.8 | 79.2±14.7 | 90.8±7.1 | -0.0 | 57.0 | 82.6±15.2 | 92.4±7.0 | -0.0 | 61.0 | 75.2±23.3 | 87.1±15.1 | -4.7 | 57.0 | 80.0±19.7 | 91.3±12.3 | -1.3 | 61.0 | 133.0 |
| NSR (5) | 89.8±9.7 | 95.1±4.3 | -0.0 | 54.5 | **85.3±13.9** | **94.6±4.8** | -0.0 | 37.1 | 87.7±13.9 | 95.8±4.6 | -0.0 | 56.9 | 90.2±9.0 | 96.6±2.8 | **-0.0** | 53.6 | 85.8±13.1 | 94.8±4.3 | **0.3** | 58.3 | 87.5 |
| NSR (3) | 89.8±9.7 | 95.1±4.3 | -0.0 | 53.8 | 84.8±13.3 | 94.2±5.0 | -0.0 | 40.5 | **88.8±12.4** | **96.1±4.2** | -0.0 | 62.6 | **90.4±8.7** | **96.8±2.5** | **-0.0** | 54.0 | **86.9±11.6** | **95.1±4.0** | 0.1 | 64.1 | **69.5** |
| Oracle | 55.7±19.1 | 77.1±15.4 | **-0.0** | 52.6 | 78.8±15.3 | 90.7±7.2 | -0.0 | 57.1 | 72.3±19.1 | 85.8±16.2 | -0.0 | 60.9 | 63.0±20.6 | 81.9±13.8 | **-0.0** | 57.2 | 72.3±19.1 | 85.8±16.2 | -0.0 | 60.9 | 150.0 |

### Image-level results — VisA

| Method | 1x12 ROC | 1x12 AP | 1x12 BWT | 1x12 FWT | 11-1 ROC | 11-1 AP | 11-1 BWT | 11-1 FWT | 8-4 ROC | 8-4 AP | 8-4 BWT | 8-4 FWT | 8-1x4 ROC | 8-1x4 AP | 8-1x4 BWT | 8-1x4 FWT | RankSum↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PaDiM NCL | 53.6±15.5 | 60.4±13.4 | -29.8 | 49.2 | 53.6±15.5 | 60.4±13.4 | -23.3 | 66.2 | 60.6±16.5 | 66.7±13.6 | -23.2 | 53.1 | 53.6±15.5 | 60.4±13.4 | -25.8 | 52.2 | 174.0 |
| PaDiM CL | 80.7±11.1 | 81.3±11.5 | -0.2 | 45.1 | 75.0±11.0 | 76.7±12.1 | -0.0 | 66.2 | **76.1±11.2** | 77.8±12.1 | -0.0 | 53.1 | 77.1±11.0 | 78.8±11.6 | **-0.0** | 53.1 | 69.5 |
| PatchCore NCL | 56.3±19.3 | 63.8±18.2 | -37.0 | 54.5 | 56.3±19.3 | 63.8±18.2 | -36.9 | **90.8** | 69.8±23.5 | 75.2±20.2 | -30.0 | 59.7 | 56.3±19.3 | 63.8±18.2 | -36.5 | 62.1 | 135.0 |
| PatchCore CL | 58.9±20.2 | 65.6±19.4 | -33.9 | 54.8 | 56.1±19.5 | 63.3±18.3 | -37.2 | 89.9 | 72.3±22.7 | 77.2±19.7 | -26.5 | **60.0** | 54.6±20.0 | 62.5±18.4 | -38.6 | **62.8** | 127.0 |
| CFA NCL | 56.4±15.7 | 63.7±16.7 | -29.6 | 49.2 | 60.8±15.8 | 66.8±18.3 | -13.9 | 41.5 | 62.8±16.6 | 70.7±16.4 | -15.1 | 50.2 | 58.9±16.7 | 64.6±18.8 | -18.9 | 50.0 | 158.0 |
| CFA CL | **83.4±12.1** | **86.3±11.6** | -1.6 | 49.1 | 74.2±10.9 | 78.9±12.6 | **3.9** | 56.5 | 71.7±12.0 | 77.4±13.0 | **5.5** | 53.5 | 75.2±12.9 | 80.9±13.3 | -2.7 | 52.4 | **66.0** |
| FastFlow NCL | 53.1±10.7 | 60.4±13.0 | -25.7 | 48.1 | 54.4±8.0 | 61.8±11.5 | -2.5 | 52.1 | 55.2±13.1 | 63.0±9.8 | -5.1 | 54.5 | 52.7±9.7 | 59.0±10.3 | -12.2 | 51.5 | 167.0 |
| FastFlow Replay | 60.2±12.2 | 65.2±11.7 | -16.4 | 48.0 | 55.7±6.1 | 62.4±8.9 | -0.1 | 50.7 | 52.3±7.6 | 60.2±9.3 | -16.1 | 48.6 | 53.8±10.1 | 61.8±9.8 | -17.8 | 46.4 | 169.0 |
| STFPM NCL | 60.8±14.6 | 65.0±15.7 | -29.4 | 54.8 | 50.5±8.0 | 58.7±10.5 | -10.8 | 51.8 | 70.4±17.7 | 76.3±14.1 | -16.3 | 46.8 | 60.7±13.0 | 65.7±14.6 | -28.0 | 55.1 | 140.0 |
| STFPM Replay | 69.5±15.5 | 75.1±13.4 | -18.1 | **55.2** | 67.4±14.9 | 72.6±14.6 | -13.3 | 60.0 | 76.1±16.0 | **80.2±14.8** | -11.6 | 53.9 | 78.6±11.8 | **82.0±12.3** | -6.0 | 51.8 | 87.0 |
| NSR | 71.2±14.8 | 77.3±14.6 | -7.9 | 47.0 | 72.8±9.8 | 75.5±10.8 | -0.0 | 66.2 | 73.6±10.2 | 74.7±13.1 | -0.0 | 53.1 | 78.8±11.7 | 79.9±11.7 | -0.1 | 53.1 | 86.0 |
| NSR (5) | 82.4±10.6 | 83.6±11.1 | -0.7 | 52.1 | 72.6±10.8 | 77.6±12.8 | -0.0 | 35.8 | 60.2±23.7 | 71.9±15.5 | -0.0 | 45.5 | 75.9±14.0 | 79.3±13.7 | **-0.0** | 47.7 | 102.5 |
| NSR (3) | 82.3±10.2 | 83.6±11.1 | -0.7 | 50.6 | **76.1±10.8** | **80.2±12.4** | -0.0 | 45.0 | 60.6±26.4 | 72.4±16.5 | -0.0 | 41.9 | **78.9±10.2** | 81.6±10.9 | **-0.0** | 44.8 | 89.5 |
| Oracle | 49.4±12.5 | 58.5±8.2 | **-0.0** | 46.5 | 72.8±9.8 | 75.5±10.8 | -0.0 | 66.2 | 67.9±16.3 | 71.5±16.4 | -0.0 | 53.1 | 67.9±16.3 | 71.5±16.4 | **-0.0** | 53.1 | 109.5 |

### Pixel-level results — MVTec AD

| Method | 1x15 ROC | 1x15 AP | 1x15 BWT | 1x15 FWT | 14-1 ROC | 14-1 AP | 14-1 BWT | 14-1 FWT | 10-5 ROC | 10-5 AP | 10-5 BWT | 10-5 FWT | 3x5 ROC | 3x5 AP | 3x5 BWT | 3x5 FWT | 10-1x5 ROC | 10-1x5 AP | 10-1x5 BWT | 10-1x5 FWT | RankSum↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PaDiM NCL | 68.2±18.6 | 8.6±9.7 | -29.3 | 65.1 | 68.2±18.6 | 8.6±9.7 | -20.7 | 72.0 | 78.6±19.5 | 19.3±16.8 | -16.5 | 59.9 | 74.1±20.2 | 15.5±16.3 | -23.0 | 68.2 | 68.2±18.6 | 8.6±9.7 | -23.9 | 58.4 | 234.0 |
| PaDiM CL | 89.4±8.8 | 23.6±17.2 | -3.2 | 64.4 | 87.2±13.1 | 27.5±16.7 | -0.0 | 72.0 | 87.3±13.3 | 26.6±17.1 | -0.3 | 59.9 | 89.0±10.2 | 25.9±18.4 | -1.4 | 68.6 | 87.7±12.9 | 25.4±17.6 | -0.2 | 60.1 | 94.0 |
| PatchCore NCL | 76.1±18.8 | 17.8±17.6 | -22.2 | 74.2 | 76.1±18.8 | 17.8±17.6 | -21.7 | **84.3** | 82.8±17.4 | 24.4±19.7 | -20.4 | 73.9 | 79.2±18.3 | 20.2±18.6 | -21.5 | 77.5 | 76.1±18.8 | 17.8±17.6 | -21.8 | 71.4 | 152.0 |
| PatchCore CL | 82.7±12.6 | 20.9±18.8 | -15.2 | **76.3** | 82.4±11.3 | 21.5±17.4 | -15.0 | 84.2 | 89.6±7.1 | 29.4±17.7 | -10.3 | 74.1 | 83.4±13.5 | 24.7±15.9 | -16.2 | **77.5** | 82.3±13.0 | 21.5±17.7 | -15.2 | 71.4 | 106.0 |
| CFA NCL | 71.0±18.8 | 12.5±12.4 | -24.9 | 70.6 | 74.6±9.3 | 10.5±8.4 | -17.4 | 65.8 | 80.6±12.3 | 20.6±16.3 | -13.3 | 62.3 | 76.7±13.6 | 17.6±15.8 | -13.8 | 70.3 | 73.4±11.3 | 9.5±7.2 | -19.6 | 63.8 | 193.0 |
| CFA CL | **92.5±4.9** | **33.6±21.4** | -1.9 | 71.3 | **92.6±5.3** | **34.9±18.9** | **3.8** | 79.0 | 90.7±5.3 | 31.0±16.5 | **4.2** | 67.7 | **89.5±6.4** | **29.8±15.5** | -1.7 | 72.3 | **88.2±6.5** | **26.6±18.1** | -5.1 | 68.1 | **48.0** |
| FastFlow NCL | 51.1±14.6 | 3.9±4.5 | -43.6 | 40.4 | 58.1±15.7 | 5.6±9.0 | -22.4 | 50.6 | 70.8±14.7 | 10.7±10.3 | -24.8 | 56.1 | 65.9±19.6 | 10.7±15.1 | -32.1 | 36.2 | 56.2±17.4 | 5.1±7.6 | -35.2 | 51.9 | 274.0 |
| FastFlow Replay | 74.0±13.5 | 11.1±8.6 | -17.6 | 40.2 | 80.2±9.7 | 14.7±8.1 | 3.3 | 54.5 | 80.5±8.2 | 14.7±8.3 | -6.4 | 56.8 | 82.3±9.3 | 15.2±9.9 | -8.1 | 36.0 | 73.1±12.4 | 10.5±6.4 | -15.8 | 55.7 | 202.0 |
| STFPM NCL | 77.6±16.2 | 17.2±16.9 | -19.8 | 70.2 | 76.0±15.7 | 12.0±12.6 | -19.3 | 70.3 | 82.6±16.2 | 24.1±18.3 | -17.7 | **76.2** | 78.9±17.1 | 18.5±17.4 | -20.7 | 71.8 | 76.5±16.0 | 12.3±12.4 | -19.2 | 74.5 | 166.0 |
| STFPM Replay | 88.5±10.8 | 29.0±18.3 | -7.2 | 72.8 | 89.1±10.6 | 28.6±19.6 | -6.2 | 77.7 | 90.3±9.7 | 32.4±18.7 | -6.2 | 75.6 | 80.4±15.0 | 19.6±18.6 | -17.3 | 72.3 | 85.4±12.6 | 24.3±17.0 | -9.2 | **78.9** | 90.0 |
| NSR | 53.9±19.6 | 15.2±16.6 | -25.8 | 56.2 | 85.9±13.5 | 26.8±17.4 | -0.0 | 72.0 | 82.1±17.4 | 22.4±18.8 | -0.0 | 59.9 | 72.9±21.6 | 20.4±18.5 | -2.6 | 69.0 | 83.5±15.5 | 24.9±18.5 | -1.2 | 59.9 | 147.5 |
| NSR (5) | 84.0±10.7 | 17.5±14.2 | -2.0 | 67.2 | 85.7±8.8 | 20.1±17.7 | -5.4 | 68.0 | 91.5±6.5 | 36.1±21.6 | -0.1 | 62.9 | 83.0±13.0 | 20.7±18.2 | -4.6 | 70.1 | 84.9±8.5 | 17.9±14.4 | -3.4 | 61.2 | 113.0 |
| NSR (3) | 84.1±10.7 | 17.6±14.2 | -1.9 | 67.1 | 85.9±7.6 | 19.2±15.5 | -5.1 | -- | **92.0±5.5** | **36.6±20.5** | -0.2 | -- | 83.5±13.5 | 22.5±20.3 | -4.5 | -- | 84.7±8.4 | 17.5±14.4 | -3.8 | -- | 78.0 |
| Oracle | 64.6±18.8 | 9.2±16.3 | **-0.0** | 62.3 | 85.8±13.6 | 26.8±17.4 | -0.0 | 72.0 | 79.0±20.3 | 21.8±19.4 | -0.0 | 59.9 | 74.2±20.4 | 15.9±16.6 | **-0.0** | 68.9 | 79.0±20.3 | 21.8±19.4 | **-0.0** | 59.9 | 146.5 |

### Pixel-level results — VisA

| Method | 1x12 ROC | 1x12 AP | 1x12 BWT | 1x12 FWT | 11-1 ROC | 11-1 AP | 11-1 BWT | 11-1 FWT | 8-4 ROC | 8-4 AP | 8-4 BWT | 8-4 FWT | 8-1x4 ROC | 8-1x4 AP | 8-1x4 BWT | 8-1x4 FWT | RankSum↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PaDiM NCL | 82.3±8.4 | 6.1±15.4 | -16.4 | 84.7 | 82.3±8.4 | 6.1±15.4 | -13.5 | 95.7 | 89.5±7.0 | 9.9±17.0 | -8.9 | 87.0 | 82.3±8.4 | 6.1±15.4 | -14.8 | 86.8 | 153.0 |
| PaDiM CL | 94.7±3.2 | 10.2±10.0 | -0.9 | 84.4 | **94.6±3.7** | 15.3±14.0 | -0.0 | 95.7 | **95.3±3.4** | 16.6±16.8 | -0.0 | 87.0 | **95.5±3.4** | 16.2±15.6 | **-0.0** | 87.0 | **53.5** |
| PatchCore NCL | 87.2±5.8 | 10.3±18.5 | -12.1 | **87.0** | 87.2±5.8 | 10.3±18.5 | -11.8 | 98.1 | 91.6±6.8 | 16.7±20.8 | -9.6 | **88.0** | 87.2±5.8 | 10.3±18.5 | -11.8 | **88.6** | 95.0 |
| PatchCore CL | 85.7±8.9 | 13.0±20.8 | -13.7 | 86.8 | 86.6±6.5 | 10.2±18.5 | -12.5 | **98.2** | 89.7±12.4 | 16.6±20.9 | -12.3 | 88.0 | 86.0±7.6 | 10.2±18.5 | -13.0 | 88.6 | 107.0 |
| CFA NCL | 83.8±9.0 | 8.5±19.4 | -9.2 | 85.4 | 86.4±7.6 | 12.8±17.4 | 1.3 | 85.3 | 84.1±9.1 | 14.7±18.4 | **2.7** | 83.1 | 83.2±9.4 | 12.0±17.2 | -6.0 | 86.7 | 115.0 |
| CFA CL | **95.8±2.5** | **24.7±24.1** | -1.2 | 83.3 | 93.7±3.8 | **25.8±26.0** | 1.0 | 91.7 | 93.2±4.2 | **23.4±24.2** | 0.2 | 85.5 | 93.7±4.4 | **21.4±19.9** | -1.4 | 86.8 | 57.0 |
| FastFlow NCL | 41.9±23.0 | 3.0±8.7 | -53.7 | 50.2 | 51.5±17.5 | 2.5±6.4 | -34.4 | 57.2 | 77.3±12.1 | 4.0±6.6 | -4.5 | 67.3 | 49.4±16.5 | 1.6±3.9 | -34.6 | 54.1 | 208.0 |
| FastFlow Replay | 65.4±11.3 | 2.4±4.2 | -24.0 | 55.5 | 82.8±6.2 | 3.3±4.1 | 8.3 | 42.1 | 58.1±6.8 | 0.7±0.8 | -29.0 | 71.9 | 71.9±11.7 | 2.5±3.8 | -15.8 | 68.6 | 196.0 |
| STFPM NCL | 82.4±7.9 | 5.4±12.0 | -16.5 | 77.7 | 83.9±5.8 | 2.6±3.3 | **14.8** | 6.7 | 88.6±7.7 | 10.8±14.4 | -9.7 | 72.7 | 82.0±8.3 | 6.0±13.2 | -15.6 | 85.1 | 169.0 |
| STFPM Replay | 89.6±9.0 | 10.2±13.6 | -8.7 | 75.2 | 87.4±6.9 | 9.2±13.1 | -7.0 | 57.7 | 93.7±3.3 | 15.7±15.6 | -3.5 | 82.5 | 90.6±5.8 | 14.6±14.8 | -4.9 | 86.4 | 115.0 |
| NSR | 69.7±11.9 | 10.0±18.2 | -18.4 | 81.4 | 94.4±3.6 | 14.3±13.2 | -0.0 | 95.7 | 92.7±4.8 | 13.4±16.4 | -0.0 | 87.0 | 93.5±3.9 | 13.9±16.2 | -0.2 | 87.0 | 95.0 |
| NSR (5) | 92.9±6.2 | 10.5±11.5 | -3.2 | 85.5 | 83.6±9.6 | 11.3±16.1 | -0.9 | 83.2 | 69.4±21.1 | 17.2±21.6 | -0.0 | 81.1 | 85.7±12.4 | 13.4±13.5 | -1.2 | 84.9 | 114.0 |
| NSR (3) | 92.6±6.8 | 10.5±11.5 | -3.0 | 85.5 | 85.4±7.8 | 12.5±16.1 | -0.8 | -- | 70.8±21.7 | 18.2±23.7 | -0.0 | -- | 87.4±10.2 | 13.6±14.2 | -1.0 | -- | 71.0 |
| Oracle | 83.9±6.1 | 2.9±3.0 | **-0.0** | 82.8 | 94.4±3.6 | 14.3±13.2 | -0.0 | 95.7 | 92.1±5.6 | 13.2±16.4 | -0.0 | 87.0 | 92.1±5.6 | 13.2±16.4 | **-0.0** | 87.0 | 89.5 |

Runtime (wall-clock minutes, mean ± population std over timed executions; 2×NVIDIA L40 + 2×NVIDIA H100):

| Dataset | Setting | Train | Eval | Total |
|---------|---------|------:|-----:|------:|
| MVTec AD | Single-model | 7.9±8.8 | 17.8±18.1 | 25.7±22.7 |
| MVTec AD | Multi-expert | 21.0±24.0 | 73.3±88.4 | 94.4±102.6 |
| VisA | Single-model | 30.3±46.9 | 34.2±40.9 | 64.4±77.7 |
| VisA | Multi-expert | 54.8±64.8 | 125.0±124.3 | 179.8±145.7 |
| All | Single-model | 17.8±33.8 | 25.1±31.5 | 42.9±57.8 |
| All | Multi-expert | 36.2±49.8 | 96.5±109.0 | 132.6±130.9 |
