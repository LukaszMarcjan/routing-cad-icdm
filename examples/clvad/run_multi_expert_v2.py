"""Multi-expert continual visual AD runner — heterogeneous expert ensemble.

Derived from ``run_continual_visual_ad_step_schedule_v2.py``. Key differences:

* Builds a :class:`HeterogeneousExpertEnsemble` (one frozen expert per
  training step) instead of a single CL model.
* New CLI flags:
    - ``--expert-types padim,patchcore,cfa,fastflow,stfpm`` — comma list
      of model types; round-robin over training steps.
    - ``--router oracle|min_score|max_conf`` — routing strategy at
      inference; default ``min_score``.
* Adds :class:`RoutingAccuracyCallback` to log per-concept oracle vs
  router-chosen expert.
* Strategy is :class:`MultiExpertStrategy` — does not use ``--strategy``
  (it's its own strategy class).

Example
-------
Homogeneous 5×PatchCore + Oracle on MVTec 3×5 (sanity ceiling):

    python examples/clvad/run_multi_expert_v2.py \\
        --expert-types patchcore --router oracle \\
        --benchmark mvtec --root /path/to/mvtec_ad \\
        --step-schedule 3x5 --eval-level both \\
        --n-runs 1 --master-seed 123

Heterogeneous round-robin (5 types) + unsupervised min_score on MVTec 10-1×5:

    python examples/clvad/run_multi_expert_v2.py \\
        --expert-types padim,patchcore,cfa,fastflow,stfpm \\
        --router min_score \\
        --benchmark mvtec --root /path/to/mvtec_ad \\
        --step-schedule 10-1x5 --eval-level both
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import pathlib
import sys
import time
from typing import Any, Callable, List

from pyclad.callbacks.callback import Callback
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.callbacks.evaluation.rectangular_concept_metric_with_mask import (
    RectangularConceptMetricCallbackWithMask,
)
from pyclad.callbacks.evaluation.routing_accuracy_callback import (
    RoutingAccuracyCallback,
)
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.data.readers.visual_step_schedule import (
    apply_step_schedule,
    compute_first_seen_step,
    parse_step_schedule,
)
from pyclad.metrics.base.average_precision import AveragePrecision
from pyclad.metrics.base.f1_score import F1Score
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.final_step_average import FinalStepAverage
from pyclad.metrics.continual.rectangular_forgetting_measure import (
    RectangularForgettingMeasure,
)
from pyclad.metrics.continual.rectangular_forward_transfer import (
    RectangularForwardTransfer,
)
from pyclad.metrics.continual.rectangular_new_task_acquisition import (
    RectangularNewTaskAcquisition,
)
from pyclad.models.model import Model
from pyclad.models.vision.heterogeneous_expert_ensemble import (
    ExpertFactory,
    HeterogeneousExpertEnsemble,
)
from pyclad.models.vision.routers import build_router
from pyclad.output.output_writer import InfoProvider
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.vision.multi_expert import MultiExpertStrategy

logger = logging.getLogger(__name__)


def _load_module(filename: str, alias: str):
    module_path = pathlib.Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_LEVELS = _load_module("run_continual_visual_ad_levels.py", "pyclad_visual_levels_runner_me")
_BASE = _LEVELS._BASE
_CL_FACTORY = None


def _get_cl_factory():
    global _CL_FACTORY
    if _CL_FACTORY is None:
        _CL_FACTORY = _load_module("cl_factory.py", "pyclad_visual_me_cl_factory")
    return _CL_FACTORY


_SUPPORTED_EXPERT_TYPES = sorted(_BASE.MODEL_SPECS.keys())
# Models that have dedicated CL strategies (use --strategy cl in cl_factory).
# Others (fastflow, stfpm, etc.) fall back to --strategy replay.
_CL_NATIVE_TYPES = {"padim", "patchcore", "cfa"}


def parse_args() -> argparse.Namespace:
    help_requested = any(argument in {"-h", "--help"} for argument in sys.argv[1:])
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument(
        "--expert-types",
        type=str,
        required=not help_requested,
        help="Comma-separated list of model types for round-robin expert "
        "factories, e.g. 'padim,patchcore,cfa,fastflow,stfpm'. "
        "Pass a single name (e.g. 'patchcore') for a homogeneous ensemble.",
    )
    extra_parser.add_argument(
        "--router",
        type=str,
        choices=["oracle", "min_score", "max_conf"],
        default="min_score",
        help="Routing strategy at inference (default: min_score).",
    )
    extra_parser.add_argument(
        "--step-schedule",
        type=str,
        default=None,
        help="CDAD-style step schedule, e.g. '14-1', '10-5', '3x5', '10-1x5', '8-1x4'. "
             "If omitted, runs standard 1-task-per-category continual learning "
             "(square N x N matrix; rectangular metrics degenerate to classic FM/FWT).",
    )
    extra_parser.add_argument(
        "--continual-memory-bank-size",
        type=int,
        default=None,
        help="Unused by multi-expert (kept for CLI parity).",
    )
    extra_parser.add_argument(
        "--log-per-sample-routing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist per-sample chosen-expert + raw/normalized scores to a "
             ".routing.npz sidecar next to the JSON snapshot.",
    )
    extra_parser.add_argument(
        "--cumulative-cl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Each new expert sees cumulative data (tasks 0..i) and is trained "
             "via the CL strategy native to its model type. For padim/patchcore/cfa: "
             "their dedicated CL variants (ContinualPaDiM/PatchCore/CFA + cl strategy). "
             "For fastflow/stfpm: base model + ReplayStrategy with buffer. "
             "Default: False (each expert trained naively on its own task only).",
    )
    extra_parser.add_argument(
        "--dynamic-growth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Adaptive ensemble: per new task, decide build-vs-update based on "
             "existing experts' normalized anomaly scores on the new task. If the "
             "lowest normalized score ≤ --growth-threshold, UPDATE that expert via "
             "its retained CL strategy; otherwise BUILD a new one. Mutually "
             "exclusive with --cumulative-cl.",
    )
    extra_parser.add_argument(
        "--growth-threshold",
        type=float,
        default=0.0,
        help="Z-score threshold for dynamic_growth decision. 0.0 = update if new "
             "task looks at least as normal as expert's training mean. Lower "
             "(e.g. -0.5) = stricter: only very-normal-looking tasks merge. "
             "Higher (e.g. 0.5) = more permissive merging.",
    )
    extra_parser.add_argument(
        "--expert-assignment",
        choices=["round_robin", "best"],
        default="round_robin",
        help="How to assign a model type to each task (naive mode only). "
             "'round_robin' (default): task i -> expert_types[i %% len]. "
             "'best': train one candidate per type on a fit-split, pick the type "
             "that best separates the task from previously-seen tasks (AUROC "
             "proxy; tightest normal distribution for the first task), then "
             "retrain the winner on full data. Candidate pool = --expert-types.",
    )
    extra_parser.add_argument(
        "--assignment-val-fraction",
        type=float,
        default=0.2,
        help="Held-out fraction of each task's normals used to score candidate "
             "experts during --expert-assignment best (default 0.2).",
    )

    extra_args, remaining = extra_parser.parse_known_args()
    if help_requested:
        print("Multi-expert (v2) options:")
        print("  --expert-types LIST                comma list, round-robin per training step")
        print("  --router {oracle,min_score,max_conf}")
        print("  --step-schedule SPEC               CDAD-style schedule")
        print()

    # Validate --expert-types early so we can use the first type as a
    # placeholder for the downstream parser's required --model arg.
    if not help_requested and extra_args.expert_types is None:
        raise SystemExit("--expert-types is required (e.g. 'patchcore' or 'padim,patchcore,cfa,fastflow,stfpm').")
    expert_types = (
        [t.strip().lower() for t in extra_args.expert_types.split(",") if t.strip()]
        if extra_args.expert_types is not None
        else []
    )

    original_argv = sys.argv[:]
    try:
        # _LEVELS / _BASE parse_args require --model and --strategy from fixed
        # sets. Multi-expert doesn't use a single --model (it has --expert-types)
        # and ignores --strategy; inject harmless placeholders so the underlying
        # parser is happy. The placeholder --model uses the first expert type so
        # build_model_config() picks up a valid spec when called for config
        # construction downstream.
        if "--model" not in remaining and expert_types:
            remaining = remaining + ["--model", expert_types[0]]
        if "--strategy" not in remaining:
            remaining = remaining + ["--strategy", "naive"]
        sys.argv = [sys.argv[0], *remaining]
        args = _LEVELS.parse_args()
    finally:
        sys.argv = original_argv

    # --step-schedule is now optional: omitted → standard 1-task-per-category CL.
    if not expert_types:
        raise SystemExit("--expert-types parsed to an empty list.")
    unknown = [t for t in expert_types if t not in _SUPPORTED_EXPERT_TYPES]
    if unknown:
        raise SystemExit(
            f"Unknown expert types {unknown}. Supported: {_SUPPORTED_EXPERT_TYPES}"
        )

    args.expert_types = expert_types
    args.router = extra_args.router
    args.step_schedule = extra_args.step_schedule
    args.continual_memory_bank_size = extra_args.continual_memory_bank_size
    args.log_per_sample_routing = bool(extra_args.log_per_sample_routing)
    args.cumulative_cl = bool(extra_args.cumulative_cl)
    args.dynamic_growth = bool(extra_args.dynamic_growth)
    args.growth_threshold = float(extra_args.growth_threshold)
    args.expert_assignment = str(extra_args.expert_assignment)
    args.assignment_val_fraction = float(extra_args.assignment_val_fraction)
    if args.expert_assignment == "best" and (args.cumulative_cl or args.dynamic_growth):
        raise SystemExit(
            "--expert-assignment best is only supported in naive mode "
            "(not with --cumulative-cl / --dynamic-growth)"
        )
    if args.cumulative_cl and args.dynamic_growth:
        raise SystemExit(
            "--cumulative-cl and --dynamic-growth are mutually exclusive (pick one)"
        )
    args.strategy = "multi_expert"  # purely a label for output filenames

    if args.step_schedule is not None:
        try:
            parse_step_schedule(args.step_schedule)
        except ValueError as exc:
            raise SystemExit(f"Invalid --step-schedule {args.step_schedule!r}: {exc}") from exc

    return args


def build_expert_factory_for_type(model_name: str, args: argparse.Namespace, seed: int) -> ExpertFactory:
    """Return a zero-arg factory that builds a fresh instance of ``model_name``."""

    def factory() -> Model:
        original_model = args.model
        try:
            args.model = model_name
            config = _BASE.build_model_config(args, seed)
            spec = _BASE.MODEL_SPECS[model_name]
            model_cls = _BASE._load_attr(spec.model_module, spec.model_name)
            return model_cls(config)
        finally:
            args.model = original_model

    return factory


def build_cl_expert_factory(args: argparse.Namespace, seed: int, replay_buffer_size: int):
    """Return a callable ``(type_name) -> (model, strategy)`` for cumulative CL mode.

    Reuses :mod:`examples/clvad/cl_factory` so each expert type gets the CL
    strategy native to its architecture:
      * padim/patchcore/cfa → ContinualPaDiM/PatchCore/CFA + dedicated CL strategy
      * fastflow/stfpm/...  → base model + ReplayStrategy with buffer
    """
    cl_factory = _get_cl_factory()

    def factory(type_name: str):
        type_name = type_name.lower()
        original_model = args.model
        original_strategy = args.strategy
        try:
            args.model = type_name
            if type_name in _CL_NATIVE_TYPES:
                args.strategy = "cl"
            else:
                args.strategy = "replay"
            model, _config = cl_factory.build_model(_BASE, args, seed)
            strategy = cl_factory.build_strategy(
                _BASE,
                args,
                model,
                model_creation_fn=lambda: cl_factory.build_model(_BASE, args, seed)[0],
                replay_buffer_size=replay_buffer_size,
                run_dataset=None,
            )
            return model, strategy
        finally:
            args.model = original_model
            args.strategy = original_strategy
    return factory


def build_ensemble(
    args: argparse.Namespace,
    seed: int,
    replay_buffer_size: int = 0,
) -> HeterogeneousExpertEnsemble:
    factories: List[ExpertFactory] = [
        build_expert_factory_for_type(name, args, seed) for name in args.expert_types
    ]
    router = build_router(args.router)
    cl_expert_factory = None
    if args.cumulative_cl or args.dynamic_growth:
        cl_expert_factory = build_cl_expert_factory(args, seed, replay_buffer_size)
    return HeterogeneousExpertEnsemble(
        expert_factories=factories,
        router=router,
        type_names=list(args.expert_types),
        cumulative_cl=bool(args.cumulative_cl),
        cl_expert_factory=cl_expert_factory,
        dynamic_growth=bool(args.dynamic_growth),
        growth_threshold=float(args.growth_threshold),
        expert_assignment=str(getattr(args, "expert_assignment", "round_robin")),
        assignment_val_fraction=float(getattr(args, "assignment_val_fraction", 0.2)),
        assignment_seed=int(seed),
    )


class _PendingConceptNameTracker(Callback):
    """Tiny callback that forwards train-concept names to the strategy."""

    def __init__(self, strategy: MultiExpertStrategy) -> None:
        self._strategy = strategy

    def before_concept_processing(self, concept: Concept) -> None:
        self._strategy.set_pending_concept_name(concept.name)


def build_callbacks(
    args: argparse.Namespace,
    dataset: ConceptsDataset,
    strategy: MultiExpertStrategy,
    dataset_root: pathlib.Path,
    first_seen_step: dict,
    per_sample_output_path: pathlib.Path | None = None,
) -> List[Callback]:
    legacy_metrics = [FinalStepAverage()]
    schedule_aware_metrics = [
        RectangularForgettingMeasure(),
        RectangularForwardTransfer(),
        RectangularNewTaskAcquisition(),
    ]

    callbacks: List[Callback] = [
        _BASE.ScenarioProgressCallback(dataset),
        _PendingConceptNameTracker(strategy),
    ]

    for base_metric in [RocAuc(), F1Score(), AveragePrecision()]:
        callbacks.append(
            RectangularConceptMetricCallbackWithMask(
                base_metric=base_metric,
                metrics=legacy_metrics,
                schedule_aware_metrics=schedule_aware_metrics,
                first_seen_step=first_seen_step,
            )
        )

    if args.eval_level in {"pixel", "both"}:
        from pyclad.callbacks.evaluation.rectangular_visual_pixel_concept_metric_callback_with_mask import (
            RectangularVisualPixelConceptMetricCallbackWithMask,
        )
        from pyclad.metrics.base.pixel_aupro import PixelAUPRO
        from pyclad.metrics.base.pixel_average_precision import PixelAveragePrecision
        from pyclad.metrics.base.pixel_dice_score import PixelDiceScore
        from pyclad.metrics.base.pixel_f1_score import PixelF1Score
        from pyclad.metrics.base.pixel_iou import PixelIoU
        from pyclad.metrics.base.pixel_roc_auc import PixelRocAuc

        callbacks.append(
            RectangularVisualPixelConceptMetricCallbackWithMask(
                strategy=strategy,
                benchmark=args.benchmark,
                root=dataset_root,
                registry_path=args.registry_path,
                categories=args.categories,
                max_test_samples_per_category=args.max_test,
                base_metrics=[
                    PixelRocAuc(),
                    PixelAveragePrecision(),
                    PixelAUPRO(),
                    PixelF1Score(threshold=args.pixel_threshold),
                    PixelDiceScore(threshold=args.pixel_threshold),
                    PixelIoU(threshold=args.pixel_threshold),
                ],
                metrics=legacy_metrics,
                schedule_aware_metrics=schedule_aware_metrics,
                first_seen_step=first_seen_step,
                skip_missing_anomaly_masks=args.pixel_skip_missing_masks,
                threshold_mode=args.pixel_threshold_mode,
                fixed_threshold=args.pixel_threshold,
                threshold_quantile=args.pixel_threshold_quantile,
            )
        )

    callbacks.append(
        RoutingAccuracyCallback(
            strategy=strategy,
            first_seen_step=first_seen_step,
            log_per_sample=bool(getattr(args, "log_per_sample_routing", False)),
            per_sample_output_path=per_sample_output_path,
        )
    )
    callbacks.extend([TimeEvaluationCallback(), MemoryUsageCallback()])
    return callbacks


def _schedule_output_suffix(args: argparse.Namespace) -> str:
    types_tag = "-".join(args.expert_types)
    if args.step_schedule is None:
        sched_tag = "noschd-1taskPerConcept"
    else:
        safe = args.step_schedule.replace("×", "x").replace("*", "x")
        sched_tag = f"schd-{safe}-perCat"
    if getattr(args, "dynamic_growth", False):
        thr = f"{args.growth_threshold:+.2f}".replace(".", "p")
        train_mode = f"dynGrow{thr}"
    elif getattr(args, "cumulative_cl", False):
        train_mode = "cumCL"
    else:
        train_mode = "naive"
    # Distinguish per-type "best" assignment from default round-robin so its
    # output files don't collide with existing round-robin runs.
    if getattr(args, "expert_assignment", "round_robin") == "best":
        train_mode = f"{train_mode}-asgnBest"
    return f"_me-{types_tag}_router-{args.router}_{sched_tag}_train-{train_mode}-v2"


def run_single(
    args: argparse.Namespace,
    seed: int,
    run_index: int,
    total_runs: int,
    dataset: ConceptsDataset,
    output_dir: pathlib.Path,
    dataset_root: pathlib.Path,
) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pixel_suffix = _LEVELS._pixel_output_suffix(args)
    schedule_suffix = _schedule_output_suffix(args)
    output_path = (
        output_dir
        / f"multi_expert_{args.eval_level}{pixel_suffix}{schedule_suffix}_seed_{seed}.json"
    )
    log_path = output_path.with_suffix(".log")

    with _BASE.seed_log_handler(log_path):
        run_status = _BASE.RunStatusProvider(output_path=output_path, log_path=log_path)
        run_payload = _BASE.create_run_config_payload(
            args=args,
            seed=seed,
            output_path=output_path,
            log_path=log_path,
        )
        run_payload["eval_level"] = args.eval_level
        run_payload["pixel_skip_missing_masks"] = args.pixel_skip_missing_masks
        run_payload["pixel_threshold"] = args.pixel_threshold
        run_payload["pixel_threshold_mode"] = args.pixel_threshold_mode
        run_payload["pixel_threshold_quantile"] = args.pixel_threshold_quantile
        run_payload["step_schedule"] = args.step_schedule
        run_payload["step_schedule_parsed"] = (
            parse_step_schedule(args.step_schedule) if args.step_schedule is not None else None
        )
        run_payload["expert_types"] = args.expert_types
        run_payload["router"] = args.router
        run_payload["runner_variant"] = "multi_expert_v2"
        run_provider = _BASE.RunConfigProvider(run_payload)

        ensemble = None
        strategy = None
        run_dataset: ConceptsDataset | None = None
        callbacks: List[Callback] = []

        def write_snapshot() -> None:
            _BASE.write_result_snapshot(
                output_path=output_path,
                providers=_BASE.collect_result_providers(
                    model=ensemble,
                    run_dataset=run_dataset,
                    strategy=strategy,
                    callbacks=callbacks,
                    extra_providers=[run_provider, run_status],
                ),
                completed_concepts=run_status.completed_concepts(),
            )

        logger.info("═══ Run %d/%d  seed=%d ═══", run_index + 1, total_runs, seed)
        _BASE.set_global_seeds(seed)

        try:
            logger.info("  [1/5] Preparing step schedule (native concept order)")
            ordered = dataset

            ordered_train_names = [c.name for c in ordered.train_concepts()]
            if args.step_schedule is not None:
                # Step-scheduled mode (rectangular T x N matrix).
                run_dataset = apply_step_schedule(
                    dataset=ordered,
                    schedule=args.step_schedule,
                    group_test=False,
                    name_mode="indexed",
                )
                first_seen_step = dict(
                    zip(
                        ordered_train_names,
                        compute_first_seen_step(ordered_train_names, args.step_schedule),
                    )
                )
                run_payload["schedule_metadata"] = {
                    "step_schedule": args.step_schedule,
                    "step_schedule_parsed": parse_step_schedule(args.step_schedule),
                    "train_steps": [c.name for c in run_dataset.train_concepts()],
                    "test_concepts": [c.name for c in run_dataset.test_concepts()],
                    "first_seen_step": first_seen_step,
                }
                mode_label = f"step-schedule={args.step_schedule}"
            else:
                # Standard 1-task-per-category CL (square N x N matrix).
                run_dataset = ordered
                first_seen_step = {
                    name: i for i, name in enumerate(ordered_train_names)
                }
                run_payload["schedule_metadata"] = {
                    "step_schedule": None,
                    "mode": "one_task_per_concept",
                    "train_steps": ordered_train_names,
                    "test_concepts": [c.name for c in run_dataset.test_concepts()],
                    "first_seen_step": first_seen_step,
                }
                mode_label = "1-task-per-concept (no step schedule)"
            run_status.set_total_concepts(len(run_dataset.train_concepts()))

            logger.info(
                "  Mode: %s  Train steps: %d  Test concepts: %d  expert_types: %s  router: %s",
                mode_label,
                len(run_dataset.train_concepts()),
                len(run_dataset.test_concepts()),
                args.expert_types,
                args.router,
            )

            logger.info("  [2/5] Building ensemble + strategy")
            # Compute replay buffer size (used in cumulative_cl / dynamic_growth
            # modes for FastFlow/STFPM experts that fall back to ReplayStrategy).
            replay_buffer_size = 0
            if args.cumulative_cl or args.dynamic_growth:
                replay_buffer_size = int(_BASE.resolve_replay_buffer_size(args, run_dataset))
                run_payload["replay_buffer_fraction"] = args.replay_buffer_fraction
                run_payload["replay_buffer_size"] = replay_buffer_size
            run_payload["cumulative_cl"] = bool(args.cumulative_cl)
            run_payload["dynamic_growth"] = bool(args.dynamic_growth)
            if args.dynamic_growth:
                run_payload["growth_threshold"] = float(args.growth_threshold)
            ensemble = build_ensemble(args, seed, replay_buffer_size=replay_buffer_size)
            strategy = MultiExpertStrategy(ensemble)

            logger.info("  [3/5] Building callbacks")
            per_sample_path = (
                output_path.with_suffix(".routing.npz")
                if args.log_per_sample_routing
                else None
            )
            callbacks = build_callbacks(
                args=args,
                dataset=run_dataset,
                strategy=strategy,
                dataset_root=dataset_root,
                first_seen_step=first_seen_step,
                per_sample_output_path=per_sample_path,
            )
            callbacks.append(
                _BASE.JsonCheckpointCallback(
                    output_path=output_path,
                    run_status_provider=run_status,
                    write_snapshot=write_snapshot,
                )
            )

            scenario = ConceptIncrementalScenario(
                dataset=run_dataset,
                strategy=strategy,
                callbacks=callbacks,
            )

            logger.info("  [4/5] Running scenario")
            started_at = time.perf_counter()
            scenario.run()
            elapsed = time.perf_counter() - started_at
        except BaseException as exc:
            run_status.mark_failed(exc, _BASE.traceback.format_exc())
            logger.exception(
                "  Run did not finish cleanly; saving partial results to %s", output_path
            )
            try:
                write_snapshot()
            except Exception:
                logger.exception("  Failed to save partial results to %s", output_path)
            raise

        logger.info("  [5/5] Saving results")
        run_status.mark_completed(elapsed_seconds=elapsed)
        skipped_callbacks = [
            type(c).__name__ for c in callbacks if not isinstance(c, InfoProvider)
        ]
        if skipped_callbacks:
            logger.info("    Skipping non-serializable callbacks: %s", skipped_callbacks)
        write_snapshot()
        logger.info("    ✓ finished in %.1fs → %s", elapsed, output_path)

    return output_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.getLogger("pyclad.scenarios.concept_incremental").setLevel(logging.WARNING)

    output_dir = _BASE.resolve_output_dir(args)
    seeds = _BASE.generate_seeds(args.master_seed, args.n_runs)
    logger.info("Master seed: %d  →  per-run seeds: %s", args.master_seed, seeds)
    logger.info("Results directory: %s", output_dir)
    logger.info("Multi-expert: expert_types=%s router=%s schedule=%s",
                args.expert_types, args.router, args.step_schedule)

    spec = _BASE.MODEL_SPECS[args.expert_types[0]]
    resize = int(args.resize) if args.resize is not None else spec.default_resize
    resize_to = (resize, resize)
    dataset_root = _BASE.resolve_visual_benchmark_root(
        benchmark=args.benchmark,
        root=args.root,
        registry_path=args.registry_path,
    )
    logger.info("Dataset root: %s", dataset_root)
    dataset = _BASE.read_registered_visual_benchmark_dataset(
        benchmark=args.benchmark,
        root=dataset_root,
        registry_path=args.registry_path,
        categories=args.categories,
        data_mode="numpy",
        resize_to=resize_to,
        color_mode="rgb",
        max_train_samples_per_category=args.max_train,
        max_test_samples_per_category=args.max_test,
    )
    _BASE.log_dataset_overview(dataset, "Loaded dataset order:")

    saved: list[pathlib.Path] = []
    for run_index, seed in enumerate(seeds):
        path = run_single(
            args=args,
            seed=seed,
            run_index=run_index,
            total_runs=len(seeds),
            dataset=dataset,
            output_dir=output_dir,
            dataset_root=dataset_root,
        )
        saved.append(path)

    logger.info("All %d runs complete.", len(saved))
    for path in saved:
        logger.info("  %s", path)


if __name__ == "__main__":
    main()
