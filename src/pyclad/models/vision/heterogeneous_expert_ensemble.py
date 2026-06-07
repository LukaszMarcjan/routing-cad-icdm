"""Meta-model: per-task frozen experts + router for continual AD.

For each training step, fits a fresh expert on that step's data,
freezes it, and appends to an ordered list. At inference, all experts
score the input batch and a :class:`Router` selects, per sample, which
expert's score to use.

Supports two factory modes through ``expert_factories``:

* Homogeneous: a single-element list — every task gets a fresh instance
  of the same model class (e.g. ``[lambda: build_patchcore()]``).
* Heterogeneous round-robin: multiple factories — task ``i`` uses
  ``expert_factories[i % len(expert_factories)]`` (e.g. PatchCore →
  PaDiM → CFA → FastFlow → STFPM → PatchCore → ...).

The ensemble is itself a :class:`Model`, so it plugs into the existing
scenario / strategy / callback pipeline without changes. Pixel-level
evaluation works via :meth:`score_maps` which routes per sample on
image-level scores then returns the chosen expert's per-sample map.

Routing requires per-expert calibration on validation normals to avoid
score-scale dominance. The ensemble does this automatically inside
:meth:`fit_new_expert` by scoring the same training-normal batch with
the new expert (training normals are the only easy-to-get in-distribution
data inside a CL scenario).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from pyclad.models.model import Model
from pyclad.models.vision.routers import OracleRouter, Router

logger = logging.getLogger(__name__)


ExpertFactory = Callable[[], Model]


class HeterogeneousExpertEnsemble(Model):
    """Per-task frozen experts + router-based selection at inference.

    Two training modes:

    * ``cumulative_cl=False`` (default) — each new expert is built fresh and
      trained on its own task data only. Pure architectural CL: no
      forgetting because experts are independent.
    * ``cumulative_cl=True`` — each new expert is built fresh and then
      trained on the cumulative data of all tasks seen so far, using the
      CL strategy native to its model type (per-task Gaussians for PaDiM,
      per-task banks for PatchCore, per-task descriptors + replay for
      CFA, generic replay buffer for FastFlow/STFPM). This combines
      architectural CL with algorithmic CL: each expert is itself a CL
      learner trained on a progressively larger task history.

    The ``cl_expert_factory`` callable handles the CL-aware construction.
    It receives ``(type_name)`` and returns ``(model, strategy)`` — the
    strategy is fed historical task data sequentially, after which only
    the model is retained (with full CL state).
    """

    def __init__(
        self,
        expert_factories: List[ExpertFactory],
        router: Router,
        *,
        type_names: Optional[List[str]] = None,
        calibration_max_samples: int = 256,
        cumulative_cl: bool = False,
        cl_expert_factory: Optional[Callable[[str], "Tuple[Model, object]"]] = None,
        dynamic_growth: bool = False,
        growth_threshold: float = 0.0,
        dynamic_decision_subsample: int = 64,
        expert_assignment: str = "round_robin",
        assignment_val_fraction: float = 0.2,
        assignment_max_pseudo_anomalies: int = 256,
        assignment_seed: int = 0,
    ) -> None:
        if not expert_factories:
            raise ValueError("expert_factories must contain at least one factory")
        self._expert_factories = list(expert_factories)
        self._router = router
        # Free-text labels for the JSON snapshot (defaults to generic names).
        self._type_names = list(type_names) if type_names is not None else [
            f"type_{i}" for i in range(len(expert_factories))
        ]
        if len(self._type_names) != len(expert_factories):
            raise ValueError(
                f"type_names has length {len(self._type_names)} but "
                f"expert_factories has length {len(expert_factories)}"
            )
        self._calibration_max_samples = int(calibration_max_samples)

        self._cumulative_cl = bool(cumulative_cl)
        self._dynamic_growth = bool(dynamic_growth)
        self._growth_threshold = float(growth_threshold)
        self._dynamic_decision_subsample = int(dynamic_decision_subsample)
        self._cl_expert_factory = cl_expert_factory

        # ── Per-type expert assignment (improvement #1) ──────────────────────
        # "round_robin": task i -> expert_factories[i % len(factories)] (default,
        #                reproduces the original heterogeneous behaviour).
        # "best":        for each task, train one candidate per factory on a
        #                fit-split of the task's normals, score a held-out
        #                val-split, and pick the type that best SEPARATES the
        #                task from previously-seen tasks (AUROC proxy); for the
        #                first task fall back to the tightest normal distribution
        #                (scale-invariant coefficient of variation). The winning
        #                type is then retrained on the FULL task data.
        assignment = str(expert_assignment).lower().strip()
        if assignment not in {"round_robin", "best"}:
            raise ValueError(
                f"expert_assignment must be 'round_robin' or 'best', got {expert_assignment!r}"
            )
        self._expert_assignment = assignment
        self._assignment_val_fraction = float(assignment_val_fraction)
        self._assignment_max_pseudo_anomalies = int(assignment_max_pseudo_anomalies)
        self._assignment_seed = int(assignment_seed)
        # Per-task assignment decisions (chosen type + candidate scores) for JSON.
        self._assignment_decisions: List[Dict[str, Any]] = []

        if self._cumulative_cl and self._dynamic_growth:
            raise ValueError(
                "cumulative_cl and dynamic_growth are mutually exclusive "
                "(both modes own task-history handling). Pick one."
            )
        if self._expert_assignment == "best" and (self._cumulative_cl or self._dynamic_growth):
            raise ValueError(
                "expert_assignment='best' is only supported in naive mode "
                "(cumulative_cl/dynamic_growth own their own per-expert training "
                "and candidate sweeping over CL histories is prohibitively costly)."
            )
        if (self._cumulative_cl or self._dynamic_growth) and self._cl_expert_factory is None:
            raise ValueError(
                "cumulative_cl=True or dynamic_growth=True requires "
                "cl_expert_factory to be provided"
            )

        self._experts: List[Model] = []
        self._expert_type_for_task: List[str] = []
        self._concept_to_expert_idx: Dict[str, int] = {}
        self._pending_concept_name: Optional[str] = None
        # Cumulative CL needs the full task history to feed each new expert.
        self._task_data_history: List[np.ndarray] = []
        # Dynamic growth keeps the CL strategy beside each expert so we can
        # call strategy.learn(new_task_data) to update an existing expert.
        self._expert_strategies: List[Optional[object]] = []
        self._expert_task_concepts: List[List[str]] = []
        # Per-task dynamic decisions (build vs update) for the JSON snapshot.
        self._dynamic_decisions: List[Dict[str, Any]] = []

    # ── lifecycle hooks used by the strategy ──────────────────────────────
    def set_pending_concept_name(self, concept_name: str) -> None:
        """Called by the runner/callback before each :meth:`fit_new_expert`.

        Lets us record the GT concept→expert mapping without changing the
        scenario API.
        """
        self._pending_concept_name = concept_name

    def fit_new_expert(self, data: np.ndarray, concept_name: Optional[str] = None) -> None:
        """Dispatch to naive / cumulative_cl / dynamic_growth handler."""
        if concept_name is None:
            concept_name = self._pending_concept_name or f"task_{len(self._experts)}"
        self._pending_concept_name = None

        if self._dynamic_growth:
            self._fit_dynamic_growth(data, concept_name)
            return

        task_idx = len(self._experts)

        if self._cumulative_cl:
            # Append BEFORE building so the CL expert sees the current task too.
            self._task_data_history.append(data)
            factory_idx = task_idx % len(self._expert_factories)
            type_name = self._type_names[factory_idx]
            logger.info(
                "MultiExpert[cumulative_cl]: fitting task %d (concept=%r) as expert type=%s "
                "on cumulative history (%d task(s))",
                task_idx, concept_name, type_name, len(self._task_data_history),
            )
            expert = self._fit_cumulative_cl_expert(type_name)
        elif self._expert_assignment == "best" and len(self._expert_factories) > 1:
            # Select the best-fitting type for this task using PAST tasks as
            # pseudo-anomalies (history at this point excludes the current task).
            expert, type_name, factory_idx, assign_meta = self._select_best_expert(
                data, task_idx, concept_name
            )
            self._assignment_decisions.append(assign_meta)
            self._task_data_history.append(data)
        else:
            factory_idx = task_idx % len(self._expert_factories)
            type_name = self._type_names[factory_idx]
            logger.info(
                "MultiExpert: fitting task %d (concept=%r) as expert type=%s",
                task_idx, concept_name, type_name,
            )
            expert = self._expert_factories[factory_idx]()
            expert.fit(data)
            self._task_data_history.append(data)

        self._freeze_expert(expert)

        self._experts.append(expert)
        self._expert_strategies.append(None)  # not needed in naive/cumulative modes
        self._expert_task_concepts.append([concept_name])
        self._expert_type_for_task.append(type_name)
        self._concept_to_expert_idx[concept_name] = task_idx

        # Calibrate router on this expert's training-normal scores.
        # In cumulative mode, calibration uses the current task's data
        # (consistent across modes — same notion of "this expert's normal").
        calib_data = self._subsample_for_calibration(data)
        try:
            normal_scores = self._score_safely(expert, calib_data)
            self._router.calibrate(task_idx, normal_scores)
        except Exception:
            logger.exception(
                "MultiExpert: router calibration failed for expert %d (type=%s); "
                "router will use uncalibrated scores for this expert.",
                task_idx,
                type_name,
            )

    def _fit_dynamic_growth(self, data: np.ndarray, concept_name: str) -> None:
        """Adaptive ensemble: decide build-vs-update from router confidence.

        Policy: score the new task's data with all existing experts, normalize
        each expert's mean score using its router calibration, and:
          * if min normalized mean ≤ growth_threshold → UPDATE that expert
            (its CL strategy.learn(new_task_data) absorbs the new task);
          * else → BUILD a new expert (with the next round-robin type).

        Threshold semantics (z-score): 0.0 = "as normal as this expert's
        training mean"; negative = "more normal than mean"; positive = "less
        normal". Default 0.0 is conservative-toward-growth.
        """
        decision, target_idx, decision_meta = self._choose_dynamic_action(data)

        if decision == "update":
            assert target_idx is not None
            self._update_existing_expert(target_idx, data, concept_name)
        else:
            self._build_dynamic_expert(data, concept_name)

        decision_meta.update({
            "concept": concept_name,
            "decision": decision,
            "target_expert_idx": target_idx,
            "ensemble_size_after": len(self._experts),
        })
        self._dynamic_decisions.append(decision_meta)

    def _choose_dynamic_action(
        self, data: np.ndarray
    ) -> Tuple[str, Optional[int], Dict[str, Any]]:
        """Return (decision, target_expert_idx, meta) for the new task."""
        if not self._experts:
            return "build", None, {"reason": "first_task"}

        sample = data[: min(self._dynamic_decision_subsample, len(data))]
        per_expert_mean = np.array(
            [float(self._score_safely(e, sample).mean()) for e in self._experts],
            dtype=np.float64,
        )

        normalize_fn = getattr(self._router, "normalize_scores", None)
        if callable(normalize_fn):
            normalized = normalize_fn(per_expert_mean[:, None])[:, 0]
        else:
            normalized = per_expert_mean.copy()

        argmin = int(np.argmin(normalized))
        min_normalized = float(normalized[argmin])

        meta: Dict[str, Any] = {
            "per_expert_mean_raw": per_expert_mean.tolist(),
            "per_expert_mean_normalized": normalized.tolist(),
            "argmin": argmin,
            "min_normalized": min_normalized,
            "threshold": self._growth_threshold,
        }

        if min_normalized <= self._growth_threshold:
            return "update", argmin, meta
        return "build", None, meta

    def _build_dynamic_expert(self, data: np.ndarray, concept_name: str) -> None:
        """Add a brand-new CL-aware expert (keep its strategy for later updates)."""
        new_task_idx = len(self._experts)
        factory_idx = new_task_idx % len(self._expert_factories)
        type_name = self._type_names[factory_idx]

        logger.info(
            "MultiExpert[dynamic_growth]: BUILD expert %d (concept=%r, type=%s)",
            new_task_idx, concept_name, type_name,
        )

        model, strategy = self._cl_expert_factory(type_name)
        strategy.learn(data=data)
        self._freeze_expert(model)

        self._experts.append(model)
        self._expert_strategies.append(strategy)
        self._expert_task_concepts.append([concept_name])
        self._expert_type_for_task.append(type_name)
        self._concept_to_expert_idx[concept_name] = new_task_idx

        calib_data = self._subsample_for_calibration(data)
        try:
            normal_scores = self._score_safely(model, calib_data)
            self._router.calibrate(new_task_idx, normal_scores)
        except Exception:
            logger.exception("Router calibration failed for newly built expert %d", new_task_idx)

    def _update_existing_expert(
        self, target_idx: int, data: np.ndarray, concept_name: str
    ) -> None:
        """Update an existing expert with this task via its retained CL strategy."""
        if target_idx < 0 or target_idx >= len(self._experts):
            raise IndexError(f"Cannot update expert {target_idx}: only {len(self._experts)} present")
        strategy = self._expert_strategies[target_idx]
        if strategy is None:
            raise RuntimeError(
                f"Expert {target_idx} has no retained CL strategy — cannot update. "
                "(This expert was created outside dynamic_growth mode.)"
            )
        type_name = self._expert_type_for_task[target_idx]
        logger.info(
            "MultiExpert[dynamic_growth]: UPDATE expert %d (concept=%r, type=%s, "
            "tasks_already_seen=%d)",
            target_idx, concept_name, type_name, len(self._expert_task_concepts[target_idx]),
        )

        # Temporarily unfreeze torch params so Lightning-based experts can train again.
        self._unfreeze_expert(self._experts[target_idx])
        strategy.learn(data=data)
        self._freeze_expert(self._experts[target_idx])

        self._expert_task_concepts[target_idx].append(concept_name)
        self._concept_to_expert_idx[concept_name] = target_idx

        # Re-calibrate router for this expert on its newest task's data
        # (its score distribution may have shifted after the update).
        calib_data = self._subsample_for_calibration(data)
        try:
            normal_scores = self._score_safely(self._experts[target_idx], calib_data)
            self._router.calibrate(target_idx, normal_scores)
        except Exception:
            logger.exception(
                "Router re-calibration failed for updated expert %d", target_idx
            )

    def _fit_cumulative_cl_expert(self, type_name: str) -> Model:
        """Build a CL-aware expert and feed it the full task history."""
        model, strategy = self._cl_expert_factory(type_name)
        learn_fn = getattr(strategy, "learn", None)
        if not callable(learn_fn):
            raise AttributeError(
                f"CL strategy for type {type_name!r} ({type(strategy).__name__}) "
                "does not expose .learn(data)"
            )
        for past_task_data in self._task_data_history:
            learn_fn(data=past_task_data)
        return model

    # ── Per-type expert assignment (improvement #1) ───────────────────────
    def _select_best_expert(
        self, data: np.ndarray, task_idx: int, concept_name: str
    ) -> Tuple[Model, str, int, Dict[str, Any]]:
        """Pick the expert type that best fits this task, then retrain on full data.

        Procedure (naive mode only):
          1. Split the task's normals into fit / val (held-out, leak-free).
          2. Train one candidate per factory on the fit-split.
          3. Score the val-split with each candidate.
             * If previous tasks exist, use them as pseudo-anomalies and pick
               the type with the highest separability AUROC (val-normals=0,
               past-task samples=1). This directly rewards experts whose own
               task is distinguishable — exactly what the router needs.
             * Otherwise (first task) pick the tightest val-normal distribution
               (lowest coefficient of variation; scale-invariant across types).
          4. Retrain the winning type on the FULL task data and return it.

        Both criteria are scale-invariant, so comparison across heterogeneous
        model families (memory-bank vs flow vs reconstruction) is fair.
        """
        rng = np.random.default_rng(self._assignment_seed + task_idx)
        n = len(data)
        n_val = int(round(n * self._assignment_val_fraction)) if n > 1 else 0
        n_val = max(0, min(n_val, n - 1))  # always keep >=1 fit sample
        if n_val > 0:
            perm = rng.permutation(n)
            val_data = data[perm[:n_val]]
            fit_data = data[perm[n_val:]]
        else:
            val_data = data
            fit_data = data

        pseudo_anomalies = self._gather_pseudo_anomalies(rng)
        use_separability = pseudo_anomalies is not None and len(pseudo_anomalies) > 0
        criterion = "separability_auroc" if use_separability else "tightness"

        candidates: List[Dict[str, Any]] = []
        for fidx, factory in enumerate(self._expert_factories):
            type_name = self._type_names[fidx]
            try:
                cand = factory()
                cand.fit(fit_data)
                val_scores = self._score_safely(cand, val_data)
                if use_separability:
                    anom_scores = self._score_safely(cand, pseudo_anomalies)
                    score = self._separability_auroc(val_scores, anom_scores)
                else:
                    score = self._tightness_score(val_scores)
                ok = True
            except Exception:
                logger.exception(
                    "MultiExpert[assignment]: candidate type=%s failed on task %d; "
                    "excluding it from selection.",
                    type_name, task_idx,
                )
                score, ok = float("-inf"), False
            candidates.append({"type": type_name, "factory_idx": fidx, "score": score, "ok": ok})

        best = max(candidates, key=lambda c: c["score"])
        best_idx = int(best["factory_idx"])
        type_name = self._type_names[best_idx]

        logger.info(
            "MultiExpert[assignment]: task %d (concept=%r) -> type=%s "
            "(criterion=%s, score=%.4f); candidates: %s",
            task_idx, concept_name, type_name, criterion, float(best["score"]),
            ", ".join(f"{c['type']}={c['score']:.3f}" for c in candidates),
        )

        # Retrain the winner on the full task data for the actual frozen expert.
        expert = self._expert_factories[best_idx]()
        expert.fit(data)

        meta: Dict[str, Any] = {
            "task_idx": task_idx,
            "concept": concept_name,
            "criterion": criterion,
            "chosen_type": type_name,
            "chosen_factory_idx": best_idx,
            "n_fit": int(len(fit_data)),
            "n_val": int(len(val_data)),
            "n_pseudo_anomalies": int(len(pseudo_anomalies)) if use_separability else 0,
            "candidate_scores": {c["type"]: float(c["score"]) for c in candidates},
        }
        return expert, type_name, best_idx, meta

    def _gather_pseudo_anomalies(self, rng: np.random.Generator) -> Optional[np.ndarray]:
        """Pool samples from previously-seen tasks as out-of-distribution proxies.

        Returns ``None`` for the first task (no history yet). Subsamples to
        ``assignment_max_pseudo_anomalies`` for efficiency.
        """
        if not self._task_data_history:
            return None
        pool = np.concatenate(list(self._task_data_history), axis=0)
        cap = self._assignment_max_pseudo_anomalies
        if cap > 0 and len(pool) > cap:
            idx = rng.choice(len(pool), size=cap, replace=False)
            pool = pool[idx]
        return pool

    @staticmethod
    def _separability_auroc(normal_scores: np.ndarray, anomaly_scores: np.ndarray) -> float:
        """AUROC separating pseudo-anomalies (label 1) from own val-normals (0).

        Higher anomaly score is treated as more anomalous (the AD convention).
        Rank-based, hence invariant to per-type score scale. Returns 0.5 on
        degenerate input.
        """
        from sklearn.metrics import roc_auc_score

        normal = np.asarray(normal_scores, dtype=np.float64).reshape(-1)
        anom = np.asarray(anomaly_scores, dtype=np.float64).reshape(-1)
        if normal.size == 0 or anom.size == 0:
            return 0.5
        y_true = np.concatenate([np.zeros(normal.size), np.ones(anom.size)])
        y_score = np.concatenate([normal, anom])
        if not np.isfinite(y_score).all() or len(np.unique(y_true)) < 2:
            return 0.5
        try:
            return float(roc_auc_score(y_true=y_true, y_score=y_score))
        except Exception:
            return 0.5

    @staticmethod
    def _tightness_score(normal_scores: np.ndarray) -> float:
        """Scale-invariant tightness of a normal score distribution.

        Returns negative coefficient of variation (std / |mean|) so that a
        higher value means a tighter, more consistent in-distribution model.
        """
        s = np.asarray(normal_scores, dtype=np.float64).reshape(-1)
        if s.size == 0:
            return float("-inf")
        mean = float(np.mean(s))
        std = float(np.std(s))
        cv = std / (abs(mean) + 1e-8)
        return -cv

    # ── Model API ─────────────────────────────────────────────────────────
    def fit(self, data: np.ndarray) -> None:
        """Default fit path delegates to :meth:`fit_new_expert`.

        The strategy normally calls :meth:`fit_new_expert` directly so it
        can pass ``concept_name``; this fallback is kept so the ensemble
        is a drop-in :class:`Model`.
        """
        self.fit_new_expert(data)

    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Standard predict: router selects per sample, returns (y_pred, scores).

        Uses unsupervised routing (no GT). For an oracle path, call
        :meth:`predict_with_oracle`.
        """
        if not self._experts:
            raise RuntimeError("HeterogeneousExpertEnsemble.predict called before any fit_new_expert")

        per_expert_scores = self._score_all_experts(data)
        chosen = self._router.select(per_expert_scores)
        chosen_scores = self._gather_per_sample(per_expert_scores, chosen)
        threshold = float(np.quantile(chosen_scores, 0.95)) if chosen_scores.size > 0 else 0.0
        y_pred = (chosen_scores > threshold).astype(int)
        return y_pred, chosen_scores

    def predict_with_oracle(
        self,
        data: np.ndarray,
        gt_concept_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using OracleRouter behaviour regardless of installed router.

        Bypasses self._router and uses the GT mapping directly so the
        oracle measurement is always available even when the production
        router is unsupervised.
        """
        if gt_concept_name not in self._concept_to_expert_idx:
            raise KeyError(
                f"Oracle predict for concept {gt_concept_name!r} but no expert "
                "was trained on it (concept not in concept_to_expert_idx)."
            )
        expert_idx = self._concept_to_expert_idx[gt_concept_name]
        expert = self._experts[expert_idx]
        return expert.predict(data)

    def score_data(self, data: np.ndarray) -> np.ndarray:
        """Return chosen-expert anomaly scores for routing-aware pipelines."""
        if not self._experts:
            return np.asarray([], dtype=np.float32)
        per_expert_scores = self._score_all_experts(data)
        chosen = self._router.select(per_expert_scores)
        return self._gather_per_sample(per_expert_scores, chosen)

    def score_maps(self, data: np.ndarray, resize_to_input: bool = True) -> np.ndarray:
        """Pixel-level maps from the router-selected expert per sample.

        Routes on image-level scores, then asks each chosen expert for the
        pixel map of its assigned samples. Output shape mirrors the score
        maps returned by each underlying model (``(N, H, W)``).
        """
        if not self._experts:
            return np.asarray([], dtype=np.float32)

        per_expert_scores = self._score_all_experts(data)
        chosen = self._router.select(per_expert_scores)
        return self._gather_maps_per_sample(data, chosen, resize_to_input=resize_to_input)

    def name(self) -> str:
        return "HeterogeneousExpertEnsemble"

    def additional_info(self) -> Dict[str, Any]:
        if self._dynamic_growth:
            mode = "dynamic_growth"
        elif self._cumulative_cl:
            mode = "cumulative_cl"
        else:
            mode = "naive"
        info: Dict[str, Any] = {
            "n_experts": len(self._experts),
            "router": {"name": self._router.name(), **self._router.additional_info()},
            "expert_types_by_task": list(self._expert_type_for_task),
            "concept_to_expert_idx": dict(self._concept_to_expert_idx),
            "expert_type_pool": list(self._type_names),
            "training_mode": mode,
            "tasks_in_history": len(self._task_data_history),
            "expert_assignment": self._expert_assignment,
        }
        if self._expert_assignment == "best":
            info["assignment_val_fraction"] = self._assignment_val_fraction
            info["assignment_decisions"] = list(self._assignment_decisions)
        if self._dynamic_growth:
            info["growth_threshold"] = self._growth_threshold
            info["expert_task_concepts"] = [list(c) for c in self._expert_task_concepts]
            info["dynamic_decisions"] = list(self._dynamic_decisions)
        return info

    # ── helpers used by pixel callbacks / routing-accuracy callback ───────
    @property
    def experts(self) -> List[Model]:
        return list(self._experts)

    @property
    def router(self) -> Router:
        return self._router

    @property
    def concept_to_expert_idx(self) -> Dict[str, int]:
        return dict(self._concept_to_expert_idx)

    def expert_type_for_task(self, task_idx: int) -> str:
        return self._expert_type_for_task[task_idx]

    def per_expert_scores(self, data: np.ndarray) -> np.ndarray:
        """Public accessor used by RoutingAccuracyCallback."""
        return self._score_all_experts(data)

    # ── internals ─────────────────────────────────────────────────────────
    def _score_all_experts(self, data: np.ndarray) -> np.ndarray:
        rows = [self._score_safely(expert, data) for expert in self._experts]
        return np.stack(rows, axis=0)

    @staticmethod
    def _score_safely(expert: Model, data: np.ndarray) -> np.ndarray:
        score_fn = getattr(expert, "score_data", None)
        if callable(score_fn):
            return np.asarray(score_fn(data), dtype=np.float64)
        # Fallback via predict()
        _, scores = expert.predict(data)
        return np.asarray(scores, dtype=np.float64)

    @staticmethod
    def _gather_per_sample(per_expert: np.ndarray, chosen: np.ndarray) -> np.ndarray:
        n_samples = per_expert.shape[1]
        idx_arr = np.asarray(chosen, dtype=np.int64).reshape(-1)
        if idx_arr.shape[0] != n_samples:
            raise ValueError(
                f"chosen length {idx_arr.shape[0]} != n_samples {n_samples}"
            )
        return per_expert[idx_arr, np.arange(n_samples)]

    def _gather_maps_per_sample(
        self,
        data: np.ndarray,
        chosen: np.ndarray,
        *,
        resize_to_input: bool,
    ) -> np.ndarray:
        """For each sample, ask its chosen expert for the pixel map.

        Groups samples by chosen expert and dispatches in batches so each
        expert is invoked at most once.
        """
        chosen = np.asarray(chosen, dtype=np.int64).reshape(-1)
        unique_experts = np.unique(chosen)
        outputs: Dict[int, np.ndarray] = {}
        for ei in unique_experts:
            mask = chosen == ei
            expert_data = data[mask]
            if len(expert_data) == 0:
                continue
            score_maps_fn = getattr(self._experts[int(ei)], "score_maps", None)
            if not callable(score_maps_fn):
                raise AttributeError(
                    f"Expert {ei} ({type(self._experts[int(ei)]).__name__}) does "
                    "not expose score_maps(); pixel-level eval unavailable for this "
                    "expert type."
                )
            outputs[int(ei)] = np.asarray(
                score_maps_fn(expert_data, resize_to_input=resize_to_input)
            )

        # Reassemble in original sample order.
        first = next(iter(outputs.values()))
        out_shape = (chosen.shape[0],) + first.shape[1:]
        result = np.empty(out_shape, dtype=first.dtype)
        cursors = {ei: 0 for ei in outputs}
        for i, ei in enumerate(chosen):
            ei = int(ei)
            result[i] = outputs[ei][cursors[ei]]
            cursors[ei] += 1
        return result

    @staticmethod
    def _freeze_expert(expert: Model) -> None:
        """Best-effort freeze for torch-based experts; no-op otherwise."""
        try:
            import torch.nn as nn  # type: ignore

            for attr_name in ("module", "model", "net", "teacher", "student", "encoder", "decoder"):
                obj = getattr(expert, attr_name, None)
                if isinstance(obj, nn.Module):
                    obj.eval()
                    for p in obj.parameters():
                        p.requires_grad_(False)
        except Exception:
            # If torch isn't available or the model isn't torch-based,
            # nothing to do — memory-bank models are inherently frozen.
            pass

    @staticmethod
    def _unfreeze_expert(expert: Model) -> None:
        """Inverse of _freeze_expert — used before strategy.learn() in dynamic mode."""
        try:
            import torch.nn as nn  # type: ignore

            for attr_name in ("module", "model", "net", "teacher", "student", "encoder", "decoder"):
                obj = getattr(expert, attr_name, None)
                if isinstance(obj, nn.Module):
                    obj.train()
                    for p in obj.parameters():
                        p.requires_grad_(True)
        except Exception:
            pass

    def _subsample_for_calibration(self, data: np.ndarray) -> np.ndarray:
        if data is None or len(data) == 0:
            return np.empty((0,) + (data.shape[1:] if data is not None and data.ndim > 1 else ()))
        if len(data) <= self._calibration_max_samples:
            return data
        idx = np.random.default_rng(seed=0).choice(
            len(data), size=self._calibration_max_samples, replace=False
        )
        return data[idx]


__all__ = ["HeterogeneousExpertEnsemble", "ExpertFactory"]
