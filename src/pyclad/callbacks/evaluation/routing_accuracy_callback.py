"""Track per-sample routing decisions of a HeterogeneousExpertEnsemble.

For each evaluated concept, computes:

* the fraction of samples that the production router sent to the
  *correct* expert (i.e. the one trained on this concept), called
  ``routing_accuracy``;
* a per-concept distribution over chosen experts (``routing_distribution``);
* a full confusion matrix
  ``routing_confusion[true_expert_idx][chosen_expert_idx]``;
* the **oracle ceiling** — what the chosen-expert score would have been
  had the router picked the GT expert (just for diagnostic logging via
  the routing distribution; the ensemble's
  :meth:`predict_with_oracle` is the canonical oracle path).

Without this callback you cannot disentangle "the experts are good" from
"the router is good" in the multi-expert experiments, which is the
single most common reviewer complaint.

This callback is **only** useful when the underlying strategy wraps a
:class:`HeterogeneousExpertEnsemble`; for other strategies it remains
silent and emits an empty payload.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from pyclad.callbacks.callback import Callback
from pyclad.data.concept import Concept
from pyclad.output.output_writer import InfoProvider

logger = logging.getLogger(__name__)


class RoutingAccuracyCallback(Callback, InfoProvider):
    """Per-concept routing accuracy + confusion matrix for the ensemble.

    Set ``log_per_sample=True`` and pass ``per_sample_output_path`` to also
    persist a ``.npz`` archive containing, for every (train_step,
    eval_concept) pair:

    * ``<train_key>/<concept>/chosen``           — int (N,) chosen expert per sample
    * ``<train_key>/<concept>/raw_scores``       — float (K, N) per-expert raw scores
    * ``<train_key>/<concept>/normalized_scores``— float (K, N) post-calibration

    plus top-level metadata arrays ``train_order`` and ``test_order``.
    """

    def __init__(
        self,
        strategy: Any,
        *,
        first_seen_step: Optional[Mapping[str, int]] = None,
        log_per_sample: bool = False,
        per_sample_output_path: Optional[Path] = None,
    ) -> None:
        """Parameters
        ----------
        strategy:
            The strategy whose ``_model`` is the
            :class:`HeterogeneousExpertEnsemble`.
        first_seen_step:
            Optional map ``concept_name -> training_step_idx``. When the
            scenario uses a step schedule that groups multiple concepts
            into one training step, the "GT expert" for a test concept is
            the expert that handled its first-seen step (not necessarily
            an expert trained on that concept individually).
        log_per_sample:
            If True, accumulate per-sample (chosen, raw, normalized) and
            write a .npz archive when the scenario completes. Requires
            ``per_sample_output_path``.
        per_sample_output_path:
            Where to write the .npz; if log_per_sample=True this must be
            set. Conventionally derived from the JSON snapshot path by
            replacing ``.json`` with ``.routing.npz``.
        """
        self._strategy = strategy
        self._first_seen_step = dict(first_seen_step or {})
        self._log_per_sample = bool(log_per_sample)
        self._per_sample_output_path = (
            Path(per_sample_output_path) if per_sample_output_path is not None else None
        )
        if self._log_per_sample and self._per_sample_output_path is None:
            raise ValueError(
                "log_per_sample=True requires per_sample_output_path to be set."
            )

        # Per learned-train-concept-key, per evaluated concept, list of dicts:
        #   {'true_expert': int|None, 'chosen': np.ndarray, 'n_samples': int, 'accuracy': float}
        self._records: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        # Per-sample buffer (only used when log_per_sample=True).
        #   self._per_sample[train_key][concept] = {'chosen': arr, 'raw_scores': arr, 'normalized_scores': arr}
        self._per_sample: Dict[str, Dict[str, Dict[str, np.ndarray]]] = defaultdict(dict)
        self._per_sample_written: bool = False
        self._train_order: List[str] = []
        self._test_order: List[str] = []

    # ── lifecycle hooks ───────────────────────────────────────────────────
    def after_training(self, learned_concept: Concept) -> None:
        self._train_order.append(learned_concept.name)

    def after_evaluation(
        self,
        evaluated_concept: Concept,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        anomaly_scores: np.ndarray,
        *args,
        **kwargs,
    ) -> None:
        ensemble = self._extract_ensemble()
        if ensemble is None:
            return

        train_key = self._train_order[-1] if self._train_order else "<unknown>"
        if evaluated_concept.name not in self._test_order:
            self._test_order.append(evaluated_concept.name)

        per_expert_scores = ensemble.per_expert_scores(evaluated_concept.data)
        chosen = ensemble.router.select(per_expert_scores)

        true_expert = self._gt_expert_idx(ensemble, evaluated_concept.name)
        accuracy = (
            float(np.mean(chosen == true_expert)) if true_expert is not None else float("nan")
        )

        self._records[train_key][evaluated_concept.name] = {
            "true_expert": true_expert,
            "chosen_counts": _bincount_to_dict(chosen, ensemble_size=len(ensemble.experts)),
            "n_samples": int(chosen.size),
            "accuracy": accuracy,
        }

        if self._log_per_sample:
            normalize_fn = getattr(ensemble.router, "normalize_scores", None)
            normalized = (
                np.asarray(normalize_fn(per_expert_scores), dtype=np.float32)
                if callable(normalize_fn)
                else np.asarray(per_expert_scores, dtype=np.float32)
            )
            self._per_sample[train_key][evaluated_concept.name] = {
                "chosen": np.asarray(chosen, dtype=np.int32),
                "raw_scores": np.asarray(per_expert_scores, dtype=np.float32),
                "normalized_scores": normalized,
                "true_expert": np.asarray(
                    -1 if true_expert is None else int(true_expert), dtype=np.int32
                ),
            }

    def after_scenario(self) -> None:
        """Flush per-sample buffer to .npz when the scenario finishes."""
        if not self._log_per_sample or self._per_sample_written:
            return
        self._write_per_sample()
        self._per_sample_written = True

    def _write_per_sample(self) -> None:
        if self._per_sample_output_path is None:
            return
        if not self._per_sample:
            logger.info("RoutingAccuracyCallback: nothing to write to %s (no per-sample data)",
                        self._per_sample_output_path)
            return
        self._per_sample_output_path.parent.mkdir(parents=True, exist_ok=True)
        flat: Dict[str, np.ndarray] = {}
        for train_key, per_concept in self._per_sample.items():
            for concept, payload in per_concept.items():
                safe_train = _safe_npz_key(train_key)
                safe_concept = _safe_npz_key(concept)
                prefix = f"{safe_train}__{safe_concept}"
                for field, arr in payload.items():
                    flat[f"{prefix}__{field}"] = arr
        flat["__train_order"] = np.asarray(self._train_order, dtype=object)
        flat["__test_order"] = np.asarray(self._test_order, dtype=object)
        flat["__schema_version"] = np.asarray([1], dtype=np.int32)
        np.savez_compressed(self._per_sample_output_path, **flat)
        logger.info("RoutingAccuracyCallback: per-sample routing data → %s",
                    self._per_sample_output_path)

    # ── snapshot ──────────────────────────────────────────────────────────
    def info(self) -> Dict[str, Any]:
        if not self._records:
            return {"routing_accuracy_callback": {"records": {}}}

        # Build overall confusion matrix + global accuracy at final step.
        ensemble = self._extract_ensemble()
        n_experts = len(ensemble.experts) if ensemble is not None else 0

        final_train_key = self._train_order[-1] if self._train_order else None
        confusion = (
            _build_confusion_matrix(
                records=self._records.get(final_train_key, {}),
                n_experts=n_experts,
            )
            if final_train_key is not None
            else []
        )
        overall_accuracy = _overall_accuracy(self._records.get(final_train_key, {}))

        payload = {
            "routing_accuracy_callback": {
                "router": (
                    ensemble.router.name() if ensemble is not None else "<no-ensemble>"
                ),
                "n_experts": n_experts,
                "final_step": final_train_key,
                "overall_accuracy_at_final_step": overall_accuracy,
                "confusion_matrix_at_final_step": confusion,
                "train_order": list(self._train_order),
                "test_order": list(self._test_order),
                "records": {
                    train_key: dict(per_test)
                    for train_key, per_test in self._records.items()
                },
            }
        }
        if self._log_per_sample and self._per_sample_output_path is not None:
            payload["routing_accuracy_callback"]["per_sample_archive"] = str(
                self._per_sample_output_path
            )
            payload["routing_accuracy_callback"]["per_sample_archive_schema"] = (
                "keys: <train_step>__<concept>__{chosen,raw_scores,normalized_scores,true_expert}; "
                "shapes: chosen (N,), raw/normalized (K, N), true_expert scalar. "
                "Metadata: __train_order, __test_order, __schema_version. "
                "Concept/step names are safe-encoded (non-alnum → '_'); see reader."
            )
        return payload

    # ── internals ─────────────────────────────────────────────────────────
    def _extract_ensemble(self):
        from pyclad.models.vision.heterogeneous_expert_ensemble import (
            HeterogeneousExpertEnsemble,
        )

        model = getattr(self._strategy, "_model", None)
        if isinstance(model, HeterogeneousExpertEnsemble):
            return model
        return None

    def _gt_expert_idx(self, ensemble, concept_name: str) -> Optional[int]:
        # Preferred path: schedule-aware first_seen_step (handles grouped CL).
        if concept_name in self._first_seen_step:
            return int(self._first_seen_step[concept_name])
        # Fallback: ensemble has a direct concept→expert mapping, but in
        # step-schedule mode concept_to_expert_idx is keyed by *training*
        # group name (e.g. ``step_0``), not by original category name.
        mapping = ensemble.concept_to_expert_idx
        if concept_name in mapping:
            return int(mapping[concept_name])
        return None


def _bincount_to_dict(chosen: np.ndarray, ensemble_size: int) -> Dict[str, int]:
    counts = np.bincount(np.asarray(chosen, dtype=np.int64), minlength=ensemble_size)
    return {str(i): int(c) for i, c in enumerate(counts)}


def _build_confusion_matrix(
    records: Mapping[str, Mapping[str, Any]],
    n_experts: int,
) -> List[List[int]]:
    if n_experts == 0:
        return []
    cm = np.zeros((n_experts, n_experts), dtype=np.int64)
    for _concept_name, rec in records.items():
        true = rec.get("true_expert")
        if true is None or true < 0 or true >= n_experts:
            continue
        for chosen_idx_str, count in rec.get("chosen_counts", {}).items():
            try:
                ci = int(chosen_idx_str)
            except (TypeError, ValueError):
                continue
            if 0 <= ci < n_experts:
                cm[true][ci] += int(count)
    return cm.tolist()


def _overall_accuracy(records: Mapping[str, Mapping[str, Any]]) -> float:
    total = 0
    correct = 0
    for rec in records.values():
        true = rec.get("true_expert")
        if true is None:
            continue
        counts = rec.get("chosen_counts", {})
        total += int(rec.get("n_samples", 0))
        correct += int(counts.get(str(true), 0))
    return float(correct / total) if total > 0 else float("nan")


def _safe_npz_key(name: str) -> str:
    """Make a name safe for use as a .npz key segment.

    .npz keys must be valid filenames inside a zip; we replace anything
    non-alphanumeric (besides ``_`` and ``-``) with ``_`` and store the
    original mapping via ``__train_order``/``__test_order`` arrays so a
    reader can recover the originals if needed.
    """
    out_chars = []
    for ch in str(name):
        if ch.isalnum() or ch in ("_", "-"):
            out_chars.append(ch)
        else:
            out_chars.append("_")
    return "".join(out_chars)


__all__ = ["RoutingAccuracyCallback"]
