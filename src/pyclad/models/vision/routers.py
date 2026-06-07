"""Router strategies for HeterogeneousExpertEnsemble.

A Router takes per-expert anomaly scores for a batch of samples and
returns, per sample, the index of the expert whose score should be used
as the final prediction.

Three strategies are provided:

* :class:`OracleRouter`        — uses ground-truth task identity (upper
                                  bound; needs concept_name → expert_idx
                                  mapping registered by the ensemble);
* :class:`MinAnomalyScoreRouter` — unsupervised; picks the expert with
                                    the lowest *z-score-normalized*
                                    anomaly score ("most confident this
                                    is in-distribution");
* :class:`MaxConfidenceRouter` — alias of MinAnomalyScoreRouter with a
                                  semantically clearer name for
                                  reconstruction-based experts (where
                                  the anomaly score is the reconstruction
                                  error).

All routers expose:

    select(per_expert_scores, *, gt_expert_idx=None) -> chosen_expert_idx

where ``per_expert_scores`` has shape ``(n_experts, n_samples)`` and the
return is shape ``(n_samples,)``. Unsupervised routers also expose
:meth:`calibrate` so the ensemble can pre-register each expert's score
distribution on validation normals (needed because different model
families produce scores on wildly different scales).
"""

from __future__ import annotations

import abc
from typing import Dict, List, Optional

import numpy as np


class Router(abc.ABC):
    """Per-sample expert selection given a per-expert score matrix."""

    @abc.abstractmethod
    def select(
        self,
        per_expert_scores: np.ndarray,
        *,
        gt_expert_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Return an array of length ``n_samples`` with chosen expert index per sample.

        Parameters
        ----------
        per_expert_scores:
            Array of shape ``(n_experts, n_samples)`` containing the
            anomaly score that each expert assigns to each sample.
        gt_expert_idx:
            Ground-truth expert index for the current batch (used by
            :class:`OracleRouter`; ignored by unsupervised routers).
        """

    def calibrate(self, expert_idx: int, normal_scores: np.ndarray) -> None:  # noqa: D401
        """Optional per-expert calibration on validation normals.

        Default no-op. Unsupervised routers override this.
        """

    @abc.abstractmethod
    def name(self) -> str: ...

    def additional_info(self) -> Dict:
        return {}


class OracleRouter(Router):
    """Always picks the expert whose ``gt_expert_idx`` is supplied at predict time.

    Acts as an upper bound: assumes the system has perfect task
    identification. Useful for measuring the ceiling of the multi-expert
    approach and isolating the cost of unsupervised routing.

    If ``gt_expert_idx`` is None (no ground-truth available), falls back
    to expert 0 with a one-time warning — this is intentionally bad so
    misuse is loud.
    """

    def __init__(self) -> None:
        self._missing_gt_warned = False

    def select(
        self,
        per_expert_scores: np.ndarray,
        *,
        gt_expert_idx: Optional[int] = None,
    ) -> np.ndarray:
        n_experts, n_samples = per_expert_scores.shape
        if gt_expert_idx is None:
            if not self._missing_gt_warned:
                import warnings

                warnings.warn(
                    "OracleRouter.select() called without gt_expert_idx; "
                    "falling back to expert 0. This is almost certainly a "
                    "wiring bug — the runner/callback should propagate the "
                    "ground-truth expert index for each evaluated concept.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._missing_gt_warned = True
            return np.zeros(n_samples, dtype=np.int64)

        gt = int(gt_expert_idx)
        if gt < 0 or gt >= n_experts:
            raise IndexError(
                f"OracleRouter received gt_expert_idx={gt} but only "
                f"{n_experts} expert(s) are present."
            )
        return np.full(n_samples, gt, dtype=np.int64)

    def name(self) -> str:
        return "OracleRouter"


class MinAnomalyScoreRouter(Router):
    """Picks the expert with the lowest z-score-normalized anomaly score per sample.

    Intuition: each frozen expert produces low scores on samples that
    look like its training distribution. Without task IDs, the sample is
    "claimed" by whichever expert finds it most normal. Scores are
    normalized per-expert because different model families (memory-bank,
    flow, reconstruction) produce wildly different score scales.

    Calibration is required: call :meth:`calibrate` once per expert after
    it has been fit, passing a vector of anomaly scores on a validation
    set of *normal* samples. The router stores per-expert ``(mean, std)``
    and applies z-score at selection time. Without calibration, the
    raw scores are used (degenerates to "pick expert with widest score
    distribution").
    """

    def __init__(self) -> None:
        self._calibration: Dict[int, tuple[float, float]] = {}

    def calibrate(self, expert_idx: int, normal_scores: np.ndarray) -> None:
        scores = np.asarray(normal_scores, dtype=np.float64).reshape(-1)
        if scores.size == 0:
            # No calibration data; leave as zero-mean unit-std.
            self._calibration[int(expert_idx)] = (0.0, 1.0)
            return
        mean = float(scores.mean())
        std = float(scores.std())
        if std < 1e-8:
            std = 1.0
        self._calibration[int(expert_idx)] = (mean, std)

    def select(
        self,
        per_expert_scores: np.ndarray,
        *,
        gt_expert_idx: Optional[int] = None,
    ) -> np.ndarray:
        normalized = self.normalize_scores(per_expert_scores)
        return np.argmin(normalized, axis=0).astype(np.int64)

    def normalize_scores(self, per_expert_scores: np.ndarray) -> np.ndarray:
        """Per-expert z-score normalization using stored calibration.

        Returns array of the same shape as input. Used by routing-analysis
        callbacks that want to inspect *why* a particular expert won.
        """
        normalized = np.empty_like(per_expert_scores, dtype=np.float64)
        for i in range(per_expert_scores.shape[0]):
            mean, std = self._calibration.get(i, (0.0, 1.0))
            normalized[i] = (per_expert_scores[i] - mean) / std
        return normalized

    def name(self) -> str:
        return "MinAnomalyScoreRouter"

    def additional_info(self) -> Dict:
        return {
            "calibration": {
                str(k): {"mean": v[0], "std": v[1]}
                for k, v in sorted(self._calibration.items())
            }
        }


class MaxConfidenceRouter(MinAnomalyScoreRouter):
    """Alias of :class:`MinAnomalyScoreRouter` with reconstruction-style naming.

    Semantically identical: the expert with the lowest anomaly score is
    the most confident the sample is in-distribution.
    """

    def name(self) -> str:
        return "MaxConfidenceRouter"


def build_router(name: str) -> Router:
    """Construct a router by short name. Used by the runner."""
    name = name.lower().strip()
    if name in {"oracle"}:
        return OracleRouter()
    if name in {"min_score", "minscore", "min_anomaly_score"}:
        return MinAnomalyScoreRouter()
    if name in {"max_conf", "maxconf", "max_confidence"}:
        return MaxConfidenceRouter()
    raise ValueError(
        f"Unknown router '{name}' (expected: oracle | min_score | max_conf)"
    )


__all__: List[str] = [
    "Router",
    "OracleRouter",
    "MinAnomalyScoreRouter",
    "MaxConfidenceRouter",
    "build_router",
]
