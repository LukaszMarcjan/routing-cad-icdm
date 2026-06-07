"""Thin strategy that delegates to HeterogeneousExpertEnsemble.

Implements :class:`ConceptIncrementalStrategy` so it plugs into the
existing :class:`ConceptIncrementalScenario` without changes.

The scenario calls ``strategy.learn(data=train_concept.data)`` without
the concept name. To still record GT concept→expert mappings (needed
for the oracle path and for routing-accuracy logging), this strategy
exposes :meth:`set_pending_concept_name`, which the runner wires up via
a tiny ``before_concept_processing`` callback. If the callback isn't
installed, the ensemble falls back to ``task_<idx>`` placeholder names
and the oracle path becomes unavailable.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from pyclad.models.vision.heterogeneous_expert_ensemble import (
    HeterogeneousExpertEnsemble,
)
from pyclad.strategies.strategy import ConceptIncrementalStrategy


class MultiExpertStrategy(ConceptIncrementalStrategy):
    def __init__(self, model: HeterogeneousExpertEnsemble) -> None:
        if not isinstance(model, HeterogeneousExpertEnsemble):
            raise TypeError(
                "MultiExpertStrategy requires HeterogeneousExpertEnsemble, "
                f"got {type(model).__name__}"
            )
        self._model = model

    def set_pending_concept_name(self, concept_name: str) -> None:
        self._model.set_pending_concept_name(concept_name)

    def learn(self, data: np.ndarray, **kwargs) -> None:
        concept_name = kwargs.get("concept_name")
        self._model.fit_new_expert(data, concept_name=concept_name)

    def predict(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return self._model.predict(data)

    # Note: pixel callback's _resolve_model_context falls back to
    # `getattr(strategy, '_model', None)` — already exposed via __init__.

    def name(self) -> str:
        router_name = self._model.router.name()
        return f"MultiExpert[{router_name}]"

    def additional_info(self) -> Dict[str, Any]:
        return {
            "model": self._model.additional_info(),
        }
