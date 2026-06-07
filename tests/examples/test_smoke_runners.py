"""
Smoke tests for the visual experiment runner CLI.

These tests verify that the visual runners accept the documented CLI options.
No experiments are run; no datasets or models are loaded.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType
from unittest.mock import patch

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples"
CLVAD_DIR = EXAMPLES_DIR / "clvad"


def _load_module(path: pathlib.Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def base_runner():
    return _load_module(CLVAD_DIR / "run_continual_visual_ad.py", "_smoke_base_runner")


@pytest.fixture(scope="module")
def levels_runner():
    return _load_module(CLVAD_DIR / "run_continual_visual_ad_levels.py", "_smoke_levels_runner")


@pytest.fixture(scope="module")
def multi_expert_runner():
    return _load_module(CLVAD_DIR / "run_multi_expert_v2.py", "_smoke_multi_expert_runner")


class TestBaseRunnerArgs:
    def test_dataset_options_accepted(self, base_runner):
        with patch("sys.argv", [
            "run_continual_visual_ad.py",
            "--model", "patchcore",
            "--benchmark", "mvtec",
            "--categories", "bottle", "cable",
            "--resize", "256",
            "--max-train", "2",
            "--max-test", "3",
        ]):
            args = base_runner.parse_args()
        assert args.categories == ["bottle", "cable"]
        assert args.resize == 256
        assert args.max_train == 2
        assert args.max_test == 3

    def test_default_strategy_is_replay(self, base_runner):
        with patch("sys.argv", [
            "run_continual_visual_ad.py",
            "--model", "patchcore",
            "--benchmark", "mvtec",
        ]):
            args = base_runner.parse_args()
        assert args.strategy == "replay"


class TestLevelsRunnerArgs:
    def test_pixel_options_accepted(self, levels_runner):
        with patch("sys.argv", [
            "run_continual_visual_ad_levels.py",
            "--model", "patchcore",
            "--benchmark", "mvtec",
            "--eval-level", "both",
            "--pixel-threshold-mode", "train-quantile",
            "--pixel-threshold-quantile", "0.99",
            "--no-pixel-skip-missing-masks",
        ]):
            args = levels_runner.parse_args()
        assert args.eval_level == "both"
        assert args.pixel_threshold_mode == "train-quantile"
        assert args.pixel_threshold_quantile == 0.99
        assert args.pixel_skip_missing_masks is False


class TestMultiExpertRunnerArgs:
    def test_assignment_nsr_options_accepted(self, multi_expert_runner):
        with patch("sys.argv", [
            "run_multi_expert_v2.py",
            "--expert-types", "padim,patchcore,cfa",
            "--expert-assignment", "best",
            "--router", "min_score",
            "--step-schedule", "10-1x5",
            "--benchmark", "mvtec",
            "--eval-level", "both",
        ]):
            args = multi_expert_runner.parse_args()
        assert args.expert_types == ["padim", "patchcore", "cfa"]
        assert args.expert_assignment == "best"
        assert args.router == "min_score"
        assert args.step_schedule == "10-1x5"
        assert args.eval_level == "both"
        assert args.strategy == "multi_expert"
