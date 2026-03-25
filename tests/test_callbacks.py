"""Tests for callbacks and framework integrations."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from golden_pendulum.callbacks import GoldenPendulumCallback, WeightLogger


class TestWeightLogger:
    def test_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "weights.jsonl"
            wl = WeightLogger(log_file=str(log_path), log_every=1)
            wl.log(0, {"a": 0.3, "b": 0.7})
            wl.log(1, {"a": 0.35, "b": 0.65})

            lines = log_path.read_text().strip().split("\n")
            assert len(lines) == 2
            record = json.loads(lines[0])
            assert record["step"] == 0
            assert "balance_ratio" in record

    def test_log_every_filtering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "weights.jsonl"
            wl = WeightLogger(log_file=str(log_path), log_every=10)
            for i in range(25):
                wl.log(i, {"a": 0.5, "b": 0.5})
            lines = log_path.read_text().strip().split("\n")
            assert len(lines) == 3  # steps 0, 10, 20


class TestGoldenPendulumCallback:
    def test_basic_usage(self):
        model = nn.Linear(10, 3)
        cb = GoldenPendulumCallback(lam=0.5, log_every=1)
        x = torch.randn(4, 10)
        out = model(x)
        losses = {
            "a": nn.functional.mse_loss(out[:, 0], torch.randn(4)),
            "b": nn.functional.mse_loss(out[:, 1], torch.randn(4)),
            "c": nn.functional.mse_loss(out[:, 2], torch.randn(4)),
        }
        weights = cb.on_train_batch(losses, model, batch_idx=0)
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 1e-4
