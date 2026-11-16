"""--device plumbing and paddle-distribution detection. Never imports the real paddle."""

import argparse
import sys
import types
from importlib.metadata import PackageNotFoundError

import pytest

import ml.ocr_engine as engine_module
from ml.ocr_engine import OCR_CONFIG, create_engine, engine_versions, parse_device

PADDLE_DISTS = {"paddlepaddle", "paddlepaddle-gpu"}


def fake_version(installed):
    def version(dist):
        if dist == "paddleocr":
            return "3.7.0"
        if dist in installed:
            return installed[dist]
        if dist in PADDLE_DISTS:
            raise PackageNotFoundError(dist)
        raise AssertionError(f"unexpected distribution {dist}")

    return version


@pytest.mark.parametrize(
    "installed",
    [{"paddlepaddle": "3.3.1"}, {"paddlepaddle-gpu": "3.3.1.post126"}],
)
def test_engine_versions_reports_installed_paddle_distribution(monkeypatch, installed):
    monkeypatch.setattr(engine_module, "version", fake_version(installed))

    ((dist, ver),) = installed.items()
    assert engine_versions() == {
        "paddleocr_version": "3.7.0",
        "paddlepaddle_distribution": dist,
        "paddlepaddle_version": ver,
    }


@pytest.mark.parametrize(
    "installed", [{}, {"paddlepaddle": "3.3.1", "paddlepaddle-gpu": "3.3.1"}]
)
def test_engine_versions_refuses_none_or_both_paddle_distributions(monkeypatch, installed):
    monkeypatch.setattr(engine_module, "version", fake_version(installed))

    with pytest.raises(RuntimeError, match="exactly one"):
        engine_versions()


@pytest.mark.parametrize("value", ["cpu", "gpu:0", "gpu:12"])
def test_parse_device_accepts_cpu_and_gpu_index(value):
    assert parse_device(value) == value


@pytest.mark.parametrize("value", ["", "gpu", "gpu:", "gpu:-1", "cuda:0", "GPU:0", "cpu ", "xpu:0"])
def test_parse_device_rejects_anything_else(value):
    with pytest.raises(argparse.ArgumentTypeError, match="gpu:<int>"):
        parse_device(value)


@pytest.fixture
def fake_paddle(monkeypatch):
    """Fake paddle + paddleocr modules; returns (state, PaddleOCR calls)."""
    state = {"cuda": True, "gpus": 1}
    calls = []

    paddle = types.ModuleType("paddle")
    paddle.device = types.SimpleNamespace(
        is_compiled_with_cuda=lambda: state["cuda"],
        cuda=types.SimpleNamespace(device_count=lambda: state["gpus"]),
    )
    paddleocr = types.ModuleType("paddleocr")
    paddleocr.PaddleOCR = lambda **kwargs: calls.append(kwargs) or "engine"
    monkeypatch.setitem(sys.modules, "paddle", paddle)
    monkeypatch.setitem(sys.modules, "paddleocr", paddleocr)
    return state, calls


@pytest.mark.parametrize("device", ["cpu", "gpu:0"])
def test_create_engine_passes_device_with_ocr_config(fake_paddle, device):
    _, calls = fake_paddle

    assert create_engine(device) == "engine"
    assert calls == [{**OCR_CONFIG, "device": device}]


def test_create_engine_defaults_to_cpu(fake_paddle):
    _, calls = fake_paddle
    create_engine()
    assert calls[0]["device"] == "cpu"


def test_create_engine_refuses_gpu_without_cuda_build(fake_paddle):
    state, calls = fake_paddle
    state["cuda"] = False

    with pytest.raises(RuntimeError, match="paddlepaddle-gpu"):
        create_engine("gpu:0")
    assert calls == []


def test_create_engine_refuses_missing_gpu_index(fake_paddle):
    _, calls = fake_paddle

    with pytest.raises(RuntimeError, match="gpu:1"):
        create_engine("gpu:1")
    assert calls == []


def test_create_engine_rejects_malformed_device(fake_paddle):
    with pytest.raises(ValueError, match="gpu:<int>"):
        create_engine("cuda")


def test_baseline_cli_passes_device_to_engine_and_document(tmp_path, monkeypatch):
    import run_ocr_baseline as baseline
    from PIL import Image

    image = tmp_path / "r.png"
    Image.new("RGB", (8, 6), "white").save(image)
    seen = {}

    def fake_create_engine(device):
        seen["create"] = device
        return "engine"

    def fake_run_ocr(engine, image_bytes, image_path, *, device):
        seen["run"] = device
        return {"num_lines": 0, "mean_confidence": None, "engine": {"inference_seconds": 0.0}}

    monkeypatch.setattr(baseline, "create_engine", fake_create_engine)
    monkeypatch.setattr(baseline, "run_ocr", fake_run_ocr)
    argv = ["run_ocr_baseline.py", str(image), "--out-dir", str(tmp_path / "o"), "--device", "gpu:0"]
    monkeypatch.setattr(sys, "argv", argv)

    baseline.main()

    assert seen == {"create": "gpu:0", "run": "gpu:0"}


def test_baseline_cli_rejects_bad_device(monkeypatch, capsys):
    import run_ocr_baseline as baseline

    monkeypatch.setattr(sys, "argv", ["run_ocr_baseline.py", "x.jpg", "--device", "cuda"])

    with pytest.raises(SystemExit) as exc_info:
        baseline.main()

    assert exc_info.value.code == 2
    assert "gpu:<int>" in capsys.readouterr().err
