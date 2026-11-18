import json
import sys
import types

import eval_ocr_detection as det
import ocr_error_report as cli
import pytest

A, B, C = [0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]
GT_ROWS = [
    {
        "key": "K1",
        "words": ["TEL : 03", "9", "CHANGE :"],
        "bboxes": [A, B, C],
        "image_size": {"width": 100, "height": 50},
    },
    {"key": "K2", "words": ["x"], "bboxes": [A], "image_size": {"width": 100, "height": 50}},
]


def write_ocr(ocr_dir, key, lines):
    ocr_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": "1.0",
        "image": {"path": f"sroie:test/{key}", "width": 100, "height": 50},
        "engine": {"name": "paddleocr", "device": "gpu:0", "inference_seconds": 1.0},
        "lines": [{"text": t, "confidence": 0.9, "bbox": b} for b, t in lines],
    }
    (ocr_dir / f"{key}.json").write_text(json.dumps(doc), encoding="utf-8")


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(json.dumps({"commit_sha": "abc123"}), encoding="utf-8")
    selected = []

    class FakeSplit(list):
        def select_columns(self, columns):
            selected.append(columns)
            return self

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_from_disk = lambda path: {"test": FakeSplit(GT_ROWS)}
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setattr(det, "DATA_DIR", data_dir)
    return tmp_path, selected


def run_cli(monkeypatch, ocr_dir, out):
    argv = ["ocr_error_report.py", "--split", "test", "--ocr-dir", str(ocr_dir), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()


def test_cli_writes_report_and_prints_counts(cli_env, monkeypatch, capsys):
    tmp_path, selected = cli_env
    ocr_dir, out = tmp_path / "ocr", tmp_path / "eval" / "ocr_errors_test.json"
    write_ocr(ocr_dir, "K1", [(A, "Tel:03"), (B, "9, Jalan"), (C, "Change")])

    run_cli(monkeypatch, ocr_dir, out)

    report = json.loads(out.read_text(encoding="utf-8"))
    assert selected == [["key", "words", "bboxes", "image_size"]]  # never decodes images
    assert report["sroie_revision"] == "abc123"
    assert report["ocr_device"] == "gpu:0"
    assert report["missing_keys"] == ["K2"]
    assert report["counts"]["wrong"] == 3
    assert report["counts"]["excluded_comma_cut"] == 1
    assert [c["category"] for c in report["categories"]] == ["punctuation", "spacing"]
    printed = capsys.readouterr().out
    assert "spacing" in printed and "K1" in printed


def test_cli_exits_without_any_ocr_json(cli_env, monkeypatch):
    tmp_path, _ = cli_env
    out = tmp_path / "out.json"
    with pytest.raises(SystemExit, match="no OCR JSON"):
        run_cli(monkeypatch, tmp_path / "empty", out)
    assert not out.exists()
