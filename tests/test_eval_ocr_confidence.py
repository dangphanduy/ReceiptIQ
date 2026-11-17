import json
import sys
import types

import eval_ocr_confidence as cli
import eval_ocr_detection as det
import pytest
from PIL import Image

A, B, C = [0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]


def gt(key, boxes_words, width=100, height=50):
    """boxes_words: [(bbox, text)] -> an in-memory SROIE row."""
    return {
        "key": key,
        "words": [w for _, w in boxes_words],
        "bboxes": [b for b, _ in boxes_words],
        "image_size": {"width": width, "height": height},
    }


def write_ocr(ocr_dir, key, lines, seconds, device="gpu:0", schema="1.0"):
    """lines: [(bbox, text, confidence)]."""
    ocr_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": schema,
        "image": {"path": f"sroie:test/{key}", "width": 100, "height": 50},
        "engine": {"name": "paddleocr", "device": device, "inference_seconds": seconds},
        "lines": [{"text": t, "confidence": c, "bbox": b} for b, t, c in lines],
    }
    (ocr_dir / f"{key}.json").write_text(json.dumps(doc), encoding="utf-8")


GT_ROWS = [
    gt("K1", [(A, "TOTAL"), (B, "9")]),
    gt("K2", [(A, "RM 1.00")]),
    gt("K3", [(A, "missing")]),
]


def write_fixture(ocr_dir, latency_dir):
    write_ocr(ocr_dir, "K1", [(A, "total", 0.99), (B, "9, Jalan", 0.6)], 2.0)
    # C matches no GT box: it is not a line pair, so its confidence is not analysed
    write_ocr(ocr_dir, "K2", [(A, "RM 1.0O", 0.7), (C, "noise", 0.1)], 1.0)
    write_ocr(latency_dir, "K1", [], 30.0, device="cpu")
    write_ocr(latency_dir, "K2", [], 10.0, device="cpu")


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """data/sroie manifest + fake datasets module; returns (paths, selected columns)."""
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
    paths = {
        "ocr": tmp_path / "ocr",
        "lat": tmp_path / "lat",
        "out": tmp_path / "eval" / "confidence_test.json",
        "chart": tmp_path / "eval" / "confidence_hist_test.png",
    }
    return paths, selected


def run_cli(monkeypatch, paths):
    argv = ["eval_ocr_confidence.py", "--split", "test", "--ocr-dir", str(paths["ocr"])]
    argv += ["--latency-dir", str(paths["lat"]), "--out", str(paths["out"])]
    argv += ["--chart", str(paths["chart"])]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()


def test_cli_writes_report_and_chart(cli_env, monkeypatch, capsys):
    paths, selected = cli_env
    write_fixture(paths["ocr"], paths["lat"])

    run_cli(monkeypatch, paths)

    report = json.loads(paths["out"].read_text(encoding="utf-8"))
    assert selected == [["key", "words", "bboxes", "image_size"]]  # never decodes images
    assert list(report) == [
        "split", "sroie_revision", "iou_threshold", "ocr_dir", "ocr_device",
        "ocr_schema_version", "size_mismatch_keys", "latency", "counts", "confidence",
        "missing_keys",
    ]  # fmt: skip
    assert report["sroie_revision"] == "abc123"
    assert report["ocr_device"] == "gpu:0"
    assert report["ocr_schema_version"] == "1.0"
    assert report["size_mismatch_keys"] == []
    assert report["missing_keys"] == ["K3"]
    assert report["counts"] == {"pairs": 3, "correct": 1, "wrong": 2, "comma_cut_removed": 1}

    gpu, cpu = report["latency"]["gpu"], report["latency"]["cpu"]
    assert gpu["device"] == "gpu:0" and cpu["device"] == "cpu"
    assert gpu["dir"] == str(paths["ocr"]) and cpu["dir"] == str(paths["lat"])
    assert gpu["stats"]["n"] == 2 and gpu["stats"]["mean"] == 1.5
    assert gpu["first_image_seconds"] == 2.0  # K1, first in split order
    assert gpu["stats_excluding_first"]["n"] == 1 and gpu["stats_excluding_first"]["max"] == 1.0
    assert cpu["first_image_seconds"] == 30.0 and cpu["stats"]["median"] == 20.0

    view_all, clean = report["confidence"]["all"], report["confidence"]["clean"]
    assert view_all["correct"] == {"mean": 0.99, "median": 0.99, "p05": 0.99}
    assert view_all["wrong"]["median"] == pytest.approx(0.65)
    assert view_all["wrong"]["p05"] == 0.6
    assert clean["wrong"] == {"mean": 0.7, "median": 0.7, "p05": 0.7}

    hist = view_all["histogram"]
    assert hist["edges"][:3] == [0.0, 0.8, 0.81] and hist["edges"][-1] == 1.0
    assert len(hist["correct"]) == len(hist["edges"]) - 1 == 21
    assert hist["wrong"][0] == 2 and hist["correct"][-1] == 1

    assert len(view_all["sweep"]) == 59
    assert list(view_all) == ["correct", "wrong", "histogram", "sweep", "candidate", "budget"]
    # every wrong line is < 0.71 and the correct one is 0.99: 0.71 is the lowest perfect t
    assert view_all["candidate"] == {
        "t": 0.71, "precision": 1.0, "recall": 1.0, "f1": 1.0, "flagged_rate": 2 / 3
    }  # fmt: skip
    assert clean["candidate"]["t"] == 0.71 and clean["candidate"]["flagged_rate"] == 0.5
    # budget: highest t flagging <= 15% of lines; with 3 / 2 lines that means flagging none
    assert view_all["budget"] == {
        "t": 0.6, "precision": None, "recall": 0.0, "f1": 0.0,
        "flagged_rate": 0.0, "max_flagged_rate": 0.15,
    }  # fmt: skip
    assert clean["budget"]["t"] == 0.7

    with Image.open(paths["chart"]) as img:
        assert img.format == "PNG" and img.width >= 800
    assert not list(paths["out"].parent.glob("*.tmp"))
    assert "candidate" in capsys.readouterr().out


def test_cli_without_latency_dir_reports_cpu_null(cli_env, monkeypatch, capsys):
    paths, _ = cli_env
    write_fixture(paths["ocr"], paths["lat"])
    paths["lat"] = paths["lat"].parent / "does-not-exist"

    run_cli(monkeypatch, paths)

    report = json.loads(paths["out"].read_text(encoding="utf-8"))
    assert report["latency"]["cpu"] is None
    assert "no latency JSON" in capsys.readouterr().out


def test_cli_fails_naming_file_with_bad_confidence(cli_env, monkeypatch):
    paths, _ = cli_env
    write_fixture(paths["ocr"], paths["lat"])
    write_ocr(paths["ocr"], "K2", [(A, "RM 1.00", "high")], 1.0)

    with pytest.raises(SystemExit, match="K2.json"):
        run_cli(monkeypatch, paths)
    assert not paths["out"].exists()


@pytest.mark.parametrize("conf", [1.5, -0.1, True])
def test_evaluate_refuses_confidence_outside_unit_interval(tmp_path, conf):
    write_ocr(tmp_path, "K1", [(A, "x", conf)], 1.0)

    with pytest.raises(ValueError, match="K1.json"):
        cli.evaluate([gt("K1", [(A, "x")])], tmp_path, tmp_path / "none", 0.5)


def test_evaluate_refuses_bad_inference_seconds(tmp_path):
    write_ocr(tmp_path, "K1", [(A, "x", 0.9)], None)

    with pytest.raises(ValueError, match="K1.json"):
        cli.evaluate([gt("K1", [(A, "x")])], tmp_path, tmp_path / "none", 0.5)


def test_evaluate_names_file_for_non_string_device(tmp_path):
    write_ocr(tmp_path, "K1", [(A, "x", 0.9)], 1.0, device=4)

    with pytest.raises(ValueError, match="K1.json"):
        cli.evaluate([gt("K1", [(A, "x")])], tmp_path, tmp_path / "none", 0.5)


def test_evaluate_refuses_mixed_devices(tmp_path):
    write_ocr(tmp_path, "K1", [(A, "x", 0.9)], 1.0, device="gpu:0")
    write_ocr(tmp_path, "K2", [(A, "x", 0.9)], 1.0, device="cpu")

    with pytest.raises(ValueError, match="devices"):
        cli.evaluate([gt("K1", [(A, "x")]), gt("K2", [(A, "x")])], tmp_path, tmp_path / "x", 0.5)


def test_evaluate_refuses_gpu_latency_sample(tmp_path):
    ocr_dir, latency_dir = tmp_path / "ocr", tmp_path / "latency"
    write_ocr(ocr_dir, "K1", [(A, "x", 0.9)], 1.0)
    write_ocr(latency_dir, "K1", [], 2.0, device="gpu:0")

    with pytest.raises(ValueError, match="CPU"):
        cli.evaluate([gt("K1", [(A, "x")])], ocr_dir, latency_dir, 0.5)


def test_evaluate_refuses_cpu_primary_dir_labeled_gpu(tmp_path):
    write_ocr(tmp_path, "K1", [(A, "x", 0.9)], 1.0, device="cpu")

    with pytest.raises(ValueError, match="GPU"):
        cli.evaluate([gt("K1", [(A, "x")])], tmp_path, tmp_path / "none", 0.5)


def test_evaluate_reports_cpu_null_for_unrelated_json(tmp_path):
    ocr_dir, latency_dir = tmp_path / "ocr", tmp_path / "latency"
    write_ocr(ocr_dir, "K1", [(A, "x", 0.9)], 1.0)
    latency_dir.mkdir()
    (latency_dir / "notes.json").write_text("{}", encoding="utf-8")

    report = cli.evaluate([gt("K1", [(A, "x")])], ocr_dir, latency_dir, 0.5)

    assert report["latency"]["cpu"] is None


def test_evaluate_reports_size_mismatch(tmp_path):
    write_ocr(tmp_path, "K1", [(A, "x", 0.9)], 1.0)

    report = cli.evaluate([gt("K1", [(A, "x")], width=101)], tmp_path, tmp_path / "none", 0.5)

    assert report["size_mismatch_keys"] == ["K1"]


def test_evaluate_zero_matched_pairs_has_no_candidate(tmp_path):
    write_ocr(tmp_path, "K1", [(A, "x", 0.9)], 1.0)

    report = cli.evaluate([gt("K1", [(B, "x")])], tmp_path, tmp_path / "none", 0.5)

    assert report["counts"] == {"pairs": 0, "correct": 0, "wrong": 0, "comma_cut_removed": 0}
    assert report["confidence"]["all"]["candidate"] is None
    assert report["confidence"]["clean"]["candidate"] is None


def test_evaluate_refuses_gt_length_mismatch(tmp_path):
    write_ocr(tmp_path, "K1", [(A, "x", 0.9)], 1.0)
    row = gt("K1", [(A, "x")])
    row["words"].append("extra")

    with pytest.raises(ValueError, match="K1"):
        cli.evaluate([row], tmp_path, tmp_path / "none", 0.5)


def test_cli_refuses_ocr_dir_without_results(cli_env, monkeypatch):
    paths, _ = cli_env
    paths["ocr"].mkdir()
    (paths["ocr"] / "notes.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="no OCR JSON"):
        run_cli(monkeypatch, paths)
    assert not paths["out"].exists()
