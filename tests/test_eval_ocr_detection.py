import json
import sys
import types

import eval_ocr_detection as cli
import pytest

A, B = [0, 0, 10, 10], [20, 0, 30, 10]


def gt(key, bboxes, width=100, height=50):
    return {"key": key, "bboxes": bboxes, "image_size": {"width": width, "height": height}}


def write_ocr(ocr_dir, key, bboxes, width=100, height=50, device="gpu:0", schema="1.0"):
    ocr_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": schema,
        "image": {"path": f"sroie:test/{key}", "width": width, "height": height},
        "engine": {"name": "paddleocr", "device": device},
        "lines": [{"text": "x", "confidence": 0.9, "bbox": b} for b in bboxes],
    }
    (ocr_dir / f"{key}.json").write_text(json.dumps(doc), encoding="utf-8")


def test_keys_without_ocr_are_listed_not_counted_as_misses(tmp_path):
    write_ocr(tmp_path, "K1", [A])

    result = cli.evaluate([gt("K1", [A]), gt("K2", [A, B])], tmp_path, 0.5)

    assert result["counts"] == {"evaluated": 1, "missing": 1}
    assert result["missing_keys"] == ["K2"]
    assert result["micro"] == {
        "tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0
    }  # fmt: skip
    assert [row["key"] for row in result["per_image"]] == ["K1"]


def test_per_image_rows(tmp_path):
    write_ocr(tmp_path, "K1", [A, [50, 0, 60, 10]])

    (row,) = cli.evaluate([gt("K1", [A, B])], tmp_path, 0.5)["per_image"]

    assert row == {
        "key": "K1", "tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5, "f1": 0.5
    }  # fmt: skip


def test_size_mismatch_is_listed_but_still_evaluated(tmp_path):
    write_ocr(tmp_path, "K1", [A], width=99)
    write_ocr(tmp_path, "K2", [A])

    result = cli.evaluate([gt("K1", [A]), gt("K2", [A])], tmp_path, 0.5)

    assert result["size_mismatch_keys"] == ["K1"]
    assert result["counts"]["evaluated"] == 2
    assert result["micro"]["tp"] == 2


def test_image_with_no_ocr_lines_counts_all_gt_as_misses(tmp_path):
    write_ocr(tmp_path, "K1", [])

    (row,) = cli.evaluate([gt("K1", [A, B])], tmp_path, 0.5)["per_image"]

    assert (row["tp"], row["fp"], row["fn"], row["precision"]) == (0, 0, 2, None)


def test_mixed_ocr_schema_versions_are_refused(tmp_path):
    write_ocr(tmp_path, "K1", [A], schema="1.0")
    write_ocr(tmp_path, "K2", [A], schema="2.0")

    with pytest.raises(ValueError, match="schema"):
        cli.evaluate([gt("K1", [A]), gt("K2", [A])], tmp_path, 0.5)


def test_unsafe_gt_key_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unsafe key"):
        cli.evaluate([gt("../K1", [A])], tmp_path, 0.5)


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """data/sroie manifest + fake datasets module; returns (ocr_dir, out, selected columns)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(
        json.dumps({"repo_id": "jsdnrs/ICDAR2019-SROIE", "commit_sha": "abc123"}), encoding="utf-8"
    )
    selected = []

    class FakeSplit(list):
        def select_columns(self, columns):
            selected.append(columns)
            return self

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_from_disk = lambda path: {
        "test": FakeSplit([gt("K1", [A, B]), gt("K2", [A])])
    }
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setattr(cli, "DATA_DIR", data_dir)
    return tmp_path / "ocr", tmp_path / "eval" / "detection_test.json", selected


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["eval_ocr_detection.py", *args])
    cli.main()


def test_cli_writes_report_with_provenance(cli_env, monkeypatch):
    ocr_dir, out, selected = cli_env
    write_ocr(ocr_dir, "K1", [A])
    manifest = {"ocr": {"device": "gpu:0", "paddlepaddle_distribution": "paddlepaddle-gpu"}}
    (ocr_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    run_cli(monkeypatch, "--split", "test", "--ocr-dir", str(ocr_dir), "--out", str(out))

    report = json.loads(out.read_text(encoding="utf-8"))
    assert selected == [["key", "bboxes", "image_size"]]  # never decodes images
    assert report["iou_threshold"] == 0.5
    assert report["split"] == "test"
    assert report["sroie_revision"] == "abc123"
    assert report["ocr_schema_version"] == "1.0"
    assert report["ocr_device"] == "gpu:0"
    assert report["counts"] == {"evaluated": 1, "missing": 1}
    assert report["missing_keys"] == ["K2"]
    assert report["micro"]["tp"] == 1 and report["micro"]["fn"] == 1
    assert report["per_image"][0]["key"] == "K1"
    assert not list(out.parent.glob("*.tmp"))


def test_cli_without_ocr_manifest_takes_device_from_ocr_json(cli_env, monkeypatch):
    ocr_dir, out, _ = cli_env
    write_ocr(ocr_dir, "K1", [A], device="gpu:0")

    run_cli(monkeypatch, "--ocr-dir", str(ocr_dir), "--out", str(out), "--iou", "0.7")

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["ocr_device"] == "gpu:0"
    assert report["iou_threshold"] == 0.7


def test_cli_refuses_mixed_devices(cli_env, monkeypatch):
    ocr_dir, out, _ = cli_env
    write_ocr(ocr_dir, "K1", [A], device="gpu:0")
    write_ocr(ocr_dir, "K2", [A], device="cpu")

    with pytest.raises(SystemExit, match="device"):
        run_cli(monkeypatch, "--ocr-dir", str(ocr_dir), "--out", str(out))
    assert not out.exists()


def test_cli_refuses_ocr_dir_without_results(cli_env, monkeypatch):
    ocr_dir, out, _ = cli_env
    ocr_dir.mkdir()

    with pytest.raises(SystemExit, match="no OCR JSON"):
        run_cli(monkeypatch, "--ocr-dir", str(ocr_dir), "--out", str(out))
    assert not out.exists()


@pytest.mark.parametrize("value", ["0", "-0.1", "1.5", "abc"])
def test_cli_rejects_iou_outside_0_1(monkeypatch, capsys, value):
    with pytest.raises(SystemExit) as exc_info:
        run_cli(monkeypatch, "--iou", value)
    assert exc_info.value.code == 2
