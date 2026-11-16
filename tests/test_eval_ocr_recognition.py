import json
import sys
import types

import eval_ocr_detection as det
import eval_ocr_recognition as cli
import pytest

A, B = [0, 0, 10, 10], [20, 0, 30, 10]


def gt(key, lines, width=100, height=50):
    """lines: [(bbox, text)] -> an in-memory SROIE row."""
    return {
        "key": key,
        "bboxes": [b for b, _ in lines],
        "words": [t for _, t in lines],
        "image_size": {"width": width, "height": height},
    }


def write_ocr(ocr_dir, key, lines, width=100, height=50, device="gpu:0", schema="1.0"):
    ocr_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": schema,
        "image": {"path": f"sroie:test/{key}", "width": width, "height": height},
        "engine": {"name": "paddleocr", "device": device},
        "lines": [{"text": t, "confidence": 0.9, "bbox": b} for b, t in lines],
    }
    (ocr_dir / f"{key}.json").write_text(json.dumps(doc), encoding="utf-8")


def test_evaluate_scores_matched_pairs_only(tmp_path):
    write_ocr(tmp_path, "K1", [(A, "total 1.00"), ([60, 0, 70, 10], "unmatched")])
    write_ocr(tmp_path, "K2", [(A, "ABCD")])

    result = cli.evaluate(
        [gt("K1", [(A, "TOTAL 1.00"), (B, "missed")]), gt("K2", [(A, "ABCE")]), gt("K3", [])],
        tmp_path,
        0.5,
    )

    assert result["counts"] == {
        "images_evaluated": 2, "missing": 1, "pairs": 2, "cer_pairs": 2, "skipped_empty_gt": 0
    }  # fmt: skip
    assert result["missing_keys"] == ["K3"]
    assert result["cer"] == {"mean": 0.125, "median": 0.125}
    assert result["exact_match_rate"] == 0.5
    worst = result["worst_lines"][0]
    assert (worst["key"], worst["gt_text"], worst["pred_text"], worst["cer"]) == (
        "K2", "ABCE", "ABCD", 0.25
    )  # fmt: skip


def test_evaluate_counts_empty_gt(tmp_path):
    write_ocr(tmp_path, "K1", [(A, ""), (B, "x")])

    result = cli.evaluate([gt("K1", [(A, " "), (B, "X")])], tmp_path, 0.5)

    assert result["counts"]["skipped_empty_gt"] == 1
    assert result["counts"]["cer_pairs"] == 1
    assert result["exact_match_rate"] == 1.0


def test_evaluate_flags_size_mismatch_but_still_scores(tmp_path):
    write_ocr(tmp_path, "K1", [(A, "x")], width=99)

    result = cli.evaluate([gt("K1", [(A, "x")])], tmp_path, 0.5)

    assert result["size_mismatch_keys"] == ["K1"]
    assert result["counts"]["pairs"] == 1


def test_evaluate_refuses_misaligned_gt(tmp_path):
    write_ocr(tmp_path, "K1", [(A, "x")])
    row = gt("K1", [(A, "x")])
    row["words"].append("extra")

    with pytest.raises(ValueError, match="K1"):
        cli.evaluate([row], tmp_path, 0.5)


@pytest.mark.parametrize("bad", [None, 5, b"x"])
def test_evaluate_refuses_non_string_gt_word(tmp_path, bad):
    write_ocr(tmp_path, "K1", [(A, "x"), (B, "y")])
    row = gt("K1", [(A, "x"), (B, "y")])
    row["words"][1] = bad

    with pytest.raises(ValueError, match=r"K1.*1"):
        cli.evaluate([row], tmp_path, 0.5)


def test_evaluate_refuses_non_string_ocr_text(tmp_path):
    write_ocr(tmp_path, "K1", [(A, 5)])

    with pytest.raises(ValueError, match="K1.json"):
        cli.evaluate([gt("K1", [(A, "5")])], tmp_path, 0.5)


def test_evaluate_refuses_mixed_schema(tmp_path):
    write_ocr(tmp_path, "K1", [(A, "x")], schema="1.0")
    write_ocr(tmp_path, "K2", [(A, "x")], schema="2.0")

    with pytest.raises(ValueError, match="schema"):
        cli.evaluate([gt("K1", [(A, "x")]), gt("K2", [(A, "x")])], tmp_path, 0.5)


def test_evaluate_refuses_unsafe_key(tmp_path):
    with pytest.raises(ValueError, match="unsafe key"):
        cli.evaluate([gt("../K1", [])], tmp_path, 0.5)


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """data/sroie manifest + fake datasets module; returns (ocr_dir, out, selected columns)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(json.dumps({"commit_sha": "abc123"}), encoding="utf-8")
    selected = []

    class FakeSplit(list):
        def select_columns(self, columns):
            selected.append(columns)
            return self

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_from_disk = lambda path: {
        "test": FakeSplit([gt("K1", [(A, "TOTAL"), (B, "")]), gt("K2", [(A, "x")])])
    }
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setattr(det, "DATA_DIR", data_dir)
    return tmp_path / "ocr", tmp_path / "eval" / "recognition_test.json", selected


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["eval_ocr_recognition.py", *args])
    cli.main()


def test_cli_writes_report(cli_env, monkeypatch, capsys):
    ocr_dir, out, selected = cli_env
    write_ocr(ocr_dir, "K1", [(A, "TOTAI"), (B, "")])
    (ocr_dir / "manifest.json").write_text(
        json.dumps({"ocr": {"device": "gpu:0"}}), encoding="utf-8"
    )

    run_cli(monkeypatch, "--split", "test", "--ocr-dir", str(ocr_dir), "--out", str(out))

    report = json.loads(out.read_text(encoding="utf-8"))
    assert selected == [["key", "words", "bboxes", "image_size"]]  # never decodes images
    assert list(report) == [
        "iou_threshold", "split", "sroie_revision", "ocr_dir", "ocr_device",
        "ocr_schema_version", "normalization", "counts", "cer", "exact_match_rate",
        "worst_lines", "missing_keys", "size_mismatch_keys",
    ]  # fmt: skip
    assert report["sroie_revision"] == "abc123"
    assert report["ocr_device"] == "gpu:0"
    assert report["ocr_schema_version"] == "1.0"
    assert report["counts"] == {
        "images_evaluated": 1, "missing": 1, "pairs": 2, "cer_pairs": 1, "skipped_empty_gt": 1
    }  # fmt: skip
    assert report["cer"] == {"mean": 0.2, "median": 0.2}
    assert report["exact_match_rate"] == 0.5
    assert report["missing_keys"] == ["K2"]
    assert not list(out.parent.glob("*.tmp"))
    stdout = capsys.readouterr().out.strip().splitlines()
    assert len(stdout) == 1 and "CER" in stdout[0]


def test_cli_refuses_mixed_devices(cli_env, monkeypatch):
    ocr_dir, out, _ = cli_env
    write_ocr(ocr_dir, "K1", [(A, "x")], device="gpu:0")
    write_ocr(ocr_dir, "K2", [(A, "x")], device="cpu")

    with pytest.raises(SystemExit, match="device"):
        run_cli(monkeypatch, "--ocr-dir", str(ocr_dir), "--out", str(out))
    assert not out.exists()


def test_cli_refuses_manifest_device_conflict(cli_env, monkeypatch):
    ocr_dir, out, _ = cli_env
    write_ocr(ocr_dir, "K1", [(A, "x")], device="gpu:0")
    (ocr_dir / "manifest.json").write_text(json.dumps({"ocr": {"device": "cpu"}}), "utf-8")

    with pytest.raises(SystemExit, match="device"):
        run_cli(monkeypatch, "--ocr-dir", str(ocr_dir), "--out", str(out))
    assert not out.exists()


@pytest.mark.parametrize("contents", ['{"schema_version": "1.0"}', "{broken"])
def test_cli_refuses_malformed_ocr_json(cli_env, monkeypatch, contents):
    ocr_dir, out, _ = cli_env
    ocr_dir.mkdir()
    (ocr_dir / "K1.json").write_text(contents, encoding="utf-8")

    with pytest.raises(SystemExit, match="K1.json"):
        run_cli(monkeypatch, "--ocr-dir", str(ocr_dir), "--out", str(out))
    assert not out.exists()


def test_cli_refuses_ocr_dir_without_results(cli_env, monkeypatch):
    ocr_dir, out, _ = cli_env
    ocr_dir.mkdir()
    (ocr_dir / "notes.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="no OCR JSON"):
        run_cli(monkeypatch, "--ocr-dir", str(ocr_dir), "--out", str(out))
    assert not out.exists()


@pytest.mark.parametrize("value", ["0", "1.5", "abc"])
def test_cli_rejects_iou_outside_0_1(monkeypatch, value):
    with pytest.raises(SystemExit) as exc_info:
        run_cli(monkeypatch, "--iou", value)
    assert exc_info.value.code == 2
