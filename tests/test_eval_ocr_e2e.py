import json
import sys
import types

import eval_ocr_detection as det
import eval_ocr_e2e as cli
import pytest

A, B = [0, 0, 10, 10], [20, 0, 30, 10]


def gt(key, words, width=100, height=50):
    """words: GT `words` strings -> an in-memory SROIE row (boxes unused by this metric)."""
    return {
        "key": key,
        "words": list(words),
        "image_size": {"width": width, "height": height},
    }


def write_ocr(ocr_dir, key, texts, width=100, height=50, device="gpu:0", schema="1.0"):
    ocr_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": schema,
        "image": {"path": f"sroie:test/{key}", "width": width, "height": height},
        "engine": {"name": "paddleocr", "device": device},
        "lines": [{"text": t, "confidence": 0.9, "bbox": A} for t in texts],
    }
    (ocr_dir / f"{key}.json").write_text(json.dumps(doc), encoding="utf-8")


def test_evaluate_reports_per_image_and_micro(tmp_path):
    write_ocr(tmp_path, "K1", ["A A", "B"])
    write_ocr(tmp_path, "K2", ["x y z"])

    result = cli.evaluate(
        [gt("K1", ["a b", "B"]), gt("K2", ["W"]), gt("K3", ["missing"])], tmp_path
    )

    assert result["counts"] == {"evaluated": 2, "missing": 1}
    assert result["missing_keys"] == ["K3"]
    assert result["per_image"] == [
        {"key": "K1", "tp": 2, "fp": 1, "fn": 1,
         "precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3},
        {"key": "K2", "tp": 0, "fp": 3, "fn": 1, "precision": 0.0, "recall": 0.0, "f1": 0.0},
    ]  # fmt: skip
    assert result["micro"] == {
        "tp": 2, "fp": 4, "fn": 2, "precision": 1 / 3, "recall": 0.5, "f1": 0.4
    }  # fmt: skip


def test_evaluate_image_without_ocr_lines_scores_zero(tmp_path):
    write_ocr(tmp_path, "K1", [])

    result = cli.evaluate([gt("K1", ["TOTAL 1.00"])], tmp_path)

    assert result["per_image"][0] == {
        "key": "K1", "tp": 0, "fp": 0, "fn": 2, "precision": 0.0, "recall": 0.0, "f1": 0.0
    }  # fmt: skip


def test_evaluate_flags_size_mismatch_but_still_scores(tmp_path):
    write_ocr(tmp_path, "K1", ["x"], width=99)

    result = cli.evaluate([gt("K1", ["x"])], tmp_path)

    assert result["size_mismatch_keys"] == ["K1"]
    assert result["micro"]["tp"] == 1


@pytest.mark.parametrize("bad", [None, 5, b"x"])
def test_evaluate_refuses_non_string_gt_word(tmp_path, bad):
    write_ocr(tmp_path, "K1", ["x", "y"])

    with pytest.raises(ValueError, match=r"K1.*1"):
        cli.evaluate([gt("K1", ["x", bad])], tmp_path)


def test_evaluate_refuses_non_string_ocr_text(tmp_path):
    write_ocr(tmp_path, "K1", [5])

    with pytest.raises(ValueError, match="K1.json"):
        cli.evaluate([gt("K1", ["5"])], tmp_path)


def test_evaluate_refuses_mixed_schema(tmp_path):
    write_ocr(tmp_path, "K1", ["x"], schema="1.0")
    write_ocr(tmp_path, "K2", ["x"], schema="2.0")

    with pytest.raises(ValueError, match="schema"):
        cli.evaluate([gt("K1", ["x"]), gt("K2", ["x"])], tmp_path)


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
        "test": FakeSplit([gt("K1", ["TOTAL", "RM 1.00"]), gt("K2", ["x"])])
    }
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setattr(det, "DATA_DIR", data_dir)
    return tmp_path / "ocr", tmp_path / "eval" / "e2e_test.json", selected


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["eval_ocr_e2e.py", *args])
    cli.main()


def test_cli_writes_report(cli_env, monkeypatch, capsys):
    ocr_dir, out, selected = cli_env
    write_ocr(ocr_dir, "K1", ["TOTAL RM 1.00", "RM"])
    (ocr_dir / "manifest.json").write_text(
        json.dumps({"ocr": {"device": "gpu:0"}}), encoding="utf-8"
    )

    run_cli(monkeypatch, "--split", "test", "--ocr-dir", str(ocr_dir), "--out", str(out))

    report = json.loads(out.read_text(encoding="utf-8"))
    assert selected == [["key", "words", "image_size"]]  # never decodes images
    assert list(report) == [
        "split", "sroie_revision", "ocr_dir", "ocr_device", "ocr_schema_version",
        "normalization", "counts", "micro", "missing_keys", "size_mismatch_keys", "per_image",
    ]  # fmt: skip
    assert report["sroie_revision"] == "abc123"
    assert report["ocr_device"] == "gpu:0"
    assert report["counts"] == {"evaluated": 1, "missing": 1}
    assert report["micro"] == {
        "tp": 3, "fp": 1, "fn": 0, "precision": 0.75, "recall": 1.0, "f1": 6 / 7
    }  # fmt: skip
    assert report["per_image"][0]["key"] == "K1"
    assert report["missing_keys"] == ["K2"]
    assert not list(out.parent.glob("*.tmp"))
    stdout = capsys.readouterr().out.strip().splitlines()
    assert len(stdout) == 1 and "F1" in stdout[0]


def test_cli_refuses_ocr_dir_without_results(cli_env, monkeypatch):
    ocr_dir, out, _ = cli_env
    ocr_dir.mkdir()
    (ocr_dir / "notes.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="no OCR JSON"):
        run_cli(monkeypatch, "--ocr-dir", str(ocr_dir), "--out", str(out))
    assert not out.exists()
