import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import run_ocr_batch as batch_module
from PIL import Image
from run_ocr_batch import run_batch

from ml.ocr_engine import OCR_CONFIG, run_ocr
from ml.ocr_schema import SCHEMA_VERSION

DATASET_MANIFEST = {
    "repo_id": "jsdnrs/ICDAR2019-SROIE",
    "requested_revision": "abc",
    "commit_sha": "abc123",
}


def encoded(image, fmt="PNG", **save_kwargs):
    buf = io.BytesIO()
    image.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def rows(*keys):
    """Rows shaped like data/sroie with the image column cast to Image(decode=False)."""
    png = encoded(Image.new("RGB", (40, 30), "white"))
    return [{"key": k, "image": {"bytes": png, "path": f"{k}.jpg"}} for k in keys]


class FakeOcr:
    """Stands in for the real engine: records calls, fails on chosen keys."""

    def __init__(self, fail_on=()):
        self.calls, self.fail_on = [], set(fail_on)

    def __call__(self, image_bytes, image_path):
        assert isinstance(image_bytes, bytes)
        key = image_path.rsplit("/", 1)[-1]
        self.calls.append(key)
        if key in self.fail_on:
            raise RuntimeError(f"boom {key}")
        return {"schema_version": SCHEMA_VERSION, "image": {"path": image_path}, "lines": []}


def batch(tmp_path, data, ocr, limit=None):
    return run_batch(
        data, ocr, tmp_path, split="test", dataset_manifest=DATASET_MANIFEST, limit=limit
    )


def test_writes_one_json_per_row(tmp_path):
    batch(tmp_path, rows("A", "B"), FakeOcr())

    doc = json.loads((tmp_path / "A.json").read_text(encoding="utf-8"))
    assert doc["image"]["path"] == "sroie:test/A"
    assert (tmp_path / "B.json").exists()


def test_resume_skips_keys_with_output(tmp_path):
    (tmp_path / "A.json").write_text("{}", encoding="utf-8")
    ocr = FakeOcr()

    manifest = batch(tmp_path, rows("A", "B"), ocr)

    assert ocr.calls == ["B"]
    assert (tmp_path / "A.json").read_text(encoding="utf-8") == "{}"
    assert manifest["counts"] == {"processed": 1, "skipped": 1, "failed": 0}


def test_leftover_temp_file_is_not_treated_as_done(tmp_path):
    (tmp_path / "A.json.tmp").write_text('{"trunc', encoding="utf-8")
    ocr = FakeOcr()

    batch(tmp_path, rows("A"), ocr)

    assert ocr.calls == ["A"]
    json.loads((tmp_path / "A.json").read_text(encoding="utf-8"))
    assert not (tmp_path / "A.json.tmp").exists()


def test_keyboard_interrupt_during_write_leaves_no_partial_output(tmp_path, monkeypatch):
    path = tmp_path / "A.json"
    real_write_text = Path.write_text

    def interrupting_write(self, data, **kwargs):
        real_write_text(self, data[:5], **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "write_text", interrupting_write)

    with pytest.raises(KeyboardInterrupt):
        batch_module._write_json_atomic(path, {"complete": True})

    assert not path.exists()
    assert not (tmp_path / "A.json.tmp").exists()


def test_failure_is_logged_and_run_continues(tmp_path, capsys):
    manifest = batch(tmp_path, rows("A", "B", "C"), FakeOcr(fail_on={"B"}))

    assert (tmp_path / "A.json").exists() and (tmp_path / "C.json").exists()
    assert not (tmp_path / "B.json").exists()
    assert not (tmp_path / "B.json.tmp").exists()
    assert manifest["counts"] == {"processed": 2, "skipped": 0, "failed": 1}
    assert manifest["failed"] == [{"key": "B", "error": "RuntimeError: boom B"}]
    assert "B" in capsys.readouterr().err


@pytest.mark.parametrize("bad_key", ["../escape", "sub/dir", "a\\b", ""])
def test_unsafe_key_is_a_failure_not_a_write_outside_out_dir(tmp_path, bad_key):
    out_dir = tmp_path / "out"
    ocr = FakeOcr()

    manifest = batch(out_dir, rows(bad_key, "OK"), ocr)

    assert ocr.calls == ["OK"]
    assert manifest["failed"][0]["key"] == bad_key
    assert sorted(p.name for p in tmp_path.rglob("*.json")) == ["OK.json", "manifest.json"]


def test_limit_takes_first_n_rows(tmp_path):
    ocr = FakeOcr()

    batch(tmp_path, rows("A", "B", "C"), ocr, limit=2)

    assert ocr.calls == ["A", "B"]
    assert not (tmp_path / "C.json").exists()


def test_limit_zero_processes_no_rows(tmp_path):
    ocr = FakeOcr()

    manifest = batch(tmp_path, rows("A"), ocr, limit=0)

    assert ocr.calls == []
    assert manifest["counts"] == {"processed": 0, "skipped": 0, "failed": 0}


def test_cli_rejects_limit_zero(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_ocr_batch.py", "--limit", "0"])

    with pytest.raises(SystemExit) as exc_info:
        batch_module.main()

    assert exc_info.value.code == 2
    assert "must be >= 1" in capsys.readouterr().err


def test_existing_out_dir_preserves_unrelated_files(tmp_path):
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")

    batch(tmp_path, rows("A"), FakeOcr())

    assert unrelated.read_text(encoding="utf-8") == "keep me"
    assert (tmp_path / "A.json").exists()


def test_cli_builds_one_engine_continues_and_exits_nonzero_on_failure(
    tmp_path, monkeypatch, capsys
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(json.dumps(DATASET_MANIFEST), encoding="utf-8")
    out_dir = tmp_path / "out"

    class FakeSplit(list):
        def cast_column(self, name, feature):
            return self

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Image = lambda **kwargs: kwargs
    fake_datasets.load_from_disk = lambda path: {"test": FakeSplit(rows("A", "B", "C"))}
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    engine = object()
    engine_builds = []

    def fake_create_engine():
        engine_builds.append(engine)
        return engine

    def fake_run_ocr(actual_engine, image_bytes, image_path):
        assert actual_engine is engine
        if image_path.endswith("/B"):
            raise RuntimeError("broken image")
        return {"schema_version": SCHEMA_VERSION, "image": {"path": image_path}, "lines": []}

    monkeypatch.setattr(batch_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(batch_module, "create_engine", fake_create_engine)
    monkeypatch.setattr(batch_module, "run_ocr", fake_run_ocr)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_ocr_batch.py", "--split", "test", "--limit", "3", "--out-dir", str(out_dir)],
    )

    with pytest.raises(SystemExit) as exc_info:
        batch_module.main()

    assert exc_info.value.code == 1
    assert engine_builds == [engine]
    assert (out_dir / "A.json").exists()
    assert not (out_dir / "B.json").exists()
    assert (out_dir / "C.json").exists()
    assert json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))["counts"] == {
        "processed": 2,
        "skipped": 0,
        "failed": 1,
    }
    assert "failed keys: B" in capsys.readouterr().err


def test_manifest_contents(tmp_path):
    batch(tmp_path, rows("A", "B"), FakeOcr(fail_on={"A"}), limit=5)

    m = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert m["dataset"] == {"repo_id": "jsdnrs/ICDAR2019-SROIE", "revision": "abc123"}
    assert m["split"] == "test"
    assert m["limit"] == 5
    assert m["ocr"]["config"] == OCR_CONFIG
    assert m["ocr"]["paddleocr_version"] and m["ocr"]["paddlepaddle_version"]
    assert m["counts"] == {"processed": 1, "skipped": 0, "failed": 1}
    assert m["started_at"] <= m["finished_at"]
    assert not list(tmp_path.glob("*.tmp"))


class FakeEngine:
    def __init__(self):
        self.inputs = []

    def predict(self, arr):
        self.inputs.append(arr)
        poly = np.array([[1, 2], [11, 2], [11, 8], [1, 8]])
        return [{"rec_texts": ["TOTAL 9.00"], "rec_scores": np.array([0.9]), "rec_polys": [poly]}]


def test_run_ocr_feeds_bgr_and_builds_document():
    engine = FakeEngine()
    png = encoded(Image.new("RGB", (40, 30), (255, 0, 0)))

    doc = run_ocr(engine, png, "x.jpg")

    assert engine.inputs[0].shape == (30, 40, 3)
    assert engine.inputs[0][0, 0].tolist() == [0, 0, 255]  # red pixel in BGR order
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["image"] == {"path": "x.jpg", "width": 40, "height": 30}
    assert doc["lines"][0]["text"] == "TOTAL 9.00"
    assert doc["engine"]["name"] == "paddleocr"
    assert doc["engine"]["inference_seconds"] >= 0


def test_run_ocr_accepts_grayscale_images():
    engine = FakeEngine()
    run_ocr(engine, encoded(Image.new("L", (5, 4))), "g.png")
    assert engine.inputs[0].shape == (4, 5, 3)


def test_run_ocr_decodes_like_paddlex_file_read(tmp_path):
    # PaddleX reads a path with cv2.imdecode; PIL decodes JPEG differently. Bytes in must
    # give the same array as handing PaddleX the file, so baseline output does not change.
    import cv2

    rng = np.random.default_rng(0)
    path = tmp_path / "noise.jpg"
    Image.fromarray(rng.integers(0, 255, (32, 48, 3), dtype=np.uint8)).save(path, quality=80)
    engine = FakeEngine()

    run_ocr(engine, path.read_bytes(), str(path))

    expected = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(engine.inputs[0], expected)


class FullFrameEngine:
    """Returns one box spanning the whole array it was given, i.e. in decoded-image coords."""

    def predict(self, arr):
        h, w = arr.shape[:2]
        poly = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        return [{"rec_texts": ["X"], "rec_scores": np.array([0.9]), "rec_polys": [poly]}]


def test_exif_oriented_jpeg_records_rotated_size_matching_boxes():
    # Deliberate: cv2 applies EXIF orientation, so size comes from the decoded (rotated)
    # image the boxes refer to. The old baseline wrote PIL's unrotated size (40x30).
    exif = Image.Exif()
    exif[274] = 6  # Orientation: rotate 90 degrees clockwise when displayed
    jpeg = encoded(Image.new("RGB", (40, 30), "white"), "JPEG", exif=exif)

    doc = run_ocr(FullFrameEngine(), jpeg, "oriented.jpg")

    width, height = doc["image"]["width"], doc["image"]["height"]
    assert (width, height) == (30, 40)
    for line in doc["lines"]:
        x0, y0, x1, y1 = line["bbox"]
        assert 0 <= x0 <= x1 <= width and 0 <= y0 <= y1 <= height


def test_run_ocr_rejects_undecodable_bytes():
    with pytest.raises(ValueError, match="cannot decode"):
        run_ocr(FakeEngine(), b"not an image", "bad.jpg")


def test_corrupt_image_in_batch_is_a_logged_failure(tmp_path):
    data = rows("A") + [{"key": "BAD", "image": {"bytes": b"garbage", "path": "BAD.jpg"}}]

    manifest = batch(tmp_path, data, lambda b, p: run_ocr(FakeEngine(), b, p))

    assert (tmp_path / "A.json").exists()
    assert manifest["counts"]["failed"] == 1
    assert manifest["failed"][0]["key"] == "BAD"


@pytest.mark.parametrize("limit", [None, 10])
def test_empty_rows_still_write_manifest(tmp_path, limit):
    manifest = batch(tmp_path, [], FakeOcr(), limit=limit)
    assert manifest["counts"] == {"processed": 0, "skipped": 0, "failed": 0}
    assert (tmp_path / "manifest.json").exists()
