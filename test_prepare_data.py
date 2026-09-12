from collections import Counter
from pathlib import Path

import prepare_data


def _make_fake_source(tmp_path):
    src = tmp_path / "fake-source"
    (src).mkdir()
    (src / "data.yaml").write_text(
        "names: [Caries, Crown, Periapical lesion]\n"
    )
    for split in ["train", "valid"]:
        (src / split / "images").mkdir(parents=True)
        (src / split / "labels").mkdir(parents=True)

    (src / "train" / "images" / "a.jpg").write_bytes(b"fake")
    (src / "train" / "labels" / "a.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n"  # caries -> kept, remapped to 0
        "1 0.5 0.5 0.1 0.1\n"  # crown -> dropped, not a target class
        "2 0.5 0.5 0.1 0.1\n"  # periapical lesion -> kept, remapped to 2
    )
    (src / "valid" / "images" / "b.jpg").write_bytes(b"fake")
    # no label file for b.jpg -> treated as a negative/background image

    return src


def test_class_name_map(tmp_path):
    src = _make_fake_source(tmp_path)
    assert prepare_data.class_name_map(src) == {0: 0, 1: None, 2: 2}


def test_remap_and_copy_drops_unmapped_classes_and_remaps_ids(tmp_path, monkeypatch):
    src = _make_fake_source(tmp_path)
    monkeypatch.setattr(prepare_data, "OUT_DIR", tmp_path / "merged")

    counts = Counter()
    prepare_data.remap_and_copy(src, counts)

    assert counts == {"caries": 1, "periapical_lesion": 1}

    train_label = tmp_path / "merged" / "labels" / "train" / "fake-source_a.txt"
    lines = train_label.read_text().splitlines()
    assert lines == ["0 0.5 0.5 0.1 0.1", "2 0.5 0.5 0.1 0.1"]

    assert (tmp_path / "merged" / "images" / "train" / "fake-source_a.jpg").exists()
    # valid split maps to val, background image still copied with empty label
    assert (tmp_path / "merged" / "images" / "val" / "fake-source_b.jpg").exists()
    assert (tmp_path / "merged" / "labels" / "val" / "fake-source_b.txt").read_text() == ""
