"""Download Phase 1 datasets from Roboflow and merge them into data.yaml's
3 classes (caries, calculus, periapical_lesion). See DATA_SOURCES.md.

Usage:
    pip install roboflow
    ROBOFLOW_API_KEY=your_key python prepare_data.py
"""

import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

RAW_DIR = Path("dataset_raw")
OUT_DIR = Path("dataset")
TARGET_CLASSES = ["caries", "calculus", "periapical_lesion"]

SOURCES = [
    {"workspace": "dentalxray-yjztn", "project": "dental-xray-analysis-zfuqf", "try_versions": [1]},
    {"workspace": "dentex-hhs9g", "project": "kaggle-periapical_truth-perrh", "try_versions": [2, 1]},
    {"workspace": "comsats-uuniversity-wah-campus", "project": "teeth-calculus", "try_versions": [2, 1]},
    {"workspace": "trial-yolo-wspd7", "project": "calculus-detection1", "try_versions": [1]},
]

SPLIT_MAP = {"train": "train", "valid": "val", "test": "val"}


def download(rf, source):
    dest = RAW_DIR / source["project"]
    if dest.exists():
        print(f"skip download, already present: {dest}")
        return dest
    project = rf.workspace(source["workspace"]).project(source["project"])
    last_error = None
    for version in source["try_versions"]:
        try:
            project.version(version).download("yolov8", location=str(dest))
            return dest
        except Exception as e:  # noqa: BLE001 - version guessing, try next
            last_error = e
    raise RuntimeError(
        f"could not download {source['project']} at versions "
        f"{source['try_versions']}: check the exact version number on its "
        f"Roboflow Universe page. Last error: {last_error}"
    )


def _normalize(name):
    return name.lower().replace(" ", "_")


def class_name_map(dataset_dir):
    with open(dataset_dir / "data.yaml") as f:
        names = yaml.safe_load(f)["names"]
    target_by_name = {_normalize(name): i for i, name in enumerate(TARGET_CLASSES)}
    return {i: target_by_name.get(_normalize(name)) for i, name in enumerate(names)}


def remap_and_copy(dataset_dir, counts):
    id_map = class_name_map(dataset_dir)
    for split, out_split in SPLIT_MAP.items():
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if not img_dir.exists():
            continue

        out_img_dir = OUT_DIR / "images" / out_split
        out_lbl_dir = OUT_DIR / "labels" / out_split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        prefix = dataset_dir.name
        for img_path in img_dir.iterdir():
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            kept_lines = []
            if lbl_path.exists():
                for line in lbl_path.read_text().splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    old_id = int(parts[0])
                    new_id = id_map.get(old_id)
                    if new_id is None:
                        continue
                    kept_lines.append(" ".join([str(new_id)] + parts[1:]))
                    counts[TARGET_CLASSES[new_id]] += 1

            out_name = f"{prefix}_{img_path.name}"
            shutil.copy(img_path, out_img_dir / out_name)
            (out_lbl_dir / (Path(out_name).stem + ".txt")).write_text(
                "\n".join(kept_lines)
            )


def main():
    if not os.environ.get("ROBOFLOW_API_KEY"):
        raise SystemExit(
            "Set ROBOFLOW_API_KEY (Settings -> API Keys on roboflow.com) "
            "before running this script."
        )
    from roboflow import Roboflow  # deferred: only needed for the actual download

    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])

    counts = Counter()
    for source in SOURCES:
        print(f"downloading {source['project']}...")
        dataset_dir = download(rf, source)
        remap_and_copy(dataset_dir, counts)

    print("\nInstance counts per class:")
    for name in TARGET_CLASSES:
        n = counts[name]
        flag = "  <-- under 100, merge into a neighbor per ROADMAP.md" if n < 100 else ""
        print(f"  {name}: {n}{flag}")


if __name__ == "__main__":
    main()
