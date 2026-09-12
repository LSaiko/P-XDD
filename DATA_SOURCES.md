# Data sources (Phase 1)

All CC BY 4.0 — attribution required if this project or any derived model/paper
is published. Every source is pulled and remapped by [`prepare_data.py`](prepare_data.py).

| Class we need | Source project | Images | License |
|---|---|---|---|
| caries, periapical_lesion | [dentalxray-yjztn/dental-xray-analysis-zfuqf](https://universe.roboflow.com/dentalxray-yjztn/dental-xray-analysis-zfuqf) | 13,590 | CC BY 4.0 |
| periapical_lesion (DENTEX-derived, larger) | [dentex-hhs9g/kaggle-periapical_truth-perrh](https://universe.roboflow.com/dentex-hhs9g/kaggle-periapical_truth-perrh) | 12,079 | CC BY 4.0 |
| calculus | [comsats-uuniversity-wah-campus/teeth-calculus](https://universe.roboflow.com/comsats-uuniversity-wah-campus/teeth-calculus) | 826 | CC BY 4.0 |
| calculus | [trial-yolo-wspd7/calculus-detection1](https://universe.roboflow.com/trial-yolo-wspd7/calculus-detection1) | 956 | CC BY 4.0 |

Only boxes labeled `caries`, `calculus`, or `periapical lesion` (case-insensitive)
are kept from each source — everything else (crown, filling, implant, missing
teeth, root canal, root piece, ...) is dropped, not merged in, since it's out
of scope for this project's 3 classes.

The proper "Dentex Challenge 2023" MICCAI dataset is the origin of the
`kaggle-periapical_truth` mirror above; use that mirror rather than the
original challenge site, since Roboflow already normalized it to YOLO format.

## Citations

```bibtex
@misc{ dental-xray-analysis-zfuqf_dataset,
  title = { Dental-Xray Analysis Dataset },
  author = { dentalxray },
  howpublished = { \url{ https://universe.roboflow.com/dentalxray-yjztn/dental-xray-analysis-zfuqf } },
  publisher = { Roboflow },
  year = { 2025 },
}

@misc{ kaggle-periapical_truth-perrh_dataset,
  title = { kaggle periapical_truth Dataset },
  author = { dentex },
  howpublished = { \url{ https://universe.roboflow.com/dentex-hhs9g/kaggle-periapical_truth-perrh } },
  publisher = { Roboflow },
  year = { 2025 },
}

@misc{ teeth-calculus_dataset,
  title = { Teeth Calculus Dataset },
  author = { Comsats Uuniversity Wah Campus },
  howpublished = { \url{ https://universe.roboflow.com/comsats-uuniversity-wah-campus/teeth-calculus } },
  publisher = { Roboflow },
  year = { 2025 },
}

@misc{ calculus-detection1_dataset,
  title = { Calculus-Detection1 Dataset },
  author = { Trial YOLO },
  howpublished = { \url{ https://universe.roboflow.com/trial-yolo-wspd7/calculus-detection1 } },
  publisher = { Roboflow },
  year = { 2025 },
}
```

## Running it

Needs a free Roboflow account API key (Settings -> API Keys on
roboflow.com), not committed anywhere:

```bash
pip install roboflow
ROBOFLOW_API_KEY=your_key python prepare_data.py
```

This downloads each source into `dataset_raw/`, remaps labels into our 3
classes, and writes the merged result into `dataset/images/{train,val}` and
`dataset/labels/{train,val}` — matching [`data.yaml`](data.yaml). Both
`dataset_raw/` and `dataset/` are gitignored; re-run any time to refresh.

At the end it prints instance counts per class. Per [ROADMAP.md](ROADMAP.md),
merge any class that comes in under 100 instances into a neighbor rather than
training on it directly — these sources shouldn't hit that, but check anyway.
