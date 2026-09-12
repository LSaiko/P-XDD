# Dev workflow

1. Branch off `main` for any change; no direct pushes to `main`.
2. Local loop before opening a PR: edit → `pytest` → `docker build` →
   manual smoke test of `/predict` with a real image.
3. PR must pass CI (pytest + docker `/health` check) before merge.
4. Dropping in new weights: put `best.pt` in `models/`, point `MODEL_PATH`
   at it if versioning multiple models, note eval metrics (mAP50 per class)
   in the PR description.
5. Tag a release on `main` after each roadmap phase lands.
