# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A FastAPI microservice for salamander detection and individual identification, deployed on Railway. It implements a four-stage computer vision pipeline:

1. **Detect** (`POST /crop-salamander`) — YOLO bounding-box detection → rectangular crop
2. **Segment** (`POST /segment-salamander`) — YOLO-seg instance segmentation → mask-based crop with background replacement
3. **Embed** (`POST /embed`) — DINOv2 GeM-pooled patch tokens → 384-dim normalized vector for vector DB storage
4. **Verify** (`POST /identify`) — Patch-level mutual nearest-neighbor matching → `is_same`/`score`

## Commands

```bash
# Install all dependencies
pip install -r requirements.txt -r requirements-dev.txt
# or
make install-dev

# Run development server (auto-reload)
make run
# Equivalent: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --log-config app/uvicorn_logging_config.json

# Format code
make format        # runs black + ruff --fix

# Lint (no auto-fix)
make lint          # runs ruff check + black --check

# Type check
make type-check    # runs mypy app/

# Run all checks
make check         # lint + type-check

# Run tests
make test                          # pytest tests/ -v
pytest tests/test_main.py::test_health_endpoint -v   # single test
make test-cov                      # with HTML coverage report

# Install pre-commit hooks
make pre-commit-install

# Regenerate openapi.yml from app routes
make generate-openapi
```

## Architecture

### App layout

```
app/
  main.py             # FastAPI app, endpoints, lifespan startup/shutdown
  models.py           # All Pydantic request/response models
  utils.py            # pil_to_base64 / base64_to_pil helpers
  detection/
    base.py           # YOLOModelBase (shared loading, inference, torch.load patch)
    config.py         # YOLOConfig dataclass (thresholds, mask_threshold, verbose)
    detector.py       # SalamanderDetector — wraps crop.pt
    segmenter.py      # SalamanderSegmenter — wraps segment.pt, applies cv2 mask
  identification/
    embedder.py       # SalamanderEmbedder — DINOv2 vits14, GeM pooling, _ResizePad
    verifier.py       # SalamanderVerifier — patch-level MNN matching, classify()
```

### Startup

`app/main.py` uses a `lifespan` async context manager to load all four model singletons at startup: `detector`, `segmenter`, `embedder`, `verifier`. These are module-level globals. All endpoints check `is_model_loaded()` and return HTTP 503 if a model is unavailable.

### YOLO model loading

`YOLOModelBase` (detection/base.py) monkey-patches `torch.load` during model loading to force `weights_only=False`, working around a PyTorch 2.6+ behavior change that breaks YOLO model deserialization.

- `SalamanderDetector` reads `YOLO_MODEL_PATH` env var (default: `models/crop.pt`)
- `SalamanderSegmenter` reads `YOLO_SEGMENT_MODEL_PATH` env var (default: `models/segment.pt`)

### DINOv2 identification design

The embedder uses **GeM-pooled patch tokens** (not the CLS token). The CLS token only captures species-level similarity (all fire salamanders score ~0.92); patch tokens encode local spot/color patterns that distinguish individuals.

`SalamanderVerifier._patch_match_score()` builds a cosine similarity matrix across all ~256 patch pairs, keeps only mutual nearest neighbors, then scores as `(match_ratio) × (mean_similarity)`. Thresholds: ≥0.25 → same/high, ≥0.15 → same/medium, ≥0.10 → different/low, <0.10 → different/high.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `YOLO_MODEL_PATH` | `models/crop.pt` | YOLO detection model |
| `YOLO_SEGMENT_MODEL_PATH` | `models/segment.pt` | YOLO-seg model |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |

Copy `.env.example` to `.env` for local development.

## Code Style

- Python 3.11+, line length 100, formatted with Black
- Ruff rules: `E, W, F, I, B, C4, UP, ARG, SIM`; `F401` ignored in `__init__.py`
- mypy with `ignore_missing_imports = true` (ultralytics and cv2 stubs are absent)
- pytest uses `asyncio_mode = "auto"` — no manual `@pytest.mark.asyncio` loop management needed
- All response models are in `app/models.py`; do not scatter Pydantic models elsewhere

## Deployment

Deployed to Railway via `Dockerfile` + `railway.toml`. The Docker image expects model files baked in at `models/crop.pt` and `models/segment.pt`. These `.pt` files are committed to the repo (up to 10 MB allowed by pre-commit hook).
