# Smart Cafeteria Waste Tracker

A computer-vision prototype that detects plate/tray presence, classifies food
waste, recognises faces for individual attribution, and writes every event to a
local database — all from a single webcam.

## ⚠️ Current Status — read this first

The waste pipeline requires **two trained models that do not ship with this
repo** (you must train them on your own cafeteria images):

| Component | Status without training | What to do |
|-----------|------------------------|------------|
| Face recognition | ✅ Works out of the box (InsightFace, auto-downloads) | Enroll people via dashboard |
| Waste classifier | ❌ **Must be trained** — pipeline disabled until then | Upload ≥25 images/class, train, activate |
| Plate detector | ❌ **Must be trained** — pipeline disabled until then | Capture + label plate images, train, activate |

**Honesty guarantees (by design):**
- No waste transactions are ever created without both real models loaded.
- The engine never fabricates detections, confidences, or waste labels.
- Optional COCO "proxy" plate mode (`models.plate.allow_coco_fallback: true`)
  is clearly labelled `plate_proxy` in logs/UI — it is a demo stand-in, not
  real plate detection.
- The dashboard home page shows a live **System Readiness** panel.

Until models are trained, the engine runs **face-recognition-only mode**
(live "who's here" panel) and creates no transactions.

---

## Architecture

```
┌──────────────────┐       IPC (JSON files)      ┌──────────────────────────────┐
│  Inference Engine│  ──────────────────────────► │     Streamlit Dashboard       │
│  (main.py)       │  ◄──────────────────────────  │  (dashboard/app.py)           │
│                  │       commands.json          └──────────────────────────────┘
│  FrameBuffer     │                              Pages:
│  PlateDetector   │  runtime_state.json + DB     1. Live Monitor (camera feed)
│  WasteDetector   │  ────────────────────────►   2. Training & Enrollment
│  FaceEngine      │                              3. Review Queue
│  StateMachine    │                              4. Transactions
│  EventManager    │                              5. Analytics
└──────────────────┘
```

**The inference engine and dashboard run as two separate processes.**
This is intentional — OpenCV and Streamlit thread models are incompatible.
They communicate via:
- `data/runtime_state.json` (engine → dashboard, every 0.5 s)
- `data/commands.json` (dashboard → engine, consumed on each read)
- `database/cafeteria.db` (SQLite WAL mode, shared reads are safe)

---

## Quick Start

### 1. Install

```bash
# Clone or unzip the project
cd smart-cafeteria-waste

# Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install
pip install -e .
```

### 2. First Run (no models trained yet)

The system gracefully starts without models — detectors report "not loaded" and 
the state machine stays in IDLE until models are available.

```bash
# Terminal 1: Start the inference engine
python -m cafeteria.main

# Terminal 2: Start the dashboard
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

### 3. Train Models

**Option A — Dashboard:**
1. Open `🧠 Training` page
2. Enroll faces: add person ID, upload photos, click "Generate Embeddings"
3. Upload waste images per category (EMPTY / LOW_WASTE / MEDIUM_WASTE / HIGH_WASTE)
4. Click "START TRAINING" and wait
5. Click "ACTIVATE THIS MODEL" — engine hot-swaps within 5 seconds

**Option B — CLI:**
```bash
# Capture waste dataset images from webcam
python scripts/capture_dataset.py

# Train the waste model
python scripts/train_waste_model.py --epochs 50

# Enroll a person
python scripts/enroll_person.py --id person_01 --name "Alice"

# Evaluate a trained model
python scripts/evaluate_model.py --task waste

# Benchmark hardware latency
python scripts/benchmark.py
```

---

## Getting to first transaction

The fastest path from "I have photos" to a live pipeline is **one command**:

```bash
PYTHONPATH=src .venv/bin/python scripts/quickstart.py
```

It runs a preflight report (image counts per waste class, plate images/labels,
enrolled persons, camera config), then trains, registers, and activates
whatever the data allows — a running engine hot-swaps within ~5 seconds.

```bash
# Report only — see what's missing without training anything
PYTHONPATH=src .venv/bin/python scripts/quickstart.py --dry-run

# Train just one model, or override epochs
PYTHONPATH=src .venv/bin/python scripts/quickstart.py --waste-only
PYTHONPATH=src .venv/bin/python scripts/quickstart.py --plate-only --epochs 30
```

Data requirements it checks for you:
- **Waste classifier**: ≥4 images total to train at all; **≥25 per class
  recommended** (`data/datasets/waste/<CLASS>/`).
- **Plate detector**: images in `data/datasets/plate/images/` **plus** matching
  YOLO `.txt` labels in `data/datasets/plate/labels/` (any labelled pair is
  enough to start; ≥10 recommended).
- **Faces** (optional): enroll via the dashboard Training page or
  `scripts/enroll_person.py` — without enrollment, transactions go to the
  review queue.

After training, it prints whether the live pipeline is fully active and the
next step (`python run.py` to start engine + dashboard together).

---

## State Machine

```
IDLE ──► PLATE_DETECTED ──► FOOD_ANALYSIS ──► WASTE_EVENT
                                                    │
                                            ┌───────┴────────┐
                                       no face engine    face engine
                                            │                │
                                            ▼                ▼
                                  FACE_RECOGNITION     FACE_CAPTURE
                                            │                │
                                            └───────┬────────┘
                                                    ▼
                                        TRANSACTION_COMMIT ──► COOLDOWN ──► IDLE
                                        REVIEW_REQUIRED    ──► COOLDOWN ──► IDLE
```

---

## Configuration

Edit `configs/config.yaml` to adjust:

| Key | Default | Description |
|-----|---------|-------------|
| `camera.source` | `0` | Webcam index or RTSP URL |
| `camera.width/height` | `1280×720` | Capture resolution |
| `event.minimum_plate_presence_seconds` | `0.20` | Debounce before food analysis |
| `event.cooldown_seconds` | `2.0` | Freeze period after each event |
| `event.timeout_seconds` | `4.0` | Max face capture window |
| `recognition.model_pack` | `buffalo_s` | InsightFace pack (`det_size` 320) |
| `recognition.similarity_threshold` | `0.40` | ArcFace match threshold |
| `recognition.frames_to_vote` | `2` | Temporal voting frames |
| `inference.frame_skip` | `2` | Process every Nth frame for plate detection |
| `inference.debug_window` | `false` | OpenCV desktop window (dashboard uses `latest.jpg`) |
| `models.plate.allow_coco_fallback` | `false` | Opt-in COCO proxy only — never silent |

Override device with environment variable:
```bash
CAFETERIA_DEVICE=cpu python -m cafeteria.main     # Force CPU
CAFETERIA_DEVICE=cuda:0 python -m cafeteria.main  # Force GPU
```

---

## Directory Layout

```
smart-cafeteria-waste/
├── configs/
│   └── config.yaml               # Main configuration
├── dashboard/
│   ├── app.py                    # Streamlit home + sidebar
│   └── pages/                    # 5 Streamlit pages
├── data/
│   ├── captures/                 # Evidence images (YYYY-MM-DD/TX-xxx/event.jpg)
│   ├── datasets/                 # Training images
│   │   └── waste/{EMPTY,LOW_WASTE,...}/
│   ├── enrollment/               # Face embeddings (.npz) + raw images
│   ├── frames/                   # latest.jpg for dashboard feed
│   ├── review_queue/             # Review evidence copies
│   └── runtime_state.json        # Live IPC state file
├── models/
│   ├── face/                     # InsightFace model cache
│   ├── plate/                    # Trained plate detector versions
│   ├── waste/                    # Trained waste classifier versions
│   └── registry.json             # Model version + activation registry
├── scripts/
│   ├── quickstart.py             # One-command preflight + train + activate
│   ├── enroll_person.py          # Webcam enrollment tool
│   ├── capture_dataset.py        # Dataset capture tool
│   ├── train_waste_model.py      # Training CLI
│   ├── evaluate_model.py         # Evaluation CLI
│   └── benchmark.py              # Hardware latency benchmarking
├── src/cafeteria/
│   ├── camera/                   # WebcamCamera, RTSPCamera, FrameBuffer
│   ├── config/                   # Settings, device auto-detection
│   ├── detection/                # PlateDetector (YOLO), WasteDetector (YOLO-cls)
│   ├── media/                    # ImageStore
│   ├── monitoring/               # health.py, metrics.py (IPC state writer)
│   ├── pipeline/                 # StateMachine, EventManager, TransactionEngine
│   ├── recognition/              # FaceEngine (InsightFace), EmbeddingMatcher, EnrollmentManager
│   ├── storage/                  # SQLAlchemy models, repositories, database init
│   ├── training/                 # DatasetManager, WasteModelTrainer, WasteModelEvaluator, ModelRegistry
│   ├── utils/                    # logging, timing, image_quality
│   └── main.py                   # Inference engine entry point
└── tests/
    ├── test_analytics.py
    ├── test_config_load.py
    ├── test_debug_overlay.py
    ├── test_e2e.py
    ├── test_event_manager.py
    ├── test_quickstart.py
    ├── test_state_machine.py
    ├── test_storage.py
    ├── test_dataset_manager.py
    ├── test_registry.py
    └── test_utils.py
```

---

## Running Tests

On some macOS + numpy combinations, a bare `pytest tests/` can hit a BLAS
floating-point exception during collection. Prefer an **explicit file list**:

```bash
pip install pytest pytest-cov
cd smart-cafeteria-waste
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_analytics.py \
  tests/test_config_load.py \
  tests/test_dataset_manager.py \
  tests/test_debug_overlay.py \
  tests/test_e2e.py \
  tests/test_event_manager.py \
  tests/test_quickstart.py \
  tests/test_registry.py \
  tests/test_state_machine.py \
  tests/test_storage.py \
  tests/test_utils.py \
  -v
```

`tests/conftest.py` also pins OpenMP/BLAS to one thread, and pytest is
configured with `norecursedirs` so `.venv` is never collected. If
`pytest tests/` works on your machine, that is fine too.

---

## Two-Camera Upgrade Path

The codebase is designed for two-camera deployment:

1. `CameraBase` is an abstract interface — add a second `WebcamCamera` instance.
2. `FrameBuffer` is per-camera — instantiate two independently.
3. Route one to `PlateDetector` and one to `FaceEngine` in `EventManager`.
4. No pipeline logic changes required.

---

## Platform Support

| Platform | Camera Backend | GPU Acceleration |
|----------|---------------|-----------------|
| macOS    | AVFoundation  | Apple MPS       |
| Windows  | DirectShow    | CUDA            |
| Linux    | V4L2          | CUDA            |

Device is auto-detected at startup. Override with `CAFETERIA_DEVICE` env var.
