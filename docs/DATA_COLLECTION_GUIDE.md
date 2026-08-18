# Data Collection Guide — do this THIS WEEK

The models are only as good as the photos you feed them. This guide tells you
exactly what to shoot, how many, and where the files go. Budget: one focused
afternoon for a demo-grade dataset; a week of lunch services for a good one.

**Golden rule for everything below: collect data with the SAME camera, at the
SAME height, angle, distance, background, and lighting as the deployed
system.** A model trained on kitchen-counter photos will not work at the
tray-return station. The easiest way to guarantee this: put the webcam in its
final position first, and capture training photos through it.

---

## 1. Waste classifier (4 categories)

### How many

| | Per category | Total |
|---|---|---|
| Minimum to demo | 25+ | 100+ |
| Good accuracy | 100+ | 400+ |

Keep the categories roughly BALANCED — 100 EMPTY photos and 10 HIGH_WASTE
photos will teach the model to just say "EMPTY".

Note: training technically starts at 4 images total (the dashboard blocks
below that), but anything under ~25 per class is a coin flip, not a model.

### What counts as which category — visual rubric

Judge by how much of the plate's food surface is still covered with food.
Sauce smears, grease, and crumbs do NOT count as food.

| Category | Rule of thumb | Looks like |
|----------|---------------|------------|
| EMPTY | ~0–5% covered | Clean or scraped plate; only smears, streaks, tiny crumbs, a stray pea |
| LOW_WASTE | ~5–25% covered | A few bites left; one small item (half a roti, a spoonful of rice); rim of dal in the bowl |
| MEDIUM_WASTE | ~25–50% covered | A recognizable portion left; half the rice, a full uneaten vegetable serving |
| HIGH_WASTE | ~50%+ covered | Most of the meal untouched; looks like a served plate someone barely started |

Decide borderline cases once, write your decision down, and apply it
consistently — consistent labels beat "correct" labels. If two teammates
label photos, have both label the same 10 photos first and compare.

### How to shoot

- **Through the deployment camera.** Two options:
  - Dashboard: **Training → 📦 Waste Dataset** — but this is for uploading
    existing files, so better:
  - The capture tool: run `python scripts/capture_dataset.py` from the
    project folder. It shows the live camera with the current category
    overlaid. **SPACE** = capture, **N** = next category, **Q** = quit. It
    saves straight into the right dataset folder.
- Vary what doesn't matter, keep constant what does: different foods,
  different plates-of-the-day, plate rotated, plate shifted a few cm left /
  right / forward — but same camera position and same lighting.
- Shoot across several lunch services if you can: real leftovers look
  different from staged ones (mixed, smeared, sauce-flooded).
- Include annoying real cases: crumpled napkin on the plate, spoon lying
  across it, hand at the plate edge.

### Common mistakes (these will silently ruin the model)

- **Stock photos or internet images.** Different camera, lighting, plates —
  the model learns the wrong world. Only photos from your setup.
- **Different backgrounds per class** (e.g. all EMPTY shot on the steel
  counter, all HIGH_WASTE on the wooden table). The model learns the
  background, not the food. Same background for every class, always.
- **Burst shots / near-duplicates.** 30 frames of the same plate half a
  second apart are 1 photo of information — worse, near-identical copies can
  land in both the training and test split, making accuracy look great while
  the model learned nothing. Change something (food, position, plate) between
  captures. Exact byte-duplicates are auto-skipped on upload, but *near*
  duplicates are on you.
- **Labeling by guilt instead of the rubric.** "They should have eaten that"
  is not a category. Percent of plate covered is.

### Where files go

Uploaded/captured images land in:

```
data/datasets/waste/EMPTY/
data/datasets/waste/LOW_WASTE/
data/datasets/waste/MEDIUM_WASTE/
data/datasets/waste/HIGH_WASTE/
```

You can also drop image files (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`)
directly into those folders. Train/val/test splitting (70/20/10) happens
automatically at training time.

---

## 2. Plate detector (~80+ labeled images)

The plate detector needs photos where you've drawn a box around every plate —
YOLO-format bounding boxes.

### How many and what

- **80+ images minimum** (150+ is better) taken through the deployment
  camera.
- Vary the plate position across the whole camera view: left, right, near,
  far, partially at the frame edge, two plates at once, plate held in hands,
  plate on a tray.
- Include **10–15 "negative" images with no plate at all** (empty counter,
  person standing there without a tray) — save these with an empty label
  file. This teaches the detector what "no plate" looks like.
- Use every plate/tray type your cafeteria actually uses.

Easy way to gather the raw frames: run `python scripts/capture_dataset.py`
and capture into any category, then move those files out — or take stills
while someone moves a plate around in front of the mounted webcam.

### How to label (free, in the browser)

Recommended tool: **makesense.ai** (free, no account, runs locally in the
browser). Alternative: **labelImg** (desktop app).

1. Go to makesense.ai → "Get Started" → drop in your plate images.
2. Choose **Object Detection**. Create ONE label named `plate`.
3. For each image, drag a rectangle around each plate. Tight box: edges of
   the plate/tray, not the food, not the hands. Every plate in the image gets
   a box; images with no plate get no boxes.
4. Export: Actions → **Export Annotations** → **"A .zip package containing
   files in YOLO format"**.
5. The zip contains one `.txt` per image. Each line is
   `0 x_center y_center width height` (all fractions of image size) — class
   `0` is `plate`, which matches the project's training config exactly
   (one class, named `plate`).

### Where files go

```
data/datasets/plate/images/   ← the photos (e.g. 00001.jpg)
data/datasets/plate/labels/   ← the YOLO .txt files, SAME filename stem (00001.txt)
```

The image and its label must share the same name apart from the extension.
Negative images get an empty `.txt` file.

### Training it — heads-up

Plate training is **not currently runnable from the dashboard** (the Train
Model tab's "Plate Detection" task will tell you to upload a labeled dataset
and stop), and there is no `scripts/train_plate_model.py`. Once your images
and labels are in the folders above, run this from the project folder:

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from cafeteria.config.settings import load_settings
from cafeteria.training.dataset_manager import DatasetManager
from cafeteria.training.trainer import PlateModelTrainer
from cafeteria.training.registry import ModelRegistry

cfg = load_settings(config_path=Path('configs/config.yaml'))
dm = DatasetManager(cfg.project_root / cfg.storage.datasets)
data_yaml = dm.prepare_yolo_det_dataset(cfg.project_root / 'data' / 'datasets' / 'plate_split')
registry = ModelRegistry(cfg.project_root / 'models' / 'registry.json')
version = registry.next_version('plate')
r = PlateModelTrainer(cfg.project_root / cfg.storage.datasets, cfg.project_root / cfg.storage.models, device=cfg.device).train(version=version, data_yaml=data_yaml, epochs=50)
registry.register_version('plate', r.version, str(r.weights_path), metrics=r.metrics, config=r.config)
registry.activate('plate', r.version)
print('Trained and activated', r.version, r.metrics)
"
```

Then confirm the home page System Readiness panel shows the plate model READY
(the running engine hot-swaps it within ~5 seconds).

---

## 3. Face enrollment (per person)

The dashboard wizard handles this — you just need people and light.

- **8–15 photos per person.** The wizard's default flow captures 3 photos in
  each of 5 poses (front, left, right, up, down) = 15, auto-advancing as it
  goes. Don't skip poses without reason — pose variety is what makes
  recognition robust at odd angles.
- Where: **Training → 👤 Face Enrollment** → create the person → follow the
  oval guide. Full steps are in `docs/RUNBOOK.md` section 3.

**Lighting tips (most "face not recognized" problems are lighting):**
- Light the face from the FRONT. Never enroll with a window or bright light
  behind the person — the camera sees a silhouette.
- Enroll under the same lighting the deployed camera will see (ideally: stand
  at the tray-return spot).
- No caps, sunglasses, or masks during enrollment; glasses are fine if the
  person always wears them.
- The wizard auto-rejects blurry and too-dark shots — if it keeps rejecting,
  add light rather than fighting it.
- If someone is unreliable at recognition later: add more images via
  **📷 Add More Images →** and regenerate embeddings, rather than starting
  over.

Get written consent before enrolling anyone. Photos and the identity vector
are stored locally in `data/enrollment/<person_id>/` and can be deleted at
any time (see RUNBOOK section 8).

---

## 4. Printable checklist — one page

```
SMART CAFETERIA WASTE TRACKER — DATA COLLECTION CHECKLIST

SETUP (once, before any photos)
[ ] Webcam mounted in its FINAL position (height, angle, distance)
[ ] Lighting arranged as it will be during real use
[ ] Test frame checked on the Live Monitor page

WASTE CLASSIFIER  — target 25+ per class (100+ ideal), balanced
[ ] EMPTY         ____ / 25    (clean/scraped, smears only)
[ ] LOW_WASTE     ____ / 25    (~5–25% of plate covered)
[ ] MEDIUM_WASTE  ____ / 25    (~25–50% covered)
[ ] HIGH_WASTE    ____ / 25    (50%+ covered, barely touched)
[ ] All shot through the deployment camera, same background
[ ] Varied: foods, plate position/rotation, several lunch services
[ ] No stock photos, no burst near-duplicates
[ ] Counts checked on Training → Waste Dataset bar chart

PLATE DETECTOR — target 80+ labeled images
[ ] Plates at varied positions: left/right/near/far/edge/held/two-at-once
[ ] 10–15 negative images (no plate) included
[ ] Labeled on makesense.ai, single class "plate", tight boxes
[ ] Exported as YOLO format
[ ] Images in  data/datasets/plate/images/
[ ] Labels in  data/datasets/plate/labels/  (matching filenames)

FACE ENROLLMENT — everyone who consented
[ ] Consent collected for every person
[ ] Each person: wizard completed, all 5 poses (15 photos)
[ ] Front-lit, no backlight, no caps/masks
[ ] "✅ READY" badge showing on the enrollment roster

TRAIN & VERIFY
[ ] Waste model trained (Training → Train Model → START TRAINING)
[ ] New version ACTIVATED (Training → Model Versions)
[ ] Plate model trained + activated (technical teammate — see section 2)
[ ] Home page System Readiness: ALL READY
[ ] End-to-end test: 1 food plate → exactly 1 transaction
```
