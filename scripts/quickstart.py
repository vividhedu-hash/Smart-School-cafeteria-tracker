"""
scripts/quickstart.py — one command from "I have photos" to "live pipeline active".

Usage:
    PYTHONPATH=src .venv/bin/python scripts/quickstart.py               # full run
    PYTHONPATH=src .venv/bin/python scripts/quickstart.py --dry-run    # report only
    PYTHONPATH=src .venv/bin/python scripts/quickstart.py --waste-only
    PYTHONPATH=src .venv/bin/python scripts/quickstart.py --plate-only --epochs 30

What it does:
  1. PREFLIGHT — reports what data exists (waste images per class, plate
     images/labels, enrolled persons, camera config) and what is missing.
  2. WASTE TRAINING — if enough waste images exist, trains the YOLO waste
     classifier, registers the version, prints held-out metrics, activates it.
  3. PLATE TRAINING — if labelled plate images exist, trains the YOLO plate
     detector, registers and activates it the same way.
  4. FINAL STATUS — recomputes readiness and tells you exactly what to do next.

Activation both updates models/registry.json (the engine polls it every 5 s)
and writes an activate_model command so a running engine hot-swaps immediately.

This script only orchestrates existing components (DatasetManager, the
trainers, ModelRegistry) — no training logic is duplicated here.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

# Minimum data requirements
MIN_WASTE_TOTAL = 4          # hard floor to train at all (trainer enforces this too)
RECOMMENDED_PER_CLASS = 25   # recommended for a usable model
RECOMMENDED_PLATE_LABELLED = 10  # warn below this; train if any images + labels exist


class UserError(Exception):
    """A problem the user must fix (missing data etc.) — printed without traceback."""


# ──────────────────────────────────────────────────────────────────────────────
# Report gathering (pure — unit-testable with a stub cfg)
# ──────────────────────────────────────────────────────────────────────────────

def gather_report(cfg) -> dict:
    """Collect a data/model readiness snapshot from disk. No side effects."""
    from cafeteria.training.dataset_manager import DatasetManager, WASTE_CLASSES
    from cafeteria.training.registry import ModelRegistry

    dm = DatasetManager(cfg.project_root / cfg.storage.datasets)
    registry = ModelRegistry(cfg.project_root / "models" / "registry.json")

    waste_counts = dm.waste_class_counts()
    plate_stats = dm.plate_dataset_stats()

    # Enrolled persons = directories with an embedding.npy
    enroll_root = cfg.project_root / cfg.recognition.embedding_dir
    enrolled = []
    if enroll_root.exists():
        enrolled = sorted(
            d.name for d in enroll_root.iterdir()
            if d.is_dir() and (d / "embedding.npy").exists()
        )

    def _model_ready(task: str, default_rel: str) -> tuple[bool, str]:
        active = registry.get_active_weights(task)
        if active:
            p = Path(active)
            if not p.is_absolute():
                p = cfg.project_root / p
            if p.exists():
                return True, f"active version {registry.active_version_string(task)}"
        default = cfg.project_root / default_rel
        if default.exists():
            return True, str(default)
        return False, "no trained model"

    waste_model_ok, waste_model_detail = _model_ready("waste", cfg.models.waste.weights)
    plate_model_ok, plate_model_detail = _model_ready("plate", cfg.models.plate.weights)

    return {
        "waste_counts": waste_counts,
        "waste_total": sum(waste_counts.values()),
        "waste_classes": list(WASTE_CLASSES),
        "plate_images": plate_stats["images"],
        "plate_labels": plate_stats["labels"],
        "plate_unlabelled": plate_stats["unlabelled"],
        "enrolled": enrolled,
        "camera_source": cfg.camera.source,
        "camera_mode": cfg.camera.mode,
        "waste_model_ready": waste_model_ok,
        "waste_model_detail": waste_model_detail,
        "plate_model_ready": plate_model_ok,
        "plate_model_detail": plate_model_detail,
    }


def waste_trainable(report: dict) -> tuple[bool, str]:
    """Can the waste classifier be trained with the current data?"""
    total = report["waste_total"]
    if total < MIN_WASTE_TOTAL:
        return False, (
            f"only {total} waste image(s) — need at least {MIN_WASTE_TOTAL} total "
            f"({RECOMMENDED_PER_CLASS}+ per class recommended). "
            "Upload via dashboard Training page or copy into "
            "data/datasets/waste/<CLASS>/"
        )
    thin = [c for c, n in report["waste_counts"].items()
            if 0 < n < RECOMMENDED_PER_CLASS]
    note = ""
    if thin:
        note = (f" (warning: classes below the recommended "
                f"{RECOMMENDED_PER_CLASS} images: {', '.join(thin)})")
    return True, f"{total} images across classes{note}"


def plate_trainable(report: dict) -> tuple[bool, str]:
    """Can the plate detector be trained with the current data?"""
    imgs, lbls = report["plate_images"], report["plate_labels"]
    if imgs == 0:
        return False, (
            "no plate images. Capture some with: "
            "PYTHONPATH=src .venv/bin/python scripts/capture_dataset.py"
        )
    if lbls == 0:
        guide = _project_root / "docs" / "DATA_COLLECTION_GUIDE.md"
        guide_hint = (f" See {guide.relative_to(_project_root)} for details."
                      if guide.exists() else "")
        return False, (
            f"{imgs} image(s) but 0 YOLO label files. Each image needs a "
            "matching .txt in data/datasets/plate/labels/ with lines like:\n"
            "        0 <cx> <cy> <w> <h>     (normalised 0–1, class 0 = plate)\n"
            "      Free labelling tools: labelImg, CVAT, Roboflow (export YOLO)."
            + guide_hint
        )
    notes: list[str] = []
    labelled = min(imgs, lbls)
    if labelled < RECOMMENDED_PLATE_LABELLED:
        notes.append(
            f"warning: {labelled} labelled image(s) — "
            f"{RECOMMENDED_PLATE_LABELLED}+ recommended for a usable detector"
        )
    if report["plate_unlabelled"] > 0:
        notes.append(
            f"{report['plate_unlabelled']} image(s) missing labels will be skipped"
        )
    note = f" ({'; '.join(notes)})" if notes else ""
    return True, f"{imgs} images / {lbls} labels{note}"


# ──────────────────────────────────────────────────────────────────────────────
# Printing
# ──────────────────────────────────────────────────────────────────────────────

def _status(ok: bool) -> str:
    return "READY  " if ok else "MISSING"


def print_report(report: dict) -> None:
    print("\n" + "=" * 66)
    print("  PREFLIGHT — Smart Cafeteria Waste Tracker")
    print("=" * 66)

    print("\n  Waste dataset (data/datasets/waste/):")
    for cls in report["waste_classes"]:
        n = report["waste_counts"].get(cls, 0)
        flag = "ok" if n >= RECOMMENDED_PER_CLASS else (
            "low" if n > 0 else "EMPTY")
        print(f"    {cls:<14} {n:>4} image(s)   [{flag}]")
    print(f"    {'TOTAL':<14} {report['waste_total']:>4}")

    ok, why = waste_trainable(report)
    print(f"  → Waste training: {'POSSIBLE' if ok else 'BLOCKED'} — {why}")

    print("\n  Plate dataset (data/datasets/plate/):")
    print(f"    images: {report['plate_images']}   labels: {report['plate_labels']}"
          f"   unlabelled: {report['plate_unlabelled']}")
    ok, why = plate_trainable(report)
    print(f"  → Plate training: {'POSSIBLE' if ok else 'BLOCKED'} — {why}")

    print("\n  Trained models:")
    print(f"    waste model: {_status(report['waste_model_ready'])} "
          f"({report['waste_model_detail']})")
    print(f"    plate model: {_status(report['plate_model_ready'])} "
          f"({report['plate_model_detail']})")

    print("\n  Face recognition:")
    if report["enrolled"]:
        print(f"    {len(report['enrolled'])} enrolled: "
              + ", ".join(report["enrolled"]))
    else:
        print("    no persons enrolled — all transactions will go to the "
              "review queue (enroll via dashboard Training page)")

    print("\n  Camera:")
    print(f"    mode={report['camera_mode']}  source={report['camera_source']} "
          "(checked live when the engine starts)")
    print("=" * 66)


def print_final_status(cfg) -> None:
    """Recompute readiness after training and print next steps."""
    report = gather_report(cfg)
    pipeline_active = report["waste_model_ready"] and report["plate_model_ready"]

    print("\n" + "=" * 66)
    print("  FINAL STATUS")
    print("=" * 66)
    print(f"  waste model: {_status(report['waste_model_ready'])} "
          f"({report['waste_model_detail']})")
    print(f"  plate model: {_status(report['plate_model_ready'])} "
          f"({report['plate_model_detail']})")
    print(f"  enrolled persons: {len(report['enrolled'])}")

    if pipeline_active:
        print("\n  ✅ LIVE PIPELINE WILL BE FULLY ACTIVE.")
        print("     A running engine hot-swaps within ~5 s; otherwise start everything:")
        print("       python run.py")
        print("     Then put a plate with food in front of the camera and watch")
        print("     the Live Monitor + Transactions pages.")
    else:
        print("\n  ⚠️  Pipeline NOT fully active yet:")
        if not report["waste_model_ready"]:
            _, why = waste_trainable(report)
            print(f"     - waste model missing — {why}")
        if not report["plate_model_ready"]:
            _, why = plate_trainable(report)
            print(f"     - plate model missing — {why}")
        print("     Fix the above, then re-run this script.")
    print("=" * 66 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Training + activation (thin orchestration over existing components)
# ──────────────────────────────────────────────────────────────────────────────

def _activate(cfg, registry, task: str, version: str) -> None:
    """Activate in the registry AND nudge a running engine to hot-swap now."""
    from cafeteria.monitoring.metrics import write_command

    if not registry.activate(task, version):
        raise UserError(f"Could not activate {task} version {version} (not in registry)")
    try:
        cmd_path = Path(cfg.project_root) / cfg.application.commands_path
        cmd_path.parent.mkdir(parents=True, exist_ok=True)
        write_command(
            cmd_path,
            {"type": "activate_model", "task": task, "version": version},
        )
    except Exception:
        pass  # engine polls the registry anyway; the command is just faster
    print(f"  ✅ Activated {task} model {version} "
          "(running engine hot-swaps within ~5 s)")


def train_waste(cfg, registry, epochs: int) -> str:
    """Train, register, and activate the waste classifier. Returns version."""
    from cafeteria.training.trainer import WasteModelTrainer

    version = registry.next_version("waste")
    print(f"\n── Training waste classifier ({version}, {epochs} epochs, "
          f"device={cfg.device}) — this can take a while on CPU …")

    trainer = WasteModelTrainer(
        datasets_dir=cfg.project_root / cfg.storage.datasets,
        models_output_dir=cfg.project_root / cfg.storage.models,
        device=cfg.device,
    )
    result = trainer.train(
        base_model=cfg.training.default_base_model,
        epochs=epochs,
        image_size=224,
        batch=cfg.training.default_batch,
        version=version,
    )

    registry.register_version(
        task="waste", version=version,
        weights_path=str(result.weights_path),
        metrics=result.metrics, config=result.config,
        dataset_stats=result.dataset_stats,
    )

    print(f"  weights: {result.weights_path}")
    if result.metrics:
        print("  held-out metrics:")
        for k, v in result.metrics.items():
            if isinstance(v, (int, float)):
                print(f"    {k}: {v}")
    _activate(cfg, registry, "waste", version)
    return version


def train_plate(cfg, registry, epochs: int) -> str:
    """Train, register, and activate the plate detector. Returns version."""
    from cafeteria.training.dataset_manager import DatasetManager
    from cafeteria.training.trainer import PlateModelTrainer

    version = registry.next_version("plate")
    print(f"\n── Training plate detector ({version}, {epochs} epochs, "
          f"device={cfg.device}) …")

    dm = DatasetManager(cfg.project_root / cfg.storage.datasets)
    with tempfile.TemporaryDirectory(prefix="plate_split_") as tmp:
        data_yaml = dm.prepare_yolo_det_dataset(Path(tmp))
        trainer = PlateModelTrainer(
            datasets_dir=cfg.project_root / cfg.storage.datasets,
            models_output_dir=cfg.project_root / cfg.storage.models,
            device=cfg.device,
        )
        result = trainer.train(
            base_model=cfg.training.plate_base_model,
            epochs=epochs,
            image_size=cfg.training.default_image_size,
            batch=cfg.training.default_batch,
            version=version,
            data_yaml=data_yaml,
        )

    registry.register_version(
        task="plate", version=version,
        weights_path=str(result.weights_path),
        metrics=result.metrics, config=result.config,
        dataset_stats=result.dataset_stats,
    )

    print(f"  weights: {result.weights_path}")
    if result.metrics:
        print("  held-out metrics:")
        for k, v in result.metrics.items():
            if isinstance(v, (int, float)):
                print(f"    {k}: {v}")
    _activate(cfg, registry, "plate", version)
    return version


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"epochs must be an integer, got {value!r}") from exc
    if n < 1:
        raise argparse.ArgumentTypeError("epochs must be >= 1")
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="One-command path from photos to a live waste pipeline.",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--waste-only", action="store_true",
                       help="train/activate only the waste classifier")
    group.add_argument("--plate-only", action="store_true",
                       help="train/activate only the plate detector")
    p.add_argument(
        "--epochs", type=_positive_int, default=None,
        help="training epochs (default: training.default_epochs from config)",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="preflight report only — no training")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from cafeteria.config.settings import load_settings
    from cafeteria.training.registry import ModelRegistry

    cfg = load_settings()
    registry = ModelRegistry(cfg.project_root / "models" / "registry.json")
    epochs = args.epochs or cfg.training.default_epochs

    report = gather_report(cfg)
    print_report(report)

    if args.dry_run:
        print("\n  (dry run — no training performed)")
        return 0

    do_waste = not args.plate_only
    do_plate = not args.waste_only
    trained_anything = False

    try:
        if do_waste:
            ok, why = waste_trainable(report)
            if ok:
                train_waste(cfg, registry, epochs)
                trained_anything = True
            else:
                print(f"\n  ⏭  Skipping waste training — {why}")
                if args.waste_only:
                    raise UserError("waste training was requested but is blocked (see above)")

        if do_plate:
            ok, why = plate_trainable(report)
            if ok:
                train_plate(cfg, registry, epochs)
                trained_anything = True
            else:
                print(f"\n  ⏭  Skipping plate training — {why}")
                if args.plate_only:
                    raise UserError("plate training was requested but is blocked (see above)")

    except UserError as e:
        print(f"\n  ❌ {e}\n")
        return 1
    except (ValueError, RuntimeError) as e:
        # Trainer-level user errors (empty dataset, bad data) — no traceback
        print(f"\n  ❌ Training failed: {e}\n")
        return 1

    if not trained_anything:
        print("\n  Nothing was trained. Add the missing data listed above and re-run.")

    print_final_status(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
