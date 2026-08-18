"""
scripts/evaluate_model.py — Evaluate a trained model on the test split.
"""
from __future__ import annotations
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

import argparse
from cafeteria.config.settings import load_settings
from cafeteria.training.registry import ModelRegistry

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--task", default="waste", choices=["waste", "plate"])
    parser.add_argument("--version", default=None, help="Version to evaluate (default: active)")
    args = parser.parse_args()

    cfg = load_settings(config_path=_project_root / "configs" / "config.yaml")
    registry = ModelRegistry(cfg.project_root / "models" / "registry.json")

    if args.version:
        versions = registry.get_all_versions(args.task)
        entry = next((v for v in versions if v["version"] == args.version), None)
        if not entry:
            print(f"Version {args.version} not found")
            sys.exit(1)
        weights = entry["weights_path"]
    else:
        weights = registry.get_active_weights(args.task)
        if not weights:
            print(f"No active model for task '{args.task}'")
            sys.exit(1)

    print(f"Evaluating {args.task} model: {weights}")

    if args.task == "waste":
        import tempfile
        from cafeteria.training.dataset_manager import DatasetManager
        from cafeteria.training.evaluator import WasteModelEvaluator

        dm = DatasetManager(cfg.project_root / cfg.storage.datasets)
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "split"
            dm.prepare_yolo_cls_dataset(split_dir)
            evaluator = WasteModelEvaluator(
                weights_path=weights,
                dataset_dir=split_dir,
                device=cfg.device,
            )
            metrics = evaluator.evaluate()

        print("\n── Evaluation Results (Test Split) ──────────────")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
    else:
        print("Plate evaluation requires data.yaml — use the dashboard training page.")

if __name__ == "__main__":
    main()
