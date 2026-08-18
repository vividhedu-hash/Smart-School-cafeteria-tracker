"""
scripts/train_waste_model.py — CLI training launcher.
"""
from __future__ import annotations
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

import argparse
from cafeteria.config.settings import load_settings
from cafeteria.training.trainer import WasteModelTrainer
from cafeteria.training.registry import ModelRegistry

def main():
    parser = argparse.ArgumentParser(description="Train the waste classification model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--base-model", default=None)
    args = parser.parse_args()

    cfg = load_settings(config_path=_project_root / "configs" / "config.yaml")
    registry = ModelRegistry(cfg.project_root / "models" / "registry.json")
    version = registry.next_version("waste")

    trainer = WasteModelTrainer(
        datasets_dir=cfg.project_root / cfg.storage.datasets,
        models_output_dir=cfg.project_root / cfg.storage.models,
        device=cfg.device,
    )
    result = trainer.train(
        base_model=args.base_model or cfg.training.default_base_model,
        epochs=args.epochs or cfg.training.default_epochs,
        batch=args.batch or cfg.training.default_batch,
        version=version,
    )
    registry.register_version("waste", result.version, str(result.weights_path),
                               metrics=result.metrics, config=result.config)
    print(f"\n✅ Training complete. Version: {result.version}")
    print(f"   Weights: {result.weights_path}")
    print(f"   Metrics: {result.metrics}")
    print(f"\nActivate with: registry.activate('waste', '{result.version}')")

if __name__ == "__main__":
    main()
