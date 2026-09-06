"""
Face enrollment — capture images and generate ArcFace embeddings.

Canonical on-disk package (what the matcher and any later trainer read):

  data/enrollment/
      index.json                 ← gallery manifest (all persons)
      {person_id}/
          images/                ← JPEG/PNG samples (InsightFace class folder)
              {pose}_{ms}_{id}.jpg
          embeddings/            ← per-image 512-d ArcFace vectors (.npy)
          embedding.npy          ← L2-normalised mean (live matcher)
          embedding.json         ← same mean as JSON (inspectable / portable)
          samples.jsonl          ← one record per image (path, pose, det_score)
          meta.json              ← person_id, name, counts, schema
"""
from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from cafeteria.recognition.quality import BiometricQualityAssessor, FaceQualityReport, PoseCategory
from cafeteria.utils.logging import get_logger

logger = get_logger("recognition.enrollment")

SCHEMA_VERSION = "cafeteria.enrollment.v1"
EMBEDDING_DIM = 512
_PERSON_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def validate_person_id(person_id: str) -> str:
    """Reject path traversal and empty IDs. Raises ValueError if invalid."""
    if not isinstance(person_id, str) or not _PERSON_ID_RE.match(person_id):
        raise ValueError(
            "Person ID must be 1–64 characters: letters, numbers, dot, underscore, hyphen."
        )
    if person_id.startswith("_"):
        raise ValueError("Person ID cannot start with '_' (reserved).")
    return person_id


def safe_filename(filename: str) -> str:
    """Return a basename with no directory components."""
    name = Path(str(filename).replace("\\", "/")).name
    if not name or name in {".", ".."}:
        raise ValueError("Invalid filename")
    return name


class EnrollmentManager:
    """
    Manages enrollment images and ArcFace embedding generation.

    Args:
        enrollment_dir: Root directory (e.g. data/enrollment/).
        face_engine:    Loaded FaceEngine instance.
        min_face_size:  Minimum face pixel size accepted during enrollment.
        audit_path:     Optional JSONL audit log path.
    """

    def __init__(
        self,
        enrollment_dir: str | Path,
        face_engine=None,
        min_face_size: int = 60,
        audit_path: Optional[str | Path] = None,
    ) -> None:
        self._root = Path(enrollment_dir)
        self._face_engine = face_engine
        self._min_face_size = min_face_size
        self._audit_path = Path(audit_path) if audit_path else None
        self._quality_assessor = BiometricQualityAssessor()
        # [AI-CoLab: Cursor] audit_path is optional so older callers (and tests) keep working.

    # ──────────────────────────────────────────────────────────────────────
    # Directory helpers
    # ──────────────────────────────────────────────────────────────────────

    def person_dir(self, person_id: str) -> Path:
        return self._root / validate_person_id(person_id)

    def images_dir(self, person_id: str) -> Path:
        return self.person_dir(person_id) / "images"

    def per_image_emb_dir(self, person_id: str) -> Path:
        return self.person_dir(person_id) / "embeddings"

    def embedding_path(self, person_id: str) -> Path:
        return self.person_dir(person_id) / "embedding.npy"

    def multi_embedding_path(self, person_id: str) -> Path:
        return self.person_dir(person_id) / "embeddings_multi.npy"

    def poses_json_path(self, person_id: str) -> Path:
        return self.person_dir(person_id) / "poses.json"

    def embedding_json_path(self, person_id: str) -> Path:
        return self.person_dir(person_id) / "embedding.json"

    def samples_path(self, person_id: str) -> Path:
        return self.person_dir(person_id) / "samples.jsonl"

    def meta_path(self, person_id: str) -> Path:
        return self.person_dir(person_id) / "meta.json"

    def gallery_index_path(self) -> Path:
        return self._root / "index.json"

    def ensure_person_dir(self, person_id: str) -> None:
        self.images_dir(person_id).mkdir(parents=True, exist_ok=True)

    def _audit(self, action: str, person_id: Optional[str] = None, **details) -> None:
        if self._audit_path is None:
            return
        from cafeteria.storage.audit import append_audit
        append_audit(
            self._audit_path,
            action=action,
            person_id=person_id,
            **details,
        )

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)

    # ──────────────────────────────────────────────────────────────────────
    # Image management
    # ──────────────────────────────────────────────────────────────────────

    def save_image(
        self,
        person_id: str,
        image: np.ndarray,
        pose: Optional[str] = None,
    ) -> Path:
        """Save a BGR enrollment image with a unique, pose-tagged name."""
        self.ensure_person_dir(person_id)
        pose_slug = re.sub(r"[^a-z0-9]+", "", (pose or "capture").lower()) or "capture"
        img_dir = self.images_dir(person_id)
        path = img_dir / f"{pose_slug}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.jpg"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Failed to write enrollment image: {path}")
        try:
            from cafeteria.recognition.crypto import lock_tree
            lock_tree(self.person_dir(person_id))
        except Exception:
            pass
        logger.debug("Saved enrollment image: %s", path)
        return path

    def list_images(self, person_id: str) -> list[Path]:
        """Return sorted list of enrollment images for a person."""
        img_dir = self.images_dir(person_id)
        if not img_dir.exists():
            return []
        files = [
            p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        ]
        return sorted(files, key=lambda p: p.name)

    def delete_image(self, person_id: str, filename: str) -> bool:
        """Delete a specific enrollment image by filename (no path traversal)."""
        pid = validate_person_id(person_id)
        name = safe_filename(filename)
        img_root = self.images_dir(pid).resolve()
        path = (img_root / name).resolve()
        if not path.is_relative_to(img_root) or not path.is_file():
            return False
        path.unlink()
        stem = path.stem
        emb = self.per_image_emb_dir(pid) / f"{stem}.npy"
        if emb.exists():
            emb.unlink()
        logger.info("Deleted enrollment image %s for %s", name, pid)
        return True

    def delete_images(self, person_id: str, filenames: list[str]) -> dict:
        """Delete several images, then invalidate the mean embedding."""
        removed, skipped = [], []
        for fn in filenames:
            try:
                name = safe_filename(fn)
                ok = self.delete_image(person_id, name)
            except ValueError:
                name = str(fn)
                ok = False
            (removed if ok else skipped).append(name)
        self.invalidate_identity(person_id)
        self.write_gallery_index()
        self._audit(
            "delete_images",
            person_id,
            removed=removed,
            skipped=skipped,
            remaining=self.image_count(person_id),
        )
        return {"removed": removed, "skipped": skipped}

    def image_count(self, person_id: str) -> int:
        return len(self.list_images(person_id))

    def invalidate_identity(self, person_id: str) -> None:
        """Drop the live matcher vector after sample changes (must rebuild)."""
        pid = validate_person_id(person_id)
        for path in (self.embedding_path(pid), self.embedding_json_path(pid)):
            if path.exists():
                path.unlink()
        meta = self.load_meta(pid) or {"person_id": pid}
        meta["schema"] = SCHEMA_VERSION
        meta["image_count"] = self.image_count(pid)
        meta["embedding_stale"] = True
        meta["updated_at"] = time.time()
        self._write_json(self.meta_path(pid), meta)

    # ──────────────────────────────────────────────────────────────────────
    # Embedding generation
    # ──────────────────────────────────────────────────────────────────────

    def generate_embedding(self, person_id: str, name: str) -> Optional[np.ndarray]:
        """
        Generate and save the mean ArcFace embedding for a person.

        Processes all enrollment images, extracts face embeddings,
        and saves the mean embedding as embedding.npy plus JSON/JSONL sidecars.
        """
        pid = validate_person_id(person_id)
        if self._face_engine is None or not self._face_engine.is_loaded:
            raise RuntimeError("FaceEngine must be loaded before generating embeddings")

        images = self.list_images(pid)
        if not images:
            logger.warning("No enrollment images found for %s", pid)
            return None

        self.per_image_emb_dir(pid).mkdir(parents=True, exist_ok=True)
        embeddings = []
        weights = []
        sample_rows = []
        failed = []
        pose_dict: dict[str, list[dict]] = {}
        quality_scores: list[float] = []

        for img_path in images:
            img = cv2.imread(str(img_path))
            pose_tag = img_path.stem.split("_")[0] if "_" in img_path.stem else None
            row = {
                "file": f"images/{img_path.name}",
                "pose": pose_tag if pose_tag and not pose_tag.isdigit() else None,
                "used_in_mean": False,
                "det_score": None,
                "quality_score": None,
                "iso_compliant": False,
                "pose_category": None,
                "ipd_pixels": None,
                "sharpness": None,
            }
            if img is None:
                failed.append(img_path.name)
                sample_rows.append(row)
                continue
            face = self._face_engine.get_largest_face(img, min_size=self._min_face_size)
            if face is None:
                logger.warning("No face found in enrollment image: %s", img_path)
                failed.append(img_path.name)
                sample_rows.append(row)
                continue

            # Biometric quality assessment (ISO/IEC 19794-5 & Big Tech standards)
            bbox = face["bbox"]
            kps = face.get("kps")
            quality = self._quality_assessor.evaluate(img, bbox, kps)

            emb = np.asarray(face["embedding"], dtype=np.float32)
            norm_e = np.linalg.norm(emb)
            if norm_e > 0:
                emb = emb / norm_e

            np.save(str(self.per_image_emb_dir(pid) / f"{img_path.stem}.npy"), emb)
            embeddings.append(emb)

            # Adaptive quality weighting (higher-quality, centered & sharp frames have higher influence)
            q_score = float(quality.overall_score)
            weight = max(0.20, q_score / 100.0)
            weights.append(weight)
            quality_scores.append(q_score)

            pose_cat = quality.pose_category.value
            if pose_cat not in pose_dict:
                pose_dict[pose_cat] = []
            pose_dict[pose_cat].append({
                "file": img_path.name,
                "quality_score": q_score,
                "yaw": quality.pose.yaw,
                "pitch": quality.pose.pitch,
                "roll": quality.pose.roll,
            })

            row["used_in_mean"] = True
            row["det_score"] = round(float(face.get("det_score") or 0.0), 4)
            row["quality_score"] = round(q_score, 1)
            row["iso_compliant"] = quality.passed
            row["pose_category"] = pose_cat
            row["ipd_pixels"] = quality.ipd_pixels
            row["sharpness"] = quality.sharpness_score
            row["yaw"] = quality.pose.yaw
            row["pitch"] = quality.pose.pitch
            row["roll"] = quality.pose.roll
            sample_rows.append(row)

        if not embeddings:
            logger.error(
                "Embedding generation failed for %s — no valid faces in %d images",
                pid,
                len(images),
            )
            return None

        # Quality-weighted composite master embedding
        weights_arr = np.array(weights, dtype=np.float32)[:, np.newaxis]
        weighted_sum = np.sum(np.stack(embeddings, axis=0) * weights_arr, axis=0)
        norm = np.linalg.norm(weighted_sum)
        if norm > 0:
            mean_emb = (weighted_sum / norm).astype(np.float32)
        else:
            mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)

        # Save multi-pose template gallery for pose-adaptive nearest-neighbor matching
        multi_embs = np.stack(embeddings, axis=0)
        np.save(str(self.multi_embedding_path(pid)), multi_embs)
        self._write_json(self.poses_json_path(pid), {
            "schema": "cafeteria.poses.v1",
            "person_id": pid,
            "poses": pose_dict,
            "exemplar_count": len(embeddings),
        })

        from cafeteria.recognition.crypto import lock_tree, save_embedding
        save_embedding(self.embedding_path(pid), mean_emb)
        self._write_json(self.embedding_json_path(pid), {
            "schema": "cafeteria.arcface.v1",
            "person_id": pid,
            "name": name,
            "dim": int(mean_emb.shape[0]),
            "dtype": "float32",
            "encrypted": True,
            "storage": "embedding.npy",
            "multi_template_storage": "embeddings_multi.npy",
        })
        with open(self.samples_path(pid), "w", encoding="utf-8") as f:
            for row in sample_rows:
                f.write(json.dumps(row) + "\n")

        avg_q = float(np.mean(quality_scores)) if quality_scores else 0.0
        meta = {
            "schema": SCHEMA_VERSION,
            "person_id": pid,
            "name": name,
            "image_count": len(images),
            "valid_faces": len(embeddings),
            "failed_images": failed,
            "embedding_dim": int(mean_emb.shape[0]),
            "embedding_stale": False,
            "model": "insightface",
            "average_quality_score": round(avg_q, 1),
            "iso_compliant": avg_q >= 68.0,
            "pose_coverage": list(pose_dict.keys()),
            "enrolled_at": (self.load_meta(pid) or {}).get("enrolled_at") or time.time(),
            "updated_at": time.time(),
        }
        self._write_json(self.meta_path(pid), meta)
        lock_tree(self.person_dir(pid))
        lock_tree(self._root)
        self.write_gallery_index()
        self._audit(
            "generate_embedding",
            pid,
            valid_faces=len(embeddings),
            image_count=len(images),
            avg_quality=round(avg_q, 1),
        )

        logger.info(
            "Embedding generated for %s (%s) — %d/%d images succeeded (Avg Quality: %.1f)",
            pid, name, len(embeddings), len(images), avg_q,
        )
        return mean_emb

    def evaluate_face_quality(
        self,
        image: np.ndarray,
        target_pose: Optional[str] = None,
    ) -> Optional[FaceQualityReport]:
        """
        Evaluate ISO biometric quality of a live frame or image for real-time UI feedback.
        """
        if self._face_engine is None or not self._face_engine.is_loaded:
            return None
        face = self._face_engine.get_largest_face(image, min_size=self._min_face_size)
        if face is None:
            return None
        t_pose = None
        if target_pose:
            try:
                t_pose = PoseCategory(target_pose.lower())
            except ValueError:
                pass
        return self._quality_assessor.evaluate(
            image, face["bbox"], face.get("kps"), target_pose=t_pose
        )

    def write_gallery_index(self) -> Path:
        """Write data/enrollment/index.json — the ML-readable gallery manifest."""
        self._root.mkdir(parents=True, exist_ok=True)
        persons = []
        for pid in self.list_person_dirs():
            meta = self.load_meta(pid) or {}
            images = [f"{pid}/images/{p.name}" for p in self.list_images(pid)]
            persons.append({
                "person_id": pid,
                "name": meta.get("name") or pid,
                "images": images,
                "image_count": len(images),
                "embedding": f"{pid}/embedding.npy" if self.has_embedding(pid) else None,
                "embedding_json": f"{pid}/embedding.json" if self.embedding_json_path(pid).exists() else None,
                "samples": f"{pid}/samples.jsonl" if self.samples_path(pid).exists() else None,
                "embedding_stale": bool(meta.get("embedding_stale")),
                "embedding_dim": meta.get("embedding_dim") or (EMBEDDING_DIM if self.has_embedding(pid) else None),
            })
            if not self.samples_path(pid).exists() and images:
                with open(self.samples_path(pid), "w", encoding="utf-8") as f:
                    for img in self.list_images(pid):
                        pose = img.stem.split("_")[0] if "_" in img.stem else None
                        f.write(json.dumps({
                            "file": f"images/{img.name}",
                            "pose": pose if pose and not pose.isdigit() else None,
                            "used_in_mean": self.has_embedding(pid),
                            "det_score": None,
                        }) + "\n")
                persons[-1]["samples"] = f"{pid}/samples.jsonl"
        payload = {
            "schema": SCHEMA_VERSION,
            "updated_at": time.time(),
            "embedding_dim": EMBEDDING_DIM,
            "layout": {
                "images": "{person_id}/images/*.jpg",
                "mean_embedding": "{person_id}/embedding.npy",
                "mean_embedding_json": "{person_id}/embedding.json",
                "per_image_embeddings": "{person_id}/embeddings/{stem}.npy",
                "samples": "{person_id}/samples.jsonl",
            },
            "persons": persons,
        }
        path = self.gallery_index_path()
        self._write_json(path, payload)
        return path

    def update_display_name(self, person_id: str, name: str) -> None:
        pid = validate_person_id(person_id)
        clean = name.strip()
        if not clean:
            raise ValueError("Display name cannot be empty")
        meta = self.load_meta(pid) or {"person_id": pid}
        meta["name"] = clean
        meta["updated_at"] = time.time()
        meta["schema"] = SCHEMA_VERSION
        self._write_json(self.meta_path(pid), meta)
        if self.embedding_json_path(pid).exists():
            try:
                payload = json.loads(self.embedding_json_path(pid).read_text(encoding="utf-8"))
                payload["name"] = clean
                self._write_json(self.embedding_json_path(pid), payload)
            except (json.JSONDecodeError, OSError):
                pass
        self.write_gallery_index()
        self._audit("rename_person", pid, name=clean)

    # ──────────────────────────────────────────────────────────────────────
    # Metadata
    # ──────────────────────────────────────────────────────────────────────

    def load_meta(self, person_id: str) -> Optional[dict]:
        """Load person metadata from meta.json."""
        path = self.meta_path(person_id)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def has_embedding(self, person_id: str) -> bool:
        return self.embedding_path(person_id).exists()

    def list_enrolled(self) -> list[str]:
        """Return list of person_ids that have embedding files."""
        return [
            pid for pid in self.list_person_dirs()
            if self.has_embedding(pid)
        ]

    def list_person_dirs(self) -> list[str]:
        """Return person_ids that have an enrollment directory (with or without embedding)."""
        if not self._root.exists():
            return []
        names = []
        for d in self._root.iterdir():
            if not d.is_dir() or d.name.startswith("_"):
                continue
            try:
                validate_person_id(d.name)
            except ValueError:
                continue
            names.append(d.name)
        return sorted(names)

    def sync_to_database(self, session) -> int:
        """
        Ensure every on-disk enrollment has a Person row.

        Returns the number of rows created. Existing rows are updated with
        image_count / embedding_path when those are missing.
        """
        from cafeteria.storage.repositories import PersonRepository

        repo = PersonRepository(session)
        created = 0
        for pid in self.list_person_dirs():
            meta = self.load_meta(pid) or {}
            name = meta.get("name") or pid
            n_imgs = self.image_count(pid)
            emb = str(self.embedding_path(pid)) if self.has_embedding(pid) else None
            meta_n = int(meta.get("image_count") or 0)
            has_history = self.samples_path(pid).exists() or meta_n > 0
            # [AI-CoLab: Cursor] Abandoned wizard stubs (no files, no embedding,
            # no enrollment history) stay on disk but must not become Person rows.
            if n_imgs == 0 and not emb and not has_history:
                continue
            existing = repo.get_by_person_id(pid)
            if existing is None:
                repo.create(person_id=pid, name=name, embedding_path=emb)
                repo.update_image_count(pid, n_imgs)
                created += 1
                continue
            if existing.image_count != n_imgs:
                repo.update_image_count(pid, n_imgs)
            if emb and not existing.embedding_path:
                repo.update_embedding_path(pid, emb)
            if not emb and existing.embedding_path:
                repo.clear_embedding(pid)
            if name and existing.name in (None, "", existing.person_id) and name != pid:
                existing.name = name
        self.write_gallery_index()
        return created

    def delete_person(self, person_id: str) -> bool:
        """Permanently delete all enrollment files for a person."""
        pid = validate_person_id(person_id)
        pdir = self.person_dir(pid)
        if not pdir.exists():
            self.write_gallery_index()
            self._audit("delete_person", pid, disk=False)
            return False
        shutil.rmtree(pdir)
        self.write_gallery_index()
        self._audit("delete_person", pid, disk=True)
        logger.info("Deleted enrollment data for %s", pid)
        return True
