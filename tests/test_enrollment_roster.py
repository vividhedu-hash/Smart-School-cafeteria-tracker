"""Roster edits: delete person/images, path safety, ML gallery index."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from cafeteria.recognition.enrollment import EnrollmentManager, validate_person_id
from cafeteria.storage.audit import append_audit
from cafeteria.storage.repositories import PersonRepository


def _bgr(color=(20, 40, 80)):
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    img[:] = color
    return img


def test_validate_person_id_rejects_traversal():
    with pytest.raises(ValueError):
        validate_person_id("../etc")
    with pytest.raises(ValueError):
        validate_person_id("_hidden")
    assert validate_person_id("Bhavmanyu_24") == "Bhavmanyu_24"


def test_save_image_unique_pose_names(tmp_path):
    mgr = EnrollmentManager(enrollment_dir=tmp_path)
    a = mgr.save_image("alice_01", _bgr(), pose="front")
    b = mgr.save_image("alice_01", _bgr((1, 2, 3)), pose="left")
    assert a.exists() and b.exists()
    assert a.name != b.name
    assert a.name.startswith("front_")
    assert b.name.startswith("left_")
    assert mgr.image_count("alice_01") == 2


def test_delete_image_blocks_path_traversal(tmp_path):
    mgr = EnrollmentManager(enrollment_dir=tmp_path)
    saved = mgr.save_image("alice_01", _bgr(), pose="front")
    outside = tmp_path / "secret.jpg"
    outside.write_bytes(b"nope")
    assert mgr.delete_image("alice_01", str(outside)) is False
    assert outside.exists()
    assert mgr.delete_image("alice_01", "../../secret.jpg") is False
    assert saved.exists()
    assert mgr.delete_image("alice_01", saved.name) is True
    assert not saved.exists()


def test_delete_images_invalidates_embedding(tmp_path):
    mgr = EnrollmentManager(enrollment_dir=tmp_path)
    p = mgr.save_image("alice_01", _bgr(), pose="front")
    np.save(str(mgr.embedding_path("alice_01")), np.ones(512, dtype=np.float32))
    assert mgr.has_embedding("alice_01")
    result = mgr.delete_images("alice_01", [p.name])
    assert p.name in result["removed"]
    assert not p.exists()
    assert not mgr.has_embedding("alice_01")
    meta = mgr.load_meta("alice_01")
    assert meta["embedding_stale"] is True
    index = json.loads(mgr.gallery_index_path().read_text())
    assert index["schema"] == "cafeteria.enrollment.v1"
    assert index["persons"][0]["person_id"] == "alice_01"
    assert index["persons"][0]["image_count"] == 0


def test_delete_person_removes_disk_and_index(tmp_path):
    mgr = EnrollmentManager(enrollment_dir=tmp_path)
    mgr.save_image("alice_01", _bgr(), pose="front")
    np.save(str(mgr.embedding_path("alice_01")), np.ones(512, dtype=np.float32))
    mgr.write_gallery_index()
    assert mgr.delete_person("alice_01") is True
    assert not mgr.person_dir("alice_01").exists()
    index = json.loads(mgr.gallery_index_path().read_text())
    assert index["persons"] == []


def test_write_gallery_index_lists_existing_jpegs(tmp_path):
    mgr = EnrollmentManager(enrollment_dir=tmp_path)
    img_dir = tmp_path / "Bhavmanyu_24" / "images"
    img_dir.mkdir(parents=True)
    cv2.imwrite(str(img_dir / "0001.jpg"), _bgr())
    np.save(str(tmp_path / "Bhavmanyu_24" / "embedding.npy"), np.ones(512, dtype=np.float32))
    (tmp_path / "Bhavmanyu_24" / "meta.json").write_text(json.dumps({
        "person_id": "Bhavmanyu_24", "name": "Bhavmanyu",
    }))
    path = mgr.write_gallery_index()
    payload = json.loads(path.read_text())
    assert payload["persons"][0]["images"] == ["Bhavmanyu_24/images/0001.jpg"]
    assert (tmp_path / "Bhavmanyu_24" / "samples.jsonl").exists()


def test_purge_person_deactivates_db_row(db_session, tmp_path):
    repo = PersonRepository(db_session)
    repo.create(person_id="alice_01", name="Alice", embedding_path="/tmp/e.npy")
    db_session.flush()
    assert repo.delete("alice_01") is True
    db_session.flush()
    assert repo.get_by_person_id("alice_01") is None
    leftover = repo.get_by_person_id_any("alice_01")
    assert leftover is not None
    assert leftover.is_active is False
    assert leftover.embedding_path is None


def test_update_name(db_session):
    repo = PersonRepository(db_session)
    repo.create(person_id="alice_01", name="Alice")
    db_session.flush()
    assert repo.update_name("alice_01", "Alice Sharma") is True
    db_session.flush()
    assert repo.get_by_person_id("alice_01").name == "Alice Sharma"


def test_audit_log_jsonl(tmp_path):
    log = tmp_path / "roster.jsonl"
    append_audit(log, action="delete_person", person_id="alice_01", actor="test")
    append_audit(log, action="delete_images", person_id="alice_01", removed=["a.jpg"])
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["action"] == "delete_person"
    assert rec["person_id"] == "alice_01"
    assert "ts" in rec
