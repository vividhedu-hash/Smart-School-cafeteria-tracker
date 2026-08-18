"""Encrypted embedding roundtrip and permission lock."""
from __future__ import annotations

import os

import numpy as np

from cafeteria.recognition.crypto import (
    is_encrypted_blob,
    load_embedding,
    lock_tree,
    save_embedding,
)
from cafeteria.recognition.matcher import EmbeddingMatcher


def test_embedding_encrypt_roundtrip(tmp_path, monkeypatch):
    monkeypatch.delenv("CAFETERIA_EMBEDDING_KEY", raising=False)
    monkeypatch.setenv("CAFETERIA_PROJECT_ROOT", str(tmp_path))
    vec = np.random.randn(512).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    path = tmp_path / "person" / "embedding.npy"
    save_embedding(path, vec, project_root=tmp_path)
    raw = path.read_bytes()
    assert is_encrypted_blob(raw)
    loaded = load_embedding(path, project_root=tmp_path, migrate=False)
    np.testing.assert_allclose(loaded, vec, rtol=1e-5, atol=1e-5)


def test_matcher_loads_encrypted_embedding(tmp_path, monkeypatch):
    monkeypatch.delenv("CAFETERIA_EMBEDDING_KEY", raising=False)
    monkeypatch.setenv("CAFETERIA_PROJECT_ROOT", str(tmp_path))
    person = tmp_path / "alice_01"
    person.mkdir()
    vec = np.ones(512, dtype=np.float32)
    vec = vec / np.linalg.norm(vec)
    save_embedding(person / "embedding.npy", vec, project_root=tmp_path)
    (person / "meta.json").write_text('{"name": "Alice"}')
    matcher = EmbeddingMatcher(tmp_path, similarity_threshold=0.52)
    assert matcher.load_embeddings() == 1
    result = matcher.match(vec)
    assert result.is_known is True
    assert result.person_id == "alice_01"
    assert result.similarity >= 0.99


def test_plaintext_npy_migrates_and_still_matches(tmp_path, monkeypatch):
    monkeypatch.delenv("CAFETERIA_EMBEDDING_KEY", raising=False)
    monkeypatch.setenv("CAFETERIA_PROJECT_ROOT", str(tmp_path))
    person = tmp_path / "bob_01"
    person.mkdir()
    vec = np.zeros(512, dtype=np.float32)
    vec[0] = 1.0
    np.save(str(person / "embedding.npy"), vec)
    matcher = EmbeddingMatcher(tmp_path, similarity_threshold=0.52)
    assert matcher.load_embeddings() == 1
    assert is_encrypted_blob((person / "embedding.npy").read_bytes())
    result = matcher.match(vec)
    assert result.is_known is True


def test_lock_tree_owner_only(tmp_path):
    root = tmp_path / "enrollment"
    person = root / "p1" / "images"
    person.mkdir(parents=True)
    img = person / "a.jpg"
    img.write_bytes(b"xx")
    lock_tree(root)
    assert oct(root.stat().st_mode)[-3:] == "700"
    assert oct(img.stat().st_mode)[-3:] == "600"
