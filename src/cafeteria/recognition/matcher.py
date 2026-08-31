"""
Identity matcher — loads stored embeddings and performs nearest-neighbour
identity lookup using cosine similarity.

Temporal voting: aggregate identity across multiple frames to reduce
false matches from blurry or partially occluded faces.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from cafeteria.recognition.live_match import face_match_as_dict as live_face_as_dict
from cafeteria.recognition.live_match import face_match_field as match_field
from cafeteria.utils.logging import get_logger

logger = get_logger("recognition.matcher")


@dataclass
class MatchResult:
    """Result of a single-frame identity match."""
    person_id: Optional[str]     # None if no match above threshold
    person_name: Optional[str]
    similarity: float             # cosine similarity ∈ [0, 1]
    is_known: bool
    candidates: list[dict]        # top-N candidates with scores


class EmbeddingMatcher:
    """
    Nearest-neighbour identity matcher over enrolled ArcFace embeddings.

    Args:
        enrollment_dir:      Path to data/enrollment/.
        similarity_threshold: Cosine similarity threshold for a positive match.
        top_n:               Number of candidates to return.
    """

    def __init__(
        self,
        enrollment_dir: str | Path,
        similarity_threshold: float = 0.52,
        top_n: int = 3,
    ) -> None:
        self._root = Path(enrollment_dir)
        self._threshold = similarity_threshold
        self._top_n = top_n
        self._embeddings: dict[str, np.ndarray] = {}   # person_id → embedding
        self._names: dict[str, str] = {}               # person_id → name

    def load_embeddings(self) -> int:
        """
        Load all stored embeddings from disk.

        Returns:
            Number of persons loaded.
        """
        self._embeddings.clear()
        self._names.clear()

        if not self._root.exists():
            return 0

        for person_dir in self._root.iterdir():
            if not person_dir.is_dir() or person_dir.name.startswith("_"):
                continue
            emb_path = person_dir / "embedding.npy"
            meta_path = person_dir / "meta.json"

            if not emb_path.exists():
                json_path = person_dir / "embedding.json"
                if json_path.exists():
                    logger.warning(
                        "%s has embedding.json but no embedding.npy — "
                        "regenerate embeddings on the Training page",
                        person_dir.name,
                    )
                continue

            from cafeteria.recognition.crypto import load_embedding
            try:
                embedding = load_embedding(emb_path)
            except Exception as exc:
                logger.warning("Could not load embedding for %s: %s", person_dir.name, exc)
                continue
            person_id = person_dir.name

            # Load name from meta.json if available
            name = person_id
            if meta_path.exists():
                import json
                with open(meta_path) as f:
                    meta = json.load(f)
                name = meta.get("name", person_id)

            self._embeddings[person_id] = embedding
            self._names[person_id] = name

            json_path = person_dir / "embedding.json"
            if json_path.exists():
                try:
                    import json
                    payload = json.loads(json_path.read_text(encoding="utf-8"))
                    if "values" in payload:
                        payload.pop("values", None)
                        payload["encrypted"] = True
                        payload["storage"] = "embedding.npy"
                        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                except Exception:
                    pass

        try:
            from cafeteria.recognition.crypto import lock_tree
            lock_tree(self._root)
        except Exception:
            pass
        logger.info("Loaded %d enrolled embeddings", len(self._embeddings))
        return len(self._embeddings)

    def reload(self) -> int:
        """Reload all embeddings from disk (hot-reload after new enrollment)."""
        return self.load_embeddings()

    @property
    def enrolled_count(self) -> int:
        return len(self._embeddings)

    @property
    def enrolled_ids(self) -> list[str]:
        return list(self._embeddings.keys())

    def match(self, query_embedding: np.ndarray) -> MatchResult:
        """
        Find the nearest enrolled identity for a query embedding.

        Args:
            query_embedding: L2-normalised ArcFace embedding, shape (512,).

        Returns:
            MatchResult with the best match (or UNKNOWN if below threshold).
        """
        if not self._embeddings:
            return MatchResult(
                person_id=None,
                person_name=None,
                similarity=0.0,
                is_known=False,
                candidates=[],
            )

        # Compute cosine similarity (embeddings are already L2-normalised)
        scores = {}
        for pid, emb in self._embeddings.items():
            sim = float(np.dot(query_embedding, emb))
            # Clip to [0, 1] (can be slightly negative due to floating point)
            scores[pid] = max(0.0, min(1.0, sim))

        # Sort by similarity descending
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Build candidates list
        candidates = [
            {
                "person_id": pid,
                "person_name": self._names.get(pid, pid),
                "similarity": round(sim, 4),
            }
            for pid, sim in sorted_scores[: self._top_n]
        ]

        best_pid, best_sim = sorted_scores[0]

        if best_sim >= self._threshold:
            return MatchResult(
                person_id=best_pid,
                person_name=self._names.get(best_pid, best_pid),
                similarity=best_sim,
                is_known=True,
                candidates=candidates,
            )

        return MatchResult(
            person_id=None,
            person_name=None,
            similarity=best_sim,
            is_known=False,
            candidates=candidates,
        )

    def update_threshold(self, threshold: float) -> None:
        """Update the similarity threshold without reloading embeddings."""
        self._threshold = threshold
        logger.info("Similarity threshold updated to %.3f", threshold)


class TemporalVoter:
    """
    Aggregate identity predictions across multiple frames using majority vote.

    Over a waste event, multiple face frames are matched independently.
    The most frequent non-UNKNOWN identity wins if it meets a minimum count.

    Args:
        frames_to_vote:    How many frames to collect before deciding.
        min_votes_ratio:   Fraction of frames that must agree (e.g., 0.4 = 40%).
    """

    def __init__(
        self,
        frames_to_vote: int = 5,
        min_votes_ratio: float = 0.40,
    ) -> None:
        self._frames_to_vote = frames_to_vote
        self._min_votes_ratio = min_votes_ratio
        self._votes: list[MatchResult] = []

    def reset(self) -> None:
        self._votes.clear()

    def add_vote(self, result: MatchResult) -> None:
        self._votes.append(result)

    def is_ready(self) -> bool:
        return len(self._votes) >= self._frames_to_vote

    def decide(self) -> MatchResult:
        """
        Return the consensus identity.

        Returns:
            MatchResult with best consensus identity, or UNKNOWN.
        """
        if not self._votes:
            return MatchResult(
                person_id=None, person_name=None,
                similarity=0.0, is_known=False, candidates=[],
            )

        known_votes = [v for v in self._votes if v.is_known]
        if not known_votes:
            # No known votes — return best candidate from last frame
            last = self._votes[-1]
            return MatchResult(
                person_id=None, person_name=None,
                similarity=last.similarity,
                is_known=False,
                candidates=last.candidates,
            )

        # Count votes per person
        counter = Counter(v.person_id for v in known_votes)
        best_pid, best_count = counter.most_common(1)[0]
        min_required = max(1, int(self._frames_to_vote * self._min_votes_ratio))

        if best_count < min_required:
            # Not enough consensus — return UNKNOWN
            all_cands = self._votes[-1].candidates if self._votes else []
            return MatchResult(
                person_id=None, person_name=None,
                similarity=0.0, is_known=False,
                candidates=all_cands,
            )

        # Pick the vote with highest similarity for this person for metadata
        best_votes = [v for v in known_votes if v.person_id == best_pid]
        best_vote = max(best_votes, key=lambda v: v.similarity)

        return MatchResult(
            person_id=best_vote.person_id,
            person_name=best_vote.person_name,
            similarity=best_vote.similarity,
            is_known=True,
            candidates=best_vote.candidates,
        )
