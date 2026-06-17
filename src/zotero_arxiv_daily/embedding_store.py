"""Persistent embedding cache for the local reranker.

Stores document embeddings keyed by a stable paper identifier (Zotero item
key for corpus papers, arXiv id for candidates). On each run, only papers
whose key is absent from the cache get encoded; the rest are read from disk.

Cache invalidation is keyed by ``model_hash`` — switching the embedding model
(e.g. jina-nano → jina-small) rebuilds the cache automatically. Papers that
disappear from the active corpus are pruned so the cache does not grow
unboundedly.

Storage format: a single ``.npz`` with three arrays::

    keys       : array of N strings (the cache key, e.g. "ABCD1234")
    vectors    : float32 array of shape [N, dim]
    model_hash : scalar string (kept so a model change invalidates the cache)

The file is rewritten atomically (write to temp, rename) so a crash mid-run
cannot corrupt the cache.
"""
from __future__ import annotations

import hashlib
import logging
import re
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Keyed, on-disk embedding cache with incremental updates."""

    def __init__(
        self,
        path: str | Path,
        encoder,
        encode_kwargs: dict | None = None,
    ):
        """
        Args:
            path: Where to read/write the ``.npz`` cache file. Parent dirs are
                  created on first save.
            encoder: Anything with ``.encode(list[str], **kwargs) -> ndarray``
                     and a ``.model_name`` (or ``.model_args``) attribute we can
                     hash to detect model changes. Typically a
                     ``sentence_transformers.SentenceTransformer``.
            encode_kwargs: Forwarded to ``encoder.encode`` for misses.
        """
        self.path = Path(path)
        self.encoder = encoder
        self.encode_kwargs = dict(encode_kwargs or {})
        self.model_hash = self._compute_model_hash(encoder)

        self._keys: list[str] = []
        self._vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._index: dict[str, int] = {}
        self._load()

    # ------------------------------------------------------------------ public

    def get_or_encode(self, texts: list[str], keys: list[str]) -> np.ndarray:
        """Return embeddings for ``texts``, fetching cached rows by ``keys``.

        Keys present in the cache are returned verbatim; only missing keys are
        passed to ``encoder.encode`` in one batch, then written back to the
        cache. The returned array matches the order of ``texts``/``keys``.

        If ``keys`` is shorter than ``texts`` (or None entries), those positions
        are always encoded and never cached — used for ad-hoc queries.
        """
        if len(texts) != len(keys):
            raise ValueError(
                f"texts and keys must have equal length ({len(texts)} vs {len(keys)})"
            )

        n = len(texts)
        # Probe a single vector to learn the embedding dimension.
        out = None
        miss_idx: list[int] = []
        miss_texts: list[str] = []
        for i, (text, key) in enumerate(zip(texts, keys)):
            pos = self._index.get(key) if key else None
            if pos is not None:
                if out is None:
                    out = np.empty((n, self._vectors.shape[1]), dtype=np.float32)
                out[i] = self._vectors[pos]
            else:
                miss_idx.append(i)
                miss_texts.append(text)

        if miss_texts:
            logger.info(
                f"Embedding cache: encoding {len(miss_texts)}/{n} new "
                f"(hit rate {n - len(miss_texts)}/{n})"
            )
            new_vecs = self.encoder.encode(miss_texts, **self.encode_kwargs)
            new_vecs = np.asarray(new_vecs, dtype=np.float32)
            dim = new_vecs.shape[1]
            if out is None:
                out = np.empty((n, dim), dtype=np.float32)
            for j, i in enumerate(miss_idx):
                out[i] = new_vecs[j]
            # Persist only entries with a real key (skip ad-hoc / None keys).
            kept = [(k, v) for k, v in zip([keys[i] for i in miss_idx], new_vecs) if k]
            if kept:
                self._add_many([k for k, _ in kept], np.stack([v for _, v in kept]))
                self._save()
        else:
            logger.info(f"Embedding cache: full hit ({n}/{n})")

        return out

    def prune_to(self, active_keys: set[str]) -> int:
        """Drop cached keys not in ``active_keys``. Returns number removed.

        Use ``prune_corpus_to`` from reranker code instead: it only evicts
        Zotero-style keys, preserving arXiv candidate embeddings. This method
        is kept for tests and the general case.
        """
        if not self._keys:
            return 0
        keep = [i for i, k in enumerate(self._keys) if k in active_keys]
        removed = len(self._keys) - len(keep)
        if removed == 0:
            return 0
        self._keys = [self._keys[i] for i in keep]
        self._vectors = self._vectors[keep] if len(keep) else np.empty((0, 0), dtype=np.float32)
        self._index = {k: i for i, k in enumerate(self._keys)}
        self._save()
        logger.info(f"Embedding cache: pruned {removed} stale entries ({len(self._keys)} remain)")
        return removed

    # Zotero item keys are 8-char uppercase alphanumeric (e.g. "ABCD1234").
    # Used to scope pruning so arXiv candidate embeddings are never evicted.
    _ZOTERO_KEY_RE = re.compile(r"^[A-Z0-9]{8}$")

    def prune_corpus_to(self, active_corpus_keys: set[str]) -> int:
        """Evict Zotero-style keys absent from the active corpus.

        Only keys matching the Zotero item-key shape are candidates for
        eviction, so arXiv candidate embeddings (e.g. ``2405.14867``) are
        always preserved across runs.
        """
        if not self._keys:
            return 0
        keep = []
        removed = 0
        for i, k in enumerate(self._keys):
            if self._ZOTERO_KEY_RE.match(k) and k not in active_corpus_keys:
                removed += 1
            else:
                keep.append(i)
        if removed == 0:
            return 0
        self._keys = [self._keys[i] for i in keep]
        self._vectors = self._vectors[keep]
        self._index = {k: i for i, k in enumerate(self._keys)}
        self._save()
        logger.info(
            f"Embedding cache: pruned {removed} stale Zotero entries "
            f"({len(self._keys)} remain)"
        )
        return removed

    @property
    def size(self) -> int:
        return len(self._keys)

    # ------------------------------------------------------------------ internal

    def _compute_model_hash(self, encoder) -> str:
        """A stable hash of the model identity. A change here invalidates the cache."""
        name = getattr(encoder, "model_name", None) or getattr(encoder, "model_args", None)
        name = repr(name) if name is not None else repr(encoder)
        return hashlib.md5(name.encode("utf-8")).hexdigest()[:12]

    def _load(self) -> None:
        if not self.path.exists():
            logger.info(f"Embedding cache: no existing file at {self.path} (fresh start)")
            return
        try:
            data = np.load(self.path, allow_pickle=False)
            cached_hash = str(data["model_hash"])
            if cached_hash != self.model_hash:
                logger.info(
                    f"Embedding cache: model changed ({cached_hash} -> {self.model_hash}), "
                    f"rebuilding from scratch"
                )
                return
            self._keys = [str(k) for k in data["keys"]]
            self._vectors = np.asarray(data["vectors"], dtype=np.float32)
            self._index = {k: i for i, k in enumerate(self._keys)}
            logger.info(f"Embedding cache: loaded {self.size} entries from {self.path}")
        except Exception as e:  # corrupt file — start fresh rather than crash
            logger.warning(f"Embedding cache: failed to load {self.path} ({e}); starting fresh")
            self._keys, self._vectors, self._index = [], np.empty((0, 0), dtype=np.float32), {}

    def _add_many(self, keys: list[str], vectors: np.ndarray) -> None:
        if not keys:
            return
        for i, k in enumerate(keys):
            if k in self._index:
                # Overwrite existing key in place (e.g. abstract edited in Zotero).
                self._vectors[self._index[k]] = vectors[i]
            else:
                if self._vectors.size == 0:
                    self._vectors = vectors[i][None, :].astype(np.float32)
                else:
                    self._vectors = np.vstack([self._vectors, vectors[i][None, :].astype(np.float32)])
                self._index[k] = len(self._keys)
                self._keys.append(k)

    def _save(self) -> None:
        if not self._keys:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkstemp(suffix=".npz", dir=str(self.path.parent))[1])
        try:
            # Fixed-width unicode dtype for keys avoids object arrays, so the
            # cache loads cleanly with allow_pickle=False (safer + portable).
            max_len = max(len(k) for k in self._keys)
            np.savez(
                tmp,
                keys=np.array(self._keys, dtype=f"<U{max(max_len, 1)}"),
                vectors=self._vectors.astype(np.float32),
                model_hash=np.array(self.model_hash, dtype="<U32"),
            )
            tmp.replace(self.path)  # atomic on POSIX
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
