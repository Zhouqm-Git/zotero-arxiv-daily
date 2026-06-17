"""Local reranker with a persistent embedding cache.

Same model and similarity math as :class:`LocalReranker`, but embeddings are
keyed by paper id and cached on disk, so:

* Zotero corpus (520+ papers) is encoded once, then read from cache on every
  subsequent run — the dominant cost (45 min on CI CPU) drops to seconds.
* Same arXiv candidate seen on consecutive days reuses its embedding too.

Cache location and device come from ``config.reranker.cached_local``.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

from ..embedding_store import EmbeddingStore
from .base import BaseReranker, register_reranker
from .local import LocalReranker

logger = logging.getLogger(__name__)


@register_reranker("cached_local")
class CachedLocalReranker(BaseReranker):
    """Local SentenceTransformer reranker backed by an on-disk embedding cache."""

    def __init__(self, config):
        super().__init__(config)
        self._encoder = None
        self._store: EmbeddingStore | None = None

    # Lazily build the encoder + store so import is cheap (tests, --help).
    def _ensure_loaded(self):
        if self._encoder is not None:
            return
        from sentence_transformers import SentenceTransformer

        if not self.config.executor.debug:
            from transformers.utils import logging as transformers_logging
            from huggingface_hub.utils import logging as hf_logging

            transformers_logging.set_verbosity_error()
            hf_logging.set_verbosity_error()
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)

        cfg = self.config.reranker.cached_local
        model_name = cfg.model
        device = cfg.get("device", "auto")
        # SentenceTransformer does not accept "auto" — resolve it ourselves:
        # MPS on Apple Silicon (fast local inference), CPU elsewhere.
        if device == "auto":
            try:
                import torch
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            except Exception:
                device = "cpu"
            logger.info(f"Embedding device auto-resolved to: {device}")
        self._encoder = SentenceTransformer(
            model_name, device=device, trust_remote_code=True
        )
        encode_kwargs = dict(cfg.get("encode_kwargs") or {})
        # Cache hits must not be re-encoded, so suppress the progress bar on
        # the encoder itself; the store logs hit/miss counts instead.
        encode_kwargs.pop("show_progress_bar", None)
        self._encode_kwargs = encode_kwargs
        self._store = EmbeddingStore(
            path=cfg.cache_path,
            encoder=self._encoder,
            encode_kwargs={**encode_kwargs, "show_progress_bar": True},
        )

    def get_similarity_score(
        self,
        s1: list[str],
        s2: list[str],
        s1_keys: list[str] | None = None,
        s2_keys: list[str] | None = None,
    ) -> np.ndarray:
        self._ensure_loaded()
        # Fall back to uncached encoding if a caller passed no keys (defensive —
        # the base rerank always supplies them now).
        s1_keys = s1_keys or [None] * len(s1)
        s2_keys = s2_keys or [None] * len(s2)

        s1_feature = self._store.get_or_encode(s1, s1_keys)
        s2_feature = self._store.get_or_encode(s2, s2_keys)

        # NOTE: we deliberately do NOT auto-prune here. s2_keys is only the
        # current corpus, but the cache legitimately also holds candidate
        # (arXiv) embeddings from today/previous days. Pruning to s2_keys would
        # wrongly evict those. Corpus-side pruning is done explicitly via
        # ``prune_corpus_cache()`` from the executor, which knows the full
        # corpus membership.

        sim = self._encoder.similarity(s1_feature, s2_feature)
        # SentenceTransformer.similarity returns a torch tensor; a numpy-backed
        # fake encoder returns ndarray directly. Handle both.
        return sim.numpy() if hasattr(sim, "numpy") else np.asarray(sim)

    def prune_corpus_cache(self, corpus_keys: set[str]) -> int:
        """Evict cached entries that are no longer in the active Zotero corpus.

        Called by the executor once per run with the full corpus key set, so
        the cache cannot grow without bound as papers are removed from Zotero.
        Only entries whose key looks like a Zotero corpus id (8-char uppercase
        alphanumeric) are considered for eviction — arXiv candidate embeddings
        are always retained.
        """
        self._ensure_loaded()
        return self._store.prune_corpus_to(corpus_keys)
