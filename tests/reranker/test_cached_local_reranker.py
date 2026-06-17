"""Tests for CachedLocalReranker — stubs the encoder + store, no model download.

Verifies: shape contract, cache key propagation from the rerank loop, and
that corpus embeddings are served from cache on the second pass (encoder
not re-invoked for the corpus side).
"""

import numpy as np
import pytest

from tests.canned_responses import make_sample_corpus, make_sample_paper
from tests.test_embedding_store import FakeEncoder
from zotero_arxiv_daily.embedding_store import EmbeddingStore
from zotero_arxiv_daily.reranker.cached_local import CachedLocalReranker


def _build_reranker(config, tmp_path, encoder):
    """Construct a CachedLocalReranker with the encoder/store pre-injected,
    bypassing SentenceTransformer download."""
    r = CachedLocalReranker(config)
    r._encoder = encoder
    r._encode_kwargs = {"show_progress_bar": False}
    r._store = EmbeddingStore(
        path=tmp_path / "cache.npz", encoder=encoder, encode_kwargs={"show_progress_bar": False}
    )
    r._ensure_loaded = lambda: None  # prevent lazy SentenceTransformer init
    return r


def test_cached_local_shape(config, tmp_path):
    enc = FakeEncoder()
    r = _build_reranker(config, tmp_path, enc)
    papers = [make_sample_paper(title=f"P{i}") for i in range(3)]
    corpus = make_sample_corpus(2)
    sim = r.get_similarity_score(
        [p.abstract for p in papers],
        [c.abstract for c in corpus],
        s1_keys=[p.cache_key for p in papers],
        s2_keys=[c.cache_key for c in corpus],
    )
    assert sim.shape == (3, 2)


def test_cached_local_keys_propagated(config, tmp_path):
    """The rerank loop must hand cache keys down to get_similarity_score."""
    enc = FakeEncoder()
    r = _build_reranker(config, tmp_path, enc)
    papers = [make_sample_paper(url="https://arxiv.org/abs/2026.00001v1")]
    corpus = make_sample_corpus(1)
    # Patch get_similarity_score to capture the keys it receives.
    seen = {}
    orig = r.get_similarity_score

    def spy(s1, s2, s1_keys=None, s2_keys=None):
        seen["s1"] = s1_keys
        seen["s2"] = s2_keys
        return orig(s1, s2, s1_keys, s2_keys)

    r.get_similarity_score = spy
    r.rerank(papers, corpus)
    assert seen["s1"] == ["2026.00001"]  # arxiv id, version-stripped
    # Corpus keys come from CorpusPaper.cache_key; make_sample_corpus sets no
    # key, so they're all None — that's the safe fallback path.
    assert seen["s2"] == [None]


def test_corpus_served_from_cache_on_second_call(config, tmp_path):
    """Second call should serve both candidate and corpus from cache."""
    enc = FakeEncoder()
    r = _build_reranker(config, tmp_path, enc)
    corpus = [make_sample_paper(url="https://arxiv.org/abs/2026.00001", title="C1")]
    cands = [make_sample_paper(url="https://arxiv.org/abs/2026.00002", title="P1")]
    ck = [p.cache_key for p in corpus]
    sk = [p.cache_key for p in cands]

    r.get_similarity_score([c.abstract for c in cands], [c.abstract for c in corpus], sk, ck)
    after_first = len(enc.calls)
    assert r._store.size == 2  # both candidate and corpus cached

    # Same candidate + corpus → full cache hit, no new encoding.
    r.get_similarity_score(
        [cands[0].abstract], [c.abstract for c in corpus],
        s1_keys=[cands[0].cache_key], s2_keys=ck,
    )
    assert len(enc.calls) == after_first  # nothing re-encoded
    assert r._store.size == 2  # candidate NOT evicted (the bug we fixed)


def test_prune_corpus_cache_preserves_candidates(config, tmp_path):
    """prune_corpus_cache must evict stale Zotero keys but keep arXiv candidates."""
    enc = FakeEncoder()
    r = _build_reranker(config, tmp_path, enc)
    # Seed cache with 2 Zotero corpus keys + 1 arXiv candidate key.
    r.get_or_encode = r._store.get_or_encode
    r._store.get_or_encode(["c1", "c2", "cand"], ["ABCD1234", "EFGH5678", "2405.14867"])
    assert r._store.size == 3

    # Corpus now only has ABCD1234 → EFGH5678 should be evicted, candidate kept.
    removed = r.prune_corpus_cache({"ABCD1234"})
    assert removed == 1
    # Candidate embedding must survive.
    before = len(enc.calls)
    r._store.get_or_encode(["cand"], ["2405.14867"])  # full hit
    assert len(enc.calls) == before
