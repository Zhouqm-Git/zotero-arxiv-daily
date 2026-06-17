"""Tests for EmbeddingStore — uses a fake encoder, no real model download.

Covers: cache miss + writeback, cache hit (no re-encode), model-change
invalidation, prune of stale keys, and atomic persistence across instances.
"""

import numpy as np
import pytest

from zotero_arxiv_daily.embedding_store import EmbeddingStore


class FakeEncoder:
    """Deterministic stand-in for SentenceTransformer.

    Maps each input string to a fixed pseudo-vector so tests can assert which
    rows were encoded vs served from cache. ``model_name`` feeds the store's
    model-hash invalidation logic.
    """

    def __init__(self, model_name="fake-model-v1", dim=4, calls=None):
        self.model_name = model_name
        self.dim = dim
        self.calls = calls if calls is not None else []  # list of encoded texts

    def encode(self, texts, **kwargs):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            # Deterministic per-text vector: hash → bucket.
            h = abs(hash(t)) % 97
            out[i] = np.linspace(h, h + self.dim, self.dim, dtype=np.float32)
        self.calls.extend(texts)
        return out

    def similarity(self, a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        an = a / np.linalg.norm(a, axis=1, keepdims=True)
        bn = b / np.linalg.norm(b, axis=1, keepdims=True)
        return an @ bn.T


@pytest.fixture()
def store(tmp_path):
    enc = FakeEncoder()
    return EmbeddingStore(path=tmp_path / "cache.npz", encoder=enc), enc


def test_miss_then_hit_roundtrip(store):
    s, enc = store
    keys = ["K1", "K2", "K3"]
    texts = ["alpha", "beta", "gamma"]
    v1 = s.get_or_encode(texts, keys)
    assert v1.shape == (3, 4)
    assert len(enc.calls) == 3  # all misses first time

    # Second call: everything cached → encoder not invoked again.
    before = len(enc.calls)
    v2 = s.get_or_encode(texts, keys)
    assert len(enc.calls) == before
    np.testing.assert_array_equal(v1, v2)


def test_partial_miss_only_encodes_new(store):
    s, enc = store
    s.get_or_encode(["a", "b"], ["K1", "K2"])
    before = len(enc.calls)
    # K2 cached, K3 new, K1 cached
    out = s.get_or_encode(["a", "c", "b"], ["K1", "K3", "K2"])
    assert len(enc.calls) == before + 1  # only K3 encoded
    assert out.shape == (3, 4)


def test_model_change_invalidates_cache(tmp_path):
    enc1 = FakeEncoder(model_name="model-A")
    path = tmp_path / "cache.npz"
    s1 = EmbeddingStore(path=path, encoder=enc1)
    s1.get_or_encode(["x"], ["K1"])
    assert s1.size == 1

    # New encoder with a different model name → cache must rebuild.
    enc2 = FakeEncoder(model_name="model-B")
    s2 = EmbeddingStore(path=path, encoder=enc2)
    assert s2.size == 0  # invalidated
    s2.get_or_encode(["x"], ["K1"])
    assert s2.size == 1


def test_prune_removes_stale_keys(store):
    s, _ = store
    s.get_or_encode(["a", "b", "c"], ["K1", "K2", "K3"])
    assert s.size == 3
    removed = s.prune_to({"K1", "K3"})
    assert removed == 1
    assert s.size == 2
    # Reload from disk to confirm persistence of prune.
    enc = FakeEncoder()
    s2 = EmbeddingStore(path=s.path, encoder=enc)
    assert s2.size == 2


def test_none_key_not_cached(store):
    s, enc = store
    # None keys must always encode and never be persisted.
    s.get_or_encode(["ad hoc"], [None])
    assert s.size == 0
    s.get_or_encode(["ad hoc"], [None])
    assert len(enc.calls) == 2  # encoded both times


def test_persistence_across_instances(tmp_path):
    enc1 = FakeEncoder()
    path = tmp_path / "cache.npz"
    s1 = EmbeddingStore(path=path, encoder=enc1)
    s1.get_or_encode(["hello", "world"], ["A", "B"])

    # Fresh store on same path loads existing entries.
    enc2 = FakeEncoder()
    s2 = EmbeddingStore(path=path, encoder=enc2)
    assert s2.size == 2
    # Hit, no encode:
    before = len(enc2.calls)
    s2.get_or_encode(["hello", "world"], ["A", "B"])
    assert len(enc2.calls) == before
