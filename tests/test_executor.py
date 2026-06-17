"""Tests for zotero_arxiv_daily.executor: normalize_path_patterns, filter_corpus, fetch_zotero_corpus, E2E."""

from datetime import datetime

import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.executor import Executor, normalize_path_patterns
from zotero_arxiv_daily.protocol import CorpusPaper


# ---------------------------------------------------------------------------
# normalize_path_patterns — migrated from test_include_path.py
# ---------------------------------------------------------------------------


def test_normalize_path_patterns_rejects_single_string_for_include_path():
    with pytest.raises(TypeError, match="config.zotero.include_path must be a list"):
        normalize_path_patterns("2026/survey/**", "include_path")


def test_normalize_path_patterns_accepts_list_config_for_include_path():
    include_path = OmegaConf.create(["2026/survey/**", "2026/reading-group/**"])
    assert normalize_path_patterns(include_path, "include_path") == [
        "2026/survey/**",
        "2026/reading-group/**",
    ]


def test_normalize_path_patterns_rejects_single_string_for_ignore_path():
    with pytest.raises(TypeError, match="config.zotero.ignore_path must be a list"):
        normalize_path_patterns("archive/**", "ignore_path")


def test_normalize_path_patterns_accepts_list_config_for_ignore_path():
    ignore_path = OmegaConf.create(["archive/**", "2025/**"])
    assert normalize_path_patterns(ignore_path, "ignore_path") == ["archive/**", "2025/**"]


def test_normalize_path_patterns_accepts_empty_list():
    assert normalize_path_patterns([], "ignore_path") == []


def test_normalize_path_patterns_accepts_none():
    assert normalize_path_patterns(None, "include_path") is None


# ---------------------------------------------------------------------------
# filter_corpus — migrated from test_include_path.py
# ---------------------------------------------------------------------------


def _make_executor(include_patterns=None, ignore_patterns=None):
    executor = Executor.__new__(Executor)
    executor.include_path_patterns = normalize_path_patterns(include_patterns, "include_path") if include_patterns else None
    executor.ignore_path_patterns = normalize_path_patterns(ignore_patterns, "ignore_path") if ignore_patterns else None
    return executor


def test_filter_corpus_matches_any_path_against_any_pattern():
    executor = _make_executor(include_patterns=["2026/survey/**", "2026/reading-group/**"])
    corpus = [
        CorpusPaper(title="Survey Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a", "archive/misc"]),
        CorpusPaper(title="Reading Group Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["notes/inbox", "2026/reading-group/week-1"]),
        CorpusPaper(title="Excluded Paper", abstract="", added_date=datetime(2026, 1, 3), paths=["2025/other/topic"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Survey Paper", "Reading Group Paper"]


def test_filter_corpus_excludes_papers_matching_ignore_path():
    executor = _make_executor(ignore_patterns=["archive/**", "2025/**"])
    corpus = [
        CorpusPaper(title="Active Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a"]),
        CorpusPaper(title="Archived Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["archive/misc"]),
        CorpusPaper(title="Old Paper", abstract="", added_date=datetime(2026, 1, 3), paths=["2025/other/topic"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Active Paper"]


def test_filter_corpus_ignore_path_takes_precedence_over_include_path():
    executor = _make_executor(include_patterns=["2026/**"], ignore_patterns=["2026/ignore/**"])
    corpus = [
        CorpusPaper(title="Included Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a"]),
        CorpusPaper(title="Ignored Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["2026/ignore/topic-b"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Included Paper"]


def test_filter_corpus_no_filters_returns_all():
    executor = _make_executor()
    corpus = [
        CorpusPaper(title="Paper A", abstract="", added_date=datetime(2026, 1, 1), paths=["foo"]),
        CorpusPaper(title="Paper B", abstract="", added_date=datetime(2026, 1, 2), paths=["bar"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert filtered == corpus


# ---------------------------------------------------------------------------
# fetch_zotero_corpus
# ---------------------------------------------------------------------------


def test_fetch_zotero_corpus(config, monkeypatch):
    from tests.canned_responses import make_stub_zotero_client

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    executor = Executor.__new__(Executor)
    executor.config = config
    corpus = executor.fetch_zotero_corpus()

    assert len(corpus) == 2
    assert corpus[0].title == "Stub Paper 1"
    assert "survey/topic-a" in corpus[0].paths[0]


def test_fetch_zotero_corpus_paper_with_zero_collections(config, monkeypatch):
    from tests.canned_responses import make_stub_zotero_client

    items = [
        {
            "data": {
                "title": "No Collection Paper",
                "abstractNote": "Abstract.",
                "dateAdded": "2026-03-01T00:00:00Z",
                "collections": [],
            }
        }
    ]
    stub_zot = make_stub_zotero_client(items=items)
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    executor = Executor.__new__(Executor)
    executor.config = config
    corpus = executor.fetch_zotero_corpus()

    assert len(corpus) == 1
    assert corpus[0].paths == []


# ---------------------------------------------------------------------------
# E2E: Executor.run()
# ---------------------------------------------------------------------------


def test_run_end_to_end(config, monkeypatch, tmp_path):
    """Full pipeline: Zotero fetch -> filter -> retrieve -> rerank -> TLDR -> email."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import (
        make_sample_corpus,
        make_sample_paper,
        make_stub_openai_client,
        make_stub_smtp,
        make_stub_zotero_client,
    )

    # Config: source=["arxiv"], reranker="api", send_empty=false
    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False

    # 1. Stub pyzotero
    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    # 2. Stub OpenAI (for reranker + TLDR/affiliations)
    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)
    retrieved = [
        make_sample_paper(title="E2E Paper 1", score=None),
        make_sample_paper(title="E2E Paper 2", score=None),
    ]

    # Import to register the arxiv retriever
    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(
        registered_retrievers["arxiv"],
        "retrieve_papers",
        lambda self: retrieved,
    )

    # 4. Stub SMTP
    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))

    # 4b. Isolate recommendation writes so the E2E test doesn't pollute the
    # real data/recommendations/ (which the history filter reads) or get
    # suppressed by a prior test run's writes.
    monkeypatch.setattr("zotero_arxiv_daily.executor.RECOMMENDATIONS_DIR", tmp_path)

    # 5. (No sleep stub needed — base.retrieve_papers no longer sleeps.)

    # 6. Run
    executor = Executor(config)
    executor.run()

    # Assertions
    assert len(sent) == 1, "Email should have been sent"
    _, _, email_body = sent[0]
    assert "text/html" in email_body


def test_run_no_papers_send_empty_false(config, monkeypatch, tmp_path):
    """When no papers are found and send_empty=false, no email is sent."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [])

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.executor.RECOMMENDATIONS_DIR", tmp_path)

    executor = Executor(config)
    executor.run()

    assert len(sent) == 0, "No email should be sent when no papers and send_empty=false"


def test_run_no_papers_send_empty_true(config, monkeypatch, tmp_path):
    """When no papers are found and send_empty=true, empty email is sent."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = True

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [])

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.executor.RECOMMENDATIONS_DIR", tmp_path)

    executor = Executor(config)
    executor.run()

    assert len(sent) == 1, "Email should be sent even with no papers when send_empty=true"
    _, _, body = sent[0]
    assert "text/html" in body


# ---------------------------------------------------------------------------
# _dedup_against_corpus
# ---------------------------------------------------------------------------


def _make_dedup_executor(config=None):
    """Bare Executor instance with only the dedup/history-filter methods exercised.

    ``config`` is needed by ``_filter_already_recommended`` (it reads
    ``config.executor.filter_recommended_history``). Dedup-only tests can omit it.
    """
    executor = Executor.__new__(Executor)
    executor.config = config
    return executor


def test_dedup_drops_candidate_already_in_corpus():
    """Candidate whose arxiv id matches a corpus paper is removed."""
    from tests.canned_responses import make_sample_paper

    executor = _make_dedup_executor()
    corpus = [
        CorpusPaper(
            title="Saved",
            abstract="x",
            added_date=datetime(2026, 1, 1),
            paths=["2026/survey"],
            key="ABCD1234",
            arxiv_id="2405.14867",
        ),
    ]
    candidates = [
        # already in Zotero (id 2405.14867, different version) -> dropped
        make_sample_paper(title="Dup", url="http://arxiv.org/abs/2405.14867v3"),
        # brand new -> kept
        make_sample_paper(title="Fresh", url="http://arxiv.org/abs/2406.00001"),
    ]
    kept = executor._dedup_against_corpus(candidates, corpus)
    assert [p.title for p in kept] == ["Fresh"]


def test_dedup_keeps_all_when_corpus_has_no_arxiv_ids():
    """Corpus papers without an arxiv id can't match anything -> keep all."""
    executor = _make_dedup_executor()
    corpus = [
        CorpusPaper(title="Plain", abstract="x", added_date=datetime(2026, 1, 1), paths=["p"], key="ABCD1234", arxiv_id=None),
    ]
    from tests.canned_responses import make_sample_paper

    candidates = [
        make_sample_paper(title="A", url="http://arxiv.org/abs/2406.00001"),
        make_sample_paper(title="B", url="http://arxiv.org/abs/2406.00002"),
    ]
    kept = executor._dedup_against_corpus(candidates, corpus)
    assert len(kept) == 2


def test_dedup_keeps_candidates_without_arxiv_id():
    """Candidate whose url has no arxiv id is always kept (no false drop)."""
    executor = _make_dedup_executor()
    corpus = [
        CorpusPaper(title="Saved", abstract="x", added_date=datetime(2026, 1, 1), paths=["p"], key="ABCD1234", arxiv_id="2406.99999"),
    ]
    from tests.canned_responses import make_sample_paper

    candidates = [
        # biorxiv-style URL with no arxiv id -> never matched
        make_sample_paper(title="NoArxivId", url="https://www.biorxiv.org/content/10.1101/x"),
    ]
    kept = executor._dedup_against_corpus(candidates, corpus)
    assert [p.title for p in kept] == ["NoArxivId"]


def test_dedup_matches_version_stripped_id():
    """A corpus entry with v2 and candidate with v5 still match (version-stripped)."""
    executor = _make_dedup_executor()
    corpus = [
        CorpusPaper(title="Saved", abstract="x", added_date=datetime(2026, 1, 1), paths=["p"], key="ABCD1234", arxiv_id="2405.14867"),
    ]
    from tests.canned_responses import make_sample_paper

    candidates = [make_sample_paper(title="V5", url="http://arxiv.org/abs/2405.14867v5")]
    kept = executor._dedup_against_corpus(candidates, corpus)
    assert kept == []


# ---------------------------------------------------------------------------
# _filter_already_recommended (history suppression)
# ---------------------------------------------------------------------------


def _write_history_json(dir_path, name, arxiv_ids):
    """Write a recommendation-history JSON file with the given arxiv ids."""
    import json

    envelope = {
        "date": name,
        "count": len(arxiv_ids),
        "papers": [{"arxiv_id": aid, "title": f"Old {aid}"} for aid in arxiv_ids],
    }
    (dir_path / f"{name}.json").write_text(json.dumps(envelope), encoding="utf-8")


def _history_filter_config():
    """A minimal OmegaConf with the history-filter flag enabled."""
    from omegaconf import OmegaConf

    return OmegaConf.create({"executor": {"filter_recommended_history": True}})


def test_filter_already_recommended_drops_prior_picks(tmp_path, monkeypatch):
    """Candidates whose arxiv_id appears in a prior recommendation JSON are dropped."""
    from tests.canned_responses import make_sample_paper

    # Seed history: 2405.14867 was recommended on 2026-01-01
    _write_history_json(tmp_path, "2026-01-01", ["2405.14867"])
    monkeypatch.setattr("zotero_arxiv_daily.executor.RECOMMENDATIONS_DIR", tmp_path)

    executor = _make_dedup_executor(_history_filter_config())
    candidates = [
        make_sample_paper(title="Old", url="http://arxiv.org/abs/2405.14867v2"),
        make_sample_paper(title="New", url="http://arxiv.org/abs/2406.00001"),
    ]
    kept = executor._filter_already_recommended(candidates)
    assert [p.title for p in kept] == ["New"]


def test_filter_already_recommended_empty_when_no_history(tmp_path, monkeypatch):
    """With no history JSONs, all candidates are kept."""
    from tests.canned_responses import make_sample_paper

    monkeypatch.setattr("zotero_arxiv_daily.executor.RECOMMENDATIONS_DIR", tmp_path)
    executor = _make_dedup_executor(_history_filter_config())
    candidates = [
        make_sample_paper(title="A", url="http://arxiv.org/abs/2406.00001"),
        make_sample_paper(title="B", url="http://arxiv.org/abs/2406.00002"),
    ]
    assert executor._filter_already_recommended(candidates) == candidates


def test_filter_already_recommended_skips_latest_json(tmp_path, monkeypatch):
    """latest.json must not contribute to history (it mirrors today's run)."""
    from tests.canned_responses import make_sample_paper

    # Only latest.json mentions 2405.14867 — it should NOT suppress the candidate.
    _write_history_json(tmp_path, "latest", ["2405.14867"])
    monkeypatch.setattr("zotero_arxiv_daily.executor.RECOMMENDATIONS_DIR", tmp_path)

    executor = _make_dedup_executor(_history_filter_config())
    candidates = [make_sample_paper(title="Keep", url="http://arxiv.org/abs/2405.14867v1")]
    kept = executor._filter_already_recommended(candidates)
    assert [p.title for p in kept] == ["Keep"]


def test_filter_already_recommended_disabled_by_config(tmp_path, monkeypatch):
    """When filter_recommended_history=false, history is ignored even if present."""
    from omegaconf import OmegaConf
    from tests.canned_responses import make_sample_paper

    _write_history_json(tmp_path, "2026-01-01", ["2405.14867"])
    monkeypatch.setattr("zotero_arxiv_daily.executor.RECOMMENDATIONS_DIR", tmp_path)

    config = OmegaConf.create({"executor": {"filter_recommended_history": False}})
    executor = _make_dedup_executor(config)
    candidates = [make_sample_paper(title="Keep", url="http://arxiv.org/abs/2405.14867v1")]
    assert executor._filter_already_recommended(candidates) == candidates


# ---------------------------------------------------------------------------
# extract_arxiv_id_from_zotero
# ---------------------------------------------------------------------------


def test_extract_arxiv_id_from_doi_datacite():
    from zotero_arxiv_daily.protocol import extract_arxiv_id_from_zotero
    assert extract_arxiv_id_from_zotero({"DOI": "10.48550/arXiv.2405.14867v2"}) == "2405.14867"


def test_extract_arxiv_id_from_arxiv_url():
    from zotero_arxiv_daily.protocol import extract_arxiv_id_from_zotero
    assert extract_arxiv_id_from_zotero({"url": "http://arxiv.org/abs/cs.AI/0701234"}) == "cs.AI/0701234"
    assert extract_arxiv_id_from_zotero({"url": "https://arxiv.org/pdf/2406.00001v1"}) == "2406.00001"


def test_extract_arxiv_id_from_extra_field():
    from zotero_arxiv_daily.protocol import extract_arxiv_id_from_zotero
    assert extract_arxiv_id_from_zotero({"extra": "arXiv: 2401.00001"}) == "2401.00001"
    assert extract_arxiv_id_from_zotero({"extra": "Some notes\narXiv:2402.00002v2\nMore"}) == "2402.00002"


def test_extract_arxiv_id_returns_none_when_absent():
    from zotero_arxiv_daily.protocol import extract_arxiv_id_from_zotero
    assert extract_arxiv_id_from_zotero({"DOI": "10.1000/foo"}) is None
    assert extract_arxiv_id_from_zotero({"url": "https://ieeexplore.ieee.org/document/12345"}) is None
    assert extract_arxiv_id_from_zotero({}) is None
    assert extract_arxiv_id_from_zotero(None) is None


def test_fetch_zotero_corpus_populates_arxiv_id(config, monkeypatch):
    """fetch_zotero_corpus should set CorpusPaper.arxiv_id from Zotero data."""
    from tests.canned_responses import make_stub_zotero_client

    items = [
        {
            "key": "ARXIV001",
            "data": {
                "title": "Arxiv Preprint",
                "abstractNote": "A.",
                "dateAdded": "2026-03-01T00:00:00Z",
                "collections": [],
                "DOI": "10.48550/arXiv.2405.14867",
            },
        },
        {
            "key": "PLAIN02",
            "data": {
                "title": "Journal Article",
                "abstractNote": "B.",
                "dateAdded": "2026-03-02T00:00:00Z",
                "collections": [],
                "DOI": "10.1000/plain",
            },
        },
    ]
    stub_zot = make_stub_zotero_client(items=items)
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    executor = Executor.__new__(Executor)
    executor.config = config
    corpus = executor.fetch_zotero_corpus()

    by_title = {c.title: c.arxiv_id for c in corpus}
    assert by_title["Arxiv Preprint"] == "2405.14867"
    assert by_title["Journal Article"] is None
