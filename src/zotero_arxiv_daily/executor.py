from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper, extract_arxiv_id_from_zotero
import random
from datetime import datetime, timezone
from pathlib import Path
import json
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm

# Directory where recommendation snapshots are written. The GitHub Action
# commits this path back to the repo so an MCP tool can read it via a raw
# GitHub URL. ``latest.json`` is always overwritten; per-date files give
# history.
RECOMMENDATIONS_DIR = Path("data/recommendations")


def _load_recommended_history() -> set[str]:
    """Scan every ``data/recommendations/*.json`` (except ``latest.json``)
    and return the set of arXiv ids that have ever appeared in a top-N list.

    Used to permanently filter candidates we have already recommended — there
    is no value in surfacing the same paper twice. ``latest.json`` is excluded
    because it mirrors today's run; including it would let a paper that was
    recommended today immediately suppress itself on a re-run.
    """
    if not RECOMMENDATIONS_DIR.exists():
        return set()
    seen: set[str] = set()
    for jf in RECOMMENDATIONS_DIR.glob("*.json"):
        if jf.name == "latest.json":
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Skipping unreadable recommendation history file {jf}: {exc}")
            continue
        for p in data.get("papers", []):
            aid = p.get("arxiv_id")
            if aid:
                seen.add(aid)
    return seen


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']:c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        # Extract arxiv ids for candidate-vs-corpus deduplication. Only
        # preprints/journal articles with a recognizable arXiv id get one;
        # others fall back to None and are simply never matched against.
        for c in corpus:
            c['_arxiv_id'] = extract_arxiv_id_from_zotero(c['data'])
        arxiv_in_corpus = sum(1 for c in corpus if c.get('_arxiv_id'))
        if arxiv_in_corpus:
            logger.info(f"  {arxiv_in_corpus}/{len(corpus)} corpus papers have an arXiv id (dedup-eligible)")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths'],
            # Zotero item key drives the embedding cache; fall back to None for
            # items missing it (legacy data, test stubs) — they re-encode each run.
            key=c.get('key'),
            # arXiv id, if this Zotero paper came from arXiv — used to drop
            # candidates the user has already added to their library.
            arxiv_id=c.get('_arxiv_id'),
        ) for c in corpus]
    
    def export_recommendations(self, papers: list) -> None:
        """Persist today's reranked papers as JSON for agent consumption.

        Writes both ``data/recommendations/{YYYY-MM-DD}.json`` (history)
        and ``data/recommendations/latest.json`` (default query target).
        Always writes ``latest.json`` — even when ``papers`` is empty —
        so an agent can distinguish "no run yet" from "no papers today".
        """
        RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        envelope = {
            "date": today,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": len(papers),
            "sources": list(self.retrievers.keys()),
            "papers": [p.to_recommendation_dict() for p in papers],
        }
        for path in (RECOMMENDATIONS_DIR / f"{today}.json", RECOMMENDATIONS_DIR / "latest.json"):
            path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(f"Wrote {len(papers)} recommendations to {path}")

    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")
        return corpus

    def _dedup_against_corpus(self, candidates: list, corpus: list[CorpusPaper]) -> list:
        """Drop candidate papers already present in the user's Zotero library.

        Matches on normalized arXiv id (version-stripped). Candidates whose
        id is not extractable are always kept — we only filter on positive
        matches, never on absence of an id. Logs how many were dropped so a
        silent regression (e.g. ids no longer extracted) is visible.
        """
        corpus_ids = {c.arxiv_id for c in corpus if c.arxiv_id}
        if not corpus_ids:
            return candidates
        kept = []
        dropped = 0
        for p in candidates:
            cand_id = p.cache_key  # version-stripped arXiv id, or None
            if cand_id and cand_id in corpus_ids:
                dropped += 1
                logger.info(f"Dedup: dropping candidate {cand_id} (already in Zotero): {p.title}")
            else:
                kept.append(p)
        if dropped:
            logger.info(f"Dedup: removed {dropped}/{len(candidates)} candidates already in Zotero ({len(kept)} remain)")
        return kept

    def _filter_already_recommended(self, candidates: list) -> list:
        """Drop candidates whose arXiv id was recommended on any prior day.

        Reads the recommendation history from ``data/recommendations/*.json``
        (excluding ``latest.json``). A paper that already surfaced in a past
        top-N is permanently suppressed — no value in recommending it again.
        Disabled when ``config.executor.filter_recommended_history`` is false.
        Candidates without an extractable arXiv id are always kept.
        """
        if not self.config.executor.get("filter_recommended_history", True):
            return candidates
        history = _load_recommended_history()
        if not history:
            return candidates
        kept: list = []
        dropped = 0
        for p in candidates:
            cand_id = p.cache_key  # version-stripped arXiv id, or None
            if cand_id and cand_id in history:
                dropped += 1
                logger.info(f"History filter: dropping {cand_id} (recommended before): {p.title}")
            else:
                kept.append(p)
        if dropped:
            logger.info(
                f"History filter: removed {dropped}/{len(candidates)} previously "
                f"recommended ({len(kept)} remain)"
            )
        return kept

    
    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        # Drop candidates the user has already saved to Zotero — there's no
        # value in recommending a paper the user has read and curated. Match
        # is by normalized arXiv id; corpus papers without one can't match.
        all_papers = self._dedup_against_corpus(all_papers, corpus)
        # Drop candidates we have already recommended on any prior day. Reads
        # data/recommendations/*.json so the suppression persists across runs
        # without a separate state file. Placed after Zotero dedup so both
        # filters reduce the rerank workload.
        all_papers = self._filter_already_recommended(all_papers)
        # Evict cached embeddings for Zotero papers no longer in the active
        # corpus (deleted from Zotero since the last run). No-op for rerankers
        # without a cache (local/api); CachedLocalReranker does the real work.
        corpus_keys = {c.key for c in corpus if c.key}
        prune = getattr(self.reranker, "prune_corpus_cache", None)
        if callable(prune) and corpus_keys:
            prune(corpus_keys)
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            reranked_papers = self.reranker.rerank(all_papers, corpus)
            reranked_papers = reranked_papers[:self.config.executor.max_paper_num]
            logger.info("Generating TLDR and affiliations...")
            for p in tqdm(reranked_papers):
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. Persisting empty recommendations; no email will be sent.")
            self.export_recommendations(reranked_papers)
            return
        # Persist recommendations (history + latest) for agent consumption.
        self.export_recommendations(reranked_papers)
        logger.info("Sending email...")
        email_content = render_email(reranked_papers)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
