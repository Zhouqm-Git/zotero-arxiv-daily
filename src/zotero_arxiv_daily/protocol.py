from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
RawPaperItem = TypeVar('RawPaperItem')

@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    @property
    def cache_key(self) -> Optional[str]:
        """Stable id for embedding cache (version-stripped arXiv id, or None).

        Same candidate paper appearing across days reuses its cached embedding,
        so only brand-new arXiv entries pay the encode cost each day.
        """
        return _extract_arxiv_id(self.url) if self.url else None

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', 'English')
        prompt = f"Given the following information of a paper, generate a one-sentence TLDR summary in {lang}:\n\n"
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"

        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)

        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user. Your answer should be in {lang}.",
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        tldr = response.choices[0].message.content
        return tldr

    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]  # truncate to 2000 tokens
            prompt = enc.decode(prompt_tokens)
            affiliations = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **llm_params.get('generation_kwargs', {})
            )
            affiliations = affiliations.choices[0].message.content

            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]

            return affiliations

    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None

    def to_recommendation_dict(self) -> dict:
        """Export a lightweight dict for JSON persistence.

        Drops the heavy ``full_text`` (can be 50KB+) — agents triage via
        ``tldr`` + ``abstract`` and fetch the full PDF later. Extracts a
        normalized ``arxiv_id`` (version-stripped) from the URL so callers
        don't have to re-parse it.
        """
        arxiv_id = _extract_arxiv_id(self.url)
        return {
            "arxiv_id": arxiv_id,
            "title": self.title,
            "authors": list(self.authors) if self.authors else [],
            "abstract": self.abstract,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "tldr": self.tldr,
            "affiliations": list(self.affiliations) if self.affiliations else [],
            "score": round(float(self.score), 4) if self.score is not None else None,
            "source": self.source,
        }


# Matches both new-style (2405.14867, optionally with /vN) and old-style
# (cs.AI/0701234) arXiv identifiers. Used for persistence export only —
# the retriever already knows its own IDs at fetch time.
_ARXIV_ID_RE = re.compile(
    r'(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z\-]+)?/\d{7})'
)
_VERSION_SUFFIX_RE = re.compile(r'v\d+$')

# DataCite DOI that arXiv mints for every paper: ``10.48550/arXiv.<id>``.
_ARXIV_DOI_RE = re.compile(r'10\.48550/arXiv\.(.+)$', re.IGNORECASE)
# Zotero's "Extra" free-text field commonly carries an ``arXiv: <id>`` line
# (typed-field syntax) for items Zotero couldn't link to an arXiv record.
_ARXIV_EXTRA_RE = re.compile(r'arXiv\s*[:\s]\s*(\S+)', re.IGNORECASE)


def _extract_arxiv_id(url: str) -> Optional[str]:
    """Extract a version-stripped arXiv ID from a URL or raw string.

    ``2405.14867v2`` becomes ``2405.14867``; ``cs.AI/0701234`` is unchanged
    (old-style IDs carry no version suffix).
    """
    if not url:
        return None
    match = _ARXIV_ID_RE.search(url)
    if match is None:
        return None
    return _VERSION_SUFFIX_RE.sub('', match.group(0))


def extract_arxiv_id_from_zotero(data: dict) -> Optional[str]:
    """Best-effort extraction of a normalized arXiv id from a Zotero item.

    Zotero has no top-level ``arxivId`` key for most item types; the id can
    land in any of three places depending on how the item was imported. We
    check them in order of reliability and return the first hit, version
    suffix stripped so it matches candidate ids (``Paper.cache_key``).

    Checked fields:
      1. ``DOI`` shaped like ``10.48550/arXiv.<id>`` (DataCite-registered)
      2. ``url`` whose host is ``arxiv.org`` (e.g. ``http://arxiv.org/abs/...``)
      3. ``extra`` containing an ``arXiv: <id>`` line

    Returns ``None`` if nothing matches — caller treats that paper as not
    dedupable (it will still be scored, just not filtered out).
    """
    if not isinstance(data, dict):
        return None
    # 1. DOI
    doi = (data.get('DOI') or '').strip()
    if doi:
        m = _ARXIV_DOI_RE.match(doi)
        if m:
            # Validate shape so a stray ``10.48550/arXiv.foo`` doesn't slip in.
            validated = _extract_arxiv_id(m.group(1).strip())
            if validated:
                return validated
    # 2. URL — require arxiv.org host so a non-arxiv URL with a numeric tail
    # (e.g. a journal article id) doesn't get false-matched.
    url = (data.get('url') or '').strip()
    if url and 'arxiv.org' in url.lower():
        aid = _extract_arxiv_id(url)
        if aid:
            return aid
    # 3. Extra
    extra = data.get('extra') or ''
    if extra:
        m = _ARXIV_EXTRA_RE.search(extra)
        if m:
            validated = _extract_arxiv_id(m.group(1).strip().rstrip('.,;'))
            if validated:
                return validated
    return None


@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
    key: Optional[str] = None  # Zotero item key — stable id for the embedding cache
    arxiv_id: Optional[str] = None  # arXiv id if this paper came from arXiv — used to dedup candidates

    @property
    def cache_key(self) -> Optional[str]:
        """Stable id for embedding cache (Zotero item key, or None).

        Aligned with ``Paper.cache_key`` so the rerank loop can treat both
        candidate and corpus keys uniformly. ``None`` disables caching for that
        row (it will be re-encoded every run) — safe fallback.
        """
        return self.key
