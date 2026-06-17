from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, CorpusPaper
import numpy as np
from typing import Type
class BaseReranker(ABC):
    def __init__(self, config:DictConfig):
        self.config = config

    def rerank(self, candidates:list[Paper], corpus:list[CorpusPaper]) -> list[Paper]:
        corpus = sorted(corpus,key=lambda x: x.added_date,reverse=True)
        time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
        time_decay_weight: np.ndarray = time_decay_weight / time_decay_weight.sum()
        # Pass candidate/corpus keys so caching rerankers can look up stored
        # vectors. Non-caching rerankers (local/api) ignore these kwargs.
        cand_keys = [c.cache_key for c in candidates]
        corpus_keys = [c.cache_key for c in corpus]
        sim = self.get_similarity_score(
            [c.abstract for c in candidates],
            [c.abstract for c in corpus],
            s1_keys=cand_keys,
            s2_keys=corpus_keys,
        )
        assert sim.shape == (len(candidates), len(corpus))
        scores = (sim * time_decay_weight).sum(axis=1) * 10 # [n_candidate]
        for s,c in zip(scores,candidates):
            c.score = s
        candidates = sorted(candidates,key=lambda x: x.score,reverse=True)
        return candidates

    @abstractmethod
    def get_similarity_score(
        self,
        s1: list[str],
        s2: list[str],
        s1_keys: list[str] | None = None,
        s2_keys: list[str] | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

registered_rerankers = {}

def register_reranker(name:str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls
    return decorator

def get_reranker_cls(name:str) -> Type[BaseReranker]:
    if name not in registered_rerankers:
        raise ValueError(f"Reranker {name} not found")
    return registered_rerankers[name]