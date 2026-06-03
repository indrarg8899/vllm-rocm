"""Prefix caching for prompt reuse optimization."""

import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    prefix_hash: str
    block_ids: List[int]
    ref_count: int = 0
    hit_count: int = 0


class PrefixCache:
    """Radix-tree inspired prefix cache for KV blocks."""

    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self._cache: Dict[str, CacheEntry] = {}
        self._lru: List[str] = []

    def _hash_prefix(self, tokens: Tuple[int, ...]) -> str:
        return hashlib.sha256(str(tokens).encode()).hexdigest()[:16]

    def lookup(
        self, tokens: List[int]
    ) -> Tuple[int, List[int]]:
        """Find longest matching prefix. Returns (matched_length, block_ids)."""
        best_len = 0
        best_blocks: List[int] = []

        for length in range(len(tokens), 0, -1):
            h = self._hash_prefix(tuple(tokens[:length]))
            entry = self._cache.get(h)
            if entry is not None and length > best_len:
                best_len = length
                best_blocks = entry.block_ids.copy()
                entry.hit_count += 1

        return best_len, best_blocks

    def insert(
        self,
        tokens: List[int],
        block_ids: List[int],
    ) -> None:
        """Cache a prefix with its block IDs."""
        h = self._hash_prefix(tuple(tokens))
        if h in self._cache:
            self._cache[h].ref_count += 1
            return

        if len(self._cache) >= self.max_entries:
            self._evict()

        self._cache[h] = CacheEntry(
            prefix_hash=h,
            block_ids=block_ids,
            ref_count=1,
        )
        self._lru.append(h)

    def _evict(self) -> None:
        if not self._lru:
            return
        evict_key = self._lru.pop(0)
        self._cache.pop(evict_key, None)

    def invalidate(self, tokens: List[int]) -> None:
        h = self._hash_prefix(tuple(tokens))
        entry = self._cache.pop(h, None)
        if entry:
            self._lru = [k for k in self._lru if k != h]

    def get_stats(self) -> Dict:
        total_hits = sum(e.hit_count for e in self._cache.values())
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "total_hits": total_hits,
            "utilization": f"{len(self._cache) / self.max_entries:.1%}",
        }
