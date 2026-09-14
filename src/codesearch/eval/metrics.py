"""Precision@k, recall@k, and MRR - pure functions over a retrieved-ids
list and a set of relevant ids, so they're trivial to unit-test against
hand-computed cases independent of the rest of the pipeline."""

from __future__ import annotations


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    # Distinct relevant items found, not occurrences - two different
    # retrieved chunks can both match the same relevant item (e.g. a class
    # summary chunk and a method chunk both overlapping one ground-truth
    # line range), which must not push recall above 1.0.
    found = {r for r in top_k if r in relevant}
    return len(found) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def macro_average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
