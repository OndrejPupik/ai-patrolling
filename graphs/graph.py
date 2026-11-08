"""Graph data structure and example graph builders for the patrolling env."""
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Graph:
    weights: List[float]
    adjacency: Dict[int, List[Tuple[int, float]]]

    @property
    def n(self) -> int:
        return len(self.weights)

    def neighbors(self, node: int) -> List[Tuple[int, float]]:
        return self.adjacency[node]


def _add_edge(adj: Dict[int, List[Tuple[int, float]]], u: int, v: int, length: float) -> None:
    adj[u].append((v, length))
    adj[v].append((u, length))


def make_cycle(n: int, weights: Optional[List[float]] = None, length: float = 1.0) -> Graph:
    weights = weights or [1.0] * n
    adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
    for i in range(n):
        _add_edge(adj, i, (i + 1) % n, length)
    return Graph(weights, adj)


def make_grid(rows: int, cols: int, weights: Optional[List[float]] = None, length: float = 1.0) -> Graph:
    n = rows * cols
    weights = weights or [1.0] * n
    adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}

    def idx(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                _add_edge(adj, idx(r, c), idx(r, c + 1), length)
            if r + 1 < rows:
                _add_edge(adj, idx(r, c), idx(r + 1, c), length)
    return Graph(weights, adj)


def make_star(n: int, weights: Optional[List[float]] = None, length: float = 1.0) -> Graph:
    weights = weights or [1.0] * n
    adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
    for i in range(1, n):
        _add_edge(adj, 0, i, length)
    return Graph(weights, adj)


def make_random(
    n: int,
    p: float = 0.3,
    seed: int = 0,
    length_range: Tuple[float, float] = (1.0, 3.0),
    weight_range: Tuple[float, float] = (1.0, 5.0),
) -> Graph:
    """A connected random graph: start from a random spanning tree (guarantees
    connectivity), then add extra edges with probability p."""
    rng = random.Random(seed)
    weights = [rng.uniform(*weight_range) for _ in range(n)]
    adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}

    nodes = list(range(n))
    rng.shuffle(nodes)
    for i in range(1, n):
        u = nodes[i]
        v = nodes[rng.randint(0, i - 1)]
        _add_edge(adj, u, v, rng.uniform(*length_range))

    for u in range(n):
        existing = {nb for nb, _ in adj[u]}
        for v in range(u + 1, n):
            if v in existing:
                continue
            if rng.random() < p:
                _add_edge(adj, u, v, rng.uniform(*length_range))

    return Graph(weights, adj)
