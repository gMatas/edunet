from edunet.core import Graph
from edunet.core.ops_utils import clear_cache


__all__ = ['clear_caches']


def clear_caches(graph: Graph, keep_outputs: bool = True):
    for op in graph:
        clear_cache(op, keep_outputs)
