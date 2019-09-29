import abc
from typing import Optional, Tuple, List, Iterable, Set, Sequence, Dict, Union

import numpy as np
from numpy import ndarray


__all__ = [
    'Variable',
    'Operation',
    'Graph',
    'Flow',
]


class Variable(object):
    def __init__(self, values: ndarray = None, shape: Sequence[int] = None, dtype: Union[type, np.dtype] = None):
        if values is None:
            if shape is None and dtype is None:
                raise ValueError(
                    'If argument `values` is None, `shape`, `dtype` arguments must be provided.')
            self.__shape: Tuple[int, ...] = tuple(shape)
            self.__dtype: np.dtype = np.dtype(dtype)
            self.__values: Optional[ndarray] = None
        else:
            self.__values: ndarray = values
            self.__shape: Tuple[int, ...] = values.shape
            self.__dtype: np.dtype = values.dtype

        self.__ndim: int = len(self.__shape)

    @property
    def values(self) -> Optional[ndarray]:
        return self.__values

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.__shape

    @property
    def ndim(self) -> int:
        return self.__ndim

    @property
    def dtype(self) -> np.dtype:
        return self.__dtype

    def set_values(self, values: Optional[ndarray]):
        if values is None:
            self.__values = None
            return

        if self.__shape != values.shape or self.__dtype != values.dtype:
            raise AssertionError('New values and the preset values `shape` and `dtype` must match.')
        self.__values = values

    def assign(self, other):
        if not isinstance(other, Variable):
            raise TypeError('Argument `other` must be an instance of `Variable` class.')
        if other.shape != self.__shape or other.dtype != self.__dtype:
            raise AssertionError('Both variables `shape` and `dtype` must match to assign values between them.')
        self.__values = other.__values

    def is_assigned(self) -> bool:
        return self.__values is not None


class Operation(abc.ABC):
    def __init__(
            self,
            inputs: list,
            var_list: List[Variable],
            shape: Sequence[int],
            dtype: Union[type, np.dtype],
            name: str
    ):
        self._inputs: List[Operation] = inputs
        self._name = name

        self.output: Variable = Variable(None, shape, dtype)
        self.var_list = var_list
        self.grads_dict: Dict[Variable, Variable] = dict()

    @property
    def inputs(self):
        return self._inputs

    @property
    def name(self):
        return self._name

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def compute_gradients(self, gradients: Variable = None):
        pass


from edunet.core.ops_utils import collect_ops, sort_ops


class Graph(object):
    def __init__(self, ops: Iterable[Operation] = None):
        self.__ops: Set[Operation] = set() if ops is None else set(ops)

    def __iter__(self):
        return iter(self.__ops)

    def add(self, op: Operation) -> Operation:
        self.__ops.add(op)
        return op

    def extend(self, ops: Iterable[Operation]):
        self.__ops.update(ops)

    def remove(self, op: Operation):
        self.__ops.remove(op)

    def contains(self, op: Operation) -> bool:
        return op in self.__ops

    def get_ops(self) -> List[Operation]:
        return list(self.__ops)

    def get_ordered_ops(self) -> List[Operation]:
        return sort_ops(self.__ops)


from edunet.core.ops import Input


class Flow(object):
    def __init__(self, graph: Graph = None):
        self.__graph = graph

    @property
    def graph(self):
        return self.__graph

    def run(
            self,
            ops: List[Operation],
            feed_dict: Dict[Operation, ndarray] = None,
    ) -> Optional[List[Optional[ndarray]]]:
        # Collect operations that must run before the specified operations do.
        running_ops_set = set()
        for op in ops:
            if op not in running_ops_set:
                running_ops_set.update(collect_ops(op))

        feed_dict = dict() if feed_dict is None else feed_dict
        feed_set = set(feed_dict.keys())

        # Handle operations that are not part of the graph.
        bad_ops = feed_set.difference(running_ops_set)
        if len(bad_ops) > 0:
            raise ValueError('Operations %s are not part of the graph.' % (str(list(bad_ops))))

        # Handle input operations that are unfed.
        missing_ops = running_ops_set.difference(feed_set)
        missing_ops = [op for op in missing_ops if type(op) is Input]
        if len(missing_ops) > 0:
            raise ValueError('Input operations %s are missing from the `feed_dict`.' % (str(missing_ops)))

        # Feed all input operations with the given values.
        for op, values in feed_dict.items():
            if not isinstance(op, Input):
                raise TypeError('Feed dictionary key operations must be an Input class instances.')
            op.feed(values)

        # Run topologically sorted operations.
        for op in sort_ops(running_ops_set):
            op.run()

        # Extract and return specified outputs values.
        outputs = [op.output.values for op in ops]
        return outputs
