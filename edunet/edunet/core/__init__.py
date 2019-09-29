import abc
from typing import Optional, Tuple, List, Iterable, Set, Sequence, Dict, Union

import numpy as np
from numpy import ndarray


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


# class AbstractGraph(abc.ABC):
#
#     @abc.abstractmethod
#     def get_ops(self) -> Sequence[Operation]:
#         pass
#
#
# class FrozenGraph(AbstractGraph):
#     def __init__(self, ordered_ops: Iterable[Operation]):
#         self.__ops = tuple(ordered_ops)
#
#     def get_ops(self) -> Tuple[Operation]:
#         return self.__ops
#
#
# class FlowGraph(AbstractGraph):
#     def __init__(self, ops: Iterable[Operation] = None):
#         self.__ops: Set[Operation] = set() if ops is None else set(ops)
#
#     def get_ops(self) -> List[Operation]:
#         return self.sorted()
#
#     def add(self, op: Operation) -> Operation:
#         self.__ops.add(op)
#         return op
#
#     def extend(self, ops: Iterable[Operation]):
#         self.__ops.update(ops)
#
#     def remove(self, op: Operation):
#         self.__ops.remove(op)
#
#     def contains(self, op: Operation) -> bool:
#         return op in self.__ops
#
#     def sorted(self) -> List[Operation]:
#         sorted_ops = list()
#         used_ops = set()
#
#         def __sort(ops: List[Operation]):
#             for op in ops:
#                 if op in used_ops:
#                     continue
#                 if len(op.inputs) > 0:
#                     __sort(op.inputs)
#                 sorted_ops.append(op)
#                 used_ops.add(op)
#
#         __sort(list(self.__ops))
#         return sorted_ops
#
#     def freeze(self) -> FrozenGraph:
#         ops = self.sorted()
#         frozen_graph = FrozenGraph(ops)
#         return frozen_graph
#
#
# class FlowControl(object):
#     def __init__(self, graph: AbstractGraph = None):
#         self.__graph = graph
#
#     def run_forward(self):
#         for op in self.__graph.get_ops():
#             op.forward()
#
#     def run_backward(self):
#         flow_map: Dict[Operation, Operation] = dict()
#         for op in self.__graph.get_ops()[::-1]:
#             gradients = None
#             if op in flow_map:
#                 next_op = flow_map[op]
#                 gradients = next_op.grads_dict[op.outputs]
#
#             op.backward(gradients)
#
#             inputs_map = dict.fromkeys(op.inputs, op)
#             flow_map.update(inputs_map)
#
#     def run(
#             self,
#             outputs: List[Operation] = None,
#             feed_dict: Dict[Input, ndarray] = None,
#             backprop=False,
#     ) -> Optional[List[Optional[ndarray]]]:
#         if feed_dict is not None:
#             feed_pairs(feed_dict)
#
#         self.run_forward()
#
#         if backprop:
#             self.run_backward()
#
#         if outputs:
#             return [op.outputs.values for op in outputs]
#
#     def feed(self, feed_dict: Dict[Operation, ndarray]):
#         feed_set = set(feed_dict.keys())
#         graph_ops_set = set(self.__graph.get_ops())
#
#         # Handle operations that are not part of the graph.
#         bad_ops = feed_set.difference(graph_ops_set)
#         if len(bad_ops) > 0:
#             raise AssertionError('Operations %s are not part of the graph.' % (str(bad_ops)))
#
#         # Handle input operations that are unfed.
#         missing_ops = graph_ops_set.difference(feed_set)
#         missing_ops = [op for op in missing_ops if type(op) is Input]
#         if len(missing_ops) > 0:
#             raise AssertionError('Input operations %s are missing from the `feed_dict`.' % (str(missing_ops)))
#
#         for op, values in feed_dict.items():
#             if not isinstance(op, Input):
#                 raise TypeError('Feed dictionary key operations must be Input class instances.')
#             op.feed(values)
