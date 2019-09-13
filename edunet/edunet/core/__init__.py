import abc
from typing import Optional, Tuple, List, Iterable, Set, Sequence, Dict, Union

import numpy as np
from numpy import ndarray


# class Variable(object):
#     def __init__(self, values: ndarray = None):
#         self.__values = values
#
#     @property
#     def values(self) -> Optional[ndarray]:
#         return self.__values
#
#     @values.setter
#     def values(self, a: ndarray):
#         self.__values = a
#
#     @property
#     def shape(self) -> Tuple[int, ...]:
#         return self.__values.shape
#
#     @property
#     def dtype(self) -> np.dtype:
#         return self.__values.dtype
#
#     def is_empty(self) -> bool:
#         return self.__values is None
#
#     def assign(self, other):
#         if not isinstance(other, Variable):
#             raise TypeError('Argument `other` must be an instance of `Variable` class.')
#         self.__values = other.values


class Variable(object):
    def __init__(
            self,
            values: Sequence[ndarray] = None,
            shape: Sequence[int] = None,
            dtype: Union[type, np.dtype] = None,
            nitems: int = None
    ):
        if values is None or len(values) == 0:
            if shape is None or dtype is None or nitems is None:
                raise ValueError('If argument `values` is None or empty, `shape`, `dtype` '
                                 'and `nitems` arguments must be provided.')

            self.__shape: Tuple[int, ...] = tuple(shape)
            self.__dtype: np.dtype = np.dtype(dtype)
            self.__nitems: int = nitems
            self.__values: List[Optional[ndarray]] = [None] * self.__nitems

        else:
            value = values[0]
            self.__shape: Tuple[int, ...] = value.shape
            self.__dtype: np.dtype = value.dtype
            self.__nitems: int = len(values)

            for i in range(1, self.__nitems):
                value = values[i]
                if value is None:
                    continue
                if value.shape != self.__shape or value.dtype != self.__dtype:
                    raise AssertionError('All non-None values must be of the same shape and dtype.')

            self.__values: List[Optional[ndarray]] = list(values)

    @property
    def nitems(self) -> int:
        return self.__nitems

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.__shape

    @property
    def dtype(self) -> np.dtype:
        return self.__dtype

    def get_values(self, index: int) -> Optional[ndarray]:
        return self.__values[index]

    def set_value(self, index: int, value: Optional[ndarray]):
        if value is not None:
            assert value.shape == self.__shape and value.dtype == self.__dtype, 'Value shape or dtype mismatch.'
        self.__values[index] = value

    def assign(self, other):
        if not isinstance(other, Variable):
            raise TypeError('Argument `other` must be an instance of `Variable` class.')
        if other.shape != self.__shape or (other.dtype != self.__dtype or other.nitems != self.__nitems):
            raise AssertionError('Both variables shape, dtype and nitems must match to assign values between them.')
        self.__values = other.__values

    def is_complete(self) -> bool:
        return all(val is None for val in self.__values)

    def as_tuple(self) -> Tuple[Optional[ndarray]]:
        return tuple(self.__values)

    def as_array(self) -> ndarray:
        return np.array(self.__values, self.__dtype)


class SingularVariable(Variable):
    def __init__(self, values: ndarray = None, shape=None, dtype=None):
        super(SingularVariable, self).__init__((values if values is None else [values]), shape, dtype, 1)

    @classmethod
    def from_variable(cls, variable: Variable):
        values = variable.get_values(0)
        return cls(values, variable.shape, variable.dtype)

    def get_values(self, **kwargs) -> Optional[ndarray]:
        return super().get_values(0)

    def set_value(self, value: Optional[ndarray], **kwargs):
        return super().set_value(0, value)


from edunet.core.operations import Operation
from edunet.core.operations import Input





# class Variable(object):
#     def __init__(
#             self,
#             shape: Sequence[int, ...],
#             dtype: Union[type, np.dtype],
#             values: Sequence[Optional[ndarray]] = None,
#             batch_size: int = None
#     ):
#         self.__shape: Tuple[int] = tuple(shape)
#         self.__dtype: np.dtype = np.dtype(dtype)
#
#         if values is None:
#             if batch_size is None:
#                 raise ValueError('Values or batch size arguments must be provided.')
#
#             self.__batch_size = batch_size
#             self.__values = [None] * self.batch_size
#         else:
#             if any(value.shape != self.__shape or value.dtype != self.__dtype for value in values):
#                 raise AssertionError('All values must be of the same shape and dtype.')
#
#             self.__values = list(values)
#             self.__batch_size = len(self.__values)
#
#     @property
#     def batch_size(self) -> int:
#         return self.__batch_size
#
#     def get_value(self, index: int) -> Optional[ndarray]:
#         return self.__values[index]
#
#     def set_value(self, index: int, value: Optional[ndarray]):
#         if value is not None:
#             assert value.shape == self.__shape and value.dtype == self.__dtype, 'Value shape or dtype mismatch.'
#         self.__values[index] = value
#
#     def as_tuple(self) -> Tuple[Optional[ndarray]]:
#         return tuple(self.__values)
#
#     @property
#     def shape(self) -> Tuple[int, ...]:
#         return self.__shape
#
#     @property
#     def dtype(self) -> np.dtype:
#         return self.__dtype
#
#     def is_complete(self) -> bool:
#         return all(val is None for val in self.__values)
#
#     def assign(self, other):
#         if not isinstance(other, Variable):
#             raise TypeError('Argument `other` must be an instance of `Variable` class.')
#         if other.shape != self.__shape or (other.dtype != self.__dtype or other.batch_size != self.__batch_size):
#             raise AssertionError('Both variables shape, dtype and batch size must match to assign values between them.')
#         self.__values = other.__values


class AbstractGraph(abc.ABC):

    @abc.abstractmethod
    def get_ops(self) -> Sequence[Operation]:
        pass


class FrozenGraph(AbstractGraph):
    def __init__(self, ordered_ops: Iterable[Operation]):
        self.__ops = tuple(ordered_ops)

    def get_ops(self) -> Tuple[Operation]:
        return self.__ops


class FlowGraph(AbstractGraph):
    def __init__(self, ops: Iterable[Operation] = None):
        self.__ops: Set[Operation] = set() if ops is None else set(ops)

    def get_ops(self) -> List[Operation]:
        return self.sorted()

    def add(self, op: Operation) -> Operation:
        self.__ops.add(op)
        return op

    def extend(self, ops: Iterable[Operation]):
        self.__ops.update(ops)

    def remove(self, op: Operation):
        self.__ops.remove(op)

    def contains(self, op: Operation) -> bool:
        return op in self.__ops

    def sorted(self) -> List[Operation]:
        sorted_ops = list()
        used_ops = set()

        def __sort(ops: List[Operation]):
            for op in ops:
                if op in used_ops:
                    continue
                if len(op.inputs) > 0:
                    __sort(op.inputs)
                sorted_ops.append(op)
                used_ops.add(op)

        __sort(list(self.__ops))
        return sorted_ops

    def freeze(self) -> FrozenGraph:
        ops = self.sorted()
        frozen_graph = FrozenGraph(ops)
        return frozen_graph


class FlowControl(object):
    def __init__(self, graph: AbstractGraph = None):
        self.__graph = graph

    def run_forward(self):
        for op in self.__graph.get_ops():
            op.forward()

    def run_backward(self):
        flow_map: Dict[Operation, Operation] = dict()
        for op in self.__graph.get_ops()[::-1]:
            gradients = None
            if op in flow_map:
                next_op = flow_map[op]
                gradients = next_op.grads_dict[op.outputs]

            op.backward(gradients)

            inputs_map = dict.fromkeys(op.inputs, op)
            flow_map.update(inputs_map)

    def run(
            self,
            outputs: List[Operation] = None,
            feed_dict: Dict[Input, ndarray] = None,
            backprop=False,
    ) -> Optional[List[Optional[ndarray]]]:
        if feed_dict is not None:
            feed_pairs(feed_dict)

        self.run_forward()

        if backprop:
            self.run_backward()

        if outputs:
            return [op.outputs.values for op in outputs]

    def feed(self, feed_dict: Dict[Operation, ndarray]):
        feed_set = set(feed_dict.keys())
        graph_ops_set = set(self.__graph.get_ops())

        # Handle operations that are not part of the graph.
        bad_ops = feed_set.difference(graph_ops_set)
        if len(bad_ops) > 0:
            raise AssertionError('Operations %s are not part of the graph.' % (str(bad_ops)))

        # Handle input operations that are unfed.
        missing_ops = graph_ops_set.difference(feed_set)
        missing_ops = [op for op in missing_ops if type(op) is Input]
        if len(missing_ops) > 0:
            raise AssertionError('Input operations %s are missing from the `feed_dict`.' % (str(missing_ops)))

        for op, values in feed_dict.items():
            if not isinstance(op, Input):
                raise TypeError('Feed dictionary key operations must be Input class instances.')
            op.feed(values)
