from typing import Set, List, Union, Iterable

from edunet.core import Operation


__all__ = [
    'collect_ops',
    'sort_ops',
    'clear_cache',
]


def collect_ops(op: Operation, include_ops: Set[Operation] = None) -> List[Operation]:
    ops = list()

    if include_ops is None:
        def __collect_ops(current_op: Operation):
            for input_op in current_op.inputs:
                __collect_ops(input_op)

            input_ops = current_op.inputs
            if len(input_ops) > 0:
                ops.extend(input_ops)

        __collect_ops(op)
        ops.append(op)

    elif len(include_ops) > 0:
        not_included_ops = include_ops.copy()

        def __collect_ops(current_op: Operation, included: bool) -> bool:
            for input_op in current_op.inputs:
                included |= __collect_ops(input_op, False)

            if current_op in not_included_ops:
                not_included_ops.remove(current_op)
                included = True

            if included:
                ops.append(current_op)

            return included

        __collect_ops(op, False)
        assert len(not_included_ops) == 0, 'Failed to include all specified operations.'

    else:
        ops.append(op)

    return ops


def sort_ops(ops: Iterable[Operation]) -> List[Operation]:
    sorted_ops = list()
    used_ops = set()

    def __sort(ops: Iterable[Operation]):
        for op in ops:
            if op in used_ops:
                continue
            if len(op.inputs) > 0:
                __sort(op.inputs)
            sorted_ops.append(op)
            used_ops.add(op)

    __sort(ops)
    return sorted_ops


def clear_cache(op: Union[Operation, Iterable[Operation]], keep_output: bool = True):
    for op in [op] if isinstance(op, Operation) else op:
        op.grads_dict.clear()
        if keep_output:
            return

        op.output.set_values(None)
