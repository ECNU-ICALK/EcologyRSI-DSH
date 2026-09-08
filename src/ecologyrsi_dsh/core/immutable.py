"""Deeply read-only JSON containers with unchanged JSON wire representation.

These prevent accidental mutation by trusted in-process consumers. They are
not an isolation boundary against arbitrary Python code.
"""
from collections.abc import Mapping


def _read_only(*args, **kwargs):
    raise TypeError("frozen_json_cannot_be_modified")


class FrozenDict(dict):
    __slots__ = ()

    def __new__(cls, value=()):
        result = dict.__new__(cls)
        dict.__init__(result, ((key, freeze_json(item)) for key, item in dict(value).items()))
        return result

    def __init__(self, value=()):
        pass  # __new__ constructs once; reinitialization cannot mutate it.

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _read_only

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class FrozenList(list):
    __slots__ = ()

    def __new__(cls, value=()):
        result = list.__new__(cls)
        list.__init__(result, (freeze_json(item) for item in value))
        return result

    def __init__(self, value=()):
        pass

    __setitem__ = __delitem__ = append = extend = insert = pop = remove = clear = reverse = sort = __iadd__ = __imul__ = _read_only

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def freeze_json(value):
    if isinstance(value, (FrozenDict, FrozenList)):
        return value
    if isinstance(value, Mapping):
        return FrozenDict(value)
    if isinstance(value, list):
        return FrozenList(value)
    if isinstance(value, tuple):
        return tuple(freeze_json(item) for item in value)
    return value


def thaw_json(value):
    """Return detached mutable JSON data for editing or serialization."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw_json(item) for item in value]
    return value
