"""Bounded, revision-keyed cache for immutable browser detail sections."""
from collections import OrderedDict
from concurrent.futures import Future
import json
import threading


class ReadSectionCache:
    def __init__(self, max_bytes=16 * 1024 * 1024, max_entries=64):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._bytes = 0
        self._values = OrderedDict()
        self._pending = {}
        self._lock = threading.Lock()

    def get_or_build(self, key, build):
        with self._lock:
            encoded = self._values.get(key)
            if encoded is not None:
                self._values.move_to_end(key)
                future, owner = None, False
            else:
                future = self._pending.get(key)
                owner = future is None
                if owner:
                    future = self._pending[key] = Future()
        if encoded is not None:
            return json.loads(encoded)
        if not owner:
            return json.loads(future.result())
        try:
            encoded = json.dumps(build(), ensure_ascii=False, allow_nan=False,
                                 separators=(',', ':')).encode('utf-8')
            with self._lock:
                if len(encoded) <= self.max_bytes:
                    self._values[key] = encoded
                    self._bytes += len(encoded)
                    while self._bytes > self.max_bytes or len(self._values) > self.max_entries:
                        _, old = self._values.popitem(last=False)
                        self._bytes -= len(old)
                self._pending.pop(key, None)
                future.set_result(encoded)
            return json.loads(encoded)
        except BaseException as exc:
            with self._lock:
                self._pending.pop(key, None)
                future.set_exception(exc)
            raise
