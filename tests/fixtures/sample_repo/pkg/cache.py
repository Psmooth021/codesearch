"""A tiny in-memory LRU cache."""

from collections import OrderedDict


class LRUCache:
    """Fixed-capacity cache that evicts the least recently used entry."""

    def __init__(self, capacity=128):
        self.capacity = capacity
        self._data = OrderedDict()

    def get(self, key):
        """Look up a value by key, marking it as recently used."""
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key, value):
        """Insert or update a value, evicting the oldest entry if full."""
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)
