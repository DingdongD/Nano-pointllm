"""
pytest conftest: compatibility shim for DynamicCache.

transformers >= 4.47 restructured DynamicCache to use a `layers` list of
DynamicLayer objects and removed the legacy `key_cache` / `value_cache`
list attributes. Tests written against the old API add tensors directly to
`cache.key_cache` and `cache.value_cache`.

This shim patches DynamicCache.__init__ so that every newly created instance
starts with `key_cache = []` and `value_cache = []`, making the legacy
test helpers work transparently.
"""
from __future__ import annotations

import transformers.cache_utils as _cu


def _patched_dynamic_cache_init(self, *args, **kwargs):
    _original_dynamic_cache_init(self, *args, **kwargs)
    if not hasattr(self, "key_cache"):
        self.key_cache = []
    if not hasattr(self, "value_cache"):
        self.value_cache = []


_original_dynamic_cache_init = _cu.DynamicCache.__init__
_cu.DynamicCache.__init__ = _patched_dynamic_cache_init
