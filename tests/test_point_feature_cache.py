import torch

from nanopointllm.engine.point_feature_cache import PointFeatureCache


def test_point_feature_cache_tracks_hits_and_lru():
    cache = PointFeatureCache(max_entries=2)
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    c = torch.randn(2, 3)

    assert cache.get("a") is None
    cache.put("a", a)
    cache.put("b", b)
    torch.testing.assert_close(cache.get("a"), a)
    cache.put("c", c)

    assert cache.get("b") is None
    torch.testing.assert_close(cache.get("a"), a)
    torch.testing.assert_close(cache.get("c"), c)
    assert cache.stats.misses >= 2
    assert cache.stats.hits >= 2
    assert cache.stats.evictions == 1
