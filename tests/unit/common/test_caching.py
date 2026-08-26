"""Unit tests for intermediate result caching.

Tests for osipy/common/caching.py.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from osipy.common.caching import (
    CacheConfig,
    IntermediateCache,
    RetentionPolicy,
    configure_cache,
    get_cache,
)

# Cache-directory hardening only applies on POSIX; on Windows there is no
# os.getuid()/chmod() and the check is skipped in IntermediateCache itself.
posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only cache directory hardening"
)


class TestRetentionPolicy:
    """Tests for RetentionPolicy enum."""

    def test_policy_values(self) -> None:
        """Test retention policy values."""
        assert RetentionPolicy.TRANSIENT.value == "transient"
        assert RetentionPolicy.CACHED.value == "cached"
        assert RetentionPolicy.PERSISTENT.value == "persistent"


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CacheConfig()
        assert config.default_policy == RetentionPolicy.CACHED
        assert config.policies == {}
        assert config.cache_dir is None
        assert config.max_memory_mb == 1024
        assert config.max_age_seconds == 3600
        assert config.compression is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = CacheConfig(
            default_policy=RetentionPolicy.PERSISTENT,
            policies={"t1_map": RetentionPolicy.TRANSIENT},
            max_memory_mb=512,
            max_age_seconds=1800,
        )
        assert config.default_policy == RetentionPolicy.PERSISTENT
        assert config.policies["t1_map"] == RetentionPolicy.TRANSIENT


class TestIntermediateCache:
    """Tests for IntermediateCache class."""

    @pytest.fixture
    def cache(self) -> IntermediateCache:
        """Create cache for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            yield IntermediateCache(config)

    def test_put_and_get(self, cache: IntermediateCache) -> None:
        """Test basic put and get operations."""
        data = np.random.rand(10, 10, 5)

        cache.put("t1_map", "test_session", data)
        retrieved = cache.get("t1_map", "test_session")

        np.testing.assert_array_equal(retrieved, data)

    def test_get_nonexistent(self, cache: IntermediateCache) -> None:
        """Test getting non-existent key returns None."""
        result = cache.get("t1_map", "nonexistent")
        assert result is None

    def test_has_method(self, cache: IntermediateCache) -> None:
        """Test has method."""
        data = np.random.rand(5, 5, 3)

        assert not cache.has("t1_map", "test")
        cache.put("t1_map", "test", data)
        assert cache.has("t1_map", "test")

    def test_invalidate(self, cache: IntermediateCache) -> None:
        """Test invalidating cached data."""
        data = np.random.rand(5, 5, 3)

        cache.put("t1_map", "test", data)
        assert cache.has("t1_map", "test")

        cache.invalidate("t1_map", "test")
        assert not cache.has("t1_map", "test")

    def test_clear_all(self, cache: IntermediateCache) -> None:
        """Test clearing all cached data."""
        cache.put("t1_map", "test1", np.random.rand(5, 5, 3))
        cache.put("concentration", "test2", np.random.rand(5, 5, 3))

        cache.clear()

        assert not cache.has("t1_map", "test1")
        assert not cache.has("concentration", "test2")

    def test_clear_by_type(self, cache: IntermediateCache) -> None:
        """Test clearing by result type."""
        cache.put("t1_map", "test1", np.random.rand(5, 5, 3))
        cache.put("concentration", "test2", np.random.rand(5, 5, 3))

        cache.clear("t1_map")

        assert not cache.has("t1_map", "test1")
        assert cache.has("concentration", "test2")

    def test_transient_policy_not_cached(self) -> None:
        """Test transient policy doesn't cache data."""
        config = CacheConfig(policies={"t1_map": RetentionPolicy.TRANSIENT})
        cache = IntermediateCache(config)

        data = np.random.rand(5, 5, 3)
        cache.put("t1_map", "test", data)

        assert not cache.has("t1_map", "test")

    def test_persistent_policy_saves_to_disk(self) -> None:
        """Test persistent policy saves to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(
                cache_dir=Path(tmpdir),
                policies={"t1_map": RetentionPolicy.PERSISTENT},
            )
            cache = IntermediateCache(config)

            data = np.random.rand(5, 5, 3)
            cache.put("t1_map", "test", data)

            # Check file exists
            cache_files = list(Path(tmpdir).glob("*.npz"))
            assert len(cache_files) >= 1

    def test_memory_eviction(self) -> None:
        """Test memory eviction when limit reached."""
        config = CacheConfig(max_memory_mb=1)  # 1 MB limit
        cache = IntermediateCache(config)

        # Add data that exceeds limit
        for i in range(10):
            # Each array is ~1 MB
            data = np.random.rand(128, 128, 10)
            cache.put("data", f"test_{i}", data)

        stats = cache.get_stats()
        # Memory should be limited
        assert stats["memory_size_mb"] <= config.max_memory_mb + 1

    def test_get_stats(self, cache: IntermediateCache) -> None:
        """Test getting cache statistics."""
        data = np.random.rand(10, 10, 5)
        cache.put("t1_map", "test", data)

        stats = cache.get_stats()

        assert "memory_entries" in stats
        assert "memory_size_mb" in stats
        assert "max_memory_mb" in stats
        assert "cache_dir" in stats

        assert stats["memory_entries"] == 1

    def test_dict_data_storage(self, cache: IntermediateCache) -> None:
        """Test storing dict data."""
        data = {
            "param1": np.random.rand(5, 5, 3),
            "param2": np.random.rand(5, 5, 3),
        }

        cache.put("params", "test", data)
        retrieved = cache.get("params", "test")

        assert isinstance(retrieved, dict)
        np.testing.assert_array_equal(retrieved["param1"], data["param1"])

    def test_metadata_storage(self, cache: IntermediateCache) -> None:
        """Test storing metadata with data."""
        data = np.random.rand(5, 5, 3)
        metadata = {"source": "test", "version": 1}

        cache.put("t1_map", "test", data, metadata=metadata)
        # Metadata is stored but not returned by get
        retrieved = cache.get("t1_map", "test")

        assert retrieved is not None


class TestGlobalCache:
    """Tests for global cache functions."""

    def test_get_cache_singleton(self) -> None:
        """Test get_cache returns singleton."""
        cache1 = get_cache()
        cache2 = get_cache()

        assert cache1 is cache2

    def test_configure_cache(self) -> None:
        """Test configuring global cache."""
        config = CacheConfig(max_memory_mb=512)
        configure_cache(config)

        cache = get_cache()
        stats = cache.get_stats()

        assert stats["max_memory_mb"] == 512


_poison_executed = False


def _poison_payload() -> None:
    """Stand-in for arbitrary code a malicious cache file could run."""
    global _poison_executed
    _poison_executed = True


class _EvilReduce:
    """Object whose pickle reconstruction runs `_poison_payload`."""

    def __reduce__(self) -> tuple:
        return (_poison_payload, ())


class TestCachePoisoningRegression:
    """Regression tests for GH-171: cache poisoning via pickle deserialization.

    A malicious ``.npz`` file placed at a cache path (e.g. by another user
    on a shared machine, since the filename is a predictable hash of the
    cache key) must not be able to execute code when loaded, and the
    default cache directory must not be a single shared location that
    every local user can write into.
    """

    def test_planted_pickle_payload_is_not_executed(self) -> None:
        """A pre-planted malicious npz at the predictable cache path must
        not execute code when read back via get()."""
        global _poison_executed
        _poison_executed = False

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(
                cache_dir=Path(tmpdir),
                policies={"t1_map": RetentionPolicy.PERSISTENT},
            )
            cache = IntermediateCache(config)

            # Plant a file at the exact path osipy would use for this key,
            # containing an object whose __reduce__ runs arbitrary code.
            full_key = "t1_map:subject01"
            key_hash = hashlib.md5(full_key.encode()).hexdigest()
            poisoned_path = Path(tmpdir) / f"{key_hash}.npz"

            np.savez(
                poisoned_path,
                data=np.array(_EvilReduce(), dtype=object),
                created_at=np.array(time.time()),
            )

            result = cache.get("t1_map", "subject01")

            assert _poison_executed is False
            assert result is None

    def test_default_cache_dir_is_scoped_per_user(self) -> None:
        """The default cache directory must not be a single name shared by
        every local user (the predictable, world-writable path in GH-171)."""
        cache_dir = IntermediateCache._default_cache_dir()

        assert cache_dir.name != "osipy_cache"
        assert cache_dir.name.startswith("osipy_cache_")

    @posix_only
    def test_default_cache_dir_is_restricted_to_owner(self) -> None:
        """The cache directory should not be readable/writable by other
        local users."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir) / "osipy_cache")
            cache = IntermediateCache(config)

            mode = cache._cache_dir.stat().st_mode & 0o777
            assert mode == 0o700

    @posix_only
    def test_symlinked_cache_dir_is_rejected(self) -> None:
        """A cache_dir that is a symlink (e.g. planted by another user to
        redirect osipy's writes elsewhere) must be refused."""
        with tempfile.TemporaryDirectory() as tmpdir:
            real_target = Path(tmpdir) / "real_target"
            real_target.mkdir()
            symlink_path = Path(tmpdir) / "osipy_cache_link"
            symlink_path.symlink_to(real_target, target_is_directory=True)

            config = CacheConfig(cache_dir=symlink_path)
            with pytest.raises(RuntimeError, match="symlink"):
                IntermediateCache(config)

    @posix_only
    def test_cache_dir_owned_by_another_user_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pre-existing cache directory owned by a different user must be
        refused, even if permissions happen to look fine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "osipy_cache"
            cache_dir.mkdir()
            real_uid = cache_dir.stat().st_uid

            monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)

            config = CacheConfig(cache_dir=cache_dir)
            with pytest.raises(RuntimeError, match="owned by another user"):
                IntermediateCache(config)


class TestCacheWithPolicies:
    """Tests for cache with different policies per type."""

    def test_per_type_policies(self) -> None:
        """Test different policies for different types."""
        config = CacheConfig(
            default_policy=RetentionPolicy.CACHED,
            policies={
                "t1_map": RetentionPolicy.PERSISTENT,
                "temp": RetentionPolicy.TRANSIENT,
            },
        )
        cache = IntermediateCache(config)

        assert cache.get_policy("t1_map") == RetentionPolicy.PERSISTENT
        assert cache.get_policy("temp") == RetentionPolicy.TRANSIENT
        assert cache.get_policy("other") == RetentionPolicy.CACHED
