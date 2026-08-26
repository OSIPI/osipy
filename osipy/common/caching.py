"""Intermediate result caching for perfusion analysis.

This module provides configurable caching for intermediate computation
results like T1 maps and concentration curves.

"""

from __future__ import annotations

import contextlib
import getpass
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

T = TypeVar("T")


class RetentionPolicy(Enum):
    """Retention policy for intermediate results.

    Attributes
    ----------
    TRANSIENT : str
        Results are discarded after use. Minimal memory footprint.
    CACHED : str
        Results are cached in memory during session.
        Cleared when session ends.
    PERSISTENT : str
        Results are saved to disk and survive between sessions.
    """

    TRANSIENT = "transient"
    CACHED = "cached"
    PERSISTENT = "persistent"


@dataclass
class CacheConfig:
    """Configuration for intermediate result caching.

    Attributes
    ----------
    default_policy : RetentionPolicy
        Default retention policy for intermediate results.
    policies : dict[str, RetentionPolicy]
        Per-type retention policies (overrides default).
    cache_dir : Path | None
        Directory for persistent cache. Uses temp dir if None.
    max_memory_mb : int
        Maximum memory for cached results in MB.
    max_age_seconds : int
        Maximum age for cached results (0 = no limit).
    compression : bool
        Whether to compress persistent cache files.
    """

    default_policy: RetentionPolicy = RetentionPolicy.CACHED
    policies: dict[str, RetentionPolicy] = field(default_factory=dict)
    cache_dir: Path | None = None
    max_memory_mb: int = 1024  # 1 GB
    max_age_seconds: int = 3600  # 1 hour
    compression: bool = True


@dataclass
class CacheEntry:
    """Single cache entry.

    Attributes
    ----------
    key : str
        Unique identifier for this entry.
    data : Any
        Cached data (numpy array or dict).
    created_at : float
        Timestamp when entry was created.
    size_bytes : int
        Approximate size of cached data in bytes.
    result_type : str
        Type of intermediate result (e.g., "t1_map", "concentration").
    metadata : dict
        Additional metadata about the cached result.
    """

    key: str
    data: Any
    created_at: float
    size_bytes: int
    result_type: str
    metadata: dict = field(default_factory=dict)


class IntermediateCache:
    """Cache for intermediate computation results.

    Provides memory and disk caching with configurable retention
    policies per result type.

    Parameters
    ----------
    config : CacheConfig | None
        Cache configuration.

    Examples
    --------
    >>> from osipy.common.caching import IntermediateCache, CacheConfig, RetentionPolicy
    >>> config = CacheConfig(
    ...     default_policy=RetentionPolicy.CACHED,
    ...     policies={"t1_map": RetentionPolicy.PERSISTENT},
    ... )
    >>> cache = IntermediateCache(config)
    >>> cache.put("t1_map", "session1_t1", t1_data)
    >>> t1 = cache.get("t1_map", "session1_t1")
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        self.config = config or CacheConfig()
        self._memory_cache: dict[str, CacheEntry] = {}
        self._total_memory_bytes: int = 0

        # Set up cache directory for persistent storage
        if self.config.cache_dir is None:
            self._cache_dir = self._default_cache_dir()
        else:
            self._cache_dir = Path(self.config.cache_dir)

        self._prepare_cache_dir()

    @staticmethod
    def _default_cache_dir() -> Path:
        """Build a cache directory scoped to the current user.

        Using a fixed, shared directory name (e.g. ``<tmp>/osipy_cache``)
        would let any other local user read or pre-plant files there.
        Scoping by username keeps the default cache private on
        multi-user systems (HPC clusters, CI runners, shared containers).
        """
        try:
            user = getpass.getuser()
        except OSError:
            user = "default"
        safe_user = re.sub(r"[^A-Za-z0-9_.-]", "_", user)
        return Path(tempfile.gettempdir()) / f"osipy_cache_{safe_user}"

    def _prepare_cache_dir(self) -> None:
        """Create the cache directory and ensure it is private to this user.

        On POSIX systems this refuses to use a cache directory that is a
        symlink or that another user owns, and restricts permissions to
        the owner only. This closes the local cache-poisoning path where
        another user on a shared machine pre-creates or hijacks the cache
        directory to plant files.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        if sys.platform == "win32":
            return

        if self._cache_dir.is_symlink():
            raise RuntimeError(
                f"Cache directory {self._cache_dir} is a symlink; refusing "
                "to use it. Configure `cache_dir` to a private, non-symlink "
                "location."
            )

        owner_uid = self._cache_dir.stat().st_uid
        if owner_uid != os.getuid():
            raise RuntimeError(
                f"Cache directory {self._cache_dir} is owned by another "
                "user; refusing to use it. Configure `cache_dir` to a "
                "private location."
            )

        self._cache_dir.chmod(0o700)

    def get_policy(self, result_type: str) -> RetentionPolicy:
        """Get retention policy for a result type.

        Parameters
        ----------
        result_type : str
            Type of intermediate result.

        Returns
        -------
        RetentionPolicy
            Applicable retention policy.
        """
        return self.config.policies.get(result_type, self.config.default_policy)

    def put(
        self,
        result_type: str,
        key: str,
        data: NDArray[Any] | dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store intermediate result.

        Parameters
        ----------
        result_type : str
            Type of result (e.g., "t1_map", "concentration").
        key : str
            Unique key for this result.
        data : NDArray | dict
            Data to cache.
        metadata : dict | None
            Additional metadata.
        """
        policy = self.get_policy(result_type)

        if policy == RetentionPolicy.TRANSIENT:
            # Don't cache transient results
            return

        full_key = f"{result_type}:{key}"
        size_bytes = self._estimate_size(data)
        metadata = metadata or {}

        entry = CacheEntry(
            key=full_key,
            data=data,
            created_at=time.time(),
            size_bytes=size_bytes,
            result_type=result_type,
            metadata=metadata,
        )

        if policy == RetentionPolicy.CACHED:
            self._put_memory(entry)
        elif policy == RetentionPolicy.PERSISTENT:
            self._put_disk(entry)
            # Also keep in memory for fast access
            self._put_memory(entry)

    def get(
        self,
        result_type: str,
        key: str,
    ) -> NDArray[Any] | dict[str, Any] | None:
        """Retrieve intermediate result.

        Parameters
        ----------
        result_type : str
            Type of result.
        key : str
            Unique key.

        Returns
        -------
        NDArray | dict | None
            Cached data or None if not found.
        """
        policy = self.get_policy(result_type)

        if policy == RetentionPolicy.TRANSIENT:
            return None

        full_key = f"{result_type}:{key}"

        # Try memory cache first
        if full_key in self._memory_cache:
            entry = self._memory_cache[full_key]
            if self._is_valid(entry):
                return entry.data

        # Try disk cache for persistent
        if policy == RetentionPolicy.PERSISTENT:
            data = self._get_disk(full_key)
            if data is not None:
                return data

        return None

    def has(self, result_type: str, key: str) -> bool:
        """Check if result is cached.

        Parameters
        ----------
        result_type : str
            Type of result.
        key : str
            Unique key.

        Returns
        -------
        bool
            True if result is cached and valid.
        """
        return self.get(result_type, key) is not None

    def invalidate(self, result_type: str, key: str) -> None:
        """Invalidate (remove) cached result.

        Parameters
        ----------
        result_type : str
            Type of result.
        key : str
            Unique key.
        """
        full_key = f"{result_type}:{key}"

        # Remove from memory
        if full_key in self._memory_cache:
            entry = self._memory_cache.pop(full_key)
            self._total_memory_bytes -= entry.size_bytes

        # Remove from disk
        cache_file = self._get_cache_path(full_key)
        if cache_file.exists():
            cache_file.unlink()

    def clear(self, result_type: str | None = None) -> None:
        """Clear cached results.

        Parameters
        ----------
        result_type : str | None
            If provided, only clear results of this type.
            If None, clear all results.
        """
        if result_type is None:
            self._memory_cache.clear()
            self._total_memory_bytes = 0

            # Clear disk cache
            for cache_file in self._cache_dir.glob("*.npz"):
                cache_file.unlink()
        else:
            # Clear specific type
            keys_to_remove = [
                k for k in self._memory_cache if k.startswith(f"{result_type}:")
            ]
            for key in keys_to_remove:
                entry = self._memory_cache.pop(key)
                self._total_memory_bytes -= entry.size_bytes

                cache_file = self._get_cache_path(key)
                if cache_file.exists():
                    cache_file.unlink()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns
        -------
        dict
            Cache statistics including size, count, hit rate.
        """
        return {
            "memory_entries": len(self._memory_cache),
            "memory_size_mb": self._total_memory_bytes / (1024 * 1024),
            "max_memory_mb": self.config.max_memory_mb,
            "cache_dir": str(self._cache_dir),
            "disk_entries": len(list(self._cache_dir.glob("*.npz"))),
        }

    def _put_memory(self, entry: CacheEntry) -> None:
        """Store entry in memory cache with eviction."""
        # Check if we need to evict
        max_bytes = self.config.max_memory_mb * 1024 * 1024
        while (
            self._total_memory_bytes + entry.size_bytes > max_bytes
            and self._memory_cache
        ):
            self._evict_oldest()

        self._memory_cache[entry.key] = entry
        self._total_memory_bytes += entry.size_bytes

    def _evict_oldest(self) -> None:
        """Evict oldest entry from memory cache."""
        if not self._memory_cache:
            return

        oldest_key = min(
            self._memory_cache.keys(),
            key=lambda k: self._memory_cache[k].created_at,
        )

        entry = self._memory_cache.pop(oldest_key)
        self._total_memory_bytes -= entry.size_bytes

    def _put_disk(self, entry: CacheEntry) -> None:
        """Store entry on disk.

        Writes to a temporary file in the cache directory first, then
        atomically renames it into place, so a reader never observes a
        partially written file and a concurrent writer can't corrupt an
        in-progress read.
        """
        cache_file = self._get_cache_path(entry.key)
        fd, tmp_name = tempfile.mkstemp(
            dir=self._cache_dir, prefix=cache_file.stem, suffix=".tmp"
        )

        try:
            with os.fdopen(fd, "wb") as tmp_file:
                if isinstance(entry.data, np.ndarray):
                    save = np.savez_compressed if self.config.compression else np.savez
                    save(
                        tmp_file,
                        data=entry.data,
                        metadata=json.dumps(entry.metadata),
                        result_type=entry.result_type,
                        created_at=entry.created_at,
                    )
                elif isinstance(entry.data, dict):
                    np.savez_compressed(
                        tmp_file,
                        **{f"data_{k}": v for k, v in entry.data.items()},
                        metadata=json.dumps(entry.metadata),
                        result_type=entry.result_type,
                        created_at=entry.created_at,
                    )
                else:
                    return
            Path(tmp_name).replace(cache_file)
        except Exception as e:
            warnings.warn(f"Failed to save cache to disk: {e}", stacklevel=2)
        finally:
            with contextlib.suppress(FileNotFoundError):
                Path(tmp_name).unlink()

    def _get_disk(self, key: str) -> NDArray[Any] | dict[str, Any] | None:
        """Retrieve entry from disk."""
        cache_file = self._get_cache_path(key)

        if not cache_file.exists():
            return None

        try:
            # allow_pickle=False is intentional: cache entries only ever
            # contain numeric arrays and JSON-encoded strings, so pickle
            # support is never needed and would let a file planted by
            # another user execute arbitrary code on load.
            loaded = np.load(cache_file, allow_pickle=False)

            # Check age
            created_at = float(loaded.get("created_at", 0))
            if self.config.max_age_seconds > 0:
                age = time.time() - created_at
                if age > self.config.max_age_seconds:
                    cache_file.unlink()
                    return None

            # Return data
            if "data" in loaded:
                return loaded["data"]
            else:
                # Dict format
                return {k[5:]: v for k, v in loaded.items() if k.startswith("data_")}

        except Exception:
            return None

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Create hash of key for filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self._cache_dir / f"{key_hash}.npz"

    def _is_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid."""
        if self.config.max_age_seconds <= 0:
            return True

        age = time.time() - entry.created_at
        return age <= self.config.max_age_seconds

    def _estimate_size(self, data: Any) -> int:
        """Estimate size of data in bytes."""
        if isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, dict):
            return sum(self._estimate_size(v) for v in data.values())
        else:
            return 1000  # Default estimate


# Global cache instance
_global_cache: IntermediateCache | None = None


def get_cache() -> IntermediateCache:
    """Get global cache instance.

    Returns
    -------
    IntermediateCache
        Global cache instance.
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = IntermediateCache()
    return _global_cache


def configure_cache(config: CacheConfig) -> None:
    """Configure global cache.

    Parameters
    ----------
    config : CacheConfig
        Cache configuration.
    """
    global _global_cache
    _global_cache = IntermediateCache(config)
