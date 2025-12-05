"""
Hyperbolic Cache - Spatially-aware cache for disk-backed memories.

Key insight: The cache is a WINDOW into hyperbolic space.
- Loaded regions: Hot neighborhoods in Hilbert space (in RAM)
- Unloaded regions: Cold areas on disk
- Window movement: Procedurally load/unload based on access patterns

This enables intelligent streaming of graph neighborhoods.
"""

import torch
from collections import OrderedDict
from typing import Optional, Any, Set, List


class HyperbolicCache:
    """
    Spatially-aware cache that tracks loaded/unloaded regions in hyperbolic space.
    
    Architecture:
    1. Loaded regions: Contiguous ranges in Hilbert space (hot in RAM)
    2. Unloaded regions: Everything else (on disk)
    3. Access pattern tracking: Where is the "hot spot" moving?
    4. Procedural loading: Stream in neighborhoods as needed
    5. Procedural unloading: Evict cold regions when full
    
    The cache is a sliding window over the hyperbolic memory manifold.
    """
    
    def __init__(self, capacity: int, poincare):
        """
        Args:
            capacity: Max number of nodes in RAM (hard limit for 5GB budget)
            poincare: PoincareManifold for distance computation
        """
        self.capacity = capacity
        self.poincare = poincare
        self.cache = OrderedDict()  # key -> bundle (CPU tensors)
        
        # 🌀 SPATIAL STATE TRACKING: Loaded vs Unloaded regions
        self.loaded_hilbert_indices: Set[int] = set()  # Hilbert indices currently in RAM
        self.key_to_hilbert = {}  # logical_idx -> hilbert_idx
        self.hilbert_to_key = {}  # hilbert_idx -> logical_idx
        
        # 📊 ACCESS PATTERN: Track hot spots for intelligent window movement
        self.recent_access_hilbert = []  # Recent Hilbert indices (FIFO, max 100)
        self.access_window_size = 100
        
        # 🌀 ADAPTIVE PREFETCH CONFIGURATION:
        # Cache usage < 50%: ±10 neighbors (20 nodes, aggressive sequential loading)
        # Cache usage 50-80%: ±5 neighbors (10 nodes, moderate)
        # Cache usage > 80%: ±2 neighbors (4 nodes, conservative)
        #
        # Why adaptive?
        # - Empty cache: Maximize sequential I/O, load entire neighborhoods
        # - Full cache: Minimize eviction thrashing, only load immediate needs
        # 
        # Graph structure: ~8-10 neighbors/node, 2-3 hop traversals
        # ±10 window covers: 1 node + 20 neighbors ≈ 2-hop neighborhood
        
    def access(self, embedding: torch.Tensor, key: Any, hilbert_idx: int = None):
        """
        Access an embedding, intelligently managing loaded regions.
        
        Strategy:
        1. Mark this Hilbert region as accessed (hot spot)
        2. If not loaded, add to load queue
        3. Predict which neighbors will be accessed next
        4. Unload cold regions if capacity exceeded
        
        Args:
            embedding: The embedding tensor (will be stored on CPU)
            key: Cache key (logical index)
            hilbert_idx: Hilbert curve index (position in hyperbolic space)
        
        Returns:
            List of neighbor keys to prefetch (unloaded neighbors in hot region)
        """
        # Store on CPU to save GPU memory
        if embedding.device.type != 'cpu':
            embedding = embedding.cpu()
        
        # Update cache (LRU order)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = embedding
        
        # 🌀 SPATIAL STATE UPDATE: Mark region as loaded
        if hilbert_idx is not None:
            self.loaded_hilbert_indices.add(hilbert_idx)
            self.key_to_hilbert[key] = hilbert_idx
            self.hilbert_to_key[hilbert_idx] = key
            
            # Track access pattern (sliding window)
            self.recent_access_hilbert.append(hilbert_idx)
            if len(self.recent_access_hilbert) > self.access_window_size:
                self.recent_access_hilbert.pop(0)
        
        # 💀 HARD CAPACITY ENFORCEMENT: Unload cold regions
        if len(self.cache) > self.capacity:
            self._unload_cold_regions()
        
        # 🔮 PREDICTIVE LOADING: Return unloaded neighbors in hot region
        neighbors_to_load = []
        if hilbert_idx is not None:
            neighbors_to_load = self._predict_unloaded_neighbors(hilbert_idx)
        
        return neighbors_to_load
    
    def _unload_cold_regions(self):
        """
        Intelligently unload cold regions when cache is full.
        
        Strategy:
        1. Identify hot region (center of recent accesses in Hilbert space)
        2. Evict nodes farthest from hot region in Hilbert space
        3. Update loaded_hilbert_indices to reflect unloaded regions
        """
        if len(self.cache) <= self.capacity:
            return
        
        # Calculate hot region center (median of recent accesses)
        if self.recent_access_hilbert:
            hot_center = sorted(self.recent_access_hilbert)[len(self.recent_access_hilbert) // 2]
        else:
            # Fallback: just evict oldest
            evicted_key = self.cache.popitem(last=False)[0]
            self._cleanup_evicted(evicted_key)
            return
        
        # Find nodes farthest from hot center in Hilbert space
        hilbert_distances = {}
        for key, h_idx in self.key_to_hilbert.items():
            if key in self.cache:
                # Hilbert distance = absolute difference (1D curve distance)
                hilbert_distances[key] = abs(h_idx - hot_center)
        
        # Evict 10% of cache (batch unload for efficiency)
        num_to_evict = max(1, (len(self.cache) - self.capacity) + len(self.cache) // 10)
        evict_keys = sorted(hilbert_distances.keys(), 
                           key=lambda k: hilbert_distances[k], 
                           reverse=True)[:num_to_evict]
        
        for key in evict_keys:
            if key in self.cache:
                del self.cache[key]
                self._cleanup_evicted(key)
    
    def _cleanup_evicted(self, key: Any):
        """Clean up spatial tracking for evicted node."""
        if key in self.key_to_hilbert:
            h_idx = self.key_to_hilbert[key]
            del self.key_to_hilbert[key]
            if h_idx in self.hilbert_to_key:
                del self.hilbert_to_key[h_idx]
            self.loaded_hilbert_indices.discard(h_idx)
    
    def _predict_unloaded_neighbors(self, hilbert_idx: int) -> List[Any]:
        """
        Predict which unloaded neighbors should be loaded next.
        
        Strategy: Load a WINDOW around current position in Hilbert space.
        - Graph has k_neighbors ≈ 8-10 edges per node
        - Graph traversal explores multiple hops (2-3 levels deep)
        - Prefetch ±10 neighbors = ~20 sequential disk reads
        
        Benefits:
        - One sequential disk read loads entire local neighborhood
        - Graph traversal becomes RAM-only after first access
        - Amortizes disk I/O cost over many memory operations
        
        Args:
            hilbert_idx: Current access position in Hilbert space
            
        Returns:
            List of keys for unloaded neighbors (max 20, adaptive)
        """
        unloaded_neighbors = []
        
        # Check spatial neighborhood in Hilbert space
        all_hilbert = sorted(self.hilbert_to_key.keys())
        if not all_hilbert or hilbert_idx not in all_hilbert:
            return []
        
        pos = all_hilbert.index(hilbert_idx)
        
        # 🌀 ADAPTIVE PREFETCH WINDOW
        # - If cache is mostly empty: Aggressively prefetch ±10 (20 nodes)
        # - If cache is full: Conservative ±2 (4 nodes)
        cache_usage = len(self.cache) / self.capacity if self.capacity > 0 else 1.0
        
        if cache_usage < 0.5:
            # Cache has room - aggressive prefetch
            prefetch_range = 10  # ±10 neighbors
            max_prefetch = 20
        elif cache_usage < 0.8:
            # Cache getting full - moderate prefetch
            prefetch_range = 5  # ±5 neighbors
            max_prefetch = 10
        else:
            # Cache nearly full - conservative prefetch
            prefetch_range = 2  # ±2 neighbors
            max_prefetch = 4
        
        # Look for UNLOADED neighbors (gaps in loaded region)
        for offset in range(-prefetch_range, prefetch_range + 1):
            if offset == 0:
                continue  # Skip current node
            
            neighbor_pos = pos + offset
            if 0 <= neighbor_pos < len(all_hilbert):
                neighbor_hilbert = all_hilbert[neighbor_pos]
                
                # Only suggest if NOT already loaded
                if neighbor_hilbert not in self.loaded_hilbert_indices:
                    if neighbor_hilbert in self.hilbert_to_key:
                        neighbor_key = self.hilbert_to_key[neighbor_hilbert]
                        unloaded_neighbors.append(neighbor_key)
        
        return unloaded_neighbors[:max_prefetch]
    
    def is_loaded(self, hilbert_idx: int) -> bool:
        """Check if a Hilbert region is currently loaded in RAM."""
        return hilbert_idx in self.loaded_hilbert_indices
    
    def get_loaded_regions(self) -> List[tuple]:
        """
        Get list of loaded regions as (start, end) Hilbert ranges.
        
        Useful for diagnostics and understanding cache state.
        """
        if not self.loaded_hilbert_indices:
            return []
        
        sorted_indices = sorted(self.loaded_hilbert_indices)
        regions = []
        start = sorted_indices[0]
        end = sorted_indices[0]
        
        for idx in sorted_indices[1:]:
            if idx == end + 1:
                # Contiguous region
                end = idx
            else:
                # Gap found, start new region
                regions.append((start, end))
                start = idx
                end = idx
        
        regions.append((start, end))
        return regions
    
    def get(self, key: Any) -> Optional[torch.Tensor]:
        """
        Retrieve embedding from cache, updating LRU order.
        
        Args:
            key: Cache key
            
        Returns:
            Embedding tensor (CPU) or None if not cached
        """
        if key not in self.cache:
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def evict_farthest(self, anchor: torch.Tensor, num_to_evict: int = 1):
        """
        Evict embeddings farthest from anchor point in hyperbolic space.
        
        This preserves local neighborhoods - if you're accessing memories
        about topic A, we keep memories near A and evict distant topics.
        
        Args:
            anchor: Reference embedding (recently accessed)
            num_to_evict: Number of embeddings to remove
        """
        if len(self.cache) <= num_to_evict:
            return  # Don't evict everything
        
        # Move anchor to CPU for distance computation
        if anchor.device.type != 'cpu':
            anchor = anchor.cpu()
        
        # Compute distances from anchor to all cached embeddings
        cached_embeddings = torch.stack(list(self.cache.values()))  # [N, dim]
        
        # Batch distance computation
        anchor_expanded = anchor.unsqueeze(0).expand(cached_embeddings.size(0), -1)
        dists = self.poincare.distance(anchor_expanded, cached_embeddings)
        
        # Map distances back to keys
        distances = {}
        for idx, key in enumerate(self.cache.keys()):
            distances[key] = dists[idx].item()
        
        # Evict farthest (highest distance)
        evict_keys = sorted(distances.keys(), key=lambda k: distances[k], reverse=True)[:num_to_evict]
        for key in evict_keys:
            del self.cache[key]
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
    
    def __len__(self):
        return len(self.cache)
    
    def size_mb(self):
        """Estimate cache size in MB."""
        if not self.cache:
            return 0.0
        sample = next(iter(self.cache.values()))
        bytes_per_emb = sample.numel() * sample.element_size()
        return (len(self.cache) * bytes_per_emb) / (1024 * 1024)
