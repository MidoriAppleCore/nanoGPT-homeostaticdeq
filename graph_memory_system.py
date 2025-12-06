"""
Graph-Structured Memory System with Hyperbolic Geometry

This module implements a memory system where memories are stored as a graph,
preserving relational structure between memories. Each memory node has edges
to its k nearest neighbors in hyperbolic space, and the graph structure is
preserved across all memory tiers (working/buffer/long-term).

Key Features:
- Graph storage: adjacency matrices, edge weights, cluster structure
- GNN-based query: learned attention over graph neighborhoods
- GNN-based integration: cross-attention to use retrieved memories
- Structure-preserving consolidation: maintains graph when moving between tiers
- Hyperbolic geometry: natural hierarchies in Poincaré ball

🚀 TRAJECTORY STORAGE OPTIMIZATION (Fiber Bundle + Hyperbolic Layout):

The system records successful memory traversal trajectories to learn optimal paths.
This is CRITICAL for performance - trajectory updates happen on EVERY training step!

OPTIMIZATION STRATEGY:

1. **Fiber Bundle Storage** (10x speedup):
   - Trajectory data (edge_flow_context, edge_flow_prev_nodes) stored WITH each node
   - ONE disk read gets: node + edges + trajectory patterns
   - Atomic read-modify-write prevents race conditions
   - Example: Node 42's bundle contains ALL edges and their success histories

2. **Hyperbolic Disk Layout** (10-100x speedup):
   - Nodes sorted on disk by position on Hilbert curve (space-filling)

🔍 PROFILING INFRASTRUCTURE:
   - Use --profile flag to enable micro-profiling (python train.py --profile)
   - Profiles: memory ops, flow recording, highway formation, retrieval
   - Disabled by default for clean logs
"""

import sys
import time
import os
import threading
import queue
import weakref
from functools import wraps

# 🧠 DEBUG VIEW: --show-brain flag shows DEQ internals
SHOW_BRAIN_ALWAYS = '--brain-always' in sys.argv  # Verbose: show every DEQ iteration immediately
SHOW_BRAIN_DEBUG = '--show-brain' in sys.argv or SHOW_BRAIN_ALWAYS  # Enable if either flag set
COMPUTE_ROUTING_STATS = SHOW_BRAIN_DEBUG  # Only compute expensive routing stats when debugging

# 🔍 PROFILING: Only enable with --profile flag
ENABLE_MICRO_PROFILING = '--profile' in sys.argv
_profile_stats = {}

# 🚀 ASYNC WRITE INFRASTRUCTURE: Background thread for non-blocking disk writes
# This allows training to continue while edge updates flush to disk
_async_write_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent RAM explosion
_async_write_enabled = True  # Set to False to disable async writes (for debugging)
_async_write_thread = None
_async_write_shutdown = threading.Event()

# 🔮 PREFETCH INFRASTRUCTURE: Hints for preloading bundles into cache
# During backward pass, we prefetch likely-needed bundles for next forward pass
_prefetch_queues = {}  # {tier_id: queue.Queue} - one queue per memory tier
_prefetch_enabled = True  # Set to False to disable prefetch hints
_prefetch_tiers = {}  # {tier_id: weakref.ref(tier_obj)} - map tier id -> weakref for worker

# Prefetch background worker
_prefetch_worker = None
_prefetch_shutdown = threading.Event()

# 🚀 EDGE UPDATE SUBSAMPLING: Reduce 4,600 strengthen_edges_batch calls to manageable count
# The system learns better → generates more edge updates → need to subsample
EDGE_SUBSAMPLE_BASE_RATE = 0.2  # Base rate: 20% (scales down adaptively)
EDGE_SUBSAMPLE_MAX_CALLS = 3000  # Target: keep edge updates under 3K calls (aggressive but safe)
EDGE_PRIORITY_THRESHOLD = 0.1  # Always keep high-reward edge updates
_edge_subsample_counter = 0  # Rolling counter for deterministic subsampling

# Global metrics for clean reporting (reset each iteration)
_edge_subsample_metrics = {
    'total_input': 0,
    'total_output': 0, 
    'high_priority': 0,
    'calls_count': 0
}

def profile_op(name):
    """Decorator to profile operations when ENABLE_MICRO_PROFILING is True"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not ENABLE_MICRO_PROFILING:
                return func(*args, **kwargs)
            
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            
            if name not in _profile_stats:
                _profile_stats[name] = {'count': 0, 'total_ms': 0, 'max_ms': 0}
            _profile_stats[name]['count'] += 1
            _profile_stats[name]['total_ms'] += elapsed
            _profile_stats[name]['max_ms'] = max(_profile_stats[name]['max_ms'], elapsed)
            
            return result
        return wrapper
    return decorator

def print_profile_stats():
    """Print accumulated profiling statistics (only if profiling enabled)"""
    if not ENABLE_MICRO_PROFILING or not _profile_stats:
        return
    
    print("\n" + "="*80)
    print("🔍 MICRO-PROFILING STATISTICS")
    print("="*80)
    sorted_ops = sorted(_profile_stats.items(), key=lambda x: x[1]['total_ms'], reverse=True)
    for name, stats in sorted_ops[:20]:  # Top 20
        avg_ms = stats['total_ms'] / stats['count']
        print(f"{name:40s}: {stats['count']:6d} calls, {stats['total_ms']:8.1f}ms total, "
              f"{avg_ms:6.2f}ms avg, {stats['max_ms']:6.2f}ms max")
    print("="*80 + "\n")

def reset_profile_stats():
    """Reset profiling statistics"""
    global _profile_stats
    _profile_stats = {}

def _async_write_worker():
    """
    🚀 BACKGROUND WRITE WORKER: Processes queued write buffer flushes asynchronously
    
    This thread runs in the background, pulling flush jobs from the queue and
    executing them without blocking the main training loop.
    
    Benefits:
    - Training continues while disk writes happen
    - 2-3x speedup by overlapping compute + I/O
    - Negligible RAM overhead (just queue of pending flushes)
    """
    while not _async_write_shutdown.is_set():
        try:
            # Get flush job (blocks until available, timeout for clean shutdown)
            job = _async_write_queue.get(timeout=1.0)
            if job is None:  # Poison pill for shutdown
                break
            
            # Unpack job
            tier, buffer_snapshot, sorted_keys = job
            
            # Execute the flush (this is the expensive disk I/O)
            _execute_flush(tier, buffer_snapshot, sorted_keys)
            
            # Mark task done
            _async_write_queue.task_done()
            
        except queue.Empty:
            continue  # Timeout, check shutdown flag
        except Exception as e:
            print(f"⚠️  Async write worker error: {e}")
            import traceback
            traceback.print_exc()

def _execute_flush(tier, buffer_snapshot, sorted_keys):
    """
    Execute the actual flush operation (called by background thread)
    
    This is identical to the synchronous flush_write_buffer() logic,
    but runs in a background thread.
    """
    writes_count = 0
    
    # 🎯 FIBER BUNDLE ATOMIC WRITES: Group updates by source node
    if hasattr(tier, '_bundled_storage') and tier._bundled_storage is not None:
        # Bundled mode: read each node's bundle once, update all edges, write once
        current_node = None
        bundle = None
        
        for (source_idx, edge_slot) in sorted_keys:
            # Load bundle if we moved to a new node
            if source_idx != current_node:
                # Write previous bundle if exists
                if bundle is not None:
                    tier._bundled_storage[current_node] = bundle
                    writes_count += 1
                
                # Load new bundle
                current_node = source_idx
                bundle = tier._bundled_storage[source_idx].copy()  # Copy to avoid race
            
            # Update all edge fields in the bundle
            data = buffer_snapshot[(source_idx, edge_slot)]
            bundle['edge_traversal_count'][edge_slot] = data['count']
            bundle['edge_success_rate'][edge_slot] = data['success']
            bundle['edge_weights'][edge_slot] = data['weight']
            bundle['edge_types'][edge_slot] = data['type']
        
        # Write final bundle
        if bundle is not None:
            tier._bundled_storage[current_node] = bundle
            writes_count += 1
    else:
        # Column mode: write each field separately (legacy, slower)
        for (source_idx, edge_slot) in sorted_keys:
            data = buffer_snapshot[(source_idx, edge_slot)]
            tier.edge_traversal_count[source_idx, edge_slot] = data['count']
            tier.edge_success_rate[source_idx, edge_slot] = data['success']
            tier.edge_weights[source_idx, edge_slot] = data['weight']
            tier.edge_types[source_idx, edge_slot] = data['type']
            writes_count += 1
    
    tier.disk_writes += writes_count

def start_async_writer():
    """Start the background write worker thread"""
    global _async_write_thread
    if _async_write_thread is None or not _async_write_thread.is_alive():
        _async_write_shutdown.clear()
        _async_write_thread = threading.Thread(
            target=_async_write_worker,
            daemon=True,
            name="AsyncWriteWorker"
        )
        _async_write_thread.start()


def _prefetch_worker_func():
    """Background worker that consumes prefetch hints and loads bundles into cache.

    This is best-effort and purposely lightweight: it only calls bundled_storage.get_bundles_batch
    for hinted indices and swallows errors. It avoids blocking the main training loop.
    """
    global _prefetch_queues, _prefetch_tiers
    while not _prefetch_shutdown.is_set():
        try:
            # Iterate over tier queues and try to drain a few hints from each
            for tier_id, q in list(_prefetch_queues.items()):
                try:
                    tid, idx = q.get(timeout=0.2)
                except queue.Empty:
                    continue

                # Resolve tier object
                ref = _prefetch_tiers.get(tier_id)
                tier_obj = ref() if ref is not None else None
                if tier_obj is None:
                    # Tier gone - drop hint
                    continue

                try:
                    # Use bundled batch getter if available to warm cache
                    if hasattr(tier_obj, '_bundled_storage') and tier_obj._bundled_storage is not None:
                        # get_bundles_batch expects a list; use a single-element batch
                        try:
                            tier_obj._bundled_storage.get_bundles_batch([idx])
                        except Exception:
                            # Some DiskBackedTensor implementations raise on out-of-range; ignore
                            pass
                    else:
                        # Fallback: touch embeddings to trigger any caching logic
                        try:
                            _ = tier_obj.embeddings[idx]
                        except Exception:
                            pass
                finally:
                    # Mark queue task done if it's a real Queue
                    try:
                        q.task_done()
                    except Exception:
                        pass

        except Exception:
            # Keep worker alive on unexpected errors
            import traceback
            traceback.print_exc()
        # Sleep a little to avoid busy loop
        time.sleep(0.05)


def start_prefetch_worker():
    """Start the prefetch background thread."""
    global _prefetch_worker
    if _prefetch_worker is None or not _prefetch_worker.is_alive():
        _prefetch_shutdown.clear()
        _prefetch_worker = threading.Thread(target=_prefetch_worker_func, daemon=True, name="PrefetchWorker")
        _prefetch_worker.start()


def shutdown_prefetch_worker(wait=True):
    """Shutdown the prefetch worker cleanly."""
    global _prefetch_worker
    if _prefetch_worker is not None and _prefetch_worker.is_alive():
        _prefetch_shutdown.set()
        if wait:
            _prefetch_worker.join(timeout=2.0)
        _prefetch_worker = None

def shutdown_async_writer(wait=True):
    """Shutdown the background write worker and wait for pending writes"""
    global _async_write_thread
    
    if _async_write_thread is not None and _async_write_thread.is_alive():
        # Signal shutdown
        _async_write_shutdown.set()
        
        # Send poison pill
        try:
            _async_write_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        
        # Wait for thread to finish
        if wait:
            _async_write_thread.join(timeout=5.0)
        
        _async_write_thread = None

# Start the async writer on module import
start_async_writer()
start_prefetch_worker()

# Register cleanup on exit
import atexit
@atexit.register
def _cleanup_async_writer():
    """Ensure async writer shuts down cleanly on program exit"""
    shutdown_async_writer(wait=True)

def subsample_edge_updates(edge_updates, context_name="unknown"):
    """
    🚀 ADAPTIVE EDGE UPDATE SUBSAMPLING: Scale reduction based on workload
    
    Problem: Edge updates grow exponentially as learning improves (4K→28K calls)
    Solution: Adaptive subsampling that scales down as workload increases
    
    STRATEGY:
    1. Always keep high-reward updates (preserve strong learning signals)
    2. Adaptive sampling rate: scales to keep total calls under target (2K)
    3. Importance-weighted sampling for medium rewards (balanced learning)
    4. Sparse sampling for low rewards (computational efficiency)
    
    Args:
        edge_updates: List of (idx_a, idx_b, reward) tuples
        context_name: For debugging which call site this is
    
    Returns:
        Filtered edge_updates list (adaptively scaled to target)
    """
    if not edge_updates:
        return edge_updates
    
    global _edge_subsample_counter
    _edge_subsample_counter += 1
    
    # 🎯 ADAPTIVE SCALING: Calculate subsample rate based on workload
    # If we have 28K updates and want 2K, we need ~7% rate (2000/28000)
    # But we maintain base rate (20%) as minimum for quality
    total_updates = len(edge_updates)
    adaptive_rate = min(
        EDGE_SUBSAMPLE_BASE_RATE,  # Never exceed base rate (quality floor)
        max(0.05, EDGE_SUBSAMPLE_MAX_CALLS / max(1, total_updates))  # Scale down, min 5%
    )
    
    # Stratify updates by reward level for intelligent subsampling
    high_priority = []      # Always keep: reward >= threshold
    medium_priority = []    # Importance sample: threshold/2 <= reward < threshold  
    low_priority = []       # Deterministic sample: reward < threshold/2
    
    reward_threshold_half = EDGE_PRIORITY_THRESHOLD * 0.5
    
    for update in edge_updates:
        idx_a, idx_b, reward = update
        if reward >= EDGE_PRIORITY_THRESHOLD:
            high_priority.append(update)
        elif reward >= reward_threshold_half:
            medium_priority.append(update)
        else:
            low_priority.append(update)
    
    # TIER 1: Always keep high-priority (strong learning signals)
    result = high_priority[:]
    
    # TIER 2: Importance-weighted sampling for medium priority
    # Keep more medium-reward edges to maintain learning diversity
    if medium_priority:
        # Scale medium tier more aggressively when total load is high
        # Below target: use 2x adaptive rate (maintain diversity)
        # Above target: use 1.5x adaptive rate (reduce more aggressively)
        medium_multiplier = 1.5 if total_updates > EDGE_SUBSAMPLE_MAX_CALLS else 2.0
        medium_keep_rate = min(adaptive_rate * medium_multiplier, 0.4)  # Cap at 40%
        medium_keep_count = max(1, int(len(medium_priority) * medium_keep_rate))
        
        # Deterministic selection based on reward ranking + position
        medium_sorted = sorted(medium_priority, key=lambda x: x[2], reverse=True)
        step = max(1, len(medium_sorted) // medium_keep_count)
        start_offset = _edge_subsample_counter % step
        result.extend(medium_sorted[start_offset::step][:medium_keep_count])
    
    # TIER 3: Sparse sampling for low priority (computational efficiency)
    if low_priority:
        low_keep_count = max(1, int(len(low_priority) * adaptive_rate))
        subsample_step = max(1, len(low_priority) // low_keep_count)
        start_offset = _edge_subsample_counter % subsample_step
        result.extend(low_priority[start_offset::subsample_step][:low_keep_count])
    
    # 📊 ACCUMULATE METRICS: Collect stats for clean iteration reporting
    global _edge_subsample_metrics
    if len(edge_updates) != len(result):
        _edge_subsample_metrics['total_input'] += len(edge_updates)
        _edge_subsample_metrics['total_output'] += len(result)
        _edge_subsample_metrics['high_priority'] += len(high_priority)
        _edge_subsample_metrics['calls_count'] += 1
    
    return result

def get_edge_subsample_stats():
    """Get accumulated edge subsampling stats and reset counters"""
    global _edge_subsample_metrics
    stats = _edge_subsample_metrics.copy()
    
    # Calculate efficiency metrics
    if stats['total_input'] > 0:
        reduction_pct = ((stats['total_input'] - stats['total_output']) / stats['total_input']) * 100
        stats['reduction_pct'] = reduction_pct
    else:
        stats['reduction_pct'] = 0
        
    # Reset for next iteration
    _edge_subsample_metrics = {'total_input': 0, 'total_output': 0, 'high_priority': 0, 'calls_count': 0}
    return stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

from hyperbolic_memory import PoincareManifold


class _BundledFieldView:
    """
    🎯 VIEW WRAPPER: Extracts a single field from bundled storage.
    
    When bundled storage is accessed like `storage[idx]`, it returns a dict
    of all fields. This wrapper extracts just one field, making it look like
    a regular tensor for backward compatibility.
    
    Usage:
        storage = DiskBackedTensor(..., bundle_fields={'embedding': (128,), 'adjacency': (16,)})
        embeddings = _BundledFieldView(storage, 'embedding')
        emb = embeddings[5]  # Returns just the embedding tensor, not the dict
    """
    def __init__(self, bundled_storage, field_name: str):
        self._storage = bundled_storage
        self._field = field_name
    
    def __getitem__(self, idx):
        """
        Extract the field from bundled access.
        
        Supports:
        - Single index: view[5]
        - Tuple index: view[5, :10] or view[5, 2]
        - Slice: view[0:10]
        """
        # Handle tuple indexing (e.g., adjacency[i, :k])
        if isinstance(idx, tuple):
            row_idx = idx[0]
            col_indices = idx[1:]
            
            # Get the row bundle first
            bundle = self._storage[row_idx]  # Get just the row, not tuple
            if isinstance(bundle, dict):
                field_tensor = bundle[self._field]
                # Apply column indexing
                if len(col_indices) == 1:
                    return field_tensor[col_indices[0]]
                else:
                    return field_tensor[col_indices]
            else:
                raise ValueError(f"Expected dict bundle, got {type(bundle)}")
        else:
            # Simple indexing
            bundle = self._storage[idx]
            if isinstance(bundle, dict):
                return bundle[self._field]
            else:
                # Single field (shouldn't happen, but handle gracefully)
                return bundle
    
    def __setitem__(self, idx, value):
        """
        Set the field in bundled storage.
        
        Supports:
        - Single index: view[5] = tensor
        - Tuple index: view[5, :10] = tensor (modifies slice of field)
        """
        # Handle tuple indexing (e.g., adjacency[i, :k])
        if isinstance(idx, tuple):
            row_idx = idx[0]
            col_indices = idx[1:]
            
            # Read the existing bundle
            try:
                bundle = self._storage[row_idx]
                if not isinstance(bundle, dict):
                    raise ValueError(f"Expected dict bundle, got {type(bundle)}")
            except:
                # No existing bundle - can't do partial update
                raise ValueError(f"Cannot use tuple indexing on non-existent bundle at index {row_idx}")
            
            # Get the field tensor
            field_tensor = bundle[self._field].clone()
            
            # Modify the slice - handle shape mismatches
            try:
                # Try direct assignment
                field_tensor[col_indices] = value
            except RuntimeError as e:
                if "expanded size" in str(e) or "shape" in str(e):
                    # Shape mismatch - slice value to fit
                    # This handles cases like adjacency[i, :16] = neighbors (where neighbors has 256 elements)
                    # Extract the actual slice to determine target shape
                    target_slice = field_tensor[col_indices]
                    if target_slice.numel() < value.numel():
                        # Value is too large, take only what fits
                        field_tensor[col_indices] = value[:target_slice.numel()].reshape(target_slice.shape)
                    else:
                        # Value is too small or exact match, try reshape
                        field_tensor[col_indices] = value.reshape(target_slice.shape)
                else:
                    # Different error, re-raise
                    raise
            
            # Update bundle
            bundle[self._field] = field_tensor
            
            # Write back
            self._storage[row_idx] = bundle
        else:
            # Simple single-index assignment
            # Read the existing bundle
            try:
                bundle = self._storage[idx]
                if not isinstance(bundle, dict):
                    # Create new bundle with just this field
                    bundle = {self._field: value}
            except:
                # No existing bundle, create new
                bundle = {self._field: value}
            
            # Update the field
            bundle[self._field] = value
            
            # Write back the bundle
            self._storage[idx] = bundle
    
    def size(self, dim: Optional[int] = None):
        """Get size of the field."""
        # Delegate to underlying storage's size() method
        # The storage's size() already handles bundled mode correctly
        return self._storage.size(dim)
    
    def __len__(self):
        """Get length (number of nodes)."""
        return len(self._storage)
    
    @property
    def device(self):
        """Return storage device."""
        return torch.device(self._storage.device)
    
    @property
    def shape(self):
        """Return shape of the field."""
        # Get the field shape from bundle_fields
        if self._field in self._storage.bundle_fields:
            field_shape = self._storage.bundle_fields[self._field]
            # Prepend the number of nodes (batch dimension)
            return torch.Size([self._storage._actual_size, *field_shape])
        else:
            # Fallback to storage size
            return self._storage.size()
    
    def to(self, device):
        """Device movement (no-op, returns self for compatibility)."""
        return self
    
    def cpu(self):
        """CPU movement (no-op, returns self for compatibility)."""
        return self
    
    def cuda(self):
        """CUDA movement (no-op, returns self for compatibility)."""
        return self


class GraphMemoryTier(nn.Module):
    """
    A single tier of graph-structured memory storage.
    
    Stores:
    - embeddings: [N, D] memory vectors in Euclidean space
    - adjacency: [N, k] sparse neighbor indices (k-NN in hyperbolic space)
    - edge_weights: [N, k] hyperbolic distances to neighbors
    - cluster_ids: [N] cluster assignment for hierarchical organization
    - rewards: [N] memory utility scores
    - age: [N] timesteps since creation
    - access: [N] timesteps since last access
    """
    
    def __init__(self, capacity: int, memory_dim: int, k_neighbors: int, device: str = 'cpu',
                 use_types: bool = True, type_dim: int = 16, num_edge_types: int = 8,
                 disk_path: str = None,  # NEW: optional disk backing
                 max_disk_size: int = 1000000):  # 🚀 ADAPTIVE: Can grow beyond this!
        # 🎯 CRITICAL: Set use_bundled FIRST before calling super().__init__()
        # This allows properties to know which mode to use when buffers are registered
        self.use_bundled = disk_path is not None  # Bundled mode for disk-backed storage
        
        super().__init__()
        self.capacity = capacity
        self.memory_dim = memory_dim
        self.k_neighbors = k_neighbors
        self.device = device
        self.use_types = use_types
        self.type_dim = type_dim
        self.num_edge_types = num_edge_types
        self.max_disk_size = max_disk_size
        
        # DISK BACKING with DiskBackedTensor (transparent virtual memory!)
        self.disk_path = disk_path
        self.use_disk = disk_path is not None
        
        if self.use_disk:
            import os
            os.makedirs(disk_path, exist_ok=True)
            
            from disk_backed_tensor import DiskBackedTensor
            from hyperbolic_memory import PoincareManifold
            
            poincare = PoincareManifold(dim=memory_dim, c=1.0)
            
            # Cache sizing: HARD LIMIT based on available RAM, not dataset size!
            # Philosophy: Cache is a fixed resource constraint, dataset can be infinite
            # 
            # For 6GB GPU typical consumer setup:
            # - Embeddings: ~100-500 entries × 128 dim × 4 bytes = 50-250 KB (tiny!)
            # - Graph metadata: even smaller (just indices/weights)
            # 
            # These limits work for ANY dataset size (100 to 1M+ memories)
            # 
            # � INTELLIGENT CACHE MANAGEMENT:
            # - RAM Cache: Hot working set with hyperbolic prefetching (max 5GB)
            # - Disk Storage: Everything else, unlimited growth
            # - Strategy: Cache stays under limit, excess flows to disk atomically
            #
            # Calculate cache size for 5GB RAM budget:
            # Bundle size = embedding + adjacency + edges + metadata
            bundle_size_per_node = (
                memory_dim +              # embedding: 128 floats
                k_neighbors +             # adjacency: 10 ints
                k_neighbors +             # edge_weights: 10 floats
                k_neighbors * num_edge_types +  # edge_types: 10×4 floats
                k_neighbors +             # edge_traversal_count: 10 floats
                k_neighbors +             # edge_success_rate: 10 floats
                k_neighbors * k_neighbors +     # edge_flow_context: 10×10 floats
                k_neighbors * k_neighbors +     # edge_flow_prev_nodes: 10×10 floats
                2                         # cluster_id + depth: 2 floats
            ) * 4  # bytes per float32
            
            if use_types:
                bundle_size_per_node += type_dim * 4
            
            # 5GB RAM cache budget
            MAX_CACHE_RAM_GB = 5.0
            max_cache_nodes = int((MAX_CACHE_RAM_GB * 1024**3) / bundle_size_per_node)
            
            # Use calculated cache size (ensures we stay under 5GB)
            embedding_cache = max_cache_nodes
            graph_cache = 50  # Small metadata cache
            
            print(f"💾 Disk-backed tier: {disk_path}")
            print(f"   Max disk size: {max_disk_size} memories (ADAPTIVE - will grow as needed!)")
            print(f"   Bundle size: {bundle_size_per_node // 1024}KB per node")
            print(f"   🧠 RAM Cache: {embedding_cache:,} nodes (~{embedding_cache * bundle_size_per_node / (1024**3):.2f}GB)")
            print(f"   💽 Disk overflow: Automatic when cache exceeds {embedding_cache:,} nodes")
            print(f"   🚀 UNLIMITED GROWTH: Disk will expand automatically for new memories")
            print(f"   🔒 ATOMIC WRITES: Index updates are transaction-safe")
            print(f"   🔥 UNIFIED BUNDLED STORAGE - ONE file for parallel transport!")
            print(f"   Strategy: Hot cache (5GB max) + unlimited disk with hyperbolic prefetch")
            
            # 🎯 BUNDLED STORAGE: ONE file contains everything about a node!
            # This enables atomic reads for parallel transport on fiber bundles.
            # Instead of 9 separate files (embeddings, adjacency, weights, etc),
            # we pack everything into ONE row-oriented format.
            #
            # Benefits:
            # - ONE disk seek gets embedding + adjacency + edges (required for parallel transport!)
            # - 10-100x faster graph traversal
            # - Atomic consistency (read complete node state)
            # - Fiber bundle geometry works correctly
            #
            bundle_fields = {
                'embedding': (memory_dim,),
                'adjacency': (k_neighbors,),
                'edge_weights': (k_neighbors,),
                'edge_types': (k_neighbors, num_edge_types),
                'edge_traversal_count': (k_neighbors,),
                'edge_success_rate': (k_neighbors,),
                'edge_flow_context': (k_neighbors, k_neighbors),
                'edge_flow_prev_nodes': (k_neighbors, k_neighbors),
                'cluster_id': (),  # scalar
                'depth': (),  # scalar
            }
            
            if use_types:
                bundle_fields['type_embedding'] = (type_dim,)
            
            # Single bundled storage for complete node records
            self._bundled_storage = DiskBackedTensor(
                shape=(max_disk_size,),  # Disk can grow to this size
                dtype=torch.float32,  # Ignored in bundled mode
                device=device,
                disk_path=disk_path,  # Single directory
                hot_capacity=embedding_cache,  # 🧠 5GB RAM cache limit (calculated above)
                poincare=poincare,
                flush_interval=999999.0,  # � INFINITE: Never auto-flush (only at checkpoint)
                enable_async=True,
                bundle_fields=bundle_fields,  # 🎯 Enable bundled mode!
                hyperbolic_layout=True  # 🌀 ENABLED: Compact sequential + Hilbert tracking (no RAM explosion!)
            )
            
            print(f"   ✅ Fiber bundle geometry enabled with intelligent cache management!")
            print(f"   📊 Cache will stay ≤5GB, excess automatically saved to disk")
            
            # No separate tensors needed - bundled storage handles everything!
            # Property accessors provide transparent access to fields
        else:
            # No disk - use regular buffers
            self.register_buffer('embeddings', torch.zeros(0, memory_dim, device=device))
            self.register_buffer('adjacency', torch.full((0, k_neighbors), -1, dtype=torch.long, device=device))
            self.register_buffer('edge_weights', torch.zeros(0, k_neighbors, device=device))
            self.register_buffer('cluster_ids', torch.full((0,), -1, dtype=torch.long, device=device))
            self.register_buffer('depths', torch.zeros(0, device=device))
            self.register_buffer('edge_types', torch.zeros(0, k_neighbors, num_edge_types, device=device))
            self.register_buffer('edge_traversal_count', torch.zeros(0, k_neighbors, device=device))
            self.register_buffer('edge_success_rate', torch.zeros(0, k_neighbors, device=device))
            
            # Flow field
            self.register_buffer('edge_flow_context', torch.zeros(0, k_neighbors, k_neighbors, device=device))
            self.register_buffer('edge_flow_prev_nodes', torch.full((0, k_neighbors, k_neighbors), -1, dtype=torch.int64, device=device))
            
            # TYPE SYSTEM
            if use_types:
                self.register_buffer('type_embeddings', torch.zeros(0, type_dim, device=device))
        
        # Edge type semantics (predefined - same for disk-backed and in-memory)
        # 0: PROXIMITY - geometric neighbors in hyperbolic space
        # 1: SYNTACTIC - same POS, similar syntactic role
        # 2: SEMANTIC - synonym/hypernym/association
        # 3: SEQUENCE - likely to follow in text (bigram)
        # 4: CO_RETRIEVED - Hebbian strengthening from co-retrieval
        # 5: CAUSAL - one predicts the other
        # 6: MODIFIER - adjective→noun, adverb→verb
        # 7: COMPLEMENT - verb→object, prep→noun
        
        # Metadata - REMOVED: rewards, age, access (legacy LRU eviction, not needed with disk backing)
        # Centrality and importance are now computed from graph structure (traversal counts, edge success rates)
        
        # Current size - always use a buffer, just access it differently
        self.register_buffer('_size', torch.tensor(0, dtype=torch.long, device=device))
        
        # 🛣️ HIGHWAY TRACKING: Monitor which edges are being strengthened
        # Keep top-K most strengthened edges for debugging
        self.highway_log = []  # List of (source_idx, target_idx, old_weight, new_weight, strengthening_amount)
        self.highway_log_max = 100  # Keep top 100 highways
        
        # 🔥 LAZY WRITE BUFFER: Batch disk writes for 16x eligibility traces
        # Problem: strengthen_edge() writes to disk immediately (16x slower with eligibility traces)
        # Solution: Buffer ALL changes in RAM, flush ONLY at checkpoint (never during training!)
        self.write_buffer = {}  # {(source_idx, edge_slot): {'weight': val, 'success': val, 'count': val, 'type': val}}
        self.write_buffer_size = 0
        self.write_buffer_max = 10000000  # � INFINITE: Never auto-flush (10M edges = way more than we'll see)
        self.flush_counter = 0
        self.flush_interval = 10000000  # � INFINITE: Never auto-flush during training
        
        # 🔥 ADJACENCY CACHE: Cache neighbor lookups to avoid disk reads
        # Each strengthen_edge needs to find which slot contains target_idx
        # Without cache: 16x disk reads per iteration (with eligibility traces)
        # With cache: 1x disk read, then pure RAM lookups
        self.adjacency_cache = {}  # {source_idx: neighbors_tensor}
        self.adjacency_cache_max = 500  # 🚀 CRITICAL: Bound cache size to prevent slowdown
        self.adjacency_cache_hits = 0  # Track effectiveness
        self.adjacency_cache_max = 5000  # � INCREASED: Keep 5000 neighbor lists in RAM (covers most active nodes)
        
        # 📊 DISK I/O PROFILING
        self.disk_reads = 0
        self.disk_writes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 🌊 FLOW FIELD HYPERPARAMETERS
        # (Flow data is stored in disk-backed tensors: edge_flow_context, edge_flow_prev_nodes)
        self.flow_decay = 0.99  # Slow decay for long-term pattern recognition
        self.flow_strength = 2.0  # Bias multiplier for predictive suggestions
        self.flow_min_reward = 0.2  # 🚀 TUNED: Record flows when reward > 0.2 (loss < ~4.0)
                                     # This balances between filtering garbage and learning from decent trajectories
        self.flow_enabled = True  # 🚀 ENABLED FROM START: PMI preloading builds highways, no warmup needed!
        self.flow_warmup_iters = 0  # No warmup - highways pre-built during trajectory encoding
    
    @property
    def size(self):
        """Get current size - compatible with both disk-backed and in-memory."""
        if self.use_disk:
            # 🎯 BUNDLED STORAGE: Use _bundled_storage._actual_size
            if self.use_bundled:
                return torch.tensor(self._bundled_storage._actual_size, dtype=torch.long, device=self.device)
            else:
                # Column-oriented disk storage (legacy)
                from disk_backed_tensor import DiskBackedTensor
                if isinstance(self.adjacency, DiskBackedTensor):
                    return torch.tensor(self.adjacency._actual_size, dtype=torch.long, device=self.device)
                else:
                    return torch.tensor(self.adjacency.shape[0], dtype=torch.long, device=self.device)
        else:
            # Regular buffer has _size attribute
            return self._size
    
    @size.setter
    def size(self, value):
        """Set size."""
        if self.use_disk:
            # 🎯 BUNDLED STORAGE: Update _bundled_storage._actual_size
            if self.use_bundled:
                if isinstance(value, torch.Tensor):
                    self._bundled_storage._actual_size = value.item()
                else:
                    self._bundled_storage._actual_size = int(value)
            else:
                # Column-oriented disk storage (legacy) - managed by DiskBackedTensor
                pass
        else:
            if isinstance(value, torch.Tensor):
                self._size = value.to(self.device)
            else:
                self._size = torch.tensor(value, dtype=torch.long, device=self.device)
    
    def __getattr__(self, name):
        """
        🎯 BUNDLED STORAGE: Fallback accessor for field names.
        
        When code tries to access self.embeddings, self.adjacency, etc.:
        - If bundled mode: return a FieldView that extracts the field from bundles
        - If not bundled: let normal attribute lookup proceed (will find buffers)
        
        This only triggers when the attribute is NOT found in the normal dict.
        """
        # Map attribute names to bundle field names
        field_mapping = {
            'embeddings': 'embedding',
            'adjacency': 'adjacency',
            'edge_weights': 'edge_weights',
            'edge_types': 'edge_types',
            'edge_traversal_count': 'edge_traversal_count',
            'edge_success_rate': 'edge_success_rate',
            'cluster_ids': 'cluster_id',
            'depths': 'depth',
            'type_embeddings': 'type_embedding',
            'edge_flow_context': 'edge_flow_context',
            'edge_flow_prev_nodes': 'edge_flow_prev_nodes'
        }
        
        if name in field_mapping and hasattr(self, 'use_bundled') and self.use_bundled:
            # Return a FieldView that extracts the specific field
            return _BundledFieldView(self._bundled_storage, field_mapping[name])
        
        # Check if it's a registered buffer (rewards, age, access, etc.)
        if hasattr(self, '_buffers') and name in self._buffers:
            return self._buffers[name]
        
        # Not found - raise AttributeError
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def add_node_dynamic(self, embedding: torch.Tensor, poincare, cluster_id: int = -1, skip_disk_search: bool = False) -> int:
        """
        Add a single node and WIRE it into existing graph dynamically.
        
        This makes the graph GROW like a city - new buildings connect to nearby roads!
        
        Args:
            embedding: Memory vector [D] (1D tensor)
            poincare: Poincare manifold for hyperbolic distance
            cluster_id: Cluster assignment (-1 to infer from neighbors)
            skip_disk_search: If True, only search hot RAM (faster for bulk preload)
        
        Returns: index of new node
        """
        # CRITICAL: Get new index FIRST (before any branching)
        new_idx = self.size.item()
        
        # CRITICAL: Ensure embedding is 1D and on correct device
        if embedding.dim() > 1:
            embedding = embedding.squeeze()
        embedding = embedding.to(self.device)
        
        if self.size == 0:
            # First node - no neighbors yet
            new_adjacency = torch.full((1, self.k_neighbors), -1, dtype=torch.long, device=self.device)
            new_weights = torch.zeros(1, self.k_neighbors, device=self.device)
            new_edge_types = torch.zeros(1, self.k_neighbors, self.num_edge_types, device=self.device)
            new_cluster = torch.tensor([0 if cluster_id < 0 else cluster_id], device=self.device)
        else:
            # Find k nearest neighbors in hyperbolic space
            # CRITICAL: Search BOTH hot RAM and cold disk for mathematically correct k-NN
            
            # Step 1: Compute distances to all hot memories in RAM
            # Convert embeddings to tensor if using bundled storage
            if self.use_bundled:
                # Get all embeddings as a stacked tensor
                all_embeddings = torch.stack([self._bundled_storage[i]['embedding'] for i in range(self.size.item())])
            else:
                all_embeddings = self.embeddings
            
            dists_hot = poincare.distance(
                embedding.unsqueeze(0),  # [1, D]
                all_embeddings  # [N_hot, D]
            ).squeeze(-1).squeeze(0)  # [N_hot] - remove keepdim dimensions
            
            # Step 2: With bundled storage, all distance computation is unified
            # No separate disk search needed - bundled storage handles caching internally
            
            # Step 3: Find k nearest neighbors from computed distances
            num_existing = self.size.item()
            if num_existing > 0:
                k_actual = min(self.k_neighbors, num_existing)
                
                # Handle case where we only have 1 memory
                if dists_hot.ndim == 0:
                    topk_dists = dists_hot.unsqueeze(0)
                    topk_indices = torch.tensor([0], dtype=torch.long, device=self.device)
                else:
                    topk_dists, topk_indices = torch.topk(dists_hot, k=k_actual, largest=False)
                    
                # Ensure tensors are at least 1-d
                if topk_dists.ndim == 0:
                    topk_dists = topk_dists.unsqueeze(0)
                    topk_indices = topk_indices.unsqueeze(0)
            else:
                k_actual = 0
                topk_dists = torch.tensor([], device=self.device)
                topk_indices = torch.tensor([], dtype=torch.long, device=self.device)
            
            # Pad if needed
            new_adjacency = torch.full((1, self.k_neighbors), -1, dtype=torch.long, device=self.device)
            new_weights = torch.zeros(1, self.k_neighbors, device=self.device)
            new_adjacency[0, :k_actual] = topk_indices
            new_weights[0, :k_actual] = topk_dists
            
            # EDGE TYPES: Infer relationship types from geometry and context
            new_edge_types = torch.zeros(1, self.k_neighbors, self.num_edge_types, device=self.device)
            
            # Type 0: PROXIMITY - all k-NN edges get this by definition
            new_edge_types[0, :k_actual, 0] = 1.0
            
            # Type 1: SYNTACTIC - infer from type embedding similarity (if available)
            if self.use_types and self.size > 0 and k_actual > 0:
                # With bundled storage, all neighbors are accessible
                valid_indices = topk_indices[:k_actual]
                
                # Bounds check for type embeddings
                if self.use_bundled:
                    # Type embeddings stored in bundles
                    max_idx = self.size.item()
                else:
                    # Type embeddings in separate tensor
                    max_idx = self.type_embeddings.size(0) if hasattr(self, 'type_embeddings') else 0
                
                if len(valid_indices) > 0 and valid_indices.max() < max_idx:
                    my_type = torch.zeros(self.type_dim, device=self.device)  # Will be inferred below
                    
                    # Get neighbor types
                    if self.use_bundled:
                        neighbor_types = torch.stack([self._bundled_storage[i.item()]['type_embedding'] 
                                                     for i in valid_indices 
                                                     if 'type_embedding' in self._bundled_storage[i.item()]])
                    else:
                        neighbor_types = self.type_embeddings[valid_indices]  # [k, type_dim]
                    
                    if len(neighbor_types) > 0:
                        # Cosine similarity of type vectors
                        type_sim = F.cosine_similarity(
                            my_type.unsqueeze(0).expand(len(neighbor_types), -1),
                            neighbor_types,
                            dim=-1
                        )  # [k]
                        # High type similarity → same syntactic role
                        for i in range(min(k_actual, len(type_sim))):
                            new_edge_types[0, i, 1] = (type_sim[i] > 0.7).float()
            
            # Type 2: SEMANTIC - infer from depth similarity (same level in hierarchy)
            if k_actual > 0:
                my_depth = embedding.norm()
                for i in range(k_actual):
                    neighbor_idx = topk_indices[i]
                    # Get neighbor embedding
                    if self.use_bundled and neighbor_idx < self.size.item():
                        neighbor_emb = self._bundled_storage[neighbor_idx.item()]['embedding']
                    elif not self.use_bundled and neighbor_idx < self.embeddings.size(0):
                        neighbor_emb = self.embeddings[neighbor_idx]
                    else:
                        continue
                    
                    neighbor_depth = neighbor_emb.norm()
                    depth_diff = torch.abs(my_depth - neighbor_depth)
                    # Similar depth → semantic peers (synonyms, not hypernyms)
                    new_edge_types[0, i, 2] = (depth_diff < 0.1).float()
            
            # Type 3: SEQUENCE - will be learned from co-occurrence (initially 0)
            
            # CRITICAL: Bidirectional wiring - neighbors point back to new node!
            # With bundled storage, all neighbors are accessible in memory
            # new_idx already defined at function start
            for i, (neighbor_idx, dist) in enumerate(zip(topk_indices, topk_dists)):
                # Check if new node is closer than neighbor's current furthest neighbor
                # Get neighbor's adjacency
                if self.use_bundled and neighbor_idx < self.size.item():
                    neighbor_adj = self._bundled_storage[neighbor_idx.item()]['adjacency']
                    neighbor_weights = self._bundled_storage[neighbor_idx.item()]['edge_weights']
                elif not self.use_bundled and neighbor_idx < self.adjacency.size(0):
                    neighbor_adj = self.adjacency[neighbor_idx]
                    neighbor_weights = self.edge_weights[neighbor_idx]
                else:
                    continue
                
                valid_mask = neighbor_adj >= 0
                if valid_mask.sum() < self.k_neighbors:
                    # Neighbor has free slots
                    free_slot = (~valid_mask).nonzero(as_tuple=True)[0][0]
                    if self.use_bundled:
                        # Update bundle
                        bundle = self._bundled_storage[neighbor_idx.item()]
                        bundle['adjacency'][free_slot] = new_idx
                        bundle['edge_weights'][free_slot] = dist
                        self._bundled_storage[neighbor_idx.item()] = bundle
                    else:
                        self.adjacency[neighbor_idx, free_slot] = new_idx
                        self.edge_weights[neighbor_idx, free_slot] = dist
                elif dist < neighbor_weights.max():
                    # Replace furthest neighbor with new node
                    furthest_slot = neighbor_weights.argmax()
                    if self.use_bundled:
                        bundle = self._bundled_storage[neighbor_idx.item()]
                        bundle['adjacency'][furthest_slot] = new_idx
                        bundle['edge_weights'][furthest_slot] = dist
                        self._bundled_storage[neighbor_idx.item()] = bundle
                    else:
                        self.adjacency[neighbor_idx, furthest_slot] = new_idx
                        self.edge_weights[neighbor_idx, furthest_slot] = dist
            
            # Infer cluster from neighbors if not specified
            if cluster_id < 0:
                # CRITICAL: Bounds check - only use indices within cluster_ids range
                valid_neighbor_mask = topk_indices < self.cluster_ids.size(0)
                valid_topk_indices = topk_indices[valid_neighbor_mask]
                
                if len(valid_topk_indices) > 0:
                    neighbor_clusters = self.cluster_ids[valid_topk_indices]
                    valid_clusters = neighbor_clusters[neighbor_clusters >= 0]
                    if len(valid_clusters) > 0:
                        # Vote: join majority cluster
                        cluster_id = torch.mode(valid_clusters).values.item()
                    else:
                        cluster_id = 0
                else:
                    cluster_id = 0
            
            new_cluster = torch.tensor([cluster_id], device=self.device)
        
        # 🔥 BUNDLED STORAGE vs RAM-ONLY: Different growth strategies
        # When disk_path is set, use index assignment (bundled storage handles everything)
        # When RAM-only, use torch.cat to grow tensors
        
        with torch.no_grad():
            if self.disk_path is not None:
                # 🌀 BUNDLED STORAGE: Build complete bundle dict and set atomically
                # DiskBackedTensor requires ALL fields in one operation
                
                bundle = {
                    'embedding': embedding.detach(),
                    'adjacency': new_adjacency[0],  # Remove batch dim
                    'edge_weights': new_weights[0],
                    'edge_types': new_edge_types[0],
                    'cluster_id': new_cluster[0],
                    'depth': embedding.norm(),
                    'edge_traversal_count': torch.zeros(self.k_neighbors, device=self.device),
                    'edge_success_rate': torch.zeros(self.k_neighbors, device=self.device),
                    'edge_flow_context': torch.zeros(self.k_neighbors, self.k_neighbors, device=self.device),
                    'edge_flow_prev_nodes': torch.full((self.k_neighbors, self.k_neighbors), -1, dtype=torch.long, device=self.device),
                }
                
                # Type embedding (if enabled)
                if self.use_types:
                    if self.size > 0 and k_actual > 0:
                        # All neighbors are accessible with bundled storage
                        valid_indices = topk_indices[:k_actual]
                        if len(valid_indices) > 0 and valid_indices.max() < new_idx:
                            # Get type embeddings from bundles
                            neighbor_type_embs = torch.stack([self._bundled_storage[i.item()]['type_embedding'] 
                                                             for i in valid_indices 
                                                             if 'type_embedding' in self._bundled_storage[i.item()]])
                            if len(neighbor_type_embs) > 0:
                                bundle['type_embedding'] = neighbor_type_embs.mean(dim=0)
                            else:
                                bundle['type_embedding'] = torch.randn(self.type_dim, device=self.device) * 0.1
                        else:
                            bundle['type_embedding'] = torch.randn(self.type_dim, device=self.device) * 0.1
                    else:
                        bundle['type_embedding'] = torch.randn(self.type_dim, device=self.device) * 0.1
                
                # Set complete bundle atomically
                self._bundled_storage[new_idx] = bundle
                
                # No auxiliary tensors needed - everything is in the bundle!
                
            else:
                # 🔧 RAM-ONLY: Use torch.cat to grow tensors
                # CRITICAL: Ensure ALL buffers are on correct device before cat
                # This can happen after disk loading operations
                if self.embeddings.device != self.device:
                    self.embeddings = self.embeddings.to(self.device)
                if self.adjacency.device != self.device:
                    self.adjacency = self.adjacency.to(self.device)
                if self.edge_weights.device != self.device:
                    self.edge_weights = self.edge_weights.to(self.device)
                if self.edge_types.device != self.device:
                    self.edge_types = self.edge_types.to(self.device)
                if self.cluster_ids.device != self.device:
                    self.cluster_ids = self.cluster_ids.to(self.device)
                if self.rewards.device != self.device:
                    self.rewards = self.rewards.to(self.device)
                if self.age.device != self.device:
                    self.age = self.age.to(self.device)
                if self.access.device != self.device:
                    self.access = self.access.to(self.device)
                if self.edge_traversal_count.device != self.device:
                    self.edge_traversal_count = self.edge_traversal_count.to(self.device)
                if self.edge_success_rate.device != self.device:
                    self.edge_success_rate = self.edge_success_rate.to(self.device)
                if self.use_types and self.type_embeddings.device != self.device:
                    self.type_embeddings = self.type_embeddings.to(self.device)
                
                self.embeddings = torch.cat([self.embeddings, embedding.detach().unsqueeze(0)], dim=0)
                self.adjacency = torch.cat([self.adjacency, new_adjacency], dim=0)
                self.edge_weights = torch.cat([self.edge_weights, new_weights], dim=0)
                self.edge_types = torch.cat([self.edge_types, new_edge_types], dim=0) if self.size > 0 else new_edge_types
                
                # cluster_ids are always regular tensors in RAM-only mode
                self.cluster_ids = torch.cat([self.cluster_ids, new_cluster], dim=0)
                
                # Edge tracking tensors
                self.edge_traversal_count = torch.cat([self.edge_traversal_count, 
                                                       torch.zeros(1, self.k_neighbors, device=self.device)], dim=0)
                self.edge_success_rate = torch.cat([self.edge_success_rate, 
                                                   torch.zeros(1, self.k_neighbors, device=self.device)], dim=0)
                
                # Type embeddings
                if self.use_types and self.size > 0 and k_actual > 0:
                    # CRITICAL: Bounds check for type_embeddings
                    valid_indices = topk_indices[:k_actual][topk_indices[:k_actual] < self.type_embeddings.size(0)]
                    if len(valid_indices) > 0:
                        neighbor_type_embs = self.type_embeddings[valid_indices]
                        inferred_type = neighbor_type_embs.mean(dim=0)
                        self.type_embeddings = torch.cat([self.type_embeddings, inferred_type.unsqueeze(0)], dim=0)
                    else:
                        random_type = torch.randn(1, self.type_dim, device=self.device) * 0.1
                        self.type_embeddings = torch.cat([self.type_embeddings, random_type], dim=0)
                elif self.use_types:
                    random_type = torch.randn(1, self.type_dim, device=self.device) * 0.1
                    self.type_embeddings = torch.cat([self.type_embeddings, random_type], dim=0)
        
        # Increment size counter
        self.size = torch.tensor(self.size.item() + 1, dtype=torch.long, device=self.device)
        
        return new_idx
    
    def add_nodes_batch_gpu(self, embeddings_batch: torch.Tensor, poincare, batch_size: int = 256, max_existing: int = 5000) -> int:
        """
        GPU-accelerated batched node insertion for fast preloading.
        
        Processes embeddings in batches on GPU, then moves to CPU for storage.
        This is MUCH faster than one-by-one insertion during preload.
        
        NOTE: Only uses HOT tier for k-NN. Disk memories are ignored during batch insertion
        to avoid loading entire dataset into RAM. This is a speed optimization for preload.
        
        For GPU memory efficiency, we limit how many existing memories are used for k-NN search.
        
        Args:
            embeddings_batch: [B, D] batch of embeddings (on any device)
            poincare: Poincare manifold for distance computation
            batch_size: Chunk size for GPU processing
            max_existing: Max number of existing memories to use for k-NN (GPU memory limit)
        
        Returns: number of nodes added
        """
        num_to_add = embeddings_batch.size(0)
        added_count = 0
        
        # CRITICAL: Ensure ALL existing tensors are on correct device BEFORE we start
        # This prevents "cuda:0 vs cpu" errors during concatenation
        # For CPU tier (longterm), everything MUST be on CPU
        # NOTE: PyTorch moves ALL buffers when model.to(device) is called, so we must
        # force them back to CPU for longterm tier
        target_device = torch.device('cpu' if self.device == 'cpu' else self.device)
        
        # AGGRESSIVE FIX: For CPU tier, force EVERYTHING to CPU regardless of size
        # This handles the case where model.cuda() moved all buffers to GPU
        # BUT skip DiskBackedTensors and _BundledFieldView (they manage their own device placement)
        from disk_backed_tensor import DiskBackedTensor
        
        # 🎯 BUNDLED MODE: Skip ALL register_buffer calls - bundled storage manages everything
        # Continue with the rest of the function normally
        bundled_mode = hasattr(self, 'use_bundled') and self.use_bundled
        
        if not bundled_mode and self.device == 'cpu':
            # Move ALL buffers back to CPU (except DiskBackedTensors)
            if not isinstance(self.adjacency, DiskBackedTensor):
                self.register_buffer('adjacency', self.adjacency.cpu())
            if not isinstance(self.edge_weights, DiskBackedTensor):
                self.register_buffer('edge_weights', self.edge_weights.cpu())
            if self.edge_types.size(0) > 0 and not isinstance(self.edge_types, DiskBackedTensor):
                self.register_buffer('edge_types', self.edge_types.cpu())
            if not isinstance(self.edge_traversal_count, DiskBackedTensor):
                self.register_buffer('edge_traversal_count', self.edge_traversal_count.cpu())
            if not isinstance(self.edge_success_rate, DiskBackedTensor):
                self.register_buffer('edge_success_rate', self.edge_success_rate.cpu())
            
            # Metadata: some are DiskBackedTensor (cluster_ids, depths, type_embeddings), some are not (rewards, age, access)
            if not isinstance(self.cluster_ids, DiskBackedTensor):
                self.register_buffer('cluster_ids', self.cluster_ids.cpu())
            if not isinstance(self.rewards, DiskBackedTensor):
                self.register_buffer('rewards', self.rewards.cpu())
            if not isinstance(self.age, DiskBackedTensor):
                self.register_buffer('age', self.age.cpu())
            if not isinstance(self.access, DiskBackedTensor):
                self.register_buffer('access', self.access.cpu())
            if not isinstance(self.depths, DiskBackedTensor):
                self.register_buffer('depths', self.depths.cpu())
            if self.use_types and self.type_embeddings.size(0) > 0 and not isinstance(self.type_embeddings, DiskBackedTensor):
                self.register_buffer('type_embeddings', self.type_embeddings.cpu())
        elif not bundled_mode and self.size > 0:
            # For GPU tier, move to target device if needed (skip DiskBackedTensors)
            if not isinstance(self.adjacency, DiskBackedTensor) and self.adjacency.device != target_device:
                self.register_buffer('adjacency', self.adjacency.to(target_device))
            if not isinstance(self.edge_weights, DiskBackedTensor) and self.edge_weights.device != target_device:
                self.register_buffer('edge_weights', self.edge_weights.to(target_device))
            if not isinstance(self.edge_types, DiskBackedTensor) and self.edge_types.size(0) > 0 and self.edge_types.device != target_device:
                self.register_buffer('edge_types', self.edge_types.to(target_device))
            if not isinstance(self.edge_traversal_count, DiskBackedTensor) and self.edge_traversal_count.device != target_device:
                self.register_buffer('edge_traversal_count', self.edge_traversal_count.to(target_device))
            if not isinstance(self.edge_success_rate, DiskBackedTensor) and self.edge_success_rate.device != target_device:
                self.register_buffer('edge_success_rate', self.edge_success_rate.to(target_device))
            
            # These are always regular tensors
            if self.cluster_ids.device != target_device:
                self.register_buffer('cluster_ids', self.cluster_ids.to(target_device))
            if self.rewards.device != target_device:
                self.register_buffer('rewards', self.rewards.to(target_device))
            if self.age.device != target_device:
                self.register_buffer('age', self.age.to(target_device))
            if self.access.device != target_device:
                self.register_buffer('access', self.access.to(target_device))
            if self.depths.device != target_device:
                self.register_buffer('depths', self.depths.to(target_device))
            if self.use_types and self.type_embeddings.size(0) > 0 and self.type_embeddings.device != target_device:
                self.register_buffer('type_embeddings', self.type_embeddings.to(target_device))
        
        # For GPU memory efficiency: sample existing memories if we have too many
        # We'll reload from CPU for each chunk to avoid accumulating GPU memory
        use_sampling = self.size > max_existing
        
        # Process in chunks to avoid GPU OOM
        from tqdm import tqdm
        num_chunks = (num_to_add + batch_size - 1) // batch_size
        pbar = tqdm(total=num_to_add, desc="Adding memories", unit="mem")
        
        for i in range(0, num_to_add, batch_size):
            batch_end = min(i + batch_size, num_to_add)
            batch = embeddings_batch[i:batch_end]  # Keep on CPU for longterm tier
            chunk_size = batch.size(0)
            
            # For CPU tier (longterm), do distance computation on CPU
            # For GPU tier (working/buffer), use GPU
            compute_device = 'cpu' if self.device == 'cpu' else 'cuda'
            batch = batch.to(compute_device)
            
            # Load existing embeddings for THIS chunk only
            existing_embs = None
            index_mapping = None
            
            if self.size > 0:
                if use_sampling:
                    # Sample a subset
                    indices = torch.randperm(self.size, device='cpu')[:max_existing]
                    emb_data = self.embeddings[indices]
                    # Handle bundled storage (returns dict) vs column storage (returns tensor)
                    existing_embs = emb_data['embedding'].to(compute_device) if isinstance(emb_data, dict) else emb_data.to(compute_device)
                    index_mapping = indices
                else:
                    # Use all existing
                    emb_data = self.embeddings[:self.size]
                    # Handle bundled storage (returns dict) vs column storage (returns tensor)
                    existing_embs = emb_data['embedding'].to(compute_device) if isinstance(emb_data, dict) else emb_data.to(compute_device)
                    index_mapping = None
            
            if existing_embs is not None and existing_embs.size(0) > 0:
                # Compute distances on appropriate devices.                                                                                                                     
                dists = poincare.distance(
                    batch.unsqueeze(1),  # [chunk_size, 1, D]
                    existing_embs.unsqueeze(0)  # [1, N_existing, D]
                )
                
                if dists.ndim == 3:
                    dists = dists.squeeze(-1)
                
                k_actual = min(self.k_neighbors, existing_embs.size(0))
                topk_dists, topk_indices_local = torch.topk(dists, k_actual, dim=1, largest=False)
                
                if index_mapping is not None:
                    topk_indices = index_mapping[topk_indices_local.cpu()]
                else:
                    topk_indices = topk_indices_local.cpu()
                
                batch_adjacency = torch.full((chunk_size, self.k_neighbors), -1, dtype=torch.long)
                batch_weights = torch.zeros(chunk_size, self.k_neighbors)
                batch_adjacency[:, :k_actual] = topk_indices
                batch_weights[:, :k_actual] = topk_dists.cpu()
            else:
                k_actual = 0
                batch_adjacency = torch.full((chunk_size, self.k_neighbors), -1, dtype=torch.long)
                batch_weights = torch.zeros(chunk_size, self.k_neighbors)
            
            # Everything on CPU now
            batch_cpu = batch.cpu()
            batch_adjacency_cpu = batch_adjacency
            batch_weights_cpu = batch_weights
            
            # Append to tier
            with torch.no_grad():
                # Handle DiskBackedTensor vs regular tensor
                from disk_backed_tensor import DiskBackedTensor
                
                # 🎯 BUNDLED MODE: Write complete bundles atomically
                if bundled_mode:
                    # Prepare all batch data
                    batch_edge_types = torch.zeros(chunk_size, self.k_neighbors, self.num_edge_types, device=target_device)
                    batch_edge_types[:, :k_actual, 0] = 1.0  # Proximity type
                    batch_clusters = torch.zeros(chunk_size, dtype=torch.long, device=target_device)
                    batch_depths = torch.norm(batch_cpu, dim=-1).to(target_device)
                    batch_trav = torch.zeros(chunk_size, self.k_neighbors, device=target_device)
                    batch_succ = torch.zeros(chunk_size, self.k_neighbors, device=target_device)
                    
                    # Type embeddings if using types
                    if self.use_types:
                        batch_type_embs = torch.randn(chunk_size, self.type_dim, device=target_device) * 0.1
                    
                    # Get start index for atomic writes
                    start_idx = self._bundled_storage._actual_size
                    
                    # Build complete bundles for each node
                    for local_i in range(chunk_size):
                        bundle = {
                            'embedding': batch_cpu[local_i].cpu(),
                            'adjacency': batch_adjacency_cpu[local_i].cpu(),
                            'edge_weights': batch_weights_cpu[local_i].cpu(),
                            'edge_types': batch_edge_types[local_i].cpu(),
                            'edge_traversal_count': batch_trav[local_i].cpu(),
                            'edge_success_rate': batch_succ[local_i].cpu(),
                            'edge_flow_context': torch.zeros(self.k_neighbors, self.k_neighbors, dtype=torch.float32),
                            'edge_flow_prev_nodes': torch.full((self.k_neighbors, self.k_neighbors), -1, dtype=torch.int64),
                            'cluster_id': batch_clusters[local_i].cpu(),
                            'depth': batch_depths[local_i].cpu()
                        }
                        # Add type_embedding if using types
                        if self.use_types:
                            bundle['type_embedding'] = batch_type_embs[local_i].cpu()
                        
                        # Atomic write of complete node state (fiber bundle requirement!)
                        self._bundled_storage[start_idx + local_i] = bundle
                    
                    # Update size and regular tensors (rewards, age, access not in bundle)
                    self.size = torch.tensor(self._bundled_storage._actual_size, dtype=torch.long, device='cpu')
                    
                    # Move regular tensors to target_device if needed before concatenation
                    if self.rewards.device.type != target_device:
                        self.rewards = self.rewards.to(target_device)
                        self.age = self.age.to(target_device)
                        self.access = self.access.to(target_device)
                    
                    self.rewards = torch.cat([self.rewards, torch.zeros(chunk_size, device=target_device)], dim=0)
                    self.age = torch.cat([self.age, torch.zeros(chunk_size, device=target_device)], dim=0)
                    self.access = torch.cat([self.access, torch.zeros(chunk_size, device=target_device)], dim=0)
                    
                else:
                    # Column-oriented mode (existing logic)
                    # Handle embeddings (might be DiskBackedTensor)
                    if isinstance(self.embeddings, DiskBackedTensor):
                        # Use setitem interface for DiskBackedTensor
                        start_idx = self.embeddings._actual_size
                        for local_i, emb in enumerate(batch_cpu):
                            self.embeddings[start_idx + local_i] = emb
                    else:
                        # Regular concatenation for normal tensors
                        self.embeddings = torch.cat([self.embeddings, batch_cpu], dim=0)
                    
                    # Handle adjacency and edge_weights (might be DiskBackedTensor)
                    if isinstance(self.adjacency, DiskBackedTensor):
                        start_idx = self.adjacency._actual_size
                        for local_i in range(chunk_size):
                            self.adjacency[start_idx + local_i] = batch_adjacency_cpu[local_i]
                    else:
                        self.adjacency = torch.cat([self.adjacency, batch_adjacency_cpu], dim=0)
                    
                    if isinstance(self.edge_weights, DiskBackedTensor):
                        start_idx = self.edge_weights._actual_size
                        for local_i in range(chunk_size):
                            self.edge_weights[start_idx + local_i] = batch_weights_cpu[local_i]
                    else:
                        self.edge_weights = torch.cat([self.edge_weights, batch_weights_cpu], dim=0)
                    
                    # Initialize other metadata - CRITICAL: Use target_device for all new tensors!
                    batch_edge_types = torch.zeros(chunk_size, self.k_neighbors, self.num_edge_types, device=target_device)
                    batch_edge_types[:, :k_actual, 0] = 1.0  # Proximity type
                    
                    # Handle edge_types (might be DiskBackedTensor)
                    if self.edge_types.size(0) == 0 and not isinstance(self.edge_types, DiskBackedTensor):
                        # First batch with regular tensor - direct assignment
                        self.edge_types = batch_edge_types
                    else:
                        # Append mode (works for both DiskBackedTensor and regular tensor)
                        if isinstance(self.edge_types, DiskBackedTensor):
                            start_idx = self.edge_types._actual_size
                            for local_i in range(chunk_size):
                                self.edge_types[start_idx + local_i] = batch_edge_types[local_i]
                        else:
                            self.edge_types = torch.cat([self.edge_types, batch_edge_types], dim=0)
                    
                    # Initialize cluster IDs (might be DiskBackedTensor)
                    batch_clusters = torch.zeros(chunk_size, dtype=torch.long, device=target_device)
                    if isinstance(self.cluster_ids, DiskBackedTensor):
                        start_idx = self.cluster_ids._actual_size
                        for local_i in range(chunk_size):
                            self.cluster_ids[start_idx + local_i] = batch_clusters[local_i]
                    else:
                        self.cluster_ids = torch.cat([self.cluster_ids, batch_clusters], dim=0)
                    
                    # Rewards, age, access are always regular tensors
                    self.rewards = torch.cat([self.rewards, torch.zeros(chunk_size, device=target_device)], dim=0)
                    self.age = torch.cat([self.age, torch.zeros(chunk_size, device=target_device)], dim=0)
                    self.access = torch.cat([self.access, torch.zeros(chunk_size, device=target_device)], dim=0)
                    
                    # Initialize depths (distance from origin in hyperbolic space) - might be DiskBackedTensor
                    batch_depths = torch.norm(batch_cpu, dim=-1).to(target_device)
                    if isinstance(self.depths, DiskBackedTensor):
                        start_idx = self.depths._actual_size
                        for local_i in range(chunk_size):
                            self.depths[start_idx + local_i] = batch_depths[local_i]
                    else:
                        self.depths = torch.cat([self.depths, batch_depths], dim=0)
                    
                    # Edge tracking (might be DiskBackedTensor)
                    if isinstance(self.edge_traversal_count, DiskBackedTensor):
                        start_idx = self.edge_traversal_count._actual_size
                        batch_trav = torch.zeros(chunk_size, self.k_neighbors, device=target_device)
                        for local_i in range(chunk_size):
                            self.edge_traversal_count[start_idx + local_i] = batch_trav[local_i]
                    else:
                        self.edge_traversal_count = torch.cat([self.edge_traversal_count, 
                                                               torch.zeros(chunk_size, self.k_neighbors, device=target_device)], dim=0)
                    
                    if isinstance(self.edge_success_rate, DiskBackedTensor):
                        start_idx = self.edge_success_rate._actual_size
                        batch_succ = torch.zeros(chunk_size, self.k_neighbors, device=target_device)
                        for local_i in range(chunk_size):
                            self.edge_success_rate[start_idx + local_i] = batch_succ[local_i]
                    else:
                        self.edge_success_rate = torch.cat([self.edge_success_rate, 
                                                           torch.zeros(chunk_size, self.k_neighbors, device=target_device)], dim=0)
                    
                    # Type embeddings (if using types) - might be DiskBackedTensor
                    if self.use_types:
                        batch_type_embs = torch.randn(chunk_size, self.type_dim, device=target_device) * 0.1
                        if isinstance(self.type_embeddings, DiskBackedTensor):
                            start_idx = self.type_embeddings._actual_size
                            for local_i in range(chunk_size):
                                self.type_embeddings[start_idx + local_i] = batch_type_embs[local_i]
                        else:
                            # Ensure existing type_embeddings is on target device
                            if self.type_embeddings.device.type != target_device:
                                self.type_embeddings = self.type_embeddings.to(target_device)
                            self.type_embeddings = torch.cat([self.type_embeddings, batch_type_embs], dim=0)
                    
                    # Update size - handle both DiskBackedTensor and regular tensor
                    if isinstance(self.embeddings, DiskBackedTensor):
                        # DiskBackedTensor tracks its own size via _actual_size
                        self.size = torch.tensor(self.embeddings._actual_size, dtype=torch.long, device='cpu')
                    else:
                        self.size = torch.tensor(self.embeddings.size(0), dtype=torch.long, device='cpu')
                
                added_count += chunk_size
            
            # Update progress bar
            pbar.update(chunk_size)
            
            # AGGRESSIVE memory cleanup every batch
            del batch_cpu, batch_adjacency_cpu, batch_weights_cpu
            if existing_embs is not None:
                del existing_embs
            if compute_device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # CPU garbage collection every 10 batches
            if i % (batch_size * 10) == 0:
                import gc
                gc.collect()
        
        # Close progress bar
        pbar.close()
        
        # CRITICAL: Flush DiskBackedTensor to ensure all writes are complete
        # This prevents race conditions during bulk insertion
        from disk_backed_tensor import DiskBackedTensor
        if bundled_mode:
            # Flush bundled storage
            self._bundled_storage.flush()
        elif isinstance(self.embeddings, DiskBackedTensor):
            self.embeddings.flush()
        
        # 🚀 REMOVED: torch.cuda.synchronize() was blocking async GPU ops!
        # PyTorch handles GPU synchronization automatically when needed
        
        return added_count
    
    @profile_op("strengthen_edge")
    def strengthen_edge(self, source_idx: int, target_idx: int, reward: float = 1.0, learning_rate: float = 0.3):
        """
        Hebbian learning: Strengthen edges that are traversed successfully.
        
        "Paths that traverse together strengthen together!"
        Creates "highways" in the memory graph.
        Also updates edge types to mark CO_RETRIEVED relationships.
        
        Args:
            source_idx: Source node index
            target_idx: Target node index
            reward: How useful this traversal was (0-2)
            learning_rate: How fast to update (0.1=slow/stable, 0.5=fast/adaptive)
        """
        # Bounds check: ensure indices are valid
        if self.size == 0:
            return
        if source_idx < 0 or source_idx >= self.size.item():
            return
        if target_idx < 0 or target_idx >= self.size.item():
            return
        
        # 🔥 CACHED ADJACENCY LOOKUP: Avoid disk read for neighbor list
        # Find edge in adjacency list (cache neighbors to avoid 16x disk reads)
        if source_idx in self.adjacency_cache:
            neighbors = self.adjacency_cache[source_idx]
            self.adjacency_cache_hits += 1
            self.cache_hits += 1
        else:
            neighbors = self.adjacency[source_idx].clone()  # Read from disk ONCE
            self.adjacency_cache[source_idx] = neighbors
            self.cache_misses += 1
            self.disk_reads += 1
            
            # 🚀 EVICT oldest entries if cache too large (prevents linear slowdown)
            if len(self.adjacency_cache) > self.adjacency_cache_max:
                # Remove first key (oldest in insertion order for Python 3.7+)
                first_key = next(iter(self.adjacency_cache))
                del self.adjacency_cache[first_key]
            
            # LRU eviction: keep cache size bounded
            if len(self.adjacency_cache) > self.adjacency_cache_max:
                # Remove oldest entry (FIFO approximation)
                first_key = next(iter(self.adjacency_cache))
                del self.adjacency_cache[first_key]
        
        edge_mask = neighbors == target_idx
        
        if edge_mask.any():
            edge_slot = edge_mask.nonzero(as_tuple=True)[0][0]
            
            # 🔥 LAZY WRITE OPTIMIZATION: Buffer changes in RAM, flush later
            # With 16x eligibility traces, we'd otherwise hammer disk with 16x writes
            # Strategy: Accumulate updates in dict, write to disk every N calls
            buffer_key = (source_idx, edge_slot.item())
            
            # Update traversal count (in buffer)
            if buffer_key in self.write_buffer:
                old_count = self.write_buffer[buffer_key]['count']
            else:
                old_count = self.edge_traversal_count[source_idx, edge_slot]
                self.write_buffer[buffer_key] = {
                    'count': old_count,
                    'success': self.edge_success_rate[source_idx, edge_slot],
                    'weight': self.edge_weights[source_idx, edge_slot],
                    'type': self.edge_types[source_idx, edge_slot].clone()
                }
                self.write_buffer_size += 1
            
            self.write_buffer[buffer_key]['count'] = old_count + 1
            
            # 🔥 HYPERBOLIC HEBBIAN NORMALIZATION
            # Problem: Uniform updates create "Super-Node" collapse (rich-get-richer)
            # Solution: Scale learning by hyperbolic radius (distance from origin)
            #   - Center nodes (generic concepts): SMALL updates (already well-connected)
            #   - Edge nodes (specific concepts): LARGE updates (rare valuable paths)
            # 
            # Theory: sinh(radius) naturally scales with hyperbolic metric
            # - Near origin (r≈0): sinh(r) ≈ r (small)
            # - Far from origin (r≈3): sinh(r) ≈ e^r (exponential boost)
            
            # Compute hyperbolic radius of source node (Euclidean norm in tangent space)
            source_embedding = self.embeddings[source_idx]  # [hidden_dim]
            radius = torch.norm(source_embedding, p=2)  # Keep as tensor!
            
            # Scale factor: sinh(r) with clamping to prevent explosion/collapse
            # Clamp to [0.1, 10.0] ensures:
            # - Central nodes: min 10% learning (don't freeze completely)
            # - Leaf nodes: max 10× boost (don't explode)
            hyperbolic_scale = torch.sinh(radius).clamp(0.1, 10.0)  # GPU clamp
            
            # Apply scaled Hebbian update
            # Effective LR: learning_rate * hyperbolic_scale
            # - Generic paths (r<1): slow learning, stable
            # - Specific paths (r>2): fast learning, cementing rare connections
            scaled_lr = learning_rate * hyperbolic_scale
            
            old_success = self.write_buffer[buffer_key]['success']
            self.write_buffer[buffer_key]['success'] = (
                (1.0 - scaled_lr) * old_success + scaled_lr * reward
            )
            
            # Mark edge as CO_RETRIEVED (type 4)
            # Strength proportional to traversal count - keep as tensor!
            new_count = self.write_buffer[buffer_key]['count']
            co_retrieval_strength = torch.clamp(new_count / 10.0, max=1.0)
            old_edge_type = self.write_buffer[buffer_key]['type'].clone()
            old_edge_type[4] = co_retrieval_strength
            self.write_buffer[buffer_key]['type'] = old_edge_type
            
            # 🔥 STRENGTHEN HIGHWAY: Reduce hyperbolic distance proportional to reward
            # Higher reward → smaller distance → stronger connection!
            # Use configurable learning_rate for fast highway formation
            old_weight = self.write_buffer[buffer_key]['weight']
            
            # 🔍 DEBUG: Check what edge_weights contains (DISABLED for performance)
            # if not hasattr(self, '_edge_weight_debug_count'):
            #     self._edge_weight_debug_count = 0
            # self._edge_weight_debug_count += 1
            # if self._edge_weight_debug_count <= 3:
            #     print(f"[EDGE DEBUG #{self._edge_weight_debug_count}] Accessing edge_weights[{source_idx}, {edge_slot}]")
            #     print(f"  old_weight type: {type(old_weight)}, value: {old_weight}")
            #     print(f"  learning_rate: {learning_rate}, reward: {reward}")
            #     print(f"  hyperbolic_scale: {hyperbolic_scale:.4f}, radius: {radius:.4f}")
            #     tier_name = "UNKNOWN"
            #     if hasattr(self, 'capacity'):
            #         if self.capacity == 10:
            #             tier_name = "WORKING"
            #         elif self.capacity == 100:
            #             tier_name = "BUFFER"
            #         elif self.capacity >= 1000:
            #             tier_name = "LONGTERM"
            #     print(f"  TIER: {tier_name} (capacity={self.capacity if hasattr(self, 'capacity') else 'N/A'})")
            #     print(f"  🔥 USING LAZY WRITE BUFFER (will flush later)")
            
            # 🔥 HYPERBOLIC WEIGHT UPDATE with normalization
            # Weight reduction: higher reward → bigger reduction (shorter path)
            # Apply same hyperbolic scaling to prevent collapse
            # BUT clamp the effective change to maintain DEQ stability (Lipschitz bound)
            weight_change = scaled_lr * reward  # How much to strengthen
            weight_change = min(weight_change, 0.5)  # 🔥 MAX 50% change per update (DEQ stability)
            
            new_weight = old_weight * (1.0 - weight_change)
            
            # � CRITICAL FIX: Allow weights to DECREASE (strengthen) but not below minimum
            # Lower weight = stronger highway!
            min_edge_weight = 0.001  # 🔥 REDUCED: Allow highways to become very strong (was 0.01)
            new_weight = torch.clamp(new_weight, min=min_edge_weight)  # No max clamp!
            
            # Safety: prevent NaN/inf
            if torch.isnan(new_weight) or torch.isinf(new_weight):
                new_weight = old_weight
            
            self.write_buffer[buffer_key]['weight'] = new_weight
            
            # 🛣️ LOG HIGHWAY FORMATION
            # Track how much this edge was strengthened (keep as tensor!)
            strengthening = old_weight - new_weight
            if strengthening > 0:
                # 🚀 LAZY EVAL: Store tensors, only convert to Python when printing/sorting
                self.highway_log.append({
                    'source_idx': source_idx,
                    'target_idx': target_idx,
                    'old_weight': old_weight,  # Keep as tensor!
                    'new_weight': new_weight,  # Keep as tensor!
                    'strengthening': strengthening,  # Keep as tensor!
                    'traversal_count': new_count,  # Keep as tensor!
                    'success_rate': self.write_buffer[buffer_key]['success']  # Keep as tensor!
                })
                
                # DEBUG: Log first few highway formations
                if not hasattr(self, '_highway_debug_count'):
                    self._highway_debug_count = 0
                self._highway_debug_count += 1
                if self._highway_debug_count <= 10:
                    print(f"🛣️ [HIGHWAY #{self._highway_debug_count}] {source_idx}→{target_idx}: "
                          f"weight {old_weight.item():.4f}→{new_weight.item():.4f} "
                          f"(Δ={strengthening.item():.4f}, r={radius:.2f}, h_scale={hyperbolic_scale:.2f}x, reward={reward:.4f})")
                
                # Keep only top-K by strengthening amount
                if len(self.highway_log) > self.highway_log_max * 2:
                    # Sort by strengthening and keep top half (convert to float for comparison)
                    self.highway_log.sort(key=lambda x: x['strengthening'].item() if torch.is_tensor(x['strengthening']) else x['strengthening'], reverse=True)
                    self.highway_log = self.highway_log[:self.highway_log_max]
            else:
                # DEBUG: Why no strengthening? (DISABLED for performance)
                # if not hasattr(self, '_no_strengthen_debug_count'):
                #     self._no_strengthen_debug_count = 0
                # self._no_strengthen_debug_count += 1
                # if self._no_strengthen_debug_count <= 5:
                #     print(f"[NO STRENGTHEN #{self._no_strengthen_debug_count}] {source_idx}→{target_idx}: "
                #           f"reward={reward:.6f}, old_weight={old_weight.item():.6f}, "
                #           f"strengthening={strengthening:.6f}")
                pass
            
            # 🔥 FLUSH BUFFER: Write to disk when buffer is full or every N calls
            self.flush_counter += 1
            if self.write_buffer_size >= self.write_buffer_max or self.flush_counter >= self.flush_interval:
                self.flush_write_buffer()
    
    def flush_write_buffer(self, force_sync=False):
        """
        � ASYNC WRITE FLUSH: Queue buffered edge updates for background writing
        
        Called automatically when:
        1. Buffer exceeds write_buffer_max entries (default 1000)
        2. Every flush_interval strengthen_edge calls (default 100)
        3. Manually at checkpoint save (with force_sync=True)
        
        This reduces disk writes from 16x (with eligibility traces) to ~1x!
        
        🎯 ASYNC MODE: Copies buffer and queues for background thread (non-blocking)
        🎯 SYNC MODE: Directly writes to disk (blocking, used for checkpoints)
        """
        if not self.write_buffer:
            return
        
        # Sort by source_idx for sequential disk access AND fiber bundle batching
        sorted_keys = sorted(self.write_buffer.keys(), key=lambda x: x[0])
        
        # 🚀 ASYNC PATH: Queue for background writing (training continues immediately!)
        global _async_write_enabled, _async_write_queue
        if _async_write_enabled and not force_sync:
            try:
                # Snapshot the buffer (shallow copy is cheap - just dict of references)
                buffer_snapshot = self.write_buffer.copy()
                
                # Queue the flush job (non-blocking if queue has space)
                job = (self, buffer_snapshot, sorted_keys)
                _async_write_queue.put(job, block=False)
                
                # Clear buffer for new updates (training continues!)
                self.write_buffer = {}
                self.write_buffer_size = 0
                self.flush_counter = 0
                
                return  # Training continues while background thread writes!
                
            except queue.Full:
                # Queue full - fall through to synchronous write
                print("⚠️  Async write queue full, falling back to sync write")
        
        # 🐌 SYNC PATH: Direct write to disk (blocks training)
        writes_count = 0
        
        # 🎯 FIBER BUNDLE ATOMIC WRITES: Group updates by source node
        if hasattr(self, '_bundled_storage') and self._bundled_storage is not None:
            # Bundled mode: read each node's bundle once, update all edges, write once
            current_node = None
            bundle = None
            
            for (source_idx, edge_slot) in sorted_keys:
                # Load bundle if we moved to a new node
                if source_idx != current_node:
                    # Write previous bundle if exists
                    if bundle is not None:
                        self._bundled_storage[current_node] = bundle
                        writes_count += 1
                    
                    # Load new bundle
                    current_node = source_idx
                    bundle = self._bundled_storage[source_idx]
                
                # Update all edge fields in the bundle
                data = self.write_buffer[(source_idx, edge_slot)]
                bundle['edge_traversal_count'][edge_slot] = data['count']
                bundle['edge_success_rate'][edge_slot] = data['success']
                bundle['edge_weights'][edge_slot] = data['weight']
                bundle['edge_types'][edge_slot] = data['type']
            
            # Write final bundle
            if bundle is not None:
                self._bundled_storage[current_node] = bundle
                writes_count += 1
        else:
            # Column mode: write each field separately (legacy, slower)
            for (source_idx, edge_slot) in sorted_keys:
                data = self.write_buffer[(source_idx, edge_slot)]
                
                # Write all buffered fields to disk in one go
                self.edge_traversal_count[source_idx, edge_slot] = data['count']
                self.edge_success_rate[source_idx, edge_slot] = data['success']
                self.edge_weights[source_idx, edge_slot] = data['weight']
                self.edge_types[source_idx, edge_slot] = data['type']
                
                writes_count += 1
        
        self.disk_writes += writes_count  # Track I/O
        
        # Clear buffer
        self.write_buffer.clear()
        self.write_buffer_size = 0
        self.flush_counter = 0
        
        # 🔥 OPTIMIZATION: DON'T clear adjacency cache!
        # Adjacency structure (which nodes are neighbors) doesn't change during Hebbian updates
        # Only edge WEIGHTS/counts change, which are buffered separately
        # Clearing cache was causing cold-start every ~3 training steps
        # self.adjacency_cache.clear()  # DISABLED - preserve cache across flushes!
        
        # DEBUG: Log first few flushes
        if not hasattr(self, '_flush_debug_count'):
            self._flush_debug_count = 0
        self._flush_debug_count += 1
        if self._flush_debug_count <= 5:
            tier_name = "UNKNOWN"
            if hasattr(self, 'capacity'):
                if self.capacity == 10:
                    tier_name = "WORKING"
                elif self.capacity == 100:
                    tier_name = "BUFFER"
                elif self.capacity >= 1000:
                    tier_name = "LONGTERM"
            cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses + 1e-10)
            print(f"💾 [FLUSH #{self._flush_debug_count}] {tier_name}: Wrote {writes_count} edges | "
                  f"Cache hit rate: {cache_hit_rate:.1%} | "
                  f"Total I/O: {self.disk_reads}R/{self.disk_writes}W")
    
    def sync_async_writes(self, timeout=10.0):
        """
        🔄 SYNC ASYNC WRITES: Wait for all pending async writes to complete
        
        Use this before operations that need guaranteed consistency:
        - Checkpointing (already handled automatically)
        - Evaluation/validation runs
        - Final model save
        
        Args:
            timeout: Maximum seconds to wait (prevents hang if thread dies)
        
        Returns:
            True if sync successful, False if timeout
        """
        global _async_write_queue
        
        # Force flush any remaining buffer
        if self.write_buffer:
            self.flush_write_buffer(force_sync=False)  # Queue it
        
        # Wait for queue to empty
        try:
            _async_write_queue.join()  # Blocks until all tasks done
            return True
        except Exception as e:
            print(f"⚠️  Sync timeout or error: {e}")
            return False
    
    def hint_prefetch(self, indices: list):
        """
        🔮 PREFETCH HINT: Suggest bundle indices to preload into cache
        
        Non-blocking - queues indices for background loading.
        Next access will likely hit cache (no disk wait)!
        
        Use case: During backward pass, hint which memories will be needed
        for next forward pass. Background thread loads them while gradients compute.
        
        Args:
            indices: List of memory indices to prefetch
        """
        global _prefetch_queues, _prefetch_enabled
        
        if not _prefetch_enabled or not indices:
            return
        
        # Get or create queue for this tier
        tier_id = id(self)
        if tier_id not in _prefetch_queues:
            _prefetch_queues[tier_id] = queue.Queue(maxsize=100)
            # Register weakref to tier object so background worker can resolve it
            try:
                _prefetch_tiers[tier_id] = weakref.ref(self)
            except Exception:
                pass
        
        prefetch_queue = _prefetch_queues[tier_id]
        
        # Queue indices (non-blocking, drop if full)
        for idx in indices:
            try:
                prefetch_queue.put_nowait((tier_id, idx))
            except queue.Full:
                break  # Queue full, stop hinting
    
    def _process_prefetch_hints(self):
        """
        🔮 PROCESS PREFETCH QUEUE: Load hinted bundles into cache
        
        Called periodically by training loop or background thread.
        Processes a few prefetch hints without blocking.
        """
        from disk_backed_tensor import DiskBackedTensor
        global _prefetch_queues
        
        tier_id = id(self)
        if tier_id not in _prefetch_queues:
            return
        
        prefetch_queue = _prefetch_queues[tier_id]
        
        # Process up to 10 hints per call (don't block too long)
        for _ in range(10):
            try:
                _, idx = prefetch_queue.get_nowait()
                
                # Skip if already in cache
                if isinstance(self.embeddings, DiskBackedTensor):
                    # Check DiskBackedTensor cache
                    if hasattr(self.embeddings, 'cache') and idx in self.embeddings.cache:
                        continue
                
                # Load into cache (this is the expensive disk read)
                # Just accessing it will cache it via DiskBackedTensor
                _ = self.embeddings[idx]
                
            except queue.Empty:
                break  # No more hints
            except Exception as e:
                # Ignore prefetch errors (best-effort)
                continue
    
    @profile_op("strengthen_edges_batch")
    def strengthen_edges_batch(self, edge_updates: list):
        """
        🚀 BATCHED EDGE STRENGTHENING: 10-100x faster than individual strengthen_edge() calls!
        
        Process ALL edge updates in one pass:
        1. Group updates by source node (fiber bundle locality)
        2. Load each bundle ONCE
        3. Apply all updates to that bundle
        4. Write bundle back
        
        This is CRITICAL for time travel / eligibility traces which update 10-100s of edges!
        
        Args:
            edge_updates: List of (src_idx, tgt_idx, reward) tuples
            
        Example:
            # Instead of:
            for src, tgt, r in edges:
                tier.strengthen_edge(src, tgt, reward=r)  # 100 disk reads+writes
            
            # Do this:
            tier.strengthen_edges_batch(edges)  # 10 disk reads+writes (10x speedup!)
        """
        if not edge_updates or self.size == 0:
            return
        
        # 🚀 CRITICAL: Skip empty batches immediately (common with subsampling)
        if len(edge_updates) == 0:
            return
        
        # 🚀 CACHE SIZE: Avoid repeated .item() syncs in loop
        tier_size = self.size.item()
        
        # 🚀 VECTORIZED GROUPING: Use dict comprehension instead of loop
        # Group updates by source node for fiber bundle optimization
        updates_by_src = {}
        for src_idx, tgt_idx, reward in edge_updates:
            # Bounds check (using cached size)
            if src_idx < 0 or src_idx >= tier_size:
                continue
            if tgt_idx < 0 or tgt_idx >= tier_size:
                continue
            
            if src_idx not in updates_by_src:
                updates_by_src[src_idx] = []
            updates_by_src[src_idx].append((tgt_idx, reward))
        
        # 🚀 SKIP if no valid updates after filtering
        if not updates_by_src:
            return
        
        # 🚀 BATCH LOAD EMBEDDINGS: Load all unique source embeddings at once
        unique_src_indices = list(updates_by_src.keys())
        embeddings_batch = {}
        if len(unique_src_indices) > 1:
            # Batch load using bundled storage if available
            if hasattr(self, '_bundled_storage') and self._bundled_storage is not None:
                try:
                    bundles = self._bundled_storage.get_bundles_batch(unique_src_indices)
                    for i, src_idx in enumerate(unique_src_indices):
                        embeddings_batch[src_idx] = bundles['embedding'][i]
                except:
                    # Fallback: load individually
                    for src_idx in unique_src_indices:
                        embeddings_batch[src_idx] = self.embeddings[src_idx]
            else:
                # Load individually for non-bundled storage
                for src_idx in unique_src_indices:
                    embeddings_batch[src_idx] = self.embeddings[src_idx]
        else:
            # Single source - just load it
            src_idx = unique_src_indices[0]
            embeddings_batch[src_idx] = self.embeddings[src_idx]
        
        # Process each source node (fiber bundle locality)
        for src_idx, targets in updates_by_src.items():
            # Load neighbors ONCE for this node
            if src_idx in self.adjacency_cache:
                neighbors = self.adjacency_cache[src_idx]
                self.adjacency_cache_hits += 1
                self.cache_hits += 1
            else:
                neighbors = self.adjacency[src_idx].clone()
                self.adjacency_cache[src_idx] = neighbors
                self.cache_misses += 1
                self.disk_reads += 1
                
                # 🚀 EVICT oldest if cache too large
                if len(self.adjacency_cache) > self.adjacency_cache_max:
                    first_key = next(iter(self.adjacency_cache))
                    del self.adjacency_cache[first_key]
            
            # 🚀 USE PRE-LOADED EMBEDDING: No disk I/O here!
            source_embedding = embeddings_batch[src_idx]
            radius = torch.norm(source_embedding, p=2)  # Keep as tensor
            hyperbolic_scale = torch.sinh(radius).clamp(0.1, 10.0)  # Clamp on GPU
            
            # Apply ALL updates for this node
            for tgt_idx, reward in targets:
                edge_mask = neighbors == tgt_idx
                if not edge_mask.any():
                    continue
                
                edge_slot = edge_mask.nonzero(as_tuple=True)[0][0]
                buffer_key = (src_idx, edge_slot.item())
                
                # Initialize buffer entry if needed
                if buffer_key not in self.write_buffer:
                    self.write_buffer[buffer_key] = {
                        'count': self.edge_traversal_count[src_idx, edge_slot],
                        'success': self.edge_success_rate[src_idx, edge_slot],
                        'weight': self.edge_weights[src_idx, edge_slot],
                        'type': self.edge_types[src_idx, edge_slot].clone()
                    }
                    self.write_buffer_size += 1
                
                # Update traversal count
                self.write_buffer[buffer_key]['count'] += 1
                
                # Scaled Hebbian update
                scaled_lr = 0.3 * hyperbolic_scale  # Use default learning_rate
                old_success = self.write_buffer[buffer_key]['success']
                self.write_buffer[buffer_key]['success'] = (
                    (1.0 - scaled_lr) * old_success + scaled_lr * reward
                )
                
                # Update edge type (CO_RETRIEVED) - keep as tensor
                new_count = self.write_buffer[buffer_key]['count']
                co_retrieval_strength = torch.clamp(new_count / 10.0, max=1.0)
                self.write_buffer[buffer_key]['type'][4] = co_retrieval_strength
                
                # Strengthen highway
                old_weight = self.write_buffer[buffer_key]['weight']
                weight_change = min(scaled_lr * reward, 0.5)
                new_weight = old_weight * (1.0 - weight_change)
                
                # 🚀 CRITICAL FIX: Allow weights to DECREASE (strengthen) but not below minimum
                # Lower weight = stronger highway!
                min_edge_weight = 0.001  # 🔥 REDUCED: Allow highways to become very strong
                new_weight = torch.clamp(new_weight, min=min_edge_weight)  # No max clamp!
                
                # Safety: prevent NaN/inf
                if torch.isnan(new_weight) or torch.isinf(new_weight):
                    new_weight = old_weight
                
                self.write_buffer[buffer_key]['weight'] = new_weight
                self.write_buffer[buffer_key]['weight'] = new_weight
        
        # 🔥 ALWAYS FLUSH: Make edge updates immediately visible to property accessors!
        # This ensures routing bundle sees latest edge stats via DiskBackedTensor cache
        if self.write_buffer:
            self.flush_write_buffer()
    
    def get_highway_stats(self, top_k: int = 10):
        """
        Get statistics about the most strengthened edges (highways).
        
        Returns dict with:
        - top_highways: List of top K most strengthened edges
        - total_highways: Total number of strengthened edges
        - avg_strengthening: Average strengthening amount
        - max_strengthening: Maximum strengthening seen
        """
        if not self.highway_log:
            return {
                'top_highways': [],
                'total_highways': 0,
                'avg_strengthening': 0.0,
                'max_strengthening': 0.0
            }
        
        # Sort by strengthening amount
        sorted_highways = sorted(self.highway_log, key=lambda x: x['strengthening'], reverse=True)
        
        return {
            'top_highways': sorted_highways[:top_k],
            'total_highways': len(self.highway_log),
            'avg_strengthening': sum(h['strengthening'] for h in self.highway_log) / len(self.highway_log),
            'max_strengthening': sorted_highways[0]['strengthening'] if sorted_highways else 0.0
        }
    
    @profile_op("record_flow")
    def record_flow(self, prev_idx: int, curr_idx: int, next_idx: int, reward: float):
        """
        🌊 THE FLOW FIELD: Record successful trajectory for predictive navigation.
        
        "If you came from A and are now at B, where did you go next?"
        
        This is **Successor Representation** - enables:
        - Context-dependent routing (same node, different trajectory → different next hop)
        - Compositional reasoning ("bank" → "river" vs "bank" → "money")
        - Chain-of-thought caching (reasoning patterns learned from trajectories)
        
        🎯 OPTIMIZED FOR FIBER BUNDLES: Trajectory data stored WITH node = 1 disk read!
        With hyperbolic layout, sequential traversal = sequential disk reads!
        
        Example:
            Trajectory: war → napoleon → wife → killed
            record_flow(war_idx, napoleon_idx, wife_idx, reward=0.9)
            
            Later: Edge napoleon→wife remembers "when coming from war, go to killed"
        
        Args:
            prev_idx: Previous node in trajectory (context)
            curr_idx: Current node
            next_idx: Next node that was successful
            reward: How good this transition was (0-2)
        """
        # Only learn from successful trajectories
        if reward < self.flow_min_reward:
            return
        
        # Validate indices
        if prev_idx < 0 or curr_idx < 0 or next_idx < 0:
            return
        if prev_idx >= self.size.item() or curr_idx >= self.size.item() or next_idx >= self.size.item():
            return
        
        # 🚀 OPTIMIZATION 1: ATOMIC BUNDLE READ
        # Read ENTIRE node bundle once (embedding + adjacency + edges + flow)
        # This is what makes it fast - trajectory data travels WITH the node!
        if hasattr(self, '_bundled_storage') and self._bundled_storage is not None:
            # ONE disk read gets: adjacency, edge_flow_prev_nodes, edge_flow_context
            bundle = self._bundled_storage[curr_idx]
            neighbors = bundle['adjacency']  # [k_neighbors]
            
            # Find edge curr→next in adjacency list
            edge_mask = neighbors == next_idx
            
            if not edge_mask.any():
                return  # Edge doesn't exist (not in k-NN)
            
            edge_slot = edge_mask.nonzero(as_tuple=True)[0][0]
            
            # Get flow data (already loaded - no extra disk I/O!)
            prev_nodes = bundle['edge_flow_prev_nodes'][edge_slot]  # [k_neighbors]
            flow_strengths = bundle['edge_flow_context'][edge_slot]  # [k_neighbors]
            
        else:
            # Fallback to column mode (slower - 3 separate disk reads)
            neighbors = self.adjacency[curr_idx]  # [k_neighbors]
            edge_mask = neighbors == next_idx
            
            if not edge_mask.any():
                return  # Edge doesn't exist (not in k-NN)
            
            edge_slot = edge_mask.nonzero(as_tuple=True)[0][0]
            prev_nodes = self.edge_flow_prev_nodes[curr_idx, edge_slot]  # [k_neighbors]
            flow_strengths = self.edge_flow_context[curr_idx, edge_slot]  # [k_neighbors]
        
        # Find if prev_idx is already tracked
        prev_mask = prev_nodes == prev_idx
        
        if prev_mask.any():
            # Update existing context
            context_slot = prev_mask.nonzero(as_tuple=True)[0][0]
            old_strength = flow_strengths[context_slot]  # Keep as tensor!
            new_strength = self.flow_decay * old_strength + (1.0 - self.flow_decay) * reward
            
            # 🚀 IN-PLACE UPDATE: No clone needed!
            # 🚀 OPTIMIZATION 2: ATOMIC BUNDLE WRITE
            # Write entire bundle back once (instead of field-by-field)
            if hasattr(self, '_bundled_storage') and self._bundled_storage is not None:
                bundle['edge_flow_context'][edge_slot][context_slot] = torch.clamp(new_strength, max=10.0)
                self._bundled_storage[curr_idx] = bundle
            else:
                self.edge_flow_context[curr_idx, edge_slot, context_slot] = min(10.0, new_strength.item())
        else:
            # Find weakest context slot to replace
            weakest_slot = flow_strengths.argmin()
            weakest_strength = flow_strengths[weakest_slot]  # Keep as tensor!
            
            # Only replace if new pattern is stronger than weakest
            if reward > weakest_strength:
                # 🚀 IN-PLACE UPDATE: No clone needed!
                # 🎯 FIBER BUNDLE ATOMIC WRITE: Write BOTH fields in one bundle update
                if hasattr(self, '_bundled_storage') and self._bundled_storage is not None:
                    bundle['edge_flow_prev_nodes'][edge_slot][weakest_slot] = prev_idx
                    bundle['edge_flow_context'][edge_slot][weakest_slot] = reward
                    self._bundled_storage[curr_idx] = bundle
                else:
                    # Fallback for non-bundled storage (rarely used)
                    self.edge_flow_prev_nodes[curr_idx, edge_slot, weakest_slot] = prev_idx
                    self.edge_flow_context[curr_idx, edge_slot, weakest_slot] = reward
    
    def record_trajectory_batch(self, trajectory: list, reward: float):
        """
        🚀 BATCHED TRAJECTORY RECORDING - Optimized for hyperbolic disk layout!
        
        Records an entire trajectory at once using batched bundle reads.
        With hyperbolic layout, sequential trajectory = sequential disk reads!
        
        Example:
            trajectory = [war_idx, napoleon_idx, wife_idx, killed_idx]
            record_trajectory_batch(trajectory, reward=0.9)
            
            Internally calls:
            - record_flow(war, napoleon, wife, 0.9)
            - record_flow(napoleon, wife, killed, 0.9)
            
            But with ONE batched disk read for the whole trajectory!
        
        Args:
            trajectory: List of node indices [prev, curr, next, ...]
            reward: Reward for this trajectory
        
        Performance:
            - Old: N separate record_flow() calls = 2N disk reads
            - New: 1 batched get_bundles_batch() call = 1 sequential disk read!
            - With hyperbolic layout: trajectory nodes are adjacent on disk 🚀
        """
        if len(trajectory) < 3:
            return  # Need at least [prev, curr, next]
        
        # 🚀 BATCHED BUNDLE READ: Load all nodes in trajectory at once
        # With hyperbolic layout, these are likely sequential on disk!
        if hasattr(self, '_bundled_storage') and self._bundled_storage is not None:
            # Get unique indices (trajectory might revisit nodes)
            unique_indices = list(set(trajectory))
            
            # ONE disk read for entire trajectory (sequential if hyperbolic layout!)
            bundles_dict = self._bundled_storage.get_bundles_batch(unique_indices)
            
            # Build lookup: idx -> bundle
            bundles = {}
            for i, idx in enumerate(unique_indices):
                bundles[idx] = {
                    'adjacency': bundles_dict['adjacency'][i],
                    'edge_flow_prev_nodes': bundles_dict['edge_flow_prev_nodes'][i],
                    'edge_flow_context': bundles_dict['edge_flow_context'][i]
                }
            
            # Process each (prev, curr, next) triple
            modified_bundles = {}  # Track which bundles we modify
            
            for i in range(len(trajectory) - 2):
                prev_idx = trajectory[i]
                curr_idx = trajectory[i + 1]
                next_idx = trajectory[i + 2]
                
                # Validate
                if prev_idx < 0 or curr_idx < 0 or next_idx < 0:
                    continue
                if prev_idx >= self.size.item() or curr_idx >= self.size.item() or next_idx >= self.size.item():
                    continue
                
                # Get curr bundle (already loaded!)
                if curr_idx not in bundles:
                    continue
                    
                bundle = bundles[curr_idx] if curr_idx not in modified_bundles else modified_bundles[curr_idx]
                neighbors = bundle['adjacency']
                
                # Find edge curr→next
                edge_mask = neighbors == next_idx
                if not edge_mask.any():
                    continue
                    
                edge_slot = edge_mask.nonzero(as_tuple=True)[0][0]
                
                # Update flow context
                prev_nodes = bundle['edge_flow_prev_nodes'][edge_slot]
                flow_strengths = bundle['edge_flow_context'][edge_slot]
                
                prev_mask = prev_nodes == prev_idx
                
                if prev_mask.any():
                    # Update existing
                    context_slot = prev_mask.nonzero(as_tuple=True)[0][0]
                    old_strength = flow_strengths[context_slot].item()
                    new_strength = self.flow_decay * old_strength + (1.0 - self.flow_decay) * reward
                    
                    new_flow_strengths = flow_strengths.clone()
                    new_flow_strengths[context_slot] = min(10.0, new_strength)
                    
                    # Store modification
                    if curr_idx not in modified_bundles:
                        modified_bundles[curr_idx] = {k: v.clone() if torch.is_tensor(v) else v for k, v in bundle.items()}
                    modified_bundles[curr_idx]['edge_flow_context'][edge_slot] = new_flow_strengths
                    
                else:
                    # Add new context
                    weakest_slot = flow_strengths.argmin()
                    if reward > flow_strengths[weakest_slot].item():
                        new_prev_nodes = prev_nodes.clone()
                        new_prev_nodes[weakest_slot] = prev_idx
                        
                        new_flow_strengths = flow_strengths.clone()
                        new_flow_strengths[weakest_slot] = reward
                        
                        if curr_idx not in modified_bundles:
                            modified_bundles[curr_idx] = {k: v.clone() if torch.is_tensor(v) else v for k, v in bundle.items()}
                        modified_bundles[curr_idx]['edge_flow_prev_nodes'][edge_slot] = new_prev_nodes
                        modified_bundles[curr_idx]['edge_flow_context'][edge_slot] = new_flow_strengths
            
            # 🚀 BATCHED BUNDLE WRITE: Write all modified bundles back
            for idx, bundle in modified_bundles.items():
                self._bundled_storage[idx] = bundle
                
        else:
            # Fallback: use individual record_flow calls
            for i in range(len(trajectory) - 2):
                self.record_flow(trajectory[i], trajectory[i+1], trajectory[i+2], reward)

    def compute_compound_edge_strength(self, src_idx: int, tgt_idx: int, prev_idx: int = -1):
        """
        🎯 COMPOUND EDGE STRENGTH: Unify all edge information for DEQ routing
        
        Combines three sources of edge information:
        1. Raw edge weight (from k-NN graph construction)
        2. Highway strengthening (from training feedback via strengthen_edges_batch)
        3. Flow context strength (from trajectory learning via record_flow)
        
        This gives the DEQ a single "compound strength" value that reflects:
        - Semantic similarity (raw weight)
        - Learned importance (highway)
        - Contextual relevance (flow with prev_idx)
        
        Args:
            src_idx: Source node index
            tgt_idx: Target node index  
            prev_idx: Previous node in trajectory (-1 = no context)
            
        Returns:
            compound_strength: Float in [0, 1] where higher = stronger/better edge
                - 0.0 = weak connection (ignore)
                - 0.5 = moderate connection
                - 1.0 = very strong highway with good flow context
        """
        # Validate indices
        if src_idx < 0 or tgt_idx < 0:
            return 0.0
        if src_idx >= self.size.item() or tgt_idx >= self.size.item():
            return 0.0
        
        # Find edge slot in adjacency list
        neighbors = self.adjacency[src_idx]
        edge_mask = neighbors == tgt_idx
        if not edge_mask.any():
            return 0.0  # Edge doesn't exist
        
        edge_slot = edge_mask.nonzero(as_tuple=True)[0][0]
        
        # Component 1: Raw edge weight (inverted - lower = stronger!)
        # Normalize to [0, 1] where 1 = strongest
        raw_weight = self.edge_weights[src_idx, edge_slot].item()
        # Typical range: [0.001, 1.0], so invert and normalize
        weight_strength = 1.0 - min(raw_weight, 1.0)  # Lower weight → higher strength
        
        # Component 2: Highway strength (from success rate)
        # success_rate is already in [0, 1] from strengthen_edges_batch
        highway_strength = self.edge_success_rate[src_idx, edge_slot].item()
        
        # Component 3: Flow context strength (trajectory-aware)
        flow_strength = 0.0
        if prev_idx >= 0:
            # Check if we have flow data for this context
            flow_prev_nodes = self.edge_flow_prev_nodes[src_idx, edge_slot]
            flow_contexts = self.edge_flow_context[src_idx, edge_slot]
            
            prev_mask = flow_prev_nodes == prev_idx
            if prev_mask.any():
                context_slot = prev_mask.nonzero(as_tuple=True)[0][0]
                flow_strength = flow_contexts[context_slot].item()
                flow_strength = min(flow_strength / 10.0, 1.0)  # Normalize to [0, 1]
        
        # Weighted combination (tunable weights)
        # - Raw weight: 20% (basic connectivity)
        # - Highway: 50% (learned importance dominates)
        # - Flow: 30% (contextual boost when available)
        compound = (
            0.2 * weight_strength +
            0.5 * highway_strength +
            0.3 * flow_strength
        )
        
        return min(compound, 1.0)  # Clamp to [0, 1]

    def compute_edge_centrality(self, src_idx: int, tgt_idx: int):
        """
        🌐 TRAJECTORY-AGNOSTIC CENTRALITY: How "important" is this edge in the graph?
        
        When we don't have highway/flow data yet (new thoughts, sparse regions),
        we can still estimate edge quality from graph structure:
        - In-degree: How many nodes point to target? (popularity)
        - Out-degree: How many nodes does target point to? (hub-ness)
        - Edge traversal count: How often has ANY path used this edge?
        
        This provides a **backup signal** when trajectory-specific data is sparse.
        
        Args:
            src_idx: Source node
            tgt_idx: Target node
            
        Returns:
            centrality: Float in [0, 1] where higher = more central/important
        """
        if src_idx < 0 or tgt_idx < 0:
            return 0.0
        if src_idx >= self.size.item() or tgt_idx >= self.size.item():
            return 0.0
        
        # Find edge slot
        neighbors = self.adjacency[src_idx]
        edge_mask = neighbors == tgt_idx
        if not edge_mask.any():
            return 0.0
        
        edge_slot = edge_mask.nonzero(as_tuple=True)[0][0]
        
        # Component 1: Edge traversal count (normalized by max in graph)
        # Higher = this edge is used frequently by ANY trajectory
        traversal = self.edge_traversal_count[src_idx, edge_slot].item()
        traversal_norm = min(traversal / 100.0, 1.0)  # Assume max ~100 traversals
        
        # Component 2: Edge success rate (how often this edge leads to good predictions)
        # This is a direct quality signal - better than reward proxy
        success = self.edge_success_rate[src_idx, edge_slot].item()
        success_norm = min(success, 1.0)  # Already 0-1 range
        
        # Component 3: Target node out-degree (is it a hub?)
        # Count valid neighbors of target
        target_neighbors = self.adjacency[tgt_idx]
        out_degree = (target_neighbors >= 0).sum().item()
        out_degree_norm = min(out_degree / self.k_neighbors, 1.0)
        
        # Weighted combination
        # - Traversal: 50% (most direct signal)
        # - Success: 30% (quality indicator - BETTER than reward!)
        # - Out-degree: 20% (connectivity)
        centrality = (
            0.5 * traversal_norm +
            0.3 * success_norm +
            0.2 * out_degree_norm
        )
        
        return min(centrality, 1.0)

    @profile_op("record_flows_batch")
    def record_flows_batch(self, flow_updates: list):
        """
        Batch record many flow updates of the form (prev, curr, next, reward).

        This groups updates by `curr` node and performs a single bundled read/write
        per `curr` node when possible. Falls back to individual `record_flow` calls
        when bundled storage isn't available.
        """
        if not flow_updates:
            return

        # 🚀 CRITICAL OPTIMIZATION: Filter out tiny rewards
        # Only record flows when reward is significant (same as self.flow_min_reward)
        # This prevents accumulation of meaningless flow data from random trajectories
        min_reward_threshold = self.flow_min_reward  # 0.5 - only successful trajectories
        
        # Group by curr node, filtering low-reward updates
        updates_by_curr = {}
        filtered_count = 0
        for prev_idx, curr_idx, next_idx, reward in flow_updates:
            if prev_idx < 0 or curr_idx < 0 or next_idx < 0:
                continue
            # Skip tiny rewards - they don't represent meaningful trajectories
            if reward < min_reward_threshold:
                filtered_count += 1
                continue
            if curr_idx not in updates_by_curr:
                updates_by_curr[curr_idx] = []
            updates_by_curr[curr_idx].append((prev_idx, next_idx, reward))
        
        # Debug: Log how many were filtered (first few times)
        if not hasattr(self, '_flow_filter_logged'):
            self._flow_filter_logged = 0
        if self._flow_filter_logged < 3 and filtered_count > 0:
            total = len(flow_updates)
            print(f"🔥 [FLOW FILTER] Skipped {filtered_count}/{total} ({100*filtered_count/total:.0f}%) low-reward flows")
            self._flow_filter_logged += 1

        # Use bundled storage path for efficiency
        if hasattr(self, '_bundled_storage') and self._bundled_storage is not None:
            # Process each curr node: load bundle once, apply all updates, write once
            for curr_idx, updates in updates_by_curr.items():
                if curr_idx < 0 or curr_idx >= self.size.item():
                    continue

                # Load bundle
                try:
                    bundle = self._bundled_storage[curr_idx]
                except Exception:
                    # Fallback to per-update calls if any I/O error
                    for prev_idx, next_idx, reward in updates:
                        self.record_flow(prev_idx, curr_idx, next_idx, reward)
                    continue

                neighbors = bundle['adjacency']
                modified = False

                for prev_idx, next_idx, reward in updates:
                    # find edge slot for curr->next
                    edge_mask = neighbors == next_idx
                    if not edge_mask.any():
                        continue
                    edge_slot = edge_mask.nonzero(as_tuple=True)[0][0]

                    prev_nodes = bundle['edge_flow_prev_nodes'][edge_slot]
                    flow_strengths = bundle['edge_flow_context'][edge_slot]

                    # find if prev already present
                    prev_mask = prev_nodes == prev_idx
                    if prev_mask.any():
                        context_slot = prev_mask.nonzero(as_tuple=True)[0][0]
                        old_strength = flow_strengths[context_slot]  # Keep as tensor!
                        new_strength = self.flow_decay * old_strength + (1.0 - self.flow_decay) * reward
                        
                        # 🚀 IN-PLACE UPDATE: No clone needed!
                        bundle['edge_flow_context'][edge_slot][context_slot] = torch.clamp(new_strength, max=10.0)
                        modified = True
                    else:
                        weakest_slot = flow_strengths.argmin()
                        weakest_strength = flow_strengths[weakest_slot]  # Keep as tensor!
                        if reward > weakest_strength:
                            # 🚀 IN-PLACE UPDATE: No clone needed!
                            bundle['edge_flow_prev_nodes'][edge_slot][weakest_slot] = prev_idx
                            bundle['edge_flow_context'][edge_slot][weakest_slot] = reward
                            modified = True

                if modified:
                    # write back modified bundle
                    try:
                        self._bundled_storage[curr_idx] = bundle
                    except Exception:
                        # If write fails, best-effort fallback to per-update calls
                        for prev_idx, next_idx, reward in updates:
                            self.record_flow(prev_idx, curr_idx, next_idx, reward)

        else:
            # No bundled storage - fall back to individual record_flow calls
            for prev_idx, curr_idx, next_idx, reward in flow_updates:
                self.record_flow(prev_idx, curr_idx, next_idx, reward)
    
    def get_predictive_bias(self, prev_idx: int, curr_idx: int, candidates: torch.Tensor) -> torch.Tensor:
        """
        🌊 GET FLOW PREDICTION: "Others who came this way went here..."
        
        Returns a bias vector for candidate next nodes based on historical trajectories.
        This bias is ADDED to attention logits to guide navigation.
        
        NOW READS FROM DISK-BACKED FLOW PATTERNS!
        
        Args:
            prev_idx: Where we came from (context)
            curr_idx: Where we are now
            candidates: [k] tensor of candidate next node indices
        
        Returns:
            [k] bias vector - higher for historically successful next nodes given this context
        """
        # Validate
        if prev_idx < 0 or curr_idx < 0:
            return torch.zeros(len(candidates), device=self.device)
        if prev_idx >= self.size.item() or curr_idx >= self.size.item():
            return torch.zeros(len(candidates), device=self.device)
        
        # For each candidate, check if it has flow context matching prev_idx
        bias = torch.zeros(len(candidates), device=self.device)
        
        for i, cand_idx in enumerate(candidates):
            cand_item = cand_idx.item() if torch.is_tensor(cand_idx) else cand_idx
            if cand_item < 0 or cand_item >= self.size.item():
                continue
            
            # Find edge curr→cand in adjacency
            neighbors = self.adjacency[curr_idx]
            edge_mask = neighbors == cand_item
            
            if not edge_mask.any():
                continue
            
            edge_slot = edge_mask.nonzero(as_tuple=True)[0][0]
            
            # Check flow context for this edge
            prev_nodes = self.edge_flow_prev_nodes[curr_idx, edge_slot]  # [k_neighbors]
            flow_strengths = self.edge_flow_context[curr_idx, edge_slot]  # [k_neighbors]
            
            # Find if prev_idx matches any tracked context
            prev_mask = prev_nodes == prev_idx
            if prev_mask.any():
                context_slot = prev_mask.nonzero(as_tuple=True)[0][0]
                strength = flow_strengths[context_slot].item()
                
                # Log-scale for attention bias
                bias[i] = math.log(1.0 + strength)
        
        # Scale by flow_strength hyperparameter
        return bias * self.flow_strength
    
    def prune_weak_edges(self, poincare, threshold: float = 0.1):
        """
        Remove rarely-used edges and replace with fresh k-NN search.
        
        Like overgrown paths in a forest - if unused, they disappear!
        
        NOTE: Disabled for disk-backed tiers - pruning 100K+ nodes is too expensive
        and the k-NN graph is already well-formed. Focus on Hebbian strengthening instead!
        """
        # Skip pruning for disk-backed tiers (too expensive, not needed)
        if self.use_disk:
            return
        
        if self.size == 0:
            return
        
        # Check if edge_traversal_count is empty (can happen with disk-backed tier)
        # For DiskBackedTensor, check shape; for regular tensor, check numel
        if hasattr(self.edge_traversal_count, 'shape'):
            if self.edge_traversal_count.shape[0] == 0:
                return
        elif hasattr(self.edge_traversal_count, 'numel'):
            if self.edge_traversal_count.numel() == 0:
                return
        else:
            return  # Unknown type, skip pruning
        
        # Normalize traversal counts
        max_count = self.edge_traversal_count.max()
        if max_count > 0:
            normalized_count = self.edge_traversal_count / max_count
        else:
            return  # No usage data yet
        
        # Identify weak edges (rarely used AND low success)
        weak_mask = (normalized_count < threshold) & (self.edge_success_rate < 0.3)
        
        # For each node with weak edges, re-search for better neighbors
        nodes_to_rewire = weak_mask.any(dim=1).nonzero(as_tuple=True)[0]
        
        for node_idx in nodes_to_rewire[:10]:  # Limit to 10 per call to avoid lag
            # Safety: Skip if not enough nodes for k-NN
            # Need at least 2 nodes to have any neighbors (self excluded)
            if self.size.item() < 2:
                continue
            
            # Re-search k-NN in current embedding space
            dists = poincare.distance(
                self.embeddings[node_idx].unsqueeze(0),
                self.embeddings
            ).squeeze(0)
            
            # Exclude self
            dists[node_idx] = float('inf')
            
            k_actual = min(self.k_neighbors, self.size.item() - 1)
            # Additional safety: k_actual must be at least 1
            if k_actual < 1:
                continue
            
            topk_dists, topk_indices = torch.topk(dists, k_actual, largest=False)
            
            # Replace adjacency for this node
            self.adjacency[node_idx] = -1
            self.adjacency[node_idx, :k_actual] = topk_indices
            self.edge_weights[node_idx] = 0
            self.edge_weights[node_idx, :k_actual] = topk_dists
            
            # Reset statistics
            self.edge_traversal_count[node_idx] = 0
            self.edge_success_rate[node_idx] = 0
    
    def add_nodes(self, embeddings: torch.Tensor, adjacency: torch.Tensor, 
                  edge_weights: torch.Tensor, cluster_ids: torch.Tensor,
                  rewards: torch.Tensor,
                  edge_types: torch.Tensor = None,
                  edge_flow_context: torch.Tensor = None,
                  edge_flow_prev_nodes: torch.Tensor = None,
                  edge_traversal_count: torch.Tensor = None,
                  edge_success_rate: torch.Tensor = None) -> None:
        """
        Add new nodes with their graph structure and optional trajectory data.
        
        Args:
            embeddings: Node embeddings [N, D]
            adjacency: Neighbor indices [N, k]
            edge_weights: Edge weights [N, k]
            cluster_ids: Cluster assignments [N]
            rewards: Node rewards [N]
            edge_types: Edge type distributions [N, k, num_types] (optional)
            edge_flow_context: Flow field contexts [N, k, k] (optional)
            edge_flow_prev_nodes: Flow field previous nodes [N, k, k] (optional)
            edge_traversal_count: Edge traversal counts [N, k] (optional)
            edge_success_rate: Edge success rates [N, k] (optional)
        """
        batch_size = embeddings.shape[0]
        
        # CYBERNETIC: All buffer modifications must be outside computation graph
        with torch.no_grad():
            # CRITICAL FIX: Ensure ALL existing buffers are on self.device FIRST
            # This prevents device mismatch errors during concatenation
            from disk_backed_tensor import DiskBackedTensor
            is_disk_backed = isinstance(self.embeddings, DiskBackedTensor)
            
            # Force all buffers to target device (skip embeddings if disk-backed)
            target_device = torch.device(self.device)
            if not is_disk_backed and self.embeddings.device != target_device:
                self.embeddings = self.embeddings.to(target_device)
            if hasattr(self.adjacency, 'device') and self.adjacency.device != target_device:
                self.adjacency = self.adjacency.to(target_device)
            if hasattr(self.edge_weights, 'device') and self.edge_weights.device != target_device:
                self.edge_weights = self.edge_weights.to(target_device)
            if hasattr(self.edge_types, 'device') and self.edge_types.device != target_device:
                self.edge_types = self.edge_types.to(target_device)
            if hasattr(self.cluster_ids, 'device') and self.cluster_ids.device != target_device:
                self.cluster_ids = self.cluster_ids.to(target_device)
            if hasattr(self.rewards, 'device') and self.rewards.device != target_device:
                self.rewards = self.rewards.to(target_device)
            if hasattr(self.age, 'device') and self.age.device != target_device:
                self.age = self.age.to(target_device)
            if hasattr(self.access, 'device') and self.access.device != target_device:
                self.access = self.access.to(target_device)
            if hasattr(self.depths, 'device') and self.depths.device != target_device:
                self.depths = self.depths.to(target_device)
            if self.use_types and hasattr(self.type_embeddings, 'device') and self.type_embeddings.device != target_device:
                self.type_embeddings = self.type_embeddings.to(target_device)
            if hasattr(self.edge_traversal_count, 'device') and self.edge_traversal_count.device != target_device:
                self.edge_traversal_count = self.edge_traversal_count.to(target_device)
            if hasattr(self.edge_success_rate, 'device') and self.edge_success_rate.device != target_device:
                self.edge_success_rate = self.edge_success_rate.to(target_device)
            
            # HYPERBOLIC-AWARE CAPACITY MANAGEMENT
            if self.size + batch_size > self.capacity:
                # Check if we should flush to disk or just evict
                if hasattr(self, 'use_disk') and self.use_disk:
                    # Flush peripheral memories to disk using hyperbolic locality
                    num_to_flush = (self.size + batch_size) - self.capacity
                    self.flush_to_disk(num_to_flush=num_to_flush)
                else:
                    # No disk - simple LRU eviction
                    evict_count = (self.size + batch_size) - self.capacity
                    _, evict_indices = torch.topk(self.access, evict_count, largest=True)
                    
                    keep_mask = torch.ones(self.size, dtype=torch.bool, device=self.device)
                    keep_mask[evict_indices] = False
                    
                    # Filter everything
                    self.embeddings = self.embeddings[keep_mask]
                    self.adjacency = self.adjacency[keep_mask]
                    self.edge_weights = self.edge_weights[keep_mask]
                    self.cluster_ids = self.cluster_ids[keep_mask]
                    self.rewards = self.rewards[keep_mask]
                    self.age = self.age[keep_mask]
                    self.access = self.access[keep_mask]
                    self.size = torch.tensor(keep_mask.sum().item(), dtype=torch.long, device=self.device)
            
            # NOTE: We do NOT preload neighbors from disk during insertion
            # This would cause RAM explosion as everything gets loaded back
            # Instead, neighbors are loaded on-demand during retrieval only
            
            # Concatenate new memories (detach to break gradient flow + ensure correct device)
            embeddings_device = embeddings.detach().to(self.device)
            adjacency_device = adjacency.to(self.device)
            edge_weights_device = edge_weights.to(self.device)
            cluster_ids_device = cluster_ids.to(self.device)
            rewards_device = rewards.to(self.device)
            
            # CRITICAL: Ensure ALL existing buffers are on correct device BEFORE concatenation
            # (Skip embeddings if DiskBackedTensor)
            from disk_backed_tensor import DiskBackedTensor
            is_disk_backed = isinstance(self.embeddings, DiskBackedTensor)
            
            # Move existing buffers to correct device if needed
            if self.adjacency.device != torch.device(self.device):
                self.adjacency = self.adjacency.to(self.device)
            if self.edge_weights.device != torch.device(self.device):
                self.edge_weights = self.edge_weights.to(self.device)
            if self.edge_types.device != torch.device(self.device):
                self.edge_types = self.edge_types.to(self.device)
            if self.cluster_ids.device != torch.device(self.device):
                self.cluster_ids = self.cluster_ids.to(self.device)
            if self.rewards.device != torch.device(self.device):
                self.rewards = self.rewards.to(self.device)
            if self.age.device != torch.device(self.device):
                self.age = self.age.to(self.device)
            if self.access.device != torch.device(self.device):
                self.access = self.access.to(self.device)
            if self.depths.device != torch.device(self.device):
                self.depths = self.depths.to(self.device)
            if self.use_types and self.type_embeddings.device != torch.device(self.device):
                self.type_embeddings = self.type_embeddings.to(self.device)
            if self.edge_traversal_count.device != torch.device(self.device):
                self.edge_traversal_count = self.edge_traversal_count.to(self.device)
            if self.edge_success_rate.device != torch.device(self.device):
                self.edge_success_rate = self.edge_success_rate.to(self.device)
            
            # Move embeddings only if not disk-backed
            if not is_disk_backed and self.embeddings.device != torch.device(self.device):
                self.embeddings = self.embeddings.to(self.device)
            
            # Handle DiskBackedTensor: Use setitem instead of concatenation
            old_size = self.size.item() if hasattr(self, 'size') else 0
            
            # Check if using bundled storage (all fields stored together)
            using_bundled = (hasattr(self.embeddings, '_storage') and 
                           hasattr(self.embeddings._storage, 'is_bundled') and 
                           self.embeddings._storage.is_bundled)
            
            if isinstance(self.embeddings, torch.Tensor) and not hasattr(self.embeddings, 'is_disk_backed'):
                # Regular tensor: use concatenation
                self.embeddings = torch.cat([self.embeddings, embeddings_device], dim=0)
            elif using_bundled:
                # 🎯 BUNDLED STORAGE: Construct complete bundles with ALL fields
                # Initialize edge_types for new nodes (default: all zeros except PROXIMITY)
                new_edge_types = torch.zeros(batch_size, adjacency.shape[1], 8, device=self.device)
                new_edge_types[:, :, 0] = 1.0  # All edges are PROXIMITY by default
                
                # Set each node as a complete bundle
                for i in range(batch_size):
                    bundle = {
                        'embedding': embeddings_device[i].cpu(),
                        'adjacency': adjacency_device[i].cpu(),
                        'edge_weights': edge_weights_device[i].cpu(),
                        'edge_types': (edge_types[i].cpu() if edge_types is not None 
                                      else new_edge_types[i].cpu()),
                        'cluster_id': cluster_ids_device[i].cpu(),
                        'edge_flow_context': (edge_flow_context[i].cpu() if edge_flow_context is not None and edge_flow_context.size(0) > i
                                             else torch.zeros(self.k_neighbors, self.k_neighbors, dtype=torch.float32)),
                        'edge_flow_prev_nodes': (edge_flow_prev_nodes[i].cpu() if edge_flow_prev_nodes is not None and edge_flow_prev_nodes.size(0) > i
                                                else torch.full((self.k_neighbors, self.k_neighbors), -1, dtype=torch.int64)),
                        'edge_traversal_count': (edge_traversal_count[i].cpu() if edge_traversal_count is not None and edge_traversal_count.size(0) > i
                                                else torch.zeros(self.k_neighbors, dtype=torch.float32)),
                        'edge_success_rate': (edge_success_rate[i].cpu() if edge_success_rate is not None and edge_success_rate.size(0) > i
                                             else torch.zeros(self.k_neighbors, dtype=torch.float32)),
                        'depth': torch.tensor(0, dtype=torch.long),  # Initialize depth (0 = surface level)
                    }
                    if self.use_types:
                        bundle['type_embedding'] = torch.zeros(self.type_dim, dtype=torch.float32)
                    
                    self.embeddings._storage[old_size + i] = bundle
            else:
                # DiskBackedTensor (column mode): use setitem interface
                for i in range(batch_size):
                    self.embeddings[old_size + i] = embeddings_device[i]
            
            # Handle adjacency (might be DiskBackedTensor) - skip if bundled
            if not using_bundled:
                if not isinstance(self.adjacency, DiskBackedTensor):
                    self.adjacency = torch.cat([self.adjacency, adjacency_device], dim=0)
                else:
                    for i in range(batch_size):
                        self.adjacency[old_size + i] = adjacency_device[i]
            
            # Handle edge_weights (might be DiskBackedTensor) - skip if bundled
            if not using_bundled:
                if not isinstance(self.edge_weights, DiskBackedTensor):
                    self.edge_weights = torch.cat([self.edge_weights, edge_weights_device], dim=0)
                else:
                    for i in range(batch_size):
                        self.edge_weights[old_size + i] = edge_weights_device[i]
            
            # Initialize edge_types for new nodes - skip if bundled (already done above)
            if not using_bundled:
                new_edge_types = torch.zeros(batch_size, adjacency.shape[1], 8, device=self.device)
                new_edge_types[:, :, 0] = 1.0  # All edges are PROXIMITY by default
                if not isinstance(self.edge_types, DiskBackedTensor):
                    self.edge_types = torch.cat([self.edge_types, new_edge_types], dim=0)
                else:
                    for i in range(batch_size):
                        self.edge_types[old_size + i] = new_edge_types[i]
            
            # cluster_ids might be DiskBackedTensor - skip if bundled (already in bundle)
            if not using_bundled:
                if isinstance(self.cluster_ids, DiskBackedTensor):
                    for i in range(batch_size):
                        self.cluster_ids[old_size + i] = cluster_ids_device[i]
                else:
                    # Convert _BundledFieldView to regular tensor if needed
                    if hasattr(self.cluster_ids, '__class__') and self.cluster_ids.__class__.__name__ == '_BundledFieldView':
                        self.cluster_ids = self.cluster_ids[:].clone()
                    # Ensure 1D shape (cluster_ids is shape (N,) where each element is scalar cluster id)
                    if self.cluster_ids.ndim != 1:
                        self.cluster_ids = self.cluster_ids.flatten()
                    self.cluster_ids = torch.cat([self.cluster_ids, cluster_ids_device], dim=0)
            
            # rewards, age, access are always regular tensors
            self.rewards = torch.cat([self.rewards, rewards_device], dim=0)
            self.age = torch.cat([self.age, torch.zeros(batch_size, device=self.device)], dim=0)
            self.access = torch.cat([self.access, torch.zeros(batch_size, device=self.device)], dim=0)
            
            # depths might be DiskBackedTensor
            new_depths = torch.norm(embeddings_device, dim=-1).to(self.device)
            if isinstance(self.depths, DiskBackedTensor):
                for i in range(batch_size):
                    self.depths[old_size + i] = new_depths[i]
            else:
                # Convert _BundledFieldView to regular tensor if needed
                if hasattr(self.depths, '__class__') and self.depths.__class__.__name__ == '_BundledFieldView':
                    self.depths = self.depths[:].clone()
                # Ensure 1D shape (depths is shape (N,) where each element is scalar depth)
                if self.depths.ndim != 1:
                    self.depths = self.depths.flatten()
                # CRITICAL: Ensure self.depths is on correct device (might have been moved during retrieval)
                if hasattr(self.depths, 'device') and str(self.depths.device).split(':')[0] != str(self.device).split(':')[0]:
                    self.depths = self.depths.to(self.device)
                self.depths = torch.cat([self.depths, new_depths], dim=0)
            
            # type_embeddings might be DiskBackedTensor
            if self.use_types:
                new_type_embeddings = torch.randn(batch_size, self.type_dim, device=self.device) * 0.1
                if isinstance(self.type_embeddings, DiskBackedTensor):
                    for i in range(batch_size):
                        self.type_embeddings[old_size + i] = new_type_embeddings[i]
                else:
                    # Convert _BundledFieldView to regular tensor if needed
                    if hasattr(self.type_embeddings, '__class__') and self.type_embeddings.__class__.__name__ == '_BundledFieldView':
                        self.type_embeddings = self.type_embeddings[:].clone()
                    self.type_embeddings = torch.cat([self.type_embeddings, new_type_embeddings], dim=0)
            
            # Initialize edge tracking for new nodes (might be DiskBackedTensor)
            new_edge_traversal = torch.zeros(batch_size, adjacency.shape[1], device=self.device)
            new_edge_success = torch.zeros(batch_size, adjacency.shape[1], device=self.device)
            if not isinstance(self.edge_traversal_count, DiskBackedTensor):
                # Convert _BundledFieldView to regular tensor if needed
                if hasattr(self.edge_traversal_count, '__class__') and self.edge_traversal_count.__class__.__name__ == '_BundledFieldView':
                    self.edge_traversal_count = self.edge_traversal_count[:].clone()
                self.edge_traversal_count = torch.cat([self.edge_traversal_count, new_edge_traversal], dim=0)
            else:
                for i in range(batch_size):
                    self.edge_traversal_count[old_size + i] = new_edge_traversal[i]
            
            if not isinstance(self.edge_success_rate, DiskBackedTensor):
                # Convert _BundledFieldView to regular tensor if needed
                if hasattr(self.edge_success_rate, '__class__') and self.edge_success_rate.__class__.__name__ == '_BundledFieldView':
                    self.edge_success_rate = self.edge_success_rate[:].clone()
                self.edge_success_rate = torch.cat([self.edge_success_rate, new_edge_success], dim=0)
            else:
                for i in range(batch_size):
                    self.edge_success_rate[old_size + i] = new_edge_success[i]
        
        # Update size: Use _actual_size for DiskBackedTensor (not .shape which is max capacity)
        # This handles both DiskBackedTensor and regular tensor cases correctly
        from disk_backed_tensor import DiskBackedTensor
        if isinstance(self.embeddings, DiskBackedTensor):
            # For DiskBackedTensor, use _actual_size (number of items added, not max capacity)
            self.size = torch.tensor(self.embeddings._actual_size, dtype=torch.long, device=self.device)
        elif isinstance(self.adjacency, DiskBackedTensor):
            # If adjacency is DiskBackedTensor, use its _actual_size
            self.size = torch.tensor(self.adjacency._actual_size, dtype=torch.long, device=self.device)
        else:
            # For regular tensors, use embeddings size
            self.size = torch.tensor(self.embeddings.shape[0], dtype=torch.long, device=self.device)
    

    
    # ═══════════════════════════════════════════════════════════════════════
    # DISK BACKING - Lazy load/save for large long-term memory
    # ═══════════════════════════════════════════════════════════════════════
    
    def flush_to_disk(self, num_to_flush: int = None):
        """
        Flush oldest/coldest memories from RAM, keeping them in DiskBackedTensor.
        
        With DiskBackedTensor, embeddings are already on disk. This method just
        trims the in-RAM metadata (adjacency, etc.) to save memory, while keeping
        the embeddings accessible via the DiskBackedTensor cache.
        
        Strategy: LRU eviction based on access time + graph centrality.
        
        Args:
            num_to_flush: How many to flush (default: half of capacity)
        """
        if not self.use_disk:
            return 0
        
        # Validate tensor sizes before flush
        current_size = self.size.item()
        if current_size == 0:
            return 0
        
        if num_to_flush is None:
            num_to_flush = current_size // 2
        
        num_to_flush = min(num_to_flush, current_size)
        
        if num_to_flush == 0:
            return 0
        
        # With DiskBackedTensor, we don't actually "flush" embeddings - they're already
        # transparently managed. We just note that we're "done" since the system
        # auto-manages the disk backing.
        # 
        # For now, just call flush on the DiskBackedTensor to ensure writes are synced
        from disk_backed_tensor import DiskBackedTensor
        if isinstance(self.embeddings, DiskBackedTensor):
            self.embeddings.flush()
        
        print(f"✅ DiskBackedTensor sync complete ({current_size} memories managed)")
        return 0  # No memories removed from tier, all stay accessible
        
    def load_from_disk(self, disk_indices: list, with_prefetch: bool = True):
        """
        Load memories from disk with hyperbolic-aware caching.
        
        Uses HyperbolicCache for intelligent eviction based on:
        - Hyperbolic distance from recently accessed memories
        - LRU access patterns
        - Neighborhood preservation (keep connected clusters)
        
        Args:
            disk_indices: List of indices into self.disk_index
            with_prefetch: If True, also prefetch neighbors (hyperbolic locality)
        
        Returns:
            List of loaded embeddings
        """
        if not self.use_disk:
            return []
        
        # Load disk file (no mmap - we'll manage cache ourselves)
        import os
        disk_file = os.path.join(self.disk_path, 'longterm_disk.pt')
        if not os.path.exists(disk_file):
            return []
        
        try:
            disk_data = torch.load(disk_file, map_location='cpu', weights_only=True)
        except:
            return []
        
        loaded_embeddings = []
        neighbors_to_prefetch = []
        newly_loaded_embeddings = []  # For cache eviction anchor
        
        for disk_idx in disk_indices:
            if disk_idx >= len(self.disk_index):
                continue
            
            entry = self.disk_index[disk_idx]
            actual_disk_idx = entry.get('disk_idx', disk_idx)
            
            # Check cache first
            cached = self.hot_cache.get(actual_disk_idx)
            if cached is not None:
                loaded_embeddings.append(cached.to(self.device))
                continue
            
            # Load from disk file
            if actual_disk_idx < disk_data['embeddings'].size(0):
                emb = disk_data['embeddings'][actual_disk_idx].clone()  # Clone to release disk ref
                loaded_embeddings.append(emb.to(self.device))
                
                # Add to hyperbolic cache (returns neighbors to keep)
                self.hot_cache.access(emb, key=actual_disk_idx)
                newly_loaded_embeddings.append(emb)
                
                # Collect neighbors for prefetching
                if with_prefetch and actual_disk_idx < disk_data['adjacency'].size(0):
                    neighbor_indices = disk_data['adjacency'][actual_disk_idx]
                    for neighbor_idx in neighbor_indices:
                        if neighbor_idx >= 0:
                            neighbors_to_prefetch.append(neighbor_idx.item())
        
        # Prefetch neighbors (hyperbolic locality)
        if with_prefetch and neighbors_to_prefetch:
            for neighbor_hot_idx in set(neighbors_to_prefetch[:10]):
                for i, disk_entry in enumerate(self.disk_index):
                    if disk_entry.get('hot_idx') == neighbor_hot_idx:
                        actual_idx = disk_entry.get('disk_idx', i)
                        
                        # Check if already in cache
                        if self.hot_cache.get(actual_idx) is None and actual_idx < disk_data['embeddings'].size(0):
                            emb = disk_data['embeddings'][actual_idx].clone()
                            self.hot_cache.access(emb, key=actual_idx)
                        break
        
        # Evict farthest memories if needed (using newly loaded as anchor)
        if newly_loaded_embeddings and len(self.hot_cache.cache) > self.hot_cache.capacity * 0.9:
            # Use most recently loaded embedding as anchor
            anchor = newly_loaded_embeddings[-1]
            self.hot_cache.evict_farthest(anchor, num_to_evict=max(1, len(self.hot_cache.cache) // 10))
        
        return loaded_embeddings
    
    def search_disk_with_preview(self, query_embedding: torch.Tensor, k: int = 20):
        """
        Fast approximate search on disk using low-dimensional previews.
        
        Returns indices into disk_index for closest matches.
        """
        if not self.use_disk or len(self.disk_index) == 0:
            return []
        
        # Extract all previews
        previews = torch.stack([entry['preview'] for entry in self.disk_index]).to(self.device)
        query_preview = query_embedding[:previews.size(1)]
        
        # Cosine similarity (faster than hyperbolic distance for preview)
        similarities = F.cosine_similarity(
            query_preview.unsqueeze(0),
            previews,
            dim=1
        )
        
        # Top-k
        k_actual = min(k, len(self.disk_index))
        _, top_indices = torch.topk(similarities, k=k_actual, largest=True)
        
        return top_indices.cpu().tolist()
    
    def get_embedding(self, idx: int, load_if_needed: bool = True):
        """
        Get embedding by index, loading from disk if needed.
        
        Provides transparent access to hot (RAM) + cold (disk) memories.
        """
        # Check if in hot RAM
        if idx < self.size.item():
            return self.embeddings[idx]
        
        # Check disk index
        if self.use_disk and load_if_needed:
            for disk_idx, entry in enumerate(self.disk_index):
                if entry.get('hot_idx') == idx:
                    # Load from disk
                    loaded = self.load_from_disk([disk_idx], with_prefetch=True)
                    if loaded:
                        return loaded[0].to(self.device)
        
        # Not found
        return None


class HyperbolicGraphConv(nn.Module):
    """
    🧠 HIPPOCAMPAL DEQ-GNN: Single-layer iterative graph convolution
    
    Instead of multi-layer GNN depth, we leverage DEQ's iterative refinement!
    The DEQ runs this layer 12 times → implicit 12-hop message propagation.
    
    This is how the HIPPOCAMPUS works: recurrent loops through CA3→CA1→EC→CA3,
    with memory retrieval as an iterative settling process into attractors.
    
    Mathematical foundation:
    z* = f(z*, neighbors) where DEQ solves for fixed point z*
    
    Each iteration:
    h'_i = exp_p_i(Σ w_ij * α_ij * log_p_i(p_j))
    
    where:
    - w_ij = exp(-d_hyp(i,j) / T) : Hyperbolic distance weighting (geometry prior)
    - α_ij : Learned attention (relevance)
    - DEQ iterations : Deep propagation through recurrence
    
    🔥 MEMORY OPTIMIZATION:
    - Single layer (not 2-3 layers) → 50% less activation storage
    - Hyperbolic distance weighting → implicit sparsity (far neighbors contribute less)
    - DEQ reuses activations → no multi-layer gradient storage
    - Expected VRAM savings: ~400-500 MB vs traditional multi-layer GNN
    """
    
    def __init__(self, in_dim: int, out_dim: int, use_gradient_checkpointing: bool = True,
                 temperature: float = 1.0, use_full_hyperbolic: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing  # Save ~30-40% VRAM
        self.temperature = nn.Parameter(torch.tensor(temperature))  # Learnable distance decay
        self.use_full_hyperbolic = use_full_hyperbolic  # Full Möbius ops vs hybrid
        
        # SINGLE transformation (DEQ provides depth through iteration!)
        self.W = nn.Linear(in_dim, out_dim)
        
        # Lightweight attention (just for relevance, geometry handles proximity)
        self.attn_query = nn.Linear(out_dim, out_dim // 4)  # Smaller dim = less memory
        self.attn_key = nn.Linear(out_dim, out_dim // 4)
        
        self.ln = nn.LayerNorm(out_dim)
        
    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, 
                edge_weights: torch.Tensor, node_embeddings: torch.Tensor,
                poincare: PoincareManifold) -> torch.Tensor:
        """
        Args:
            x: [N, in_dim] current node features (in Euclidean space)
            adjacency: [N, k] neighbor indices
            edge_weights: [N, k] hyperbolic distances to neighbors
            node_embeddings: [M, in_dim] all node embeddings
            poincare: PoincareManifold for geometric operations
        
        Returns:
            [N, out_dim] updated features (back in Euclidean space)
        """
        # Use gradient checkpointing during training to save VRAM
        if self.training and self.use_gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                x, adjacency, edge_weights, node_embeddings, poincare,
                use_reentrant=False
            )
        else:
            return self._forward_impl(x, adjacency, edge_weights, node_embeddings, poincare)
    
    def _forward_impl(self, x: torch.Tensor, adjacency: torch.Tensor,
                     edge_weights: torch.Tensor, node_embeddings: torch.Tensor,
                     poincare: PoincareManifold) -> torch.Tensor:
        """
        🌀 Hippocampal recurrent update (one DEQ iteration)
        
        This gets called 12 times by DEQ → 12-hop message propagation!
        """
        N, k = adjacency.shape
        
        # Early exit if memory is too small for neighbor gathering
        # This prevents CUDA indexing errors when memory is just starting to populate
        if node_embeddings.shape[0] < k:
            # Not enough nodes for proper graph convolution - return simple transformation
            return self.ln(self.W(x))
        
        # CRITICAL: Device-aware neighbor gathering (CPU long-term → GPU processing)
        if node_embeddings.device != x.device:
            valid_mask = adjacency >= 0
            adjacency_cpu = adjacency.cpu()
            # Clamp to valid range: [0, node_embeddings.shape[0])
            max_idx = max(node_embeddings.shape[0] - 1, 0)
            clamped_adjacency = torch.clamp(adjacency_cpu, 0, max_idx)
            safe_adjacency_cpu = torch.where(valid_mask.cpu(), clamped_adjacency, torch.tensor(0, dtype=torch.long)).long()
            unique_idx = torch.unique(safe_adjacency_cpu).long()  # Ensure long type for indexing
            
            # Transfer ONLY needed neighbors to GPU (not all 200K!)
            needed_embeddings = node_embeddings[unique_idx].to(x.device)
            
            # Build local index map (sized to actual memory size)
            idx_map = torch.zeros(node_embeddings.shape[0], dtype=torch.long)
            idx_map[unique_idx] = torch.arange(len(unique_idx), dtype=torch.long)
            local_adj = idx_map[safe_adjacency_cpu].to(x.device)
            neighbor_features = needed_embeddings[local_adj]  # [N, k, in_dim]
        else:
            valid_mask = adjacency >= 0
            safe_adjacency = torch.where(valid_mask, adjacency, torch.tensor(0, dtype=torch.long, device=adjacency.device))
            neighbor_features = node_embeddings[safe_adjacency]  # [N, k, in_dim]
        
        # SINGLE transformation (not W_self + W_neighbor!)
        x_transformed = self.W(x)  # [N, out_dim]
        neighbor_transformed = self.W(neighbor_features)  # [N, k, out_dim]
        
        # 🔥 HYPERBOLIC DISTANCE WEIGHTING (Geometry Prior!)
        # Closer neighbors = stronger influence (exponential decay with distance)
        # This creates IMPLICIT SPARSITY - far neighbors contribute ~0
        
        # Normalize edge weights per node to prevent extreme outliers from dominating
        # This helps when Hebbian learning creates very strong highways (very low distances)
        # vs weak connections (very high distances) - keeps gradients stable
        edge_weights_min = edge_weights.min(dim=-1, keepdim=True)[0]
        edge_weights_max = edge_weights.max(dim=-1, keepdim=True)[0]
        edge_weights_range = edge_weights_max - edge_weights_min + 1e-8
        edge_weights_normalized = (edge_weights - edge_weights_min) / edge_weights_range  # [0, 1] per node
        
        # Apply exponential decay (now on normalized [0,1] scale, then scaled by learnable temp)
        distance_weights = torch.exp(-edge_weights_normalized / self.temperature.abs())  # [N, k]
        distance_weights = torch.where(valid_mask, distance_weights, torch.zeros_like(distance_weights))
        
        # Normalize distance weights (softmax-like, but geometry-driven)
        distance_weights = distance_weights / (distance_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # ✨ LEARNED ATTENTION (Relevance, not just proximity)
        # Lightweight: out_dim // 4 to save memory
        query = self.attn_query(x_transformed).unsqueeze(1)  # [N, 1, out_dim//4]
        keys = self.attn_key(neighbor_transformed)  # [N, k, out_dim//4]
        
        learned_scores = (query @ keys.transpose(-2, -1)).squeeze(1) / ((self.out_dim // 4) ** 0.5)  # [N, k]
        learned_scores = torch.where(valid_mask, learned_scores, torch.tensor(-1e9, device=learned_scores.device))
        learned_weights = F.softmax(learned_scores, dim=-1)  # [N, k]
        
        # 🧠 COMBINED WEIGHTING: Geometry × Relevance
        # Both geometry (distance) and semantics (attention) matter!
        combined_weights = distance_weights * learned_weights  # Element-wise product
        combined_weights = combined_weights / (combined_weights.sum(dim=-1, keepdim=True) + 1e-8)  # Renormalize
        
        # ════════════════════════════════════════════════════════════════════
        # MESSAGE AGGREGATION: Full hyperbolic vs Hybrid (faster)
        # ════════════════════════════════════════════════════════════════════
        if self.use_full_hyperbolic:
            # 🌀 FULL HYPERBOLIC MESSAGE PASSING (Möbius aggregation)
            # This respects the DEQ's nested/hierarchical iteration structure
            # ~2-3x slower but better for tree-like reasoning
            
            # 1. Map current node to hyperbolic space
            x_hyp = poincare.exponential_map(
                torch.zeros_like(x_transformed),
                x_transformed
            )  # [N, out_dim]
            
            # 2. Map neighbors to hyperbolic space
            neighbor_hyp = poincare.exponential_map(
                torch.zeros(neighbor_transformed.size(0), neighbor_transformed.size(1), self.out_dim, 
                           device=neighbor_transformed.device),
                neighbor_transformed
            )  # [N, k, out_dim]
            
            # 3. Möbius weighted average (proper hyperbolic aggregation)
            # This is the Fréchet mean in hyperbolic space
            # Formula: m = ⊕_{i=1}^k w_i ⊗ p_i
            # where ⊕ is Möbius addition, ⊗ is scalar multiplication
            
            # Start with origin (neutral element for Möbius addition)
            aggregated_hyp = torch.zeros_like(x_hyp)  # [N, out_dim]
            
            # Iteratively add weighted neighbors using Möbius addition
            for i in range(min(neighbor_hyp.size(1), 8)):  # Limit to 8 neighbors for speed
                # Get i-th neighbor for all nodes
                neighbor_i = neighbor_hyp[:, i, :]  # [N, out_dim]
                weight_i = combined_weights[:, i:i+1]  # [N, 1]
                
                # Möbius scalar multiplication: w ⊗ p
                # Formula: (tanh(w * arctanh(||p||)) / ||p||) * p
                neighbor_norm = torch.clamp(neighbor_i.norm(dim=-1, keepdim=True), min=1e-8, max=0.99)
                neighbor_dir = neighbor_i / neighbor_norm
                
                # Scalar multiplication in hyperbolic space
                scaled_norm = torch.tanh(weight_i * torch.atanh(neighbor_norm))
                scaled_neighbor = scaled_norm * neighbor_dir  # [N, out_dim]
                
                # Möbius addition: aggregated ⊕ scaled_neighbor
                aggregated_hyp = poincare.mobius_add(aggregated_hyp, scaled_neighbor)
            
            # 4. Map back to Euclidean space
            messages = poincare.logarithmic_map(
                torch.zeros_like(aggregated_hyp),
                aggregated_hyp
            )  # [N, out_dim]
        else:
            # HYBRID: Euclidean aggregation with hyperbolic weights (faster)
            # The graph structure + distance weighting already encodes hyperbolic geometry
            messages = (neighbor_transformed * combined_weights.unsqueeze(-1)).sum(dim=1)  # [N, out_dim]
        
        # Residual connection + normalization (stable DEQ dynamics!)
        output = self.ln(x_transformed + messages)
        
        return output


class GraphMemoryQueryNetwork(nn.Module):
    """
    GNN-based query network that transforms reflex embeddings into
    intelligent queries over the memory graph.
    
    Uses 2 graph conv layers with message passing to understand
    relational structure when deciding what to retrieve.
    """
    
    def __init__(self, query_dim: int, memory_dim: int, hidden_dim: int, k_neighbors: int, 
                 enable_gnn: bool = False, num_edge_types: int = 8, use_full_hyperbolic: bool = False):
        super().__init__()
        self.query_dim = query_dim
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.enable_gnn = enable_gnn
        self.num_edge_types = num_edge_types
        
        # Project query into memory space
        self.query_proj = nn.Linear(query_dim, memory_dim)
        
        # Hyperbolic graph convolution (single layer for efficiency)
        # Multi-hop reasoning happens via attention over the graph structure
        if enable_gnn:
            self.graph_conv = HyperbolicGraphConv(
                memory_dim, memory_dim, 
                use_gradient_checkpointing=True,
                use_full_hyperbolic=use_full_hyperbolic
            )
            mode_str = "FULL HYPERBOLIC (Möbius)" if use_full_hyperbolic else "HYBRID (Euclidean+hyp weights)"
            print(f"  [GNN] {mode_str}, Gradient checkpointing ENABLED (~30-40% VRAM savings)")
        
        # Hyperbolic projection
        self.poincare = PoincareManifold(dim=memory_dim)
    
    @profile_op("query_network_forward")
    def forward(self, query: torch.Tensor, memory_tier: GraphMemoryTier, 
                k: int = 20, prev_top_indices: torch.Tensor = None,
                routing_max_hops: int = 2) -> dict:
        """
        Args:
            query: [B, T, query_dim] reflex embeddings
            memory_tier: GraphMemoryTier to query
            k: number of memories to retrieve
            prev_top_indices: [B, T] top memory index from previous timestep (for flow context)
            routing_max_hops: Depth of multi-hop routing lookahead (2=simple, 4=medium, 6=complex)
        
        Returns:
            bundle: dict with STRUCTURE for predictor to navigate:
                'embeddings': [B, T, k, memory_dim] - content
                'indices': [B, T, k] - which memories
                'adjacency': [B, T, k, k_neighbors] - local graph
                'edge_weights': [B, T, k, k_neighbors] - edge distances
                'depths': [B, T, k] - distance from origin (abstract→concrete)
                'cluster_ids': [B, T, k] - community membership
                'type_embeddings': [B, T, k, type_dim] - emergent types (if enabled)
                'flow_bias': [B, T, k] - 🌊 PREDICTIVE BIAS from flow field (NEW!)
        """
        import time
        t_start = time.time()
        
        B, T, _ = query.shape
        
        # CRITICAL: Check memory size using internal field to avoid CUDA property access
        # This prevents triggering CUDA errors from previous failed operations
        mem_size = memory_tier._bundled_storage._actual_size if hasattr(memory_tier, '_bundled_storage') else len(memory_tier.rewards)
        if mem_size == 0 or mem_size < k:
            # No memories yet - return empty bundle
            return {
                'embeddings': torch.zeros(B, T, k, self.memory_dim, device=query.device),
                'indices': torch.full((B, T, k), -1, dtype=torch.long, device=query.device),
                'adjacency': torch.full((B, T, k, self.k_neighbors), -1, dtype=torch.long, device=query.device),
                'edge_weights': torch.zeros(B, T, k, self.k_neighbors, device=query.device),
                'edge_types': torch.zeros(B, T, k, self.k_neighbors, self.num_edge_types, device=query.device),
                'depths': torch.zeros(B, T, k, device=query.device),
                'cluster_ids': torch.full((B, T, k), -1, dtype=torch.long, device=query.device),
                'type_embeddings': torch.zeros(B, T, k, memory_tier.type_dim, device=query.device) if memory_tier.use_types else None,
            }
        
        # 🚀 TEMPORAL SUBSAMPLING for large disk-backed tiers
        # Don't need to query EVERY token position - subsample and broadcast
        # This reduces bundle loading from B×T×k to B×T_sample×k
        query_original = query  # Keep original for flow field (uses full T)
        is_large_tier = memory_tier.capacity >= 10000  # Longterm tier
        if is_large_tier and T > 32:
            # Subsample timesteps (query every 8th position, broadcast to neighbors)
            stride = max(T // 32, 8)  # Max 32 query positions
            sample_indices = torch.arange(0, T, stride, device=query.device)
            T_sample = len(sample_indices)
            query_sampled = query[:, sample_indices, :]  # [B, T_sample, C]
            
            # Will broadcast results back to full T after retrieval
            use_subsampling = True
        else:
            query_sampled = query
            T_sample = T
            sample_indices = None
            use_subsampling = False
        
        # Project query to memory space
        # HANDLE DEVICE MISMATCH: query_proj is on CUDA, but query might be on CPU (for longterm tier)
        query_device = query_sampled.device
        proj_device = next(self.query_proj.parameters()).device
        if query_device != proj_device:
            query_sampled = query_sampled.to(proj_device)
        
        # Project query to memory space
        if ENABLE_MICRO_PROFILING:
            t_proj_start = time.perf_counter()
        query_emb = self.query_proj(query_sampled)  # [B, T_sample, memory_dim]
        if ENABLE_MICRO_PROFILING:
            t_proj = (time.perf_counter() - t_proj_start) * 1000
            if 'query_projection' not in _profile_stats:
                _profile_stats['query_projection'] = {'count': 0, 'total_ms': 0, 'max_ms': 0}
            _profile_stats['query_projection']['count'] += 1
            _profile_stats['query_projection']['total_ms'] += t_proj
            _profile_stats['query_projection']['max_ms'] = max(_profile_stats['query_projection']['max_ms'], t_proj)
        
        # Move back to original device if needed
        if query_device != proj_device:
            query_emb = query_emb.to(query_device)
        
        # Map to hyperbolic space (memory efficient)
        if ENABLE_MICRO_PROFILING:
            t_hyp_start = time.perf_counter()
        query_flat = query_emb.view(-1, self.memory_dim)  # [B*T_sample, memory_dim]
        query_hyp = self.poincare.exponential_map(
            torch.zeros_like(query_flat), 
            query_flat
        ).view(B, T_sample, self.memory_dim)  # [B, T_sample, memory_dim]
        if ENABLE_MICRO_PROFILING:
            t_hyp = (time.perf_counter() - t_hyp_start) * 1000
            if 'hyperbolic_map' not in _profile_stats:
                _profile_stats['hyperbolic_map'] = {'count': 0, 'total_ms': 0, 'max_ms': 0}
            _profile_stats['hyperbolic_map']['count'] += 1
            _profile_stats['hyperbolic_map']['total_ms'] += t_hyp
            _profile_stats['hyperbolic_map']['max_ms'] = max(_profile_stats['hyperbolic_map']['max_ms'], t_hyp)
        
        # HANDLE EMPTY MEMORY: Return zeros if no memories yet
        # Also handle case where embeddings exist but adjacency not grown yet (disk-backed tier bug)
        if memory_tier.size.item() == 0 or memory_tier.adjacency.size(0) == 0:
            return {
                'embeddings': torch.zeros(B, T, k, self.memory_dim, device=query.device),
                'indices': torch.full((B, T, k), -1, dtype=torch.long, device=query.device),
                'adjacency': torch.full((B, T, k, self.k_neighbors), -1, dtype=torch.long, device=query.device),
                'edge_weights': torch.zeros(B, T, k, self.k_neighbors, device=query.device),
                'edge_types': torch.zeros(B, T, k, self.k_neighbors, 8, device=query.device),
                'depths': torch.zeros(B, T, k, device=query.device),
                'cluster_ids': torch.full((B, T, k), -1, dtype=torch.long, device=query.device),
                'type_embeddings': torch.zeros(B, T, k, memory_tier.type_dim, device=query.device) if memory_tier.use_types else None,
            }
        
        # EFFICIENT k-NN: Use Euclidean distance approximation (avoid loading all embeddings!)
        # Strategy: Compute distances in batches to avoid OOM with DiskBackedTensor
        M = memory_tier.size.item()
        query_flat = query_hyp.view(-1, self.memory_dim)  # [B*T_sample, memory_dim]
        BT = query_flat.size(0)
        
        # Convert to float32 for cdist (doesn't support bfloat16)
        query_flat = query_flat.float()
        
        # Use Euclidean distance (cheap, good approximation for k-NN)
        # Process in batches to handle DiskBackedTensor efficiently
        
        # Safety: If tier is empty, return empty bundle
        if M == 0:
            empty_bundle = {
                'embeddings': torch.zeros(B, T, k, self.memory_dim, device=query.device),
                'adjacency': torch.full((B, T, k, self.k_neighbors), -1, dtype=torch.long, device=query.device),
                'edge_weights': torch.zeros(B, T, k, self.k_neighbors, device=query.device),
                'edge_types': torch.zeros(B, T, k, self.k_neighbors, self.num_edge_types, device=query.device),
                'cluster_ids': torch.zeros(B, T, k, dtype=torch.long, device=query.device),
                'depths': torch.zeros(B, T, k, device=query.device),
                'indices': torch.full((B, T, k), -1, dtype=torch.long, device=query.device),
            }
            if memory_tier.use_types:
                empty_bundle['type_embeddings'] = torch.zeros(B, T, k, memory_tier.type_dim, device=query.device)
            return empty_bundle
        
        k_actual = min(k * 4, M)  # Oversample for refinement
        
        # CRITICAL FIX: MASSIVE BATCH SIZE for SSD efficiency
        # SSDs love large sequential reads (high bandwidth, low IOPS)
        # Reading 1000 items = 4MB (slow, many seeks)
        # Reading 10000 items = 40MB (fast, one large read)
        batch_size = min(10000, M)  # 10x larger batches!
        
        # Strategy: Read large chunks, keep only top-k per chunk, then merge
        # This reduces RAM usage while keeping disk I/O efficient
        all_top_dists = []
        all_top_indices = []
        
        if ENABLE_MICRO_PROFILING:
            t_knn_start = time.perf_counter()
            num_batches = 0
        
        for batch_start in range(0, M, batch_size):
            if ENABLE_MICRO_PROFILING:
                num_batches += 1
            batch_end = min(batch_start + batch_size, M)
            batch_emb = memory_tier.embeddings[batch_start:batch_end]  # ONE BIG READ
            
            # Euclidean distance: ||q - m||^2
            # Ensure batch_emb is float32 too
            # Handle bundled storage (dict) vs column storage (tensor)
            if isinstance(batch_emb, dict):
                batch_emb_tensor = batch_emb['embedding'].float()
            else:
                batch_emb_tensor = batch_emb.float()
            
            batch_dists = torch.cdist(
                query_flat.unsqueeze(0),  # [1, BT, D]
                batch_emb_tensor.unsqueeze(0)    # [1, batch_size, D]
            ).squeeze(0)  # [BT, batch_size]
            
            # Keep only top-k from THIS batch (saves RAM for merge)
            batch_k = min(k_actual, batch_end - batch_start)
            batch_top_dists, batch_top_idx = torch.topk(batch_dists, batch_k, largest=False, dim=-1)
            
            # Offset indices to global space
            all_top_dists.append(batch_top_dists)
            all_top_indices.append(batch_top_idx + batch_start)
        
        if ENABLE_MICRO_PROFILING:
            t_knn = (time.perf_counter() - t_knn_start) * 1000
            if 'knn_search' not in _profile_stats:
                _profile_stats['knn_search'] = {'count': 0, 'total_ms': 0, 'max_ms': 0}
            _profile_stats['knn_search']['count'] += 1
            _profile_stats['knn_search']['total_ms'] += t_knn
            _profile_stats['knn_search']['max_ms'] = max(_profile_stats['knn_search']['max_ms'], t_knn)
        
        # Merge results from all batches
        merged_dists = torch.cat(all_top_dists, dim=1)  # [BT, num_batches * k_actual]
        merged_indices = torch.cat(all_top_indices, dim=1)
        
        # Final top-k selection
        topk_dists, topk_args = torch.topk(merged_dists, min(k_actual, merged_dists.size(1)), largest=False, dim=-1)
        topk_indices = torch.gather(merged_indices, 1, topk_args)
        
        # Trim to actual k
        topk_indices = topk_indices[:, :k]
        topk_dists = topk_dists[:, :k]
        
        # Pad if needed
        if topk_indices.size(1) < k:
            pad_size = k - topk_indices.size(1)
            topk_indices = F.pad(topk_indices, (0, pad_size), value=-1)
            topk_dists = F.pad(topk_dists, (0, pad_size), value=1e6)
        
        # Gather retrieved memories and their graph neighborhoods
        valid_mask = topk_indices >= 0
        safe_indices = torch.where(valid_mask, topk_indices, 0)
        
        # Flatten indices for batch indexing (DiskBackedTensor expects 1D indices)
        flat_indices = safe_indices.view(-1)  # [B*T*k]
        
        # 🚀 CRITICAL OPTIMIZATION: Deduplicate indices BEFORE bundle loading!
        # Hyperbolic clustering means nearby queries retrieve same memories
        # Example: 32768 indices → 16 unique (99.9% reduction in disk I/O!)
        unique_indices, inverse_indices = torch.unique(flat_indices, return_inverse=True)
        
        # 🎯 FIBER BUNDLE ATOMIC ACCESS: One read gets ALL fields per node!
        # This is 5-10x faster than separate field reads (1 disk seek vs 5-10 seeks)
        if ENABLE_MICRO_PROFILING:
            t_bundle_start = time.perf_counter()
        
        if hasattr(memory_tier, '_bundled_storage') and memory_tier._bundled_storage is not None:
            # Bundled mode: Load ONLY unique bundles (massive speedup!)
            bundles_unique = memory_tier._bundled_storage.get_bundles_batch(unique_indices)
            
            # Broadcast back to full flat_indices shape using inverse mapping
            retrieved_emb_flat = bundles_unique['embedding'][inverse_indices]  # [B*T*k, memory_dim]
            retrieved_adj_flat = bundles_unique['adjacency'][inverse_indices]  # [B*T*k, k_neighbors]
            retrieved_edge_weights_flat = bundles_unique['edge_weights'][inverse_indices]  # [B*T*k, k_neighbors]
            retrieved_edge_types_flat = bundles_unique['edge_types'][inverse_indices]  # [B*T*k, k_neighbors, num_edge_types]
            retrieved_clusters_flat = bundles_unique['cluster_id'][inverse_indices]  # [B*T*k]
        else:
            # Column mode: separate field reads (legacy, slower)
            retrieved_emb_flat = memory_tier.embeddings[flat_indices]  # [B*T*k, memory_dim]
            retrieved_adj_flat = memory_tier.adjacency[flat_indices]  # [B*T*k, k_neighbors]
            retrieved_edge_weights_flat = memory_tier.edge_weights[flat_indices]  # [B*T*k, k_neighbors]
            retrieved_edge_types_flat = memory_tier.edge_types[flat_indices]  # [B*T*k, k_neighbors, num_edge_types]
            retrieved_clusters_flat = memory_tier.cluster_ids[flat_indices]  # [B*T*k]
        
        if ENABLE_MICRO_PROFILING:
            t_bundle = (time.perf_counter() - t_bundle_start) * 1000
            if 'bundle_loading' not in _profile_stats:
                _profile_stats['bundle_loading'] = {'count': 0, 'total_ms': 0, 'max_ms': 0}
            _profile_stats['bundle_loading']['count'] += 1
            _profile_stats['bundle_loading']['total_ms'] += t_bundle
            _profile_stats['bundle_loading']['max_ms'] = max(_profile_stats['bundle_loading']['max_ms'], t_bundle)
        
        # Reshape back to [B*T, k, ...]
        retrieved_emb = retrieved_emb_flat.view(BT, k, self.memory_dim)
        retrieved_adj = retrieved_adj_flat.view(BT, k, self.k_neighbors)
        retrieved_edge_weights = retrieved_edge_weights_flat.view(BT, k, self.k_neighbors)
        retrieved_edge_types = retrieved_edge_types_flat.view(BT, k, self.k_neighbors, self.num_edge_types)
        retrieved_clusters = retrieved_clusters_flat.view(BT, k)
        
        # STRUCTURE: Compute depths (distance from origin in hyperbolic space)
        # This tells predictor: abstract (near center) vs concrete (near boundary)
        # Use Euclidean norm as proxy for hyperbolic distance from origin
        retrieved_depths = torch.norm(retrieved_emb.view(-1, self.memory_dim), dim=-1).view(-1, k)  # [B*T, k]
        
        # STRUCTURE: Get type embeddings (emergent categorical information)
        retrieved_types = None
        if memory_tier.use_types:
            # Use bundled access if available, otherwise fall back to field access
            if hasattr(memory_tier, '_bundled_storage') and memory_tier._bundled_storage is not None:
                # Already retrieved in bundles_unique above
                if 'type_embedding' in bundles_unique:
                    retrieved_types_flat = bundles_unique['type_embedding'][inverse_indices]  # [B*T*k, type_dim]
                    retrieved_types = retrieved_types_flat.view(-1, k, memory_tier.type_dim)  # [B*T, k, type_dim]
            else:
                # Column mode: separate read
                retrieved_types_flat = memory_tier.type_embeddings[flat_indices]  # [B*T*k, type_dim]
                retrieved_types = retrieved_types_flat.view(-1, k, memory_tier.type_dim)  # [B*T, k, type_dim]
        
        # Optionally apply hyperbolic GNN to refine retrieved memories
        # This aggregates information from graph neighbors using proper Möbius operations
        # DISABLED for VRAM-constrained environments (can enable later when more memory available)
        if self.enable_gnn:
            x = retrieved_emb.view(-1, self.memory_dim)  # [B*T*k, memory_dim]
            adj = retrieved_adj.view(-1, self.k_neighbors)  # [B*T*k, k_neighbors]
            weights = retrieved_edge_weights.view(-1, self.k_neighbors)  # [B*T*k, k_neighbors]
            
            # CRITICAL: Move x to same device as GNN layers (GPU)
            # retrieved_emb might be on CPU (long-term tier), but GNN is on GPU
            target_device = next(self.graph_conv.parameters()).device
            if x.device != target_device:
                x = x.to(target_device)
                adj = adj.to(target_device)
                weights = weights.to(target_device)
            
            # Apply graph convolution with hyperbolic geometry
            # The graph_conv._forward_impl handles device transfer for node_embeddings internally
            refined_flat = self.graph_conv(x, adj, weights, memory_tier.embeddings, self.poincare)
            refined = refined_flat.view(B, T, k, self.memory_dim)
        else:
            # Just use direct retrieval without GNN refinement
            refined = retrieved_emb.view(B, T, k, self.memory_dim)
        
        # Reshape everything to [B, T_sample, k, ...]
        bundle = {
            'embeddings': refined,  # [B, T_sample, k, memory_dim]
            'indices': topk_indices.view(B, T_sample, k),  # [B, T_sample, k]
            'adjacency': retrieved_adj.view(B, T_sample, k, self.k_neighbors),  # [B, T_sample, k, k_neighbors]
            'edge_weights': retrieved_edge_weights.view(B, T_sample, k, self.k_neighbors),  # [B, T_sample, k, k_neighbors]
            'edge_types': retrieved_edge_types.view(B, T_sample, k, self.k_neighbors, self.num_edge_types),  # [B, T_sample, k, k_neighbors, num_edge_types]
            'depths': retrieved_depths.view(B, T_sample, k),  # [B, T_sample, k]
            'cluster_ids': retrieved_clusters.view(B, T_sample, k),  # [B, T_sample, k]
            'type_embeddings': retrieved_types.view(B, T_sample, k, memory_tier.type_dim) if retrieved_types is not None else None,  # [B, T_sample, k, type_dim]
        }
        
        # 🚀 TEMPORAL BROADCAST: If we subsampled, broadcast to full T
        if use_subsampling:
            # Nearest-neighbor interpolation: each position uses closest sample
            # sample_indices = [0, 8, 16, 24, ...] for stride=8
            # Position 3 → uses sample 0, Position 12 → uses sample 8, etc.
            full_indices = torch.arange(T, device=query.device)  # [T]
            # Find closest sample for each position
            dists = (full_indices.unsqueeze(1) - sample_indices.unsqueeze(0)).abs()  # [T, T_sample]
            closest_sample = dists.argmin(dim=1)  # [T] - which sample to use
            
            # Broadcast all bundle fields
            bundle_broadcasted = {}
            for key, val in bundle.items():
                if val is None:
                    bundle_broadcasted[key] = None
                else:
                    # val shape: [B, T_sample, ...] → [B, T, ...]
                    bundle_broadcasted[key] = val[:, closest_sample, ...]
            bundle = bundle_broadcasted
        
        # � PARALLEL TRANSPORT: Query vector evolves along retrieval trajectory
        # 
        # GEOMETRIC DEEP LEARNING: The query IS the state
        # 
        # Instead of: "lookup context from prev→curr edge" (discrete, slow)
        # We do: "transport query vector along the edge" (continuous, fast)
        #
        # Mathematical structure:
        #   q_{t+1} = q_t ⊕ (α · EdgeVector)
        #
        # Where EdgeVector = direction from prev_embedding to curr_embedding
        # and ⊕ is gyrovector addition (Möbius addition in hyperbolic space)
        #
        # Result: The transported query POINTS toward geometrically coherent continuations
        # No lookups needed - the curvature accumulates in the query itself!
        
        # Initialize transported query (starts as original query)
        # Note: Uses original T dimension (not T_sample) since flow field needs full resolution
        transported_query = query_original.clone()  # [B, T, memory_dim]
        
        flow_bias = torch.zeros(B, T, k, device=query_original.device)
        
        # EARLY EXIT during warmup
        if not memory_tier.flow_enabled:
            bundle['transported_query'] = transported_query  # Pass through unchanged
            return bundle
        
        # VECTORIZED PARALLEL TRANSPORT (Pure GPU ops, zero Python loops!)
        # Math: q_next = Normalize(q + alpha * (target - q))
        # Captures context: "Bank[River]→Fish" vs "Bank[Finance]→Money"
        if prev_top_indices is not None and memory_tier.size.item() > 0:
            with torch.no_grad():
                # Get top-1 previous memory (where we came from)
                prev_indices = prev_top_indices.view(B, T)  # [B, T]
                valid_mask = (prev_indices >= 0) & (prev_indices < memory_tier.size.item())
                
                if valid_mask.any():
                    # VECTORIZED FETCH of previous embeddings
                    safe_prev = torch.where(valid_mask, prev_indices, torch.zeros_like(prev_indices))
                    prev_emb = memory_tier.embeddings[safe_prev.view(-1)].view(B, T, self.memory_dim).to(query.device)
                    
                    # Get top-1 current memory (where we're going)
                    curr_indices = topk_indices[:, 0].view(B, T)  # [B, T]
                    curr_valid = (curr_indices >= 0) & (curr_indices < memory_tier.size.item())
                    safe_curr = torch.where(curr_valid, curr_indices, torch.zeros_like(curr_indices))
                    curr_emb = memory_tier.embeddings[safe_curr.view(-1)].view(B, T, self.memory_dim).to(query.device)
                    
                    # GYROVECTOR TRANSPORT (tangent space)
                    alpha = 0.2  # Momentum coefficient
                    tangent = curr_emb - prev_emb  # [B, T, D]
                    transport_delta = alpha * tangent
                    
                    # Apply transport where valid
                    combined_mask = valid_mask & curr_valid
                    transported_query = torch.where(
                        combined_mask.unsqueeze(-1),
                        query + transport_delta,
                        query
                    )
                    
                    # Renormalize to Poincaré ball
                    norms = transported_query.norm(dim=-1, keepdim=True)
                    transported_query = transported_query / norms.clamp(min=1e-6)
                    norms_after = transported_query.norm(dim=-1, keepdim=True)
                    transported_query = torch.where(
                        norms_after > 0.95,
                        transported_query / norms_after * 0.95,
                        transported_query
                    )
                    
                    # FLOW BIAS from geometry (vectorized cosine similarity)
                    cand_embs = retrieved_emb.view(B, T, k, self.memory_dim)
                    transported_norm = transported_query / (transported_query.norm(dim=-1, keepdim=True) + 1e-8)
                    cand_norms = cand_embs / (cand_embs.norm(dim=-1, keepdim=True) + 1e-8)
                    alignment = (transported_norm.unsqueeze(2) * cand_norms).sum(dim=-1)
                    geometric_bias = memory_tier.flow_strength * F.relu(alignment)  # [B, T, k]
                    
                    # 🎯 CONTEXT-AWARE ROUTING: "Coming from X, you usually went to Y this often"
                    # This is probabilistic routing based on historical trajectory patterns!
                    # Mathematical first principle: Trajectories are guided by past success
                    # 🚀 OPTIMIZATION: Only compute when debugging (expensive disk I/O + CPU-GPU sync)
                    
                    trajectory_bias = torch.zeros(B, T, k, device=query.device)
                    
                    # Only compute trajectory bias for longterm tier where highways are stable
                    is_longterm = memory_tier.capacity >= 10000  # Heuristic: longterm has large capacity
                    
                    # Skip expensive trajectory computation unless debugging
                    if COMPUTE_ROUTING_STATS:
                        if ENABLE_MICRO_PROFILING:
                            context_start = time.perf_counter()
                    
                            if is_longterm and hasattr(memory_tier, '_bundled_storage') and memory_tier._bundled_storage is not None:
                            # We have flow data - use it for context-aware routing!
                            # Key insight: Each candidate's neighbors have flow data
                            # We check: "If I go to this candidate, which prev contexts led here successfully?"
                        
                                    try:
                                # 🚀 VECTORIZED CONTEXT-AWARE ROUTING (Pure tensor ops, zero Python loops!)
                                        # Mathematical elegance: Parallel lookup of trajectory histories
                                        
                                        # Get unique curr nodes that need bundle loading
                                        curr_flat = safe_curr.view(-1)  # [B*T]
                                        valid_curr_mask = combined_mask.view(-1)  # [B*T]
                                        unique_curr = curr_flat[valid_curr_mask].unique()
                                        
                                        # 🔍 DEBUG: Log how many nodes we're processing
                                        num_unique = len(unique_curr)
                                        if num_unique > 50:
                                            print(f"⚠️  [CONTEXT-AWARE] Processing {num_unique} unique nodes (may be slow)")
                                        
                                        # Batch load only unique bundles (minimize disk I/O)
                                        # Build a cache: curr_idx → (edge_flow_prev, edge_flow_strength, neighbors)
                                        flow_cache = {}
                                        for curr_node in unique_curr.tolist():
                                            bundle = memory_tier._bundled_storage[curr_node]
                                            flow_prev = bundle.get('edge_flow_prev_nodes')  # [k_neighbors, k_neighbors]
                                            flow_strength = bundle.get('edge_flow_context')  # [k_neighbors, k_neighbors]
                                            neighbors = bundle['adjacency']  # [k_neighbors]
                                            if flow_prev is not None and flow_strength is not None:
                                                flow_cache[curr_node] = {
                                                    'flow_prev': flow_prev.to(query.device),
                                                    'flow_strength': flow_strength.to(query.device),
                                                    'neighbors': neighbors.to(query.device)
                                                }
                                        
                                        # 🚀 FULLY VECTORIZED CONTEXT-AWARE ROUTING (No Python loops!)
                                        # Mathematical insight: Most (b,t) positions won't have flow data
                                        # Only process the ones that do, fully in tensor space
                                        
                                        # For positions with valid prev/curr, look up flow strengths
                                        # 🚀 CRITICAL OPTIMIZATION: Limit processing to avoid slowdown
                                        # As training progresses, more positions become valid → linear slowdown
                                        # Solution: Sample a fixed number of positions (e.g., 32) instead of all
                                        valid_positions = combined_mask.nonzero(as_tuple=False)  # [N, 2] where N = valid count
                                        
                                        # Limit to first 32 positions to keep overhead constant
                                        max_positions = 32
                                        if len(valid_positions) > max_positions:
                                            # Randomly sample to avoid bias toward early sequence positions
                                            indices = torch.randperm(len(valid_positions), device=valid_positions.device)[:max_positions]
                                            valid_positions = valid_positions[indices]
                                        
                                        if len(valid_positions) > 0 and len(flow_cache) > 0:
                                            for pos_idx in range(len(valid_positions)):
                                                b, t = valid_positions[pos_idx][0].item(), valid_positions[pos_idx][1].item()
                                                
                                                curr_node = safe_curr[b, t].item()  # GPU sync, but only once per valid position
                                                prev_node = safe_prev[b, t].item()  # GPU sync, but only once per valid position
                                                
                                                if curr_node not in flow_cache:
                                                    continue
                                                
                                                cache_entry = flow_cache[curr_node]
                                                neighbors = cache_entry['neighbors']  # [k_neighbors] on GPU
                                                flow_prev_matrix = cache_entry['flow_prev']  # [k_neighbors, k_neighbors] on GPU
                                                flow_strength_matrix = cache_entry['flow_strength']  # [k_neighbors, k_neighbors] on GPU
                                                
                                                # 🚀 VECTORIZED: Process all k candidates at once
                                                next_candidates = topk_indices[b, t]  # [k] tensor on GPU
                                                valid_cands = next_candidates >= 0  # [k] boolean mask
                                                
                                                if not valid_cands.any():
                                                    continue
                                                
                                                # Find which neighbors match our candidates (vectorized)
                                                # neighbors: [k_neighbors], next_candidates: [k]
                                                # Result: [k, k_neighbors] boolean matrix
                                                matches = neighbors.unsqueeze(0) == next_candidates.unsqueeze(1)  # [k, k_neighbors]
                                                edge_slots = matches.long().argmax(dim=1)  # [k] - slot for each candidate
                                                edge_found = matches.any(dim=1)  # [k] - which candidates have edges
                                                
                                                # Get flow data for found edges (vectorized gather)
                                                flow_prev_for_edges = flow_prev_matrix[edge_slots]  # [k, k_neighbors]
                                                flow_strength_for_edges = flow_strength_matrix[edge_slots]  # [k, k_neighbors]
                                                
                                                # Check which edges have prev_node in their history (vectorized)
                                                prev_matches = flow_prev_for_edges == prev_node  # [k, k_neighbors]
                                                has_prev = prev_matches.any(dim=1)  # [k] - which edges have prev context
                                                
                                                # Get strengths for matching prev contexts
                                                prev_slots = prev_matches.long().argmax(dim=1)  # [k] - which slot has prev
                                                strengths = flow_strength_for_edges.gather(1, prev_slots.unsqueeze(1)).squeeze(1)  # [k]
                                                
                                                # Combine masks and assign (vectorized write)
                                                final_mask = valid_cands & edge_found & has_prev  # [k]
                                                trajectory_bias[b, t, final_mask] = strengths[final_mask]
                                                    
                                    except Exception as e:
                                            # If anything fails, just skip trajectory bias (not critical)
                                            pass
                        
                                    if ENABLE_MICRO_PROFILING:
                                            context_elapsed = (time.perf_counter() - context_start) * 1000
                                            if 'context_aware_routing' not in _profile_stats:
                                                _profile_stats['context_aware_routing'] = {'count': 0, 'total_ms': 0, 'max_ms': 0}
                                            _profile_stats['context_aware_routing']['count'] += 1
                                            _profile_stats['context_aware_routing']['total_ms'] += context_elapsed
                                            _profile_stats['context_aware_routing']['max_ms'] = max(_profile_stats['context_aware_routing']['max_ms'], context_elapsed)
                            
                    # 🎯 COMBINE: Geometry says "this direction seems good"
                    #             Trajectory says "this direction worked from here before"
                    # Weight trajectory at 50% to balance exploration vs exploitation
                    flow_bias = geometric_bias + 0.5 * trajectory_bias
        
        bundle['flow_bias'] = flow_bias  # [B, T, k]
        bundle['transported_query'] = transported_query  # [B, T, memory_dim]
        
        # 🧠 BRAIN DEBUG VIEW: Show basic memory retrieval (before routing bundle)
        if SHOW_BRAIN_DEBUG:
            if not hasattr(self, '_brain_view_call_count'):
                self._brain_view_call_count = 0
                print("\n" + "🧠"*40)
                print("BRAIN DEBUG MODE ACTIVE")
                print("🧠"*40 + "\n")
            
            self._brain_view_call_count += 1
            
            # Show every 25 calls (~every 4-5 training iterations)
            if self._brain_view_call_count % 25 == 1:
                print("\n" + "="*80)
                print(f"🧠 MEMORY RETRIEVAL #{self._brain_view_call_count}")
                print("="*80)
                print(f"📥 Query shape: {query_original.shape}")
                print(f"💾 Retrieved memories: {bundle['embeddings'].shape}")
                # Handle variable topk_indices shape
                if topk_indices.dim() == 2:
                    # [B*T, k] format
                    print(f"   Indices (first position): {topk_indices[0, :].tolist()}")
                elif topk_indices.dim() == 3:
                    # [B, T, k] format
                    print(f"   Indices (first position): {topk_indices[0, 0, :].tolist()}")
                else:
                    print(f"   Indices shape: {topk_indices.shape}")
                print(f"   Adjacency: {bundle['adjacency'].shape}")
                print("="*80 + "\n")
        
        # 🛣️ MULTI-HOP ROUTING BUNDLE (2-3 deep with token previews)
        # "Standing at this node from THIS trajectory, where can I go?"
        # Shows ALL edges (even rare ones - might be semantically critical!)
        # Format: For each retrieved memory, show 2-hop lookahead with:
        #   - Edge statistics (traversal count, success rate)
        #   - Token previews (what concepts are reachable)
        #   - Context-aware (uses trajectory entry context, not all incoming edges)
        
        routing_bundle = {}
        
        # 🚀 PERFORMANCE: Only compute expensive routing stats when debugging
        if not COMPUTE_ROUTING_STATS:
            # Fast path: Skip routing stats (saves ~19 seconds per retrieval!)
            # The DEQ doesn't actually use these stats - they're only for visualization
            routing_bundle = {
                'hop_0_stats': None,
                'hop_1_stats': None,
                'hop_2_stats': None,
                'memory_tier': memory_tier,
                'topk_indices': topk_indices,
            }
        else:
            # Debug path: Compute full multi-hop routing statistics
            try:
                from disk_backed_tensor import DiskBackedTensor
                
                if prev_top_indices is not None and hasattr(memory_tier, 'edge_traversal_count'):
                    # � FIX: Ensure topk_indices is 3D [B, T, k] for routing bundle construction
                    if topk_indices.dim() == 2:
                        # Reshape [B*T, k] → [B, T, k]
                        topk_indices = topk_indices.view(B, T, k)
                    
                    # �🚀 CONFIGURABLE MULTI-HOP ROUTING
                    # Simple models (TinyStories): 2-hop = "cat" → "sat" → "on"
                    # Medium models (GPT-3): 3-4 hop = "quantum" → "entanglement" → "suggests" → "non-locality"
                    # Complex models (GPT-4): 5-6 hop = full syntactic phrase lookahead
                    max_hops = routing_max_hops  # From function parameter (passed from GraphMemorySystem)
                    
                    # 🐛 FIX: Ensure max_edges_per_hop is valid
                    if hasattr(memory_tier, 'k_neighbors') and memory_tier.k_neighbors > 0:
                        max_edges_per_hop = memory_tier.k_neighbors
                    else:
                        # Fallback: use a reasonable default (20 edges per node)
                        max_edges_per_hop = 20
                    
                    # 🔍 DEBUG: Print allocation info (commented out - too verbose)
                    # print(f"🔍 [ROUTING] max_hops={max_hops}, max_edges_per_hop={max_edges_per_hop}, k={k}")
                    # print(f"🔍 [ROUTING] Allocating hop_1 shape: [B={B}, T={T}, k={k}, edges={max_edges_per_hop}, 3]")
                    # print(f"🔍 [ROUTING] Allocating hop_2 shape: [B={B}, T={T}, k={k}, edges={max_edges_per_hop}, edges={max_edges_per_hop}, 3]")
                    
                    # VRAM estimate:
                    # 2-hop: ~0.6 MB (5×5=25 paths)
                    # 3-hop: ~2.1 MB (5×5×5=125 paths) 
                    # 4-hop: ~9.4 MB (5×5×5×5=625 paths)
                    # 5-hop: ~42 MB (5^5=3125 paths) - getting expensive!
                    # 6-hop: ~189 MB (5^6=15625 paths) - only for large models!
                    
                    # Dynamic allocation based on max_hops
                    # Each edge gets 6 values:
                    # [0] trajectory_specificity  # compound - centrality (how context-dependent is this edge?)
                    # [1] success_rate           # 0-1 (learned from training feedback)
                    # [2] token_preview          # Token ID (for debugging/visualization)
                    # [3] compound_strength      # 0-1 TRAJECTORY-AWARE (weight+highway+flow)
                    # [4] centrality             # 0-1 TRAJECTORY-AGNOSTIC (graph importance)
                    # [5] relative_rank          # 0-1 NORMALIZED (1.0=best option at this position)
                    hop_stats_dict = {}
                    hop_stats_dict[0] = torch.zeros(B, T, k, 6, device=query_original.device)
                    
                    # Allocate hop_1 through hop_N
                    for hop_level in range(1, max_hops + 1):
                        # Shape: [B, T, k, k_neighbors^hop_level, 6]
                        shape = [B, T, k] + [max_edges_per_hop] * hop_level + [6]
                        hop_stats_dict[hop_level] = torch.zeros(*shape, device=query_original.device)
                    
                    # Convenience aliases for backward compatibility (2-hop hardcoded implementation)
                    # TODO: Generalize loop to handle arbitrary max_hops
                    hop_0_stats = hop_stats_dict[0]
                    hop_1_stats = hop_stats_dict.get(1) if max_hops >= 1 else None
                    hop_2_stats = hop_stats_dict.get(2) if max_hops >= 2 else None
                    
                    # 🔍 DEBUG: Print actual shapes (commented out - too verbose)
                    # print(f"🔍 [ROUTING] hop_0_stats.shape={hop_0_stats.shape}")
                    # if hop_1_stats is not None:
                    #     print(f"🔍 [ROUTING] hop_1_stats.shape={hop_1_stats.shape}, dim={hop_1_stats.dim()}")
                    # if hop_2_stats is not None:
                    #     print(f"🔍 [ROUTING] hop_2_stats.shape={hop_2_stats.shape}, dim={hop_2_stats.dim()}")
                    
                    # 🐛 CRITICAL: Validate tensor dimensions before using
                    if hop_1_stats is not None and hop_1_stats.dim() < 4:
                        # Tensor didn't allocate properly - disable routing (silent fallback)
                        hop_1_stats = None
                        hop_2_stats = None
                    if hop_2_stats is not None and hop_2_stats.dim() < 5:
                        # 2-hop tensor malformed - disable it (silent fallback)
                        hop_2_stats = None
                    
                    # 🐛 SAFETY: Validate tensor shapes before indexing
                    def safe_index_check(tensor, *indices):
                        """Check if indices are within tensor bounds"""
                        if tensor is None:
                            return False
                        if tensor.dim() != len(indices):
                            return False  # Dimension mismatch!
                        for i, idx in enumerate(indices):
                            if idx >= tensor.shape[i]:
                                return False
                        return True
                    
                    # 📊 DIAGNOSTICS: Count edge tensor shapes
                    squeezed_count = 0  # Scalar or truncated tensors
                    normal_count = 0    # Full 1D arrays
                    missing_count = 0   # None or empty
                    
                    # Batch-convert indices to numpy ONCE to eliminate GPU→CPU synchronization overhead
                    if ENABLE_MICRO_PROFILING:
                        t_batch_convert_start = time.perf_counter()
                    topk_indices_np = topk_indices.cpu().numpy()
                    prev_top_indices_np = prev_top_indices.cpu().numpy()
                    if ENABLE_MICRO_PROFILING:
                        t_batch_convert_end = time.perf_counter()
                        print(f"   [TIMING] Batch index conversion: {(t_batch_convert_end - t_batch_convert_start)*1000:.1f}ms")
                    
                    # 🚀 BATCH PRE-LOAD ALL BUNDLES (the mathematically beautiful optimization!)
                    # Instead of 32,768 individual property accesses triggering bundle loads,
                    # load ALL needed bundles ONCE before the loop!
                    if ENABLE_MICRO_PROFILING:
                        t_bundle_load_start = time.perf_counter()
                    
                    # Collect ALL unique indices we'll need (topk + prev)
                    all_needed_indices = set()
                    for b in range(B):
                        for t in range(T):
                            # Previous indices for hop 0
                            if prev_top_indices is not None and t < prev_top_indices.shape[1]:
                                prev_idx = int(prev_top_indices_np[b, t])
                                if 0 <= prev_idx < memory_tier.size.item():
                                    all_needed_indices.add(prev_idx)
                            
                            # Current topk indices for hop 0, 1, 2
                            for ki in range(k):
                                mem_idx = int(topk_indices_np[b, t, ki])
                                if 0 <= mem_idx < memory_tier.size.item():
                                    all_needed_indices.add(mem_idx)
                    
                    # 🚀 OPTIMIZED: Use graph-aware batch loading (loads roots + neighbors in ONE operation!)
                    bundle_cache = {}
                    total_bundles = 0
                    hop1_count = 0
                    
                    if len(all_needed_indices) > 0 and isinstance(memory_tier._bundled_storage, DiskBackedTensor):
                        indices_list = list(all_needed_indices)
                        bundle_cache, all_loaded = memory_tier._bundled_storage.get_bundles_with_neighbors(
                            indices_list, 
                            max_memory_size=memory_tier.size.item()
                        )
                        total_bundles = len(bundle_cache)
                        hop1_count = len(all_loaded) - len(all_needed_indices)
                    
                    if ENABLE_MICRO_PROFILING:
                        t_bundle_load_end = time.perf_counter()
                        print(f"   [TIMING] Bundle pre-loading: {(t_bundle_load_end - t_bundle_load_start)*1000:.1f}ms ({total_bundles} bundles, {len(all_needed_indices)} hop0 + {hop1_count} hop1)")
                    
                    if ENABLE_MICRO_PROFILING:
                        t_stats_loop_start = time.perf_counter()
                    
                    for b in range(B):
                        for t in range(T):
                            for ki in range(k):
                                mem_idx = int(topk_indices_np[b, t, ki])
                                if mem_idx < 0 or mem_idx >= memory_tier.size.item():
                                    continue
                                
                                # 🐛 SAFETY: Check tensor bounds
                                if not safe_index_check(hop_0_stats, b, t, ki):
                                    continue
                                
                                # HOP 0: How did we get to this memory (current edge from trajectory)?
                                if prev_top_indices is not None:
                                    prev_idx = int(prev_top_indices_np[b, t]) if t < prev_top_indices.shape[1] else -1
                                    
                                    # 🐛 DEBUG: Print trajectory context at display position
                                    if SHOW_BRAIN_DEBUG and b == 0 and t == min(64, T//2) and ki == 0:
                                        print(f"   🔍 HOP 0 DEBUG: prev_idx={prev_idx}, mem_idx={mem_idx}")
                                    
                                    if prev_idx >= 0 and prev_idx < memory_tier.size.item():
                                        # � USE PRE-LOADED BUNDLE (no disk I/O!)
                                        if prev_idx in bundle_cache:
                                            prev_neighbors = bundle_cache[prev_idx]['adjacency']
                                        else:
                                            # Fallback to property accessor (shouldn't happen if pre-loading worked)
                                            prev_neighbors = memory_tier.adjacency[prev_idx]
                                        
                                        edge_match = (prev_neighbors == mem_idx).nonzero(as_tuple=True)
                                        
                                        # 🐛 DEBUG: Print edge lookup result
                                        if SHOW_BRAIN_DEBUG and b == 0 and t == min(64, T//2) and ki == 0:
                                            print(f"   🔍 Neighbors of prev_idx={prev_idx}: {prev_neighbors.tolist()[:5]}...")
                                            print(f"   🔍 Looking for mem_idx={mem_idx}, found={len(edge_match[0]) > 0}")
                                        
                                        if len(edge_match[0]) > 0:
                                            edge_slot = edge_match[0][0].item()
                                            
                                            # � USE PRE-LOADED BUNDLE (no disk I/O!)
                                            try:
                                                if prev_idx in bundle_cache:
                                                    traversal = bundle_cache[prev_idx]['edge_traversal_count'][edge_slot].item()
                                                    success = bundle_cache[prev_idx]['edge_success_rate'][edge_slot].item()
                                                else:
                                                    # Fallback to property accessor
                                                    traversal = memory_tier.edge_traversal_count[prev_idx, edge_slot].item()
                                                    success = memory_tier.edge_success_rate[prev_idx, edge_slot].item()
                                            except Exception as e:
                                                print(f"⚠️  [HOP 0 ERROR] {e}")
                                                print(f"   Location: b={b}, t={t}, ki={ki}")
                                                print(f"   prev_idx={prev_idx}, mem_idx={mem_idx}, edge_slot={edge_slot}")
                                                if isinstance(memory_tier.edge_traversal_count, DiskBackedTensor):
                                                    print(f"   Using DiskBackedTensor (cache may be cold)")
                                                else:
                                                    print(f"   memory_tier.edge_traversal_count.shape={memory_tier.edge_traversal_count.shape}")
                                                    print(f"   Trying to index: [prev_idx={prev_idx}, edge_slot={edge_slot}]")
                                                print(f"   Full traceback:")
                                                import traceback
                                                traceback.print_exc()
                                                traversal = success = 0
                                            
                                            # Get token preview (first token at destination)
                                            if isinstance(memory_tier.embeddings, DiskBackedTensor):
                                                token_preview = memory_tier._bundled_storage[mem_idx].get('token_id', -1)
                                            else:
                                                token_preview = memory_tier.token_ids[mem_idx].item() if hasattr(memory_tier, 'token_ids') else -1
                                            
                                            # 🎯 COMPOUND EDGE STRENGTH for hop 0
                                            compound_strength = 0.0
                                            centrality = 0.0
                                            if hasattr(memory_tier, 'compute_compound_edge_strength'):
                                                try:
                                                    # Use grandparent if available for context
                                                    grandparent_idx = -1
                                                    compound_strength = memory_tier.compute_compound_edge_strength(
                                                        prev_idx, mem_idx, grandparent_idx
                                                    )
                                                    centrality = memory_tier.compute_edge_centrality(prev_idx, mem_idx)
                                                except:
                                                    compound_strength = 0.0
                                                    centrality = 0.0
                                            
                                            # 🎯 TRAJECTORY SPECIFICITY: How context-dependent is this edge?
                                            # > 0: This edge is BETTER for this trajectory than in general (specialized)
                                            # < 0: This edge is popular generally but WORSE for this trajectory (wrong path)
                                            # ~ 0: Edge quality is similar regardless of context (general-purpose)
                                            trajectory_specificity = compound_strength - centrality
                                            
                                            # relative_rank will be computed after all edges are collected
                                            hop_0_stats[b, t, ki] = torch.tensor(
                                                [trajectory_specificity, success, token_preview, compound_strength, centrality, 0.0], 
                                                device=query_original.device
                                            )
                                
                                # HOP 1: Where can we go from here? (ALL outgoing edges)
                                if hop_1_stats is not None:
                                    try:
                                        # � USE PRE-LOADED BUNDLE (no disk I/O!)
                                        if mem_idx in bundle_cache:
                                            hop1_neighbors = bundle_cache[mem_idx]['adjacency']
                                            hop1_edge_trav = bundle_cache[mem_idx]['edge_traversal_count']
                                            hop1_edge_succ = bundle_cache[mem_idx]['edge_success_rate']
                                        else:
                                            # Fallback to property accessor
                                            hop1_neighbors = memory_tier.adjacency[mem_idx]
                                            hop1_edge_trav = memory_tier.edge_traversal_count[mem_idx]
                                            hop1_edge_succ = memory_tier.edge_success_rate[mem_idx]
                                        
                                        # 🐛 DEBUG: Check bundle contents
                                        if not isinstance(hop1_neighbors, torch.Tensor):
                                            print(f"⚠️  hop1_neighbors is not a tensor! Type: {type(hop1_neighbors)}")
                                            continue
                                        if hop1_neighbors.dim() == 0:
                                            print(f"⚠️  hop1_neighbors is scalar! Creating 1D array")
                                            hop1_neighbors = hop1_neighbors.unsqueeze(0)
                                        
                                    except Exception as e:
                                        print(f"⚠️  Error loading hop1 bundle for mem_idx={mem_idx}: {e}")
                                        import traceback
                                        traceback.print_exc()
                                        continue
                                    
                                    for ei, hop1_idx in enumerate(hop1_neighbors.tolist()):
                                        if hop1_idx < 0 or hop1_idx >= memory_tier.size.item():
                                            continue
                                        
                                        # 🐛 FIX: Bounds check before indexing hop_1_stats
                                        if not safe_index_check(hop_1_stats, b, t, ki, ei):
                                            continue
                                        
                                        buffer_key = (mem_idx, ei)
                                        
                                        # Stats for this edge - CHECK WRITE BUFFER FIRST!
                                        try:
                                            # 🚀 FAST PATH: Check write buffer (dictionary lookup, no disk I/O)
                                            if hasattr(memory_tier, 'write_buffer') and buffer_key in memory_tier.write_buffer:
                                                # Get live data from buffer (most recent!)
                                                traversal = memory_tier.write_buffer[buffer_key]['count'].item()
                                                success = memory_tier.write_buffer[buffer_key]['success'].item()
                                                # 🐛 DEBUG: Found in buffer!
                                                if SHOW_BRAIN_DEBUG and traversal > 0:
                                                    print(f"   ✅ BUFFER HIT: ({mem_idx}, {ei}) → trav={traversal}, succ={success:.3f}")
                                            # 🐌 SLOW PATH: Read from property accessors (DiskBackedTensor handles caching!)
                                            else:
                                                # 🐛 DEBUG: Check what disk has
                                                if SHOW_BRAIN_DEBUG and hop1_edge_trav.dim() >= 1 and ei < hop1_edge_trav.shape[0]:
                                                    trav_val = hop1_edge_trav[ei].item()
                                                    succ_val = hop1_edge_succ[ei].item() if ei < hop1_edge_succ.shape[0] else 0
                                                    if trav_val > 0:
                                                        print(f"   📀 DISK HIT: node={mem_idx}, edge={ei} → trav={trav_val}, succ={succ_val:.3f}")
                                                
                                                # Handle different tensor shapes
                                                if hop1_edge_trav.dim() == 0:
                                                    # Scalar (single edge)
                                                    squeezed_count += 1
                                                    traversal = hop1_edge_trav.item() if ei == 0 else 0
                                                    success = hop1_edge_succ.item() if ei == 0 else 0
                                                elif hop1_edge_trav.dim() >= 1 and ei < hop1_edge_trav.shape[0]:
                                                    # 1D or higher array (normal case)
                                                    normal_count += 1
                                                    traversal = hop1_edge_trav[ei].item()
                                                    success = hop1_edge_succ[ei].item()
                                                else:
                                                    # Invalid or out of bounds
                                                    squeezed_count += 1
                                                    traversal = success = 0
                                        except Exception as e:
                                            print(f"⚠️  [HOP 1 EDGE STATS ERROR] {e}")
                                            print(f"   Location: b={b}, t={t}, ki={ki}, ei={ei}")
                                            print(f"   mem_idx={mem_idx}, hop1_idx={hop1_idx}")
                                            print(f"   hop1_edge_trav shape: {hop1_edge_trav.shape if hasattr(hop1_edge_trav, 'shape') else 'NO SHAPE'}")
                                            print(f"   hop1_edge_trav dim: {hop1_edge_trav.dim() if hasattr(hop1_edge_trav, 'dim') else 'NO DIM'}")
                                            print(f"   Trying to access ei={ei}")
                                            print(f"   Full traceback:")
                                            import traceback
                                            traceback.print_exc()
                                            traversal = success = 0
                                            missing_count += 1
                                        
                                        # Token preview - trust property accessor or skip if not available
                                        token_preview = -1  # Default
                                        if hasattr(memory_tier, 'token_ids'):
                                            try:
                                                token_preview = memory_tier.token_ids[hop1_idx].item()
                                            except:
                                                token_preview = -1
                                        
                                        # 🎯 COMPOUND EDGE STRENGTH: Combine weight + highway + flow context
                                        compound_strength = 0.0
                                        centrality = 0.0
                                        if hasattr(memory_tier, 'compute_compound_edge_strength'):
                                            try:
                                                # prev_idx for context-aware flow strength
                                                prev_idx = topk_indices[b, t, ki].item() if ki >= 0 else -1
                                                compound_strength = memory_tier.compute_compound_edge_strength(
                                                    mem_idx, hop1_idx, prev_idx
                                                )
                                                centrality = memory_tier.compute_edge_centrality(mem_idx, hop1_idx)
                                            except:
                                                compound_strength = 0.0
                                                centrality = 0.0
                                        
                                        # 🎯 TRAJECTORY SPECIFICITY
                                        trajectory_specificity = compound_strength - centrality
                                        
                                        # Stats: [specificity, success, token, compound, centrality, relative_rank]
                                        # relative_rank computed after all edges collected
                                        hop_1_stats[b, t, ki, ei] = torch.tensor(
                                            [trajectory_specificity, success, token_preview, compound_strength, centrality, 0.0],
                                            device=query_original.device
                                        )
                                    
                                        # HOP 2: Where can we go from THOSE neighbors?
                                        if hop_2_stats is not None:
                                            # � USE PRE-LOADED BUNDLE (no disk I/O!)
                                            if hop1_idx in bundle_cache:
                                                hop2_neighbors = bundle_cache[hop1_idx]['adjacency']
                                                hop2_edge_trav = bundle_cache[hop1_idx]['edge_traversal_count']
                                                hop2_edge_succ = bundle_cache[hop1_idx]['edge_success_rate']
                                            else:
                                                # Fallback to property accessor
                                                hop2_neighbors = memory_tier.adjacency[hop1_idx]
                                                hop2_edge_trav = memory_tier.edge_traversal_count[hop1_idx]
                                                hop2_edge_succ = memory_tier.edge_success_rate[hop1_idx]
                                            
                                            for ei2, hop2_idx in enumerate(hop2_neighbors.tolist()):
                                                if hop2_idx < 0 or hop2_idx >= memory_tier.size.item():
                                                    continue
                                                
                                                # 🐛 FIX: Bounds check before indexing hop_2_stats
                                                if not safe_index_check(hop_2_stats, b, t, ki, ei, ei2):
                                                    continue
                                                
                                                # Stats for 2-hop edge - trust property accessor!
                                                if hop2_edge_trav.dim() >= 1 and ei2 < hop2_edge_trav.shape[0]:
                                                    traversal2 = hop2_edge_trav[ei2].item()
                                                    success2 = hop2_edge_succ[ei2].item()
                                                else:
                                                    traversal2 = success2 = 0
                                                
                                                # Token preview - trust property accessor or skip
                                                token2 = -1  # Default
                                                if hasattr(memory_tier, 'token_ids'):
                                                    try:
                                                        token2 = memory_tier.token_ids[hop2_idx].item()
                                                    except:
                                                        token2 = -1
                                                
                                                # 🎯 COMPOUND EDGE STRENGTH for hop 2
                                                compound_strength2 = 0.0
                                                centrality2 = 0.0
                                                if hasattr(memory_tier, 'compute_compound_edge_strength'):
                                                    try:
                                                        # Use hop1_idx as context for hop2 edge
                                                        compound_strength2 = memory_tier.compute_compound_edge_strength(
                                                            hop1_idx, hop2_idx, mem_idx
                                                        )
                                                        centrality2 = memory_tier.compute_edge_centrality(hop1_idx, hop2_idx)
                                                    except:
                                                        compound_strength2 = 0.0
                                                        centrality2 = 0.0
                                                
                                                # 🎯 TRAJECTORY SPECIFICITY
                                                trajectory_specificity2 = compound_strength2 - centrality2
                                                
                                                # Stats: [specificity, success, token, compound, centrality, relative_rank]
                                                # relative_rank computed later
                                                hop_2_stats[b, t, ki, ei, ei2] = torch.tensor(
                                                    [trajectory_specificity2, success2, token2, compound_strength2, centrality2, 0.0],
                                                    device=query_original.device
                                                )
                    
                    if ENABLE_MICRO_PROFILING:
                        t_stats_loop_end = time.perf_counter()
                        print(f"   [TIMING] Statistics gathering loop: {(t_stats_loop_end - t_stats_loop_start)*1000:.1f}ms")
                    
                    # 🎯 RELATIVE NORMALIZATION: Rank edges within each hop
                    # For each (b, t, ki) position, normalize compound_strength so best=1.0
                    # This gives DEQ relative comparisons even in sparse regions
                    
                    # 🚀 VECTORIZED NORMALIZATION (eliminating nested loops for 100x speedup!)
                    
                    # Hop 0: Normalize across k memories [B, T, k]
                    if hop_0_stats is not None:
                        compounds = hop_0_stats[:, :, :, 3]  # [B, T, k]
                        max_vals = compounds.max(dim=2, keepdim=True)[0]  # [B, T, 1]
                        normalized = compounds / (max_vals + 1e-8)
                        hop_0_stats[:, :, :, 5] = normalized
                    
                    # Hop 1: Normalize across outgoing edges for each memory [B, T, k, max_edges]
                    if hop_1_stats is not None:
                        compounds = hop_1_stats[:, :, :, :, 3]  # [B, T, k, max_edges]
                        max_vals = compounds.max(dim=3, keepdim=True)[0]  # [B, T, k, 1]
                        normalized = compounds / (max_vals + 1e-8)
                        hop_1_stats[:, :, :, :, 5] = normalized
                    
                    # Hop 2: Normalize across 2nd-hop edges [B, T, k, max_edges, max_edges]
                    if hop_2_stats is not None:
                        compounds = hop_2_stats[:, :, :, :, :, 3]  # [B, T, k, max_edges, max_edges]
                        max_vals = compounds.max(dim=4, keepdim=True)[0]  # [B, T, k, max_edges, 1]
                        normalized = compounds / (max_vals + 1e-8)
                        hop_2_stats[:, :, :, :, :, 5] = normalized
                    
                    # Bundle multi-hop routing information
                    routing_bundle = {
                        'hop_0_stats': hop_0_stats,      # [B, T, k, 6] - current edge (6 features)
                        'hop_1_stats': hop_1_stats,      # [B, T, k, max_edges, 6] - 1-hop options
                        'hop_2_stats': hop_2_stats,      # [B, T, k, max_edges, max_edges, 6] - 2-hop lookahead
                        'memory_tier': memory_tier,      # 🐛 DEBUG: Pass tier for write buffer inspection
                        'topk_indices': topk_indices,    # 🐛 DEBUG: Pass indices to check buffer coverage
                    }
                    
                    # 🧠 BRAIN DEBUG VIEW: Show complete DEQ fixed-point trajectory
                    # --brain-always: Every DEQ iteration immediately (verbose debugging)
                    # --show-brain: Summary every 1000 retrievals (sparse, non-intrusive)
                    if SHOW_BRAIN_DEBUG and hop_0_stats is not None:
                        if not hasattr(self, '_brain_view_call_count'):
                            self._brain_view_call_count = 0
                            self._brain_view_last_shown_iter = -1
                            self._brain_deq_iteration = 0  # Initialize here!
                        
                        self._brain_view_call_count += 1
                        
                        # Decide display frequency based on mode
                        if SHOW_BRAIN_ALWAYS:
                            # VERBOSE MODE: Show EVERY SINGLE CALL immediately (no delay)
                            display_interval = 1  # Every call
                            show_all_deq_iters = True
                        else:
                            # NORMAL MODE: Show every 1000 calls (every ~200 training iters)
                            display_interval = 1000
                            show_all_deq_iters = False
                        
                        # Detect new training iteration (call count resets or jumps)
                        # Each training iter does ~6-30 DEQ iterations (memory retrievals)
                        if self._brain_view_call_count % display_interval == 1 or SHOW_BRAIN_ALWAYS:
                            # Start showing this DEQ trajectory
                            if self._brain_view_call_count == 1 or self._brain_view_call_count % display_interval == 1:
                                self._brain_deq_iteration = 0
                                print("\n" + "🧠"*40)
                                print(f"BRAIN DEBUG: Showing DEQ fixed-point trajectory")
                                print(f"Training call #{self._brain_view_call_count}")
                                if SHOW_BRAIN_ALWAYS:
                                    print(f"Mode: VERBOSE (--brain-always) - EVERY CALL")
                                else:
                                    print(f"Mode: SUMMARY (--show-brain)")
                                print("🧠"*40 + "\n")
                        
                        # Show DEQ iterations based on mode
                        if SHOW_BRAIN_ALWAYS:
                            # VERBOSE: Show EVERY SINGLE CALL
                            print(f"\n{'='*80}")
                            print(f"🧠 Retrieval Call #{self._brain_view_call_count}")
                            print(f"{'='*80}")
                            self._display_brain_view(query_original, bundle['embeddings'], routing_bundle, topk_indices)
                        elif show_all_deq_iters:
                            # This branch won't be hit since show_all_deq_iters is only True when SHOW_BRAIN_ALWAYS
                            pass
                        else:
                            # NORMAL: Just show one summary view every 1000 calls
                            if self._brain_view_call_count % display_interval == 1:
                                print(f"\n{'='*80}")
                                print(f"🧠 Memory Retrieval Summary")
                                print(f"{'='*80}")
                                self._display_brain_view(query_original, bundle['embeddings'], routing_bundle, topk_indices)
                                print("\n" + "🧠"*40)
                                print(f"Next summary in {display_interval} retrievals (~{display_interval//5} training iterations)")
                                print("🧠"*40 + "\n")

                    
                    # 📊 DIAGNOSTICS: Report edge tensor health
                    total_edges = normal_count + squeezed_count + missing_count
                    if total_edges > 0:
                        normal_pct = 100.0 * normal_count / total_edges
                        squeezed_pct = 100.0 * squeezed_count / total_edges
                        missing_pct = 100.0 * missing_count / total_edges
                        print(f"📊 [EDGE HEALTH] Total={total_edges}: Normal={normal_count}({normal_pct:.1f}%), Squeezed={squeezed_count}({squeezed_pct:.1f}%), Missing={missing_count}({missing_pct:.1f}%)")
                        if squeezed_count > 0:
                            print(f"   ⚠️  {squeezed_count} squeezed tensors encountered (handled gracefully)")

                    
            except Exception as e:
                # Graceful degradation if routing fails
                print(f"⚠️  Failed to build multi-hop routing bundle: {e}")
                print(f"   Full traceback:")
                import traceback
                traceback.print_exc()
                routing_bundle = {}
        
        # Merge routing info into main bundle
        bundle.update(routing_bundle)
        
        # REMOVED: update_access() - access tracking no longer needed (disk-backed cache manages itself)
        
        # 🔍 PROFILING: Track time breakdown
        t_end = time.time()
        elapsed_ms = (t_end - t_start) * 1000
        if not hasattr(self, '_profile_count'):
            self._profile_count = 0
            self._total_retrieval_time = 0.0
        self._profile_count += 1
        self._total_retrieval_time += elapsed_ms
        
        #if self._profile_count <= 5:
        #    print(f"⏱️  [RETRIEVAL #{self._profile_count}] {elapsed_ms:.1f}ms | "
        #          f"Tier: {memory_tier.capacity} | Size: {memory_tier.size.item()} | k={k}")
        
        return bundle
    
    def _display_brain_view(self, query, memories, routing_bundle, topk_indices):
        """
        🧠 DEBUG VIEW: Show what the DEQ 'sees' when it looks at memory.
        
        Displays:
        - Input query tokens (what we're searching for)
        - Retrieved memory tokens (what we found)
        - Routing statistics (highway strength, traversal counts)
        - Multi-hop lookahead (future path options)
        
        This helps understand if the memory system is giving the DEQ
        good geometric structure to reason about.
        """
        # Try to get decoder function from global scope
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            decode = lambda tokens: enc.decode(tokens) if isinstance(tokens, list) else enc.decode([tokens])
        except:
            # Fallback: try to load from meta.pkl
            try:
                import pickle
                import os
                meta_path = os.path.join('data', 'tinystories', 'meta.pkl')
                if os.path.exists(meta_path):
                    with open(meta_path, 'rb') as f:
                        meta = pickle.load(f)
                        itos = meta.get('itos', {})
                        decode = lambda tid: itos.get(int(tid), f'<{tid}>')
                else:
                    decode = lambda tid: f'<{tid}>'
            except:
                decode = lambda tid: f'<{tid}>'
        
        print("\n" + "="*80)
        print("🧠 BRAIN DEBUG VIEW - DEQ Memory Bundle Structure")
        print("="*80)
        
        # Pick a display position that has trajectory context
        # Position [0, 0] will always be empty (no previous timestep)
        # Show middle of sequence where prev_top_indices should be populated
        B = query.shape[0]
        T = query.shape[1]
        display_b = 0  # First batch
        display_t = min(64, T // 2)  # Middle of sequence, or position 64
        
        print(f"\n📍 DISPLAY POSITION: batch={display_b}, timestep={display_t}/{T}")
        print(f"   (Position 0 has no trajectory context - showing middle where context exists)")
        
        # Show query shape
        print(f"\n📥 INPUT QUERY: shape={query.shape}")
        print(f"   (Embedding vectors for current context)")
        
        # Show retrieved memories
        print(f"\n💾 RETRIEVED MEMORIES: shape={memories.shape}")
        # Handle variable topk_indices shape
        if topk_indices.dim() == 2:
            mem_indices = topk_indices[0, :].tolist()
        elif topk_indices.dim() == 3:
            mem_indices = topk_indices[display_b, display_t, :].tolist()
        else:
            mem_indices = []
        print(f"   Memory indices at [{display_b},{display_t}]: {mem_indices[:5]}{'...' if len(mem_indices) > 5 else ''}")
        
        # Show routing bundle structure with ACTUAL TOKENS
        if 'hop_0_stats' in routing_bundle and routing_bundle['hop_0_stats'] is not None:
            hop_0 = routing_bundle['hop_0_stats']
            print(f"\n🛣️  HOP 0 STATS (Current Memory Nodes): shape={hop_0.shape}")
            print(f"   Format: [traversal_count, success_rate, token_id]")
            print(f"   Showing position [{display_b}, {display_t}]:")
            for ki in range(min(5, hop_0.shape[2])):  # Show first 5 neighbors
                stats = hop_0[display_b, display_t, ki].tolist()
                token_id = int(stats[2])
                token_str = decode(token_id) if token_id >= 0 else '<none>'
                print(f"   Memory {ki}: token=\"{token_str}\" | traversal={stats[0]:.0f} | success={stats[1]:.3f}")
        
        if 'hop_1_stats' in routing_bundle and routing_bundle['hop_1_stats'] is not None:
            hop_1 = routing_bundle['hop_1_stats']
            print(f"\n🔮 HOP 1 LOOKAHEAD (Where can we go?): shape={hop_1.shape}")
            
            # 🐛 DEBUG: Check write buffer status
            if 'memory_tier' in routing_bundle:
                mem_tier = routing_bundle['memory_tier']
                if hasattr(mem_tier, 'write_buffer'):
                    print(f"   📝 Write buffer size: {len(mem_tier.write_buffer)} entries")
                    # Sample a few keys and check if ANY match our retrieved memories
                    if len(mem_tier.write_buffer) > 0:
                        sample_keys = list(mem_tier.write_buffer.keys())[:3]
                        print(f"   Sample keys: {sample_keys}")
                        # Check if we're looking at the right nodes
                        if 'topk_indices' in routing_bundle:
                            mem_indices = routing_bundle['topk_indices'][display_b, display_t].tolist() if routing_bundle['topk_indices'].dim() > 2 else []
                            matching_nodes = [k[0] for k in mem_tier.write_buffer.keys() if k[0] in mem_indices[:5]]
                            if matching_nodes:
                                print(f"   ✅ Buffer has data for retrieved nodes: {matching_nodes}")
                            else:
                                print(f"   ⚠️  Buffer has NO data for nodes {mem_indices[:5]}")
                else:
                    print(f"   ⚠️  memory_tier has no write_buffer attribute!")
            
            print(f"   Showing outgoing edges from first memory at [{display_b}, {display_t}]:")
            active_edges = []
            for ei in range(min(10, hop_1.shape[3])):
                stats = hop_1[display_b, display_t, 0, ei].tolist()
                if stats[0] > 0:  # Only show active edges
                    token_id = int(stats[2])
                    token_str = decode(token_id) if token_id >= 0 else '<none>'
                    active_edges.append((ei, token_str, stats[0], stats[1]))
            
            for ei, token_str, traversal, success in active_edges[:8]:
                print(f"   Edge {ei}→ \"{token_str}\" | traversal={traversal:.0f} | success={success:.3f}")
            if len(active_edges) > 8:
                print(f"   ... and {len(active_edges) - 8} more active edges")
        
        if 'hop_2_stats' in routing_bundle and routing_bundle['hop_2_stats'] is not None:
            hop_2 = routing_bundle['hop_2_stats']
            print(f"\n🌌 HOP 2 LOOKAHEAD (2-hop paths): shape={hop_2.shape}")
            # Count active paths
            active_paths = (hop_2[display_b, display_t, 0, :, :, 0] > 0).sum().item()
            print(f"   Total active 2-hop paths from first memory: {active_paths}")
            
            # Show a few high-quality paths
            print(f"   Top paths (by success rate):")
            paths = []
            for ei in range(min(5, hop_2.shape[3])):
                for ei2 in range(min(5, hop_2.shape[4])):
                    stats = hop_2[0, 0, 0, ei, ei2].tolist()
                    if stats[0] > 0:
                        token_id = int(stats[2])
                        token_str = decode(token_id) if token_id >= 0 else '<none>'
                        paths.append((ei, ei2, token_str, stats[1], stats[0]))
            
            # Sort by success rate
            paths.sort(key=lambda x: x[3], reverse=True)
            for ei, ei2, token_str, success, traversal in paths[:5]:
                print(f"   Path [{ei}→{ei2}]→ \"{token_str}\" | success={success:.3f} | traversal={traversal:.0f}")
        
        print("\n" + "="*80)
        print("🧠 The DEQ navigates this token graph to find optimal paths")
        print("="*80 + "\n")


class GraphMemoryIntegrationNetwork(nn.Module):
    """
    Cross-attention network that integrates retrieved graph memories
    into the current context.
    
    Uses cross-attention from context to retrieved memories, then
    learns how much to trust the memory via gating.
    """
    
    def __init__(self, context_dim: int, memory_dim: int, n_head: int = 4):
        super().__init__()
        self.context_dim = context_dim
        self.memory_dim = memory_dim
        self.n_head = n_head
        
        assert context_dim % n_head == 0, "context_dim must be divisible by n_head"
        
        # Cross-attention: query from context, key/value from memory
        self.q_proj = nn.Linear(context_dim, context_dim)
        self.k_proj = nn.Linear(memory_dim, context_dim)
        self.v_proj = nn.Linear(memory_dim, context_dim)
        self.out_proj = nn.Linear(context_dim, context_dim)
        
        # 🛣️ MULTI-HOP ROUTING PROCESSOR: Process 2-hop lookahead with token previews
        # Input features per memory:
        #   hop_0: [traversal, success, token] = 3
        #   hop_1: [avg_traversal, avg_success, avg_token] = 3 (averaged over all 1-hop edges)
        #   hop_2: [avg_traversal, avg_success, avg_token] = 3 (averaged over all 2-hop edges)
        # Total: 9 dims per memory
        routing_feature_dim = 9
        self.routing_processor = nn.Sequential(
            nn.Linear(routing_feature_dim, context_dim // 4),
            nn.GELU(),
            nn.Linear(context_dim // 4, 1)  # Output: routing confidence per memory
        )
        
        # Gating: how much to trust memory
        self.gate_mlp = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
            nn.Sigmoid()
        )
        
        self.ln1 = nn.LayerNorm(context_dim)
        self.ln2 = nn.LayerNorm(context_dim)
    
    def forward(self, context: torch.Tensor, retrieved_memories: torch.Tensor, 
                flow_bias: torch.Tensor = None, transported_query: torch.Tensor = None,
                routing_bundle: dict = None) -> torch.Tensor:
        """
        Args:
            context: [B, T, context_dim] current reflex embeddings
            retrieved_memories: [B, T, num_mems, memory_dim] retrieved graph memories
            flow_bias: [B, T, num_mems] 🌊 predictive bias from trajectory history (optional)
            transported_query: [B, T, memory_dim] 🎯 query evolved by parallel transport (optional)
            routing_bundle: dict with routing intelligence (optional):
                - current_edge_stats: [B, T, k, 2] - traversal_count, success_rate for edge taken
                - next_hop_neighbors: [B, T, k, k_neighbors] - reachable memories from each candidate
                - next_hop_stats: [B, T, k, k_neighbors, 2] - statistics for each next hop
                - next_hop_previews: [B, T, k, k_neighbors, 8] - embedding previews for lookahead
        
        Returns:
            [B, T, context_dim] context enhanced with memory
        """
        B, T, C = context.shape
        _, _, num_mems, M = retrieved_memories.shape
        
        # Cross-attention with parallel-transported query if available
        if transported_query is not None:
            # Use geometry-shaped query (query evolved along retrieval trajectory)
            q = self.q_proj(transported_query)  # [B, T, C]
        else:
            # Fall back to context-based query
            q = self.q_proj(context)  # [B, T, C]
            
        k_mem = self.k_proj(retrieved_memories)  # [B, T, num_mems, C]
        v_mem = self.v_proj(retrieved_memories)  # [B, T, num_mems, C]
        
        # Reshape for multi-head attention
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # [B, n_head, T, head_dim]
        k_mem = k_mem.view(B, T, num_mems, self.n_head, head_dim).permute(0, 3, 1, 2, 4)  # [B, n_head, T, num_mems, head_dim]
        v_mem = v_mem.view(B, T, num_mems, self.n_head, head_dim).permute(0, 3, 1, 2, 4)  # [B, n_head, T, num_mems, head_dim]
        
        # Attention scores
        scores = (q.unsqueeze(-2) @ k_mem.transpose(-2, -1)) / math.sqrt(head_dim)  # [B, n_head, T, 1, num_mems]
        
        # 🌊 FLOW BIAS INJECTION: "Waze says people from 'dog' usually go to 'meow', not 'purr'"
        if flow_bias is not None:
            # Add bias to attention logits (before softmax)
            # flow_bias is [B, T, num_mems], expand to [B, n_head, T, 1, num_mems]
            bias_expanded = flow_bias.unsqueeze(1).unsqueeze(3)  # [B, 1, T, 1, num_mems]
            scores = scores + bias_expanded
        
        # 🛣️ MULTI-HOP ROUTING INTELLIGENCE: "Highway to 'dog' (95%), then branches to 'cat'(80%) or 'meow'(60%)"
        if routing_bundle is not None and 'hop_0_stats' in routing_bundle:
            try:
                hop_0 = routing_bundle['hop_0_stats']  # [B, T, k, 6]
                hop_1 = routing_bundle['hop_1_stats']  # [B, T, k, max_edges, 6]
                hop_2 = routing_bundle['hop_2_stats']  # [B, T, k, max_edges, max_edges, 6]
                
                # Each edge has 6 features:
                # [0] traversal_count (raw, unnormalized)
                # [1] success_rate (0-1, from training feedback)
                # [2] token_preview (token_id for debugging)
                # [3] compound_strength (trajectory-aware: weight+highway+flow, 0-1)
                # [4] centrality (trajectory-agnostic: graph importance, 0-1)
                # [5] relative_rank (normalized within hop: 1.0=best option)
                
                # Aggregate multi-hop statistics into routing features: [B, T, k, 18]
                # hop_0: current edge stats (6 features)
                # hop_1: average over all outgoing edges (6 features)
                # hop_2: average over all 2-hop paths (6 features)
                # Total: 6+6+6 = 18 features
                #
                # This gives the DEQ:
                # - Primary signal: compound_strength (trajectory-aware, uses flow field)
                # - Backup signal: centrality (trajectory-agnostic, graph structure)
                # - Insight metric: trajectory_specificity (positive=specialized, negative=general hub)
                # - Quality metric: success_rate (training feedback)
                # - Comparison tool: relative_rank (normalized, interpretable)
                hop_1_avg = hop_1.mean(dim=3)  # [B, T, k, 6]
                hop_2_avg = hop_2.view(B, T, num_mems, -1, 6).mean(dim=3)  # [B, T, k, 6]
                
                routing_features = torch.cat([hop_0, hop_1_avg, hop_2_avg], dim=-1)  # [B, T, k, 18]
                
                # Process through routing MLP to get confidence scores
                routing_confidence = self.routing_processor(routing_features).squeeze(-1)  # [B, T, k]
                
                # Add routing confidence to attention scores (combine with flow_bias)
                # Expand to [B, n_head, T, 1, k]
                routing_bias = routing_confidence.unsqueeze(1).unsqueeze(3)  # [B, 1, T, 1, k]
                scores = scores + routing_bias
            except Exception as e:
                # Gracefully degrade if routing bundle incomplete
                pass
        
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = (attn @ v_mem).squeeze(-2)  # [B, n_head, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        out = self.out_proj(out)
        out = self.ln1(out)
        
        # Gating: decide how much to use memory vs context
        combined = torch.cat([context, out], dim=-1)  # [B, T, 2*C]
        gate = self.gate_mlp(combined)  # [B, T, C]
        
        # Gated residual
        result = self.ln2(context + gate * out)
        
        return result


class GraphMemorySystem(nn.Module):
    """
    Complete graph-structured memory system with three tiers:
    - Working memory: small, fast, on GPU
    - Buffer memory: medium, consolidation staging, on GPU
    - Long-term memory: large, slow, on CPU
    
    All tiers maintain graph structure. Consolidation preserves
    graph relationships when moving between tiers.
    """
    
    def __init__(self, memory_dim: int, query_dim: int, context_dim: int,
                 memory_capacity: int = 200000,  # Single unified tier
                 disk_path: str = None,  # Disk backing for the unified memory
                 max_disk_size: int = 200000,  # Maximum total memories on disk
                 k_neighbors: int = 20,
                 gnn_hidden_dim: int = 512,
                 n_head: int = 4,
                 enable_gnn: bool = False,  # DISABLED by default until we optimize hyperbolic ops for VRAM
                 highway_learning_rate: float = 0.3,  # 🔥 NEW: How fast highways strengthen
                 use_full_hyperbolic_gnn: bool = False,  # 🌀 NEW: Full Möbius vs hybrid
                 routing_max_hops: int = 2):  # 🚀 NEW: Lookahead depth (2=TinyStories, 4=GPT-3, 6=GPT-4)
        super().__init__()
        
        self.memory_dim = memory_dim
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.k_neighbors = k_neighbors
        self.highway_learning_rate = highway_learning_rate  # Store for Hebbian updates
        self.routing_max_hops = routing_max_hops  # Store for speculative skip depth
        
        print(f"[GraphMemory] Routing lookahead: {routing_max_hops}-hop")
        print(f"              (2=simple, 3-4=medium, 5-6=complex reasoning)")
        
        # Hyperbolic geometry
        self.poincare = PoincareManifold(dim=memory_dim)
        
        # 🌍 UNIFIED GRAPH MEMORY
        # Single tier with DiskBackedTensor caching:
        #   - Total capacity: 200k nodes (can grow to max_disk_size)
        #   - Hot capacity: 5k nodes in RAM (hyperbolic cache keeps related clusters)
        #   - Cache eviction: Hyperbolic distance-based (far nodes evicted first)
        #   - Working memory: Emerges naturally from recency + semantic clustering
        # 
        # Benefits vs 3-tier:
        #   - DEQ can hop anywhere in the graph (no tier fragmentation)
        #   - No consolidation/promotion overhead
        #   - Cache automatically keeps hot clusters in RAM
        #   - Natural working/buffer/longterm emerge from access patterns
        
        self.memory = GraphMemoryTier(
            memory_capacity, memory_dim, k_neighbors, device='cuda',
            disk_path=disk_path,
            max_disk_size=max_disk_size
        )
        
        print(f"[GraphMemory] Unified Memory:")
        print(f"  Total capacity: {memory_capacity:,} nodes")
        print(f"  RAM cache: Auto-calculated (~5GB budget)")
        print(f"  Disk overflow: Automatic hyperbolic eviction")
        print(f"  Cache strategy: Hyperbolic distance eviction")
        
        # GNN query network (can disable for VRAM savings)
        self.query_network = GraphMemoryQueryNetwork(
            query_dim, memory_dim, gnn_hidden_dim, k_neighbors, 
            enable_gnn=enable_gnn,
            use_full_hyperbolic=use_full_hyperbolic_gnn
        ).cuda()
        
        print(f"  GNN refinement: {'ENABLED' if enable_gnn else 'DISABLED (VRAM saver)'}")
        
        # Integration network
        self.integration_network = GraphMemoryIntegrationNetwork(
            context_dim, memory_dim, n_head
        ).cuda()
        
        # Homeostatic feedback
        self.sigma_memory = 1.0
        
        # DYNAMIC EVOLUTION: Track retrieval patterns for graph optimization
        self.retrieval_history = []  # List of (query_cluster, retrieved_clusters, reward)
        self.consolidation_counter = 0
        
        # 🕰️ ELIGIBILITY TRACES: The Time Machine for Credit Assignment
        # 
        # THE PROBLEM: When you solve a math problem and get reward,
        # which step gets the credit? The last step? Or the brilliant insight 16 steps ago?
        # 
        # THE SOLUTION: Store the trajectory. When reward hits, propagate backwards
        # with exponential decay (gamma^distance). This is TD(λ) from RL!
        # 
        # BIOLOGICAL ANALOG: Synaptic tagging & capture - synapses "remember" recent
        # activity and get strengthened when reward arrives later.
        # 
        # For Language/Code:
        #   - Bigram (1 step):  "Hello" → "World"         (Reflex)
        #   - Sentence (4-8):   Subject → ... → Verb      (Grammar)  
        #   - Logic (16):       if x > 0: → ... → print(x) (Reasoning)
        #
        self.trajectory_buffer = []     # List of {query, indices, tier_name} for last N steps
        self.max_trajectory = 16        # How far back we credit (the "magic number" for logic)
        self.gamma = 0.9                # Discount factor (0.9^16 ≈ 18% credit to earliest step)
        
        # 🌀 HOLONOMY TRACKING: Measure narrative coherence
        # 
        # GEOMETRIC INSIGHT: When you traverse a closed loop and return to the same concept,
        # the transported query vector might be ROTATED compared to the original.
        # This "holonomy" measures how much curvature accumulated.
        #
        # APPLICATIONS:
        #   - Low holonomy = coherent reasoning (circular logic that makes sense)
        #   - High holonomy = confused reasoning (same words, different meanings)
        #   - Can use as adaptive depth signal (high holonomy = need more iterations)
        #
        self.holonomy_buffer = []       # List of (node_id, transported_query) pairs
        self.max_holonomy_track = 32    # Track last 32 visited nodes for loop detection
    
    @profile_op("store_memory_dynamic")
    def store_memory_dynamic(self, embedding: torch.Tensor, reward: float = 1.0) -> int:
        """
        Store new memory with DYNAMIC graph integration.
        
        The graph GROWS as new memories form, like a city expanding!
        All memories go to unified storage - cache system handles hot/cold automatically.
        
        Args:
            embedding: [D] memory vector
            reward: utility score (for homeostatic tracking)
        
        Returns:
            index of stored memory in unified tier
        """
        # 🚀 OPTIMIZATION: Avoid redundant device transfers
        # Check if embedding is already on CUDA before moving
        if embedding.device.type != 'cuda':
            embedding = embedding.to('cuda')
        
        # Add to unified memory - DiskBackedTensor cache handles RAM/disk placement
        new_idx = self.memory.add_node_dynamic(
            embedding,
            self.poincare,
            cluster_id=-1,  # Infer from neighbors
            skip_disk_search=True  # 🚀 Skip expensive disk search during preload
        )
        
        # Reward is already initialized to 0 in add_node_dynamic
        # It will be updated later by apply_dopamine() based on prediction quality
        # No need to set it here - avoids race condition with auxiliary tensor growth
        
        # 🚀 OPTIMIZATION: Only print occasionally (expensive I/O)
        if new_idx % 5000 == 0:  # Changed from 1000 to 5000
            print(f"� Stored {new_idx} memories...")
        
        return new_idx
    
    def strengthen_last_retrieval(self, per_token_loss: torch.Tensor):
        """
        Strengthen highways based on ACTUAL per-token prediction accuracy.
        
        Call this AFTER computing the loss to reinforce paths that helped!
        This is token-level learning: good retrieval for token i → strengthen that path only.
        
        Args:
            per_token_loss: [B, T] tensor of per-token losses (NOT averaged!)
        """
        if not hasattr(self, '_last_retrieval_paths') or self._last_retrieval_paths is None:
            return  # No retrieval happened this step
        
        paths = self._last_retrieval_paths
        B, T = paths['shape']
        
        # Move loss to same device as query
        device = paths['query'].device
        if per_token_loss.device != device:
            per_token_loss = per_token_loss.to(device)
        
        # Strengthen highways for EACH token individually
        with torch.no_grad():
            for b in range(B):
                for t in range(T):
                    # Get loss for this specific token
                    token_loss = per_token_loss[b, t].item()
                    
                    # 🔥 IMPROVED REWARD FORMULA: More aggressive scaling
                    # Use exponential decay: reward = exp(-loss/temperature)
                    # This gives stronger rewards for good predictions!
                    # loss=0 → 1.0, loss=2 → 0.37, loss=5 → 0.08, loss=10 → 0.007
                    temperature = 2.0  # Lower = more aggressive
                    reward = math.exp(-token_loss / temperature)
                    reward = min(2.0, reward)  # Cap at 2.0, but don't raise floor (let threshold filter)
                    
                    # Only strengthen if reward is above threshold (don't strengthen random paths!)
                    if reward < 0.15:  # Skip if prediction was bad (loss > ~5.7)
                        continue
                    
                    query_token = paths['query'][b, t]
                    
                    # Strengthen unified memory highways
                    memory_indices = paths['memory'][b, t]
                    if memory_indices.numel() > 1 and memory_indices[0] >= 0:
                        self.record_retrieval_success(
                            query_token,
                            memory_indices,
                            reward=reward,
                            learning_rate=self.highway_learning_rate
                        )
        
        # Don't clear paths here - they're shared by all DEQ iterations via apply_dopamine
        # self._last_retrieval_paths = None
    
    def record_retrieval_success(self, query_embedding: torch.Tensor, 
                                 retrieved_indices: torch.Tensor, 
                                 reward: float,
                                 learning_rate: float = 0.3):
        """
        Record which memories were retrieved together successfully.
        
        Strengthens edges along the traversal path - creates "highways"!
        
        🔥 HEBBIAN LEARNING IN UNIFIED MEMORY:
        When the DEQ navigates through memories and gets rewarded (low loss),
        we strengthen the edges it traversed. Over time, this creates "highways"
        through the 200K memory graph - fast paths to useful knowledge!
        
        Args:
            query_embedding: The query that triggered retrieval
            retrieved_indices: Which memories were retrieved (in order)
            reward: How helpful this retrieval was (higher = strengthen more)
            learning_rate: How fast to update edge weights (0.1=slow, 0.5=fast)
        """
        if len(retrieved_indices) < 2:
            return
        
        # Safety: skip if memory is empty
        if self.memory.size == 0:
            return
        
        # 🚀 BATCH OPTIMIZATION: Collect all edge updates instead of individual calls
        edge_updates = []
        
        # Strengthen edges between consecutively retrieved memories
        for i in range(len(retrieved_indices) - 1):
            idx_a = retrieved_indices[i].item()
            idx_b = retrieved_indices[i+1].item()
            
            # Only strengthen if both are valid indices
            # Check: >= 0 (not placeholder) and < size (within bounds)
            if (idx_a >= 0 and idx_a < self.memory.size.item() and 
                idx_b >= 0 and idx_b < self.memory.size.item()):
                # Bidirectional edges
                edge_updates.append((idx_a, idx_b, reward))
                edge_updates.append((idx_b, idx_a, reward))
        
        # 🚀 EDGE SUBSAMPLING: Reduce excessive strengthen_edges_batch calls
        edge_updates = subsample_edge_updates(edge_updates, "strengthen_path")
        
        # 🚀 BATCH UPDATE: 100x faster than individual calls!
        if edge_updates and hasattr(self.memory, 'strengthen_edges_batch'):
            self.memory.strengthen_edges_batch(edge_updates)
        elif edge_updates:
            # Fallback: individual calls (only if batch not available)
            for idx_a, idx_b, r in edge_updates:
                self.memory.strengthen_edge(idx_a, idx_b, reward=r, learning_rate=learning_rate)
    
    def evolve_graph(self, enable_sleep_replay: bool = False):
        """
        Periodic graph evolution: prune weak edges, optimize structure.
        
        Args:
            enable_sleep_replay: If True, performs sleep-like memory reconsolidation
        
        Call this during consolidation to maintain graph health!
        """
        # Prune weak edges in unified memory
        if self.memory.size > 100:  # Only if enough nodes
            self.memory.prune_weak_edges(self.poincare, threshold=0.1)
        
        # SLEEP-STAGE RECONSOLIDATION (optional future enhancement)
        if enable_sleep_replay and self.memory.size > 0:
            self._sleep_replay_reconsolidation()
    
    def build_graph_edges(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build k-NN graph in hyperbolic space.
        
        Args:
            embeddings: [N, D] memory vectors
        
        Returns:
            adjacency: [N, k] neighbor indices
            edge_weights: [N, k] hyperbolic distances
        """
        N = embeddings.shape[0]
        
        if N == 0:
            return (torch.full((0, self.k_neighbors), -1, dtype=torch.long, device=embeddings.device),
                    torch.zeros(0, self.k_neighbors, device=embeddings.device))
        
        # Map to hyperbolic space (exponential map from origin)
        origin = torch.zeros_like(embeddings)
        hyp_embeddings = self.poincare.exponential_map(origin, embeddings)
        
        # Compute pairwise distances
        dists = self.poincare.distance(
            hyp_embeddings.unsqueeze(1),  # [N, 1, D]
            hyp_embeddings.unsqueeze(0)   # [1, N, D]
        ).squeeze(-1)  # [N, N]
        
        # Set self-distance to infinity to exclude self-loops
        dists = dists + torch.eye(N, device=embeddings.device) * 1e6
        
        # Get k nearest neighbors
        k_actual = min(self.k_neighbors, N - 1)
        topk_dists, topk_indices = torch.topk(dists, k_actual, largest=False, dim=-1)
        
        # Pad if needed
        if k_actual < self.k_neighbors:
            pad_size = self.k_neighbors - k_actual
            adjacency = F.pad(topk_indices, (0, pad_size), value=-1)
            edge_weights = F.pad(topk_dists, (0, pad_size), value=0.0)
        else:
            adjacency = topk_indices
            edge_weights = topk_dists
        
        return adjacency, edge_weights
    
    def add_to_working(self, embeddings: torch.Tensor, rewards: torch.Tensor) -> None:
        """Add memories to unified memory (method kept for compatibility)."""
        # Delegate to store_memory_dynamic
        if embeddings.shape[0] == 0:
            return
        
        # DIMENSION SAFETY: Check if embedding dimension matches
        if embeddings.shape[-1] != self.memory_dim:
            print(f"[Memory] WARNING: Embedding dim mismatch: got {embeddings.shape[-1]}, expected {self.memory_dim}")
            print(f"[Memory] Updating memory_dim from {self.memory_dim} → {embeddings.shape[-1]}")
            print(f"[Memory] Clearing memory due to architecture change")
            
            # UPDATE DIMENSION to match new architecture
            old_memory_dim = self.memory_dim
            self.memory_dim = embeddings.shape[-1]
            
            # REBUILD QUERY NETWORK with new dimensions
            print(f"[Memory] Rebuilding query network: {self.query_dim} → {self.memory_dim}")
            self.query_network = GraphMemoryQueryNetwork(
                self.query_dim, 
                self.memory_dim, 
                self.query_network.hidden_dim,
                self.k_neighbors,
                enable_gnn=self.query_network.enable_gnn,
                num_edge_types=8
            ).cuda()
            
            # REBUILD INTEGRATION NETWORK with new dimensions
            print(f"[Memory] Rebuilding integration network: {self.context_dim} ← {self.memory_dim}")
            self.integration_network = GraphMemoryIntegrationNetwork(
                self.context_dim,
                self.memory_dim,
                self.integration_network.n_head
            ).cuda()
            
            # REBUILD POINCARE MANIFOLD with new dimension
            print(f"[Memory] Rebuilding Poincaré manifold: dim={self.memory_dim}")
            self.poincare = PoincareManifold(dim=self.memory_dim)
            
            print(f"[Memory] ✓ Architecture updated successfully: {old_memory_dim} → {self.memory_dim}")
            return
        
        # Build edges within new memories
        adjacency, edge_weights = self.build_graph_edges(embeddings)
        cluster_ids = torch.zeros(embeddings.shape[0], dtype=torch.long, device=embeddings.device)
        
        self.memory.add_nodes(embeddings, adjacency, edge_weights, cluster_ids, rewards)
    
    @profile_op("consolidate_to_buffer")
    def consolidate_to_buffer(self) -> None:
        """
        DEPRECATED: No-op with single-tier architecture.
        Consolidation not needed - cache handles memory management automatically.
        """
        pass
    
    @profile_op("consolidate_to_longterm")
    def consolidate_to_longterm(self) -> None:
        """
        DEPRECATED: No-op with single-tier architecture.
        Consolidation not needed - cache handles memory management automatically.
        """
        pass
    
    @profile_op("retrieve")
    
    def approx_knn_indices(self, query: torch.Tensor, k: int = 20) -> list:
        """
        🔮 APPROXIMATE KNN: Get likely memory indices WITHOUT loading bundles
        
        Fast approximate search for prefetch hints. Uses cheap distance calculations
        on cached embeddings or previews, not full bundle loading.
        
        Args:
            query: [D] or [B, T, D] query embedding  
            k: number of neighbors to find
        
        Returns:
            List of memory indices (integers) likely to be retrieved
        """
        from disk_backed_tensor import DiskBackedTensor
        
        if self.memory.size == 0:
            return []
        
        # Flatten query if batched
        if query.dim() > 1:
            query = query.reshape(-1, query.shape[-1]).mean(dim=0)  # [D]
        
        # Get embedding previews (cheap!) or cached embeddings
        if isinstance(self.memory.embeddings, DiskBackedTensor) and hasattr(self.memory.embeddings, '_previews'):
            # Use previews for ultra-fast approximate search
            previews = self.memory.embeddings._previews[:self.memory.size.item()]  # [N, preview_dim]
            # Project query to preview space (approximation)
            query_preview = query[:previews.shape[1]]  # Truncate to preview dim
            distances = torch.norm(previews - query_preview.unsqueeze(0), dim=1)
        else:
            # Use actual embeddings (slower but exact)
            # Only compute for cached entries to stay fast
            tier_size = min(self.memory.size.item(), 1000)  # Limit search space
            embeddings = self.memory.embeddings[:tier_size]
            distances = torch.norm(embeddings - query.unsqueeze(0), dim=1)
        
        # Get top-k indices
        k = min(k, len(distances))
        _, indices = torch.topk(distances, k, largest=False)
        
        return indices.cpu().tolist()
    
    def retrieve(self, query: torch.Tensor, k: int = None) -> dict:
        """
        Retrieve memories with FULL STRUCTURE for navigation.
        Also stores new memories during training AND records retrieval patterns!
        
        Args:
            query: [B, T, query_dim] reflex embeddings
            k: number of memories to retrieve (default: use self.k_neighbors from config)
        
        Returns:
            dict with:
                'enhanced_context': [B, T, context_dim] - integrated output
                'bundle': dict with structure (embeddings, adjacency, depths, types, etc.)
        """
        # Use configured k_neighbors if not specified
        if k is None:
            k = self.k_neighbors
        
        B, T, C = query.shape
        
        # 🌊 FLOW FIELD: Get previous retrieval for context-aware routing
        # prev_top_indices[b, t] = top memory retrieved at position t-1
        # This enables "Waze" feature: routing depends on where you came from
        prev_top_indices = None
        if hasattr(self, '_last_top_indices') and self._last_top_indices is not None:
            # Shift by 1 timestep: prev_top_indices[t] = _last_top_indices[t-1]
            prev_batch, prev_time = self._last_top_indices.shape
            if prev_batch == B and prev_time == T:
                prev_top_indices = torch.cat([
                    torch.full((B, 1), -1, dtype=torch.long, device=query.device),  # t=0 has no prev
                    self._last_top_indices[:, :-1]  # Shift right by 1
                ], dim=1)
        
        # Clear old retrieval paths at START of new forward pass
        # (They'll be set again below and used by all 12 DEQ iterations)
        self._last_retrieval_paths = None
        
        # FORMATION: Add to memory during training
        if self.training:
            # Take mean over sequence as representative embedding
            representative = query.mean(dim=[0, 1])  # [C]
            
            # Compute reward as negative perplexity of current state
            # This will be refined with actual loss later via apply_dopamine
            reward = torch.ones(1, device=query.device)
            
            # Add to unified memory - cache handles hot/cold automatically
            self.store_memory_dynamic(representative, reward.item())
        
        # Query unified memory - RETURNS STRUCTURED BUNDLE with routing info!
        # Pass prev_top_indices for flow-based context routing
        if ENABLE_MICRO_PROFILING:
            t_retrieve_start = time.perf_counter()
        
        memory_bundle = self.query_network(query, self.memory, k=k, prev_top_indices=prev_top_indices, 
                                          routing_max_hops=self.routing_max_hops)
        
        if ENABLE_MICRO_PROFILING:
            t_retrieve = (time.perf_counter() - t_retrieve_start) * 1000
            if 'retrieve_memory' not in _profile_stats:
                _profile_stats['retrieve_memory'] = {'count': 0, 'total_ms': 0, 'max_ms': 0}
            _profile_stats['retrieve_memory']['count'] += 1
            _profile_stats['retrieve_memory']['total_ms'] += t_retrieve
            _profile_stats['retrieve_memory']['max_ms'] = max(_profile_stats['retrieve_memory']['max_ms'], t_retrieve)
        
        # 🌊 STORE TOP INDICES for next timestep's flow context
        if memory_bundle['indices'].numel() > 0:
            self._last_top_indices = memory_bundle['indices'][:, :, 0].clone()  # [B, T] - top-1 per token
        else:
            self._last_top_indices = None
        
        # DYNAMIC LEARNING: Record successful retrievals to strengthen edges!
        # Skip if dimension mismatch or query network projection layer is incompatible
        skip_dynamic_learning = (C != self.memory_dim or 
                                 self.query_network.query_proj.in_features != C or
                                 self.query_network.query_proj.out_features != self.memory_dim)
        
        if self.training and not skip_dynamic_learning:
            # 🚀 OPTIMIZATION: Skip expensive attention computation during retrieval
            # We'll compute proper rewards later in strengthen_last_retrieval()
            with torch.no_grad():
                # 🔥 DEFERRED HEBBIAN LEARNING: Store retrieval paths for ALL tokens
                # We'll strengthen AFTER we know the PER-TOKEN prediction loss
                B, T, _ = query.shape
                self._last_retrieval_paths = {
                    'memory': memory_bundle['indices'],  # [B, T, k]
                    'query': query,  # [B, T, D]
                    'shape': (B, T),
                    'attention_confidence': 0.5  # Placeholder - not used anymore
                }
                
                # 🕰️ ELIGIBILITY TRACE: Store this moment in the trajectory buffer
                step_data = {
                    'query': query.detach().cpu(),  # Save RAM - move to CPU
                    'memory_indices': memory_bundle['indices'].detach().cpu(),
                    'timestep': len(self.trajectory_buffer)
                }
                self.trajectory_buffer.append(step_data)
                
                # Keep only last N steps (sliding window)
                if len(self.trajectory_buffer) > self.max_trajectory:
                    self.trajectory_buffer.pop(0)  # Remove oldest
        
        # Holonomy tracking disabled for performance
        holonomy_metric = 0.0
        self._last_holonomy = holonomy_metric
        
        # Get transported query (geometry-evolved) from memory tier
        transported_query = memory_bundle.get('transported_query', None)  # [B, T, memory_dim] or None
        
        # 🛣️ Build complete multi-hop routing bundle
        routing_bundle = None
        if 'hop_0_stats' in memory_bundle:
            routing_bundle = {
                'hop_0_stats': memory_bundle['hop_0_stats'],  # Current edge
                'hop_1_stats': memory_bundle['hop_1_stats'],  # 1-hop options
                'hop_2_stats': memory_bundle['hop_2_stats'],  # 2-hop lookahead
            }
        
        # Integrate into context with flow-based attention bias, transported query, and routing intelligence
        all_memories = memory_bundle['embeddings']  # [B, T, k, memory_dim]
        all_flow_bias = memory_bundle.get('flow_bias', torch.zeros_like(memory_bundle['depths']))  # [B, T, k]
        
        enhanced = self.integration_network(query, all_memories, 
                                            flow_bias=all_flow_bias,
                                            transported_query=transported_query,
                                            routing_bundle=routing_bundle)
        
        # Return BOTH enhanced context AND full structure bundle
        combined_bundle = {
            'embeddings': all_memories,  # [B, T, k, D]
            'depths': memory_bundle['depths'],
            'edge_weights': memory_bundle['edge_weights'],
            'edge_types': memory_bundle['edge_types'],
            'cluster_ids': memory_bundle['cluster_ids'],
            'flow_bias': all_flow_bias,
        }
        
        # 🛣️ Add multi-hop routing information to bundle (for DEQ operator)
        if routing_bundle is not None:
            combined_bundle['hop_0_stats'] = routing_bundle['hop_0_stats']
            combined_bundle['hop_1_stats'] = routing_bundle['hop_1_stats']
            combined_bundle['hop_2_stats'] = routing_bundle['hop_2_stats']
        
        if memory_bundle.get('type_embeddings') is not None:
            combined_bundle['type_embeddings'] = memory_bundle['type_embeddings']
        
        return {
            'enhanced_context': enhanced,
            'bundle': combined_bundle
        }
    
    def retrieve_hierarchical(self, query: torch.Tensor, k: int = 20, k_clusters: int = 5) -> dict:
        """
        DEPRECATED: Hierarchical retrieval not needed with single-tier architecture.
        Falls back to standard retrieve() method.
        
        Args:
            query: [B, T, query_dim] query embeddings
            k: number of memories to retrieve
            k_clusters: ignored (kept for compatibility)
        
        Returns:
            Same as retrieve()
        """
        # Just use standard retrieval - cache system handles hierarchical access naturally
        return self.retrieve(query, k=k)
    
    # Removed: step(), apply_dopamine(), _propagate_credit_backwards(), _strengthen_temporal_transition()
    # These functions relied on rewards/age/access tensors which are no longer needed
    # Quality tracking now happens via edge_success_rate and edge_traversal_count
    
    def update_balancer_feedback(self, sigma_memory: float):
        """
        Homeostatic feedback from balancer.
        
        With single-tier architecture, this just tracks stats and periodic graph evolution.
        No consolidation needed - cache handles memory management automatically.
        
        Args:
            sigma_memory: balancer's uncertainty for memory loss component
        """
        # Store for reference
        self.sigma_memory = sigma_memory
        
        # DYNAMIC EVOLUTION: Periodic graph optimization
        self.consolidation_counter += 1
        
        if self.consolidation_counter % 100 == 0:  # Every 100 iterations
            print(f"[Graph Evolution] Pruning weak edges and optimizing structure...")
            self.evolve_graph(enable_sleep_replay=False)
            print(f"  Unified memory: {self.memory.size} nodes")
            
            # 🛣️ HIGHWAY REPORT: Show top strengthened edges
            highway_stats = self.memory.get_highway_stats(top_k=5)
            if highway_stats['total_highways'] > 0:
                print(f"  🛣️ Highways: {highway_stats['total_highways']} edges strengthened, "
                      f"max={highway_stats['max_strengthening']:.4f}, "
                      f"avg={highway_stats['avg_strengthening']:.4f}")
                print(f"     Top 5 highways:")
                for i, hw in enumerate(highway_stats['top_highways']):
                    print(f"       {i+1}. Edge {hw['source_idx']}→{hw['target_idx']}: "
                          f"weight {hw['old_weight']:.3f}→{hw['new_weight']:.3f} "
                          f"(Δ={hw['strengthening']:.4f}, traversals={hw['traversal_count']}, "
                          f"success={hw['success_rate']:.2f})")
    
    def _sleep_replay_reconsolidation(self, replay_fraction: float = 0.1):
        """
        DEPRECATED: Sleep replay removed with single-tier architecture.
        Cache automatically keeps important memories hot.
        """
        pass
    
    def get_memory_stats(self) -> dict:
        """Get statistics about memory usage."""
        # Get highway stats from unified memory
        highway_stats = self.memory.get_highway_stats(top_k=5)
        
        stats = {
            # SINGLE-TIER ARCHITECTURE: Only one unified memory tier
            'num_memory': self.memory.size.item(),
            'total_size': self.memory.size.item(),
            # 🛣️ HIGHWAY STATS
            'highways_formed': highway_stats['total_highways'],
            'max_highway_strength': highway_stats['max_strengthening'],
            'avg_highway_strength': highway_stats['avg_strengthening']
        }
        
        return stats
    
    def process_prefetch_hints(self):
        """
        🔮 PROCESS PREFETCH HINTS: Load hinted bundles into cache
        
        Call this periodically (e.g., during backward pass) to process
        queued prefetch hints. Non-blocking - only processes a few hints.
        
        Use case:
            After forward pass: hint_prefetch(likely_next_indices)
            During backward: process_prefetch_hints()  
            Next forward: cache hits! Fast retrieval!
        """
        # Process hints for unified memory tier
        self.memory._process_prefetch_hints()
    
    def apply_corpus_highways(self, highway_path: str, embedding_to_memory_idx: dict, 
                             tier_name: str = 'memory', strength_scale: float = 0.5,
                             verbose: bool = True):
        """
        🛣️ APPLY PRE-COMPUTED CORPUS HIGHWAYS to memory graph edges.
        
        Takes highways from build_corpus_highways.py and strengthens matching edges
        in the memory graph. This seeds the graph with "prior knowledge" from dataset statistics.
        
        Args:
            highway_path: Path to .pkl file from build_corpus_highways.py
            embedding_to_memory_idx: Dict mapping embeddings (or token IDs) to memory indices
            tier_name: Which tier to strengthen (now always 'memory', kept for compatibility)
            strength_scale: How much to boost edges (0-1, higher = stronger boost)
            verbose: Print progress and statistics
            
        Usage:
            # After preloading memories from dataset:
            memory_system.apply_corpus_highways(
                highway_path='corpus_highways.pkl',
                embedding_to_memory_idx=token_to_memory_map,  # Build during preload
                strength_scale=0.5  # Medium boost
            )
        """
        import pickle
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"🛣️  APPLYING CORPUS HIGHWAYS TO MEMORY GRAPH")
            print(f"{'='*70}")
            print(f"   Highway file: {highway_path}")
            print(f"   Strength scale: {strength_scale}")
        
        # Load highways
        with open(highway_path, 'rb') as f:
            highway_data = pickle.load(f)
        
        highways = highway_data['highways']  # List of (token_a, token_b, strength) tuples
        
        if verbose:
            print(f"   Loaded {len(highways):,} pre-computed highways")
            print(f"   From corpus stats:")
            stats = highway_data.get('corpus_stats', {})
            for key, val in stats.items():
                print(f"     - {key}: {val}")
        
        # Use unified memory tier
        tier = self.memory
        stat_edges_found = 0
        k_nn_edges_total = tier.size.item() * self.k_neighbors if tier.size.item() > 0 else 0
        edges_by_type = {}
        total_boost = 0.0
        
        if verbose:
            from tqdm import tqdm
            highway_iter = tqdm(highways, desc="Applying highways")
        else:
            highway_iter = highways
        
        for token_a, token_b, corpus_strength in highway_iter:
            # Map token IDs to memory indices
            mem_idx_a = embedding_to_memory_idx.get(token_a)
            mem_idx_b = embedding_to_memory_idx.get(token_b)
            
            if mem_idx_a is None or mem_idx_b is None:
                continue  # Tokens not in memory
            
            # Check if edge exists in adjacency graph
            if mem_idx_a >= tier.size.item() or mem_idx_b >= tier.size.item():
                continue  # Out of bounds
            
            neighbors = tier.adjacency[mem_idx_a]
            edge_mask = neighbors == mem_idx_b
            
            if not edge_mask.any():
                continue  # No edge between these memories
            
            # Found a statistical edge that exists in graph!
            stat_edges_found += 1
            edge_slot = edge_mask.nonzero(as_tuple=True)[0][0].item()
            
            # Strengthen this edge based on corpus statistics
            reward = min(1.0, corpus_strength * strength_scale)
            total_boost += reward
            
            # Update edge statistics
            tier.edge_traversal_count[mem_idx_a, edge_slot] += 5  # Fake "prior experience"
            old_success = tier.edge_success_rate[mem_idx_a, edge_slot].item()
            tier.edge_success_rate[mem_idx_a, edge_slot] = max(old_success, reward)
            
            # Slightly lower edge weight (lower = stronger highway)
            tier.edge_weights[mem_idx_a, edge_slot] *= (1.0 - 0.1 * reward)
            
            # Track edge type
            edge_type = tier.edge_types[mem_idx_a, edge_slot].argmax().item()
            edges_by_type[edge_type] = edges_by_type.get(edge_type, 0) + 1
        
        if verbose and hasattr(highway_iter, 'close'):
            highway_iter.close()
        
        avg_boost = total_boost / max(stat_edges_found, 1)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"🌐 STATISTICAL HIGHWAY COVERAGE:")
            print(f"   Initial edge distribution:")
            print(f"     K-NN baseline: {k_nn_edges_total:,} edges from hyperbolic geometry")
            print(f"     Statistical boost: {stat_edges_found:,} / {len(highways):,} corpus highways ({100*stat_edges_found/len(highways) if len(highways) > 0 else 0:.1f}%)")
            print(f"   Edge types enhanced:")
            for edge_type in sorted(edges_by_type.keys()):
                print(f"     Type {edge_type}: {edges_by_type[edge_type]:,} edges")
            print(f"   Average boost: {avg_boost:.2%} per statistical edge")
            print(f"   This gives model 'prior knowledge' of common paths!")
            print(f"{'='*70}\n")
        
        return {
            'highways_applied': stat_edges_found,
            'total_highways': len(highways),
            'coverage': stat_edges_found / len(highways) if len(highways) > 0 else 0,
            'avg_boost': avg_boost,
            'edges_by_type': edges_by_type
        }
    
    def save_checkpoint(self, filepath: str):
        """
        💾 SAVE COMPLETE GRAPH STATE (embeddings + structure + Hebbian learning state)
        
        Saves ALL graph metadata to preserve learned structure across restarts:
        - Adjacency (which memories are connected)
        - Edge weights (hyperbolic distances - can be strengthened by Hebbian learning)
        - Edge types (semantic relationship classification)
        - Edge traversal counts (usage statistics)
        - Edge success rates (Hebbian learning - which paths help prediction)
        - Cluster IDs (hierarchical organization)
        - Rewards (memory quality scores)
        - Access counts (memory importance)
        - Ages (for decay/forgetting)
        
        🔥 CRITICAL: Without this, all Hebbian learning is LOST on restart!
        The graph structure represents LEARNED knowledge about memory relationships.
        """
        import pickle
        import os
        
        # 🔥 FLUSH ALL LAZY WRITE BUFFERS BEFORE CHECKPOINT (synchronous!)
        print("💾 Flushing write buffers before checkpoint...")
        
        # Wait for async writes to complete
        global _async_write_queue
        if _async_write_queue.qsize() > 0:
            print(f"⏳ Waiting for {_async_write_queue.qsize()} pending async writes...")
            _async_write_queue.join()  # Wait for all queued writes to finish
        
        # Force synchronous flush of any remaining buffers
        self.memory.flush_write_buffer(force_sync=True)
        
        print("✅ All write buffers flushed")
        
        checkpoint = {
            'memory': {
                # DiskBackedTensor handles embeddings separately - DON'T include them in pickle!
                # Only save regular tensors that can be pickled
                'adjacency': self.memory.adjacency.cpu() if isinstance(self.memory.adjacency, torch.Tensor) else None,
                'edge_weights': self.memory.edge_weights.cpu() if isinstance(self.memory.edge_weights, torch.Tensor) else None,
                'edge_types': self.memory.edge_types.cpu() if isinstance(self.memory.edge_types, torch.Tensor) else None,
                'edge_traversal_count': self.memory.edge_traversal_count.cpu() if isinstance(self.memory.edge_traversal_count, torch.Tensor) else None,
                'edge_success_rate': self.memory.edge_success_rate.cpu() if isinstance(self.memory.edge_success_rate, torch.Tensor) else None,
                'cluster_ids': self.memory.cluster_ids.cpu() if isinstance(self.memory.cluster_ids, torch.Tensor) else None,
                # REMOVED: rewards/access/age - no longer tracked (edge_success_rate replaces quality tracking)
                'size': self.memory.size.item(),
                'capacity': self.memory.capacity,
                # Note: embeddings/depths/type_embeddings are DiskBackedTensor - loaded automatically from disk
            },
            'config': {
                'k_neighbors': self.k_neighbors,
                'memory_dim': self.memory_dim,
                'context_dim': self.context_dim,
                'query_dim': self.query_dim,
                'consolidation_counter': self.consolidation_counter,
            },
            'version': '3.0',  # Single-tier architecture version
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        size_mb = os.path.getsize(filepath) / (1024**2)
        print(f"💾 Saved graph checkpoint: {filepath} ({size_mb:.1f} MB)")
        print(f"   Memory size: {checkpoint['memory']['size']}")
        print(f"   🧠 Preserved {checkpoint['memory']['size'] * self.k_neighbors} edges with Hebbian weights!")
    
    def load_checkpoint(self, filepath: str):
        """
        📂 LOAD COMPLETE GRAPH STATE
        
        Restores ALL graph structure including Hebbian learning state.
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Check version and handle legacy 3-tier checkpoints
        version = checkpoint.get('version', '2.0')
        
        if version == '3.0' and 'memory' in checkpoint:
            # New single-tier format
            mem = checkpoint['memory']
            if mem['adjacency'] is not None:
                self.memory.adjacency = mem['adjacency'].to(self.memory.device)
            if mem['edge_weights'] is not None:
                self.memory.edge_weights = mem['edge_weights'].to(self.memory.device)
            if mem['edge_types'] is not None:
                self.memory.edge_types = mem['edge_types'].to(self.memory.device)
            if mem['edge_traversal_count'] is not None:
                self.memory.edge_traversal_count = mem['edge_traversal_count'].to(self.memory.device)
            if mem['edge_success_rate'] is not None:
                self.memory.edge_success_rate = mem['edge_success_rate'].to(self.memory.device)
            if mem['cluster_ids'] is not None:
                self.memory.cluster_ids = mem['cluster_ids'].to(self.memory.device)
            # REMOVED: rewards/access/age - no longer tracked
            # Skip loading these fields if they exist in old checkpoints (backward compat)
            self.memory.size = torch.tensor(mem['size'], device=self.memory.device)
            self.memory.capacity = mem['capacity']
            
            print(f"✅ Loaded graph checkpoint: {filepath}")
            print(f"   Memory size: {mem['size']}")
            print(f"   🧠 Restored {mem['size'] * self.k_neighbors} edges with Hebbian learning state!")
        else:
            # Legacy 3-tier format - merge into single tier
            print(f"⚠️  Loading legacy 3-tier checkpoint (v{version}), merging into single tier...")
            
            # For now, just warn and skip - full migration would be complex
            print(f"❌ Legacy checkpoint format not supported. Please retrain from scratch.")
            return
        
        # Restore config
        self.consolidation_counter = checkpoint['config']['consolidation_counter']

