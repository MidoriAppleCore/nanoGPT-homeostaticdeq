"""
Disk-Backed Tensor - Transparent virtual memory for PyTorch tensors.

Acts like a normal tensor, but automatically pages data in/out from disk.
The API is completely transparent - just index it like embeddings[i] and it works!

Features:
- Write-back caching: Writes go to RAM immediately, disk I/O is async
- Hyperbolic eviction: Keeps semantically related memories hot
- Non-blocking: Training never waits for disk
"""

import torch
import os
import threading
import queue
import time
from typing import Optional, Tuple
from hyperbolic_cache import HyperbolicCache


class DiskBackedTensor:
    """
    A tensor that transparently pages data to/from disk with write-back caching.
    
    Strategy:
    1. Reads: Check RAM cache → load from disk if needed (blocks)
    2. Writes: Write to RAM cache immediately (non-blocking), queue for disk
    3. Background thread: Flushes dirty cache entries to disk asynchronously
    4. Eviction: Use hyperbolic distance to evict cold memories
    
    Usage:
        tensor = DiskBackedTensor(
            shape=(50000, 128),
            dtype=torch.float32,
            device='cpu',
            disk_path='./memories',
            hot_capacity=1000,  # Keep 1K hot in RAM
            flush_interval=5.0  # Flush every 5 seconds
        )
        
        # Just use it like a normal tensor!
        tensor[100] = some_embedding  # Instant (RAM write)
        emb = tensor[100]  # Fast if cached, loads from disk if needed
        batch = tensor[0:500]  # Batch access
        
        # Shutdown (flushes pending writes)
        tensor.shutdown()
    """
    
    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu',
        disk_path: str = None,
        hot_capacity: int = 1000,
        poincare = None,
        flush_interval: float = 5.0,  # Flush to disk every N seconds
        enable_async: bool = True  # Enable background flushing
    ):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.disk_path = disk_path
        self.hot_capacity = hot_capacity
        self.flush_interval = flush_interval
        self.enable_async = enable_async
        
        # Total logical size
        self.total_size = shape[0]
        self.row_shape = shape[1:]
        
        # Current actual size (grows as you add data)
        self._actual_size = 0
        
        # Hot cache with hyperbolic eviction
        if poincare is not None:
            self.cache = HyperbolicCache(capacity=hot_capacity, poincare=poincare)
        else:
            # Fallback to simple dict if no poincare manifold
            self.cache = {}
            self._cache_order = []  # LRU tracking
        
        # Dirty tracking for write-back cache
        self._dirty = set()  # Indices that need to be written to disk
        self._write_lock = threading.Lock()  # Protect cache writes
        
        # Background flush thread
        self._flush_queue = queue.Queue()
        self._shutdown_event = threading.Event()
        if enable_async and disk_path:
            self._flush_thread = threading.Thread(target=self._background_flusher, daemon=True)
            self._flush_thread.start()
        else:
            self._flush_thread = None
        
        # Disk storage
        if disk_path:
            os.makedirs(disk_path, exist_ok=True)
            self.disk_file = os.path.join(disk_path, 'tensor_disk.pt')
            
            # Load existing data if present
            if os.path.exists(self.disk_file):
                self._load_metadata()
            else:
                # Initialize empty disk storage
                self.disk_data = {
                    'embeddings': torch.zeros(0, *self.row_shape, dtype=dtype),
                    'valid_mask': torch.zeros(0, dtype=torch.bool)
                }
        else:
            self.disk_file = None
            self.disk_data = None
    
    def __len__(self):
        """Return the logical size (not actual size)."""
        return self._actual_size
    
    def size(self, dim: Optional[int] = None):
        """PyTorch-compatible size() method."""
        if dim is None:
            return torch.Size([self._actual_size, *self.row_shape])
        elif dim == 0:
            return self._actual_size
        else:
            return self.row_shape[dim - 1]
    
    def __getitem__(self, idx):
        """
        Transparent access - loads from disk if needed.
        
        Supports:
        - Single index: tensor[5]
        - Slice: tensor[0:100]
        - List: tensor[[1,2,3]]
        - Tensor: tensor[torch.tensor([1,2,3])]
        - Tuple: tensor[5, :10] or tensor[0:100, :]
        """
        if isinstance(idx, tuple):
            # Handle tuple indexing like tensor[row, col_slice] or tensor[i, j, k]
            row_idx = idx[0]
            col_indices = idx[1:] if len(idx) > 1 else (slice(None),)
            
            # Get the row(s) first
            if isinstance(row_idx, int):
                row_data = self._get_single(row_idx)
            elif isinstance(row_idx, slice):
                row_data = self._get_slice(row_idx)
            elif isinstance(row_idx, (list, torch.Tensor)):
                row_data = self._get_batch(row_idx)
            else:
                raise TypeError(f"Unsupported row index type: {type(row_idx)}")
            
            # Apply remaining indices (unpack tuple for correct indexing)
            return row_data[col_indices] if len(col_indices) > 1 else row_data[col_indices[0]]
        elif isinstance(idx, int):
            return self._get_single(idx)
        elif isinstance(idx, slice):
            return self._get_slice(idx)
        elif isinstance(idx, (list, torch.Tensor)):
            return self._get_batch(idx)
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")
    
    def _get_single(self, idx: int):
        """Get a single row, loading from disk if needed."""
        if idx >= self._actual_size:
            raise IndexError(f"Index {idx} out of range for size {self._actual_size}")
        
        # Check cache first (fast path - no disk I/O)
        with self._write_lock:
            if isinstance(self.cache, HyperbolicCache):
                cached = self.cache.get(idx)
            else:
                cached = self.cache.get(idx)
                if cached is not None and idx in self._cache_order:
                    # LRU update
                    self._cache_order.remove(idx)
                    self._cache_order.append(idx)
        
        if cached is not None:
            return cached.to(self.device)
        
        # Load from disk (slow path - blocks on I/O)
        if self.disk_data is not None and idx < len(self.disk_data['embeddings']):
            emb = self.disk_data['embeddings'][idx].clone()
            
            # Add to cache
            with self._write_lock:
                if isinstance(self.cache, HyperbolicCache):
                    self.cache.access(emb, key=idx)
                else:
                    self.cache[idx] = emb
                    self._cache_order.append(idx)
                    # Simple LRU eviction for fallback
                    if len(self.cache) > self.hot_capacity:
                        evict_idx = self._cache_order.pop(0)
                        # Don't evict dirty entries!
                        if evict_idx not in self._dirty:
                            del self.cache[evict_idx]
            
            return emb.to(self.device)
        
        # Not found
        return torch.zeros(self.row_shape, dtype=self.dtype, device=self.device)
    
    def _get_slice(self, s: slice):
        """Get a slice of rows."""
        start, stop, step = s.indices(self._actual_size)
        indices = list(range(start, stop, step or 1))
        
        # Safety: Return empty tensor if slice is empty
        if len(indices) == 0:
            return torch.zeros((0,) + self.row_shape, dtype=self.dtype, device=self.device)
        
        return torch.stack([self._get_single(i) for i in indices])
    
    def _get_batch(self, indices):
        """Get multiple rows by index list."""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return torch.stack([self._get_single(i) for i in indices])
    
    def __setitem__(self, idx, value):
        """
        Transparent write - automatically expands storage and flushes to disk.
        """
        if isinstance(idx, tuple):
            # Handle multi-dimensional indexing: tensor[i, j] or tensor[i, j, k] = value
            # Read-modify-write pattern to preserve row
            row_idx = idx[0]
            col_indices = idx[1:]  # Remaining indices (might be multiple for 3D+)
            row = self._get_single(row_idx).clone()  # Load full row
            row[col_indices] = value  # Modify using remaining indices
            self._set_single(row_idx, row)  # Write back full row
        elif isinstance(idx, int):
            self._set_single(idx, value)
        elif isinstance(idx, slice):
            self._set_slice(idx, value)
        elif isinstance(idx, (list, torch.Tensor)):
            self._set_batch(idx, value)
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")
    
    def _set_single(self, idx: int, value: torch.Tensor):
        """
        Set a single row - INSTANT write to RAM cache, async flush to disk.
        
        This is the key to non-blocking writes:
        1. Write to cache immediately (fast)
        2. Mark as dirty
        3. Background thread flushes to disk later
        """
        # Ensure value is correct shape and type
        value = value.detach().to(dtype=self.dtype, device='cpu')
        
        # Expand actual size if needed
        if idx >= self._actual_size:
            self._actual_size = idx + 1
        
        # INSTANT write to cache (no blocking!)
        with self._write_lock:
            if isinstance(self.cache, HyperbolicCache):
                self.cache.access(value, key=idx)
                
                # Trigger eviction if cache getting full
                if len(self.cache) > self.hot_capacity * 0.9:
                    # Only evict clean (non-dirty) entries
                    self.cache.evict_farthest(value, num_to_evict=max(1, len(self.cache) // 10))
            else:
                self.cache[idx] = value
                if idx not in self._cache_order:
                    self._cache_order.append(idx)
                if len(self.cache) > self.hot_capacity:
                    # LRU eviction - skip dirty entries
                    for evict_idx in list(self._cache_order):
                        if evict_idx not in self._dirty:
                            self._cache_order.remove(evict_idx)
                            del self.cache[evict_idx]
                            break
            
            # Mark as dirty (needs disk write)
            self._dirty.add(idx)
        
        # Queue for background flush (non-blocking!)
        if self._flush_thread is not None:
            self._flush_queue.put(('write', idx))
        elif len(self._dirty) > 100:
            # No background thread - flush synchronously when buffer fills
            self.flush()
    
    def _set_slice(self, s: slice, value: torch.Tensor):
        """Set a slice of rows."""
        start, stop, step = s.indices(self.total_size)
        indices = range(start, stop, step or 1)
        for i, idx in enumerate(indices):
            self._set_single(idx, value[i])
    
    def _set_batch(self, indices, value: torch.Tensor):
        """Set multiple rows."""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        for i, idx in enumerate(indices):
            self._set_single(idx, value[i])
    
    def flush(self):
        """
        Flush dirty cache entries to disk.
        
        This can be called manually or by the background thread.
        Uses atomic operations to avoid race conditions.
        """
        if self.disk_file is None:
            return
        
        # Get snapshot of dirty indices AND current size atomically
        with self._write_lock:
            if not self._dirty:
                return
            dirty_snapshot = set(self._dirty)
            current_size = self._actual_size  # Capture size under lock
        
        # Load existing disk data
        if os.path.exists(self.disk_file):
            try:
                self.disk_data = torch.load(self.disk_file, map_location='cpu', weights_only=True)
                
                # Validate loaded data has correct structure
                if 'embeddings' not in self.disk_data or 'valid_mask' not in self.disk_data:
                    raise ValueError("Corrupted disk data - missing keys")
                    
                # Ensure embeddings and mask sizes match (fix corruption)
                emb_size = self.disk_data['embeddings'].size(0)
                mask_size = self.disk_data['valid_mask'].size(0)
                if emb_size != mask_size:
                    print(f"⚠️  Size mismatch in disk data: emb={emb_size}, mask={mask_size}. Fixing...")
                    # Resize mask to match embeddings
                    new_mask = torch.zeros(emb_size, dtype=torch.bool)
                    copy_size = min(emb_size, mask_size)
                    new_mask[:copy_size] = self.disk_data['valid_mask'][:copy_size]
                    self.disk_data['valid_mask'] = new_mask
                    
            except Exception as e:
                print(f"⚠️  Failed to load disk data: {e}. Reinitializing...")
                # Corrupted file - reinitialize
                self.disk_data = {
                    'embeddings': torch.zeros(0, *self.row_shape, dtype=self.dtype),
                    'valid_mask': torch.zeros(0, dtype=torch.bool)
                }
        
        # Determine the actual size we need (max of current_size and any dirty indices)
        max_dirty_idx = max(dirty_snapshot) if dirty_snapshot else 0
        required_size = max(current_size, max_dirty_idx + 1, self.disk_data['embeddings'].size(0))
        
        # Expand disk storage if needed
        if self.disk_data['embeddings'].size(0) < required_size:
            new_embeddings = torch.zeros(required_size, *self.row_shape, dtype=self.dtype)
            new_mask = torch.zeros(required_size, dtype=torch.bool)
            
            old_size = self.disk_data['embeddings'].size(0)
            old_mask_size = self.disk_data['valid_mask'].size(0)
            
            if old_size > 0:
                # Copy existing embeddings (use actual old_size, not required_size)
                copy_size = min(old_size, required_size)
                new_embeddings[:copy_size] = self.disk_data['embeddings'][:copy_size]
                
                # Copy existing mask (handle size mismatch between embeddings and mask)
                mask_copy_size = min(old_mask_size, copy_size, required_size)
                new_mask[:mask_copy_size] = self.disk_data['valid_mask'][:mask_copy_size]
            
            self.disk_data['embeddings'] = new_embeddings
            self.disk_data['valid_mask'] = new_mask
        
        # Write dirty entries from cache to disk (bounds-checked)
        with self._write_lock:
            for idx in dirty_snapshot:
                # Skip if index is somehow out of bounds
                if idx >= self.disk_data['embeddings'].size(0):
                    continue
                    
                if isinstance(self.cache, HyperbolicCache):
                    cached = self.cache.get(idx)
                else:
                    cached = self.cache.get(idx)
                
                if cached is not None:
                    self.disk_data['embeddings'][idx] = cached.cpu()
                    self.disk_data['valid_mask'][idx] = True
        
        # Save to disk (this is the only blocking I/O)
        try:
            torch.save(self.disk_data, self.disk_file)
            
            # Clear dirty flags only after successful write
            with self._write_lock:
                self._dirty -= dirty_snapshot
        except Exception as e:
            print(f"⚠️  Disk write failed: {e}")
    
    def _background_flusher(self):
        """
        Background thread that periodically flushes dirty cache to disk.
        
        This runs in the background so training never waits for disk I/O!
        """
        last_flush = time.time()
        
        while not self._shutdown_event.is_set():
            try:
                # Wait for flush trigger or timeout
                msg = self._flush_queue.get(timeout=1.0)
                
                # Check if it's time to flush
                if time.time() - last_flush >= self.flush_interval:
                    self.flush()
                    last_flush = time.time()
                    
            except queue.Empty:
                # Timeout - check if we should flush anyway
                if time.time() - last_flush >= self.flush_interval:
                    self.flush()
                    last_flush = time.time()
        
        # Final flush on shutdown
        self.flush()
    
    def shutdown(self):
        """
        Shutdown the background flusher and ensure all data is written.
        
        Call this before exiting to avoid data loss!
        """
        if self._flush_thread is not None:
            self._shutdown_event.set()
            self._flush_thread.join(timeout=10.0)
        
        # Final flush
        self.flush()
    
    def _load_metadata(self):
        """Load metadata from disk on startup."""
        if not os.path.exists(self.disk_file):
            # No disk file yet - start fresh
            self._actual_size = 0
            self.disk_data = {
                'embeddings': torch.zeros(0, *self.row_shape, dtype=self.dtype),
                'valid_mask': torch.zeros(0, dtype=torch.bool)
            }
            return
        
        try:
            self.disk_data = torch.load(self.disk_file, map_location='cpu', weights_only=True)
            
            # Validate structure
            if 'embeddings' not in self.disk_data or 'valid_mask' not in self.disk_data:
                raise ValueError("Missing keys in disk data")
            
            # Validate sizes match
            emb_size = self.disk_data['embeddings'].size(0)
            mask_size = self.disk_data['valid_mask'].size(0)
            if emb_size != mask_size:
                print(f"⚠️  Disk data size mismatch (emb={emb_size}, mask={mask_size}). Fixing...")
                new_mask = torch.zeros(emb_size, dtype=torch.bool)
                copy_size = min(emb_size, mask_size)
                new_mask[:copy_size] = self.disk_data['valid_mask'][:copy_size]
                self.disk_data['valid_mask'] = new_mask
            
            self._actual_size = self.disk_data['valid_mask'].sum().item()
            
        except Exception as e:
            print(f"⚠️  Failed to load disk file {self.disk_file}: {e}")
            print(f"    Deleting corrupted file and starting fresh...")
            
            # Delete corrupted file
            try:
                os.remove(self.disk_file)
            except:
                pass
            
            # Start fresh
            self._actual_size = 0
            self.disk_data = {
                'embeddings': torch.zeros(0, *self.row_shape, dtype=self.dtype),
                'valid_mask': torch.zeros(0, dtype=torch.bool)
            }
    
    def to(self, device):
        """Change device (doesn't move disk data, only cache)."""
        self.device = device
        return self
    
    def dim(self):
        """Return number of dimensions."""
        return len(self.shape)
    
    @property
    def device_type(self):
        """PyTorch compatibility."""
        return torch.device(self.device)
