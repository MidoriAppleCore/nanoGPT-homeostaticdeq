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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

from hyperbolic_memory import PoincareManifold


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
                 max_disk_size: int = 100000):  # Max memories (hot + disk)
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
            embedding_cache = 100  # ~50KB for dim=128 (working set for one batch + neighbors)
            graph_cache = 50       # ~2KB (adjacency/weights for active nodes)
            
            print(f"💾 Disk-backed tier: {disk_path}")
            print(f"   Max size: {max_disk_size} memories")
            print(f"   Hot cache: {embedding_cache} entries per tensor (~{embedding_cache * memory_dim * 4 // 1024}KB)")
            print(f"   🔥 GRAPH STRUCTURE ON DISK (scales to millions!)")
            print(f"   Strategy: Transparent write-back cache with async I/O")
            
            # 🔥 DISK-BACKED EVERYTHING (enables scaling to millions of memories!)
            # Embeddings (larger cache for graph traversal efficiency)
            self.embeddings = DiskBackedTensor(
                shape=(max_disk_size, memory_dim),
                dtype=torch.float32,
                device=device,
                disk_path=os.path.join(disk_path, 'embeddings'),
                hot_capacity=embedding_cache,
                poincare=poincare,
                flush_interval=5.0,
                enable_async=True
            )
            
            # Graph structure (adjacency matrix)
            self.adjacency = DiskBackedTensor(
                shape=(max_disk_size, k_neighbors),
                dtype=torch.int64,
                device=device,
                disk_path=os.path.join(disk_path, 'adjacency'),
                hot_capacity=graph_cache,
                flush_interval=5.0,
                enable_async=True
            )
            
            # Edge weights (hyperbolic distances - Hebbian strengthened!)
            self.edge_weights = DiskBackedTensor(
                shape=(max_disk_size, k_neighbors),
                dtype=torch.float32,
                device=device,
                disk_path=os.path.join(disk_path, 'edge_weights'),
                hot_capacity=graph_cache,
                flush_interval=5.0,
                enable_async=True
            )
            
            # Edge types (semantic relationships)
            self.edge_types = DiskBackedTensor(
                shape=(max_disk_size, k_neighbors, num_edge_types),
                dtype=torch.float32,
                device=device,
                disk_path=os.path.join(disk_path, 'edge_types'),
                hot_capacity=graph_cache,
                flush_interval=5.0,
                enable_async=True
            )
            
            # 🧠 HEBBIAN LEARNING STATE (usage statistics)
            self.edge_traversal_count = DiskBackedTensor(
                shape=(max_disk_size, k_neighbors),
                dtype=torch.float32,
                device=device,
                disk_path=os.path.join(disk_path, 'edge_traversal_count'),
                hot_capacity=graph_cache,
                flush_interval=5.0,
                enable_async=True
            )
            
            self.edge_success_rate = DiskBackedTensor(
                shape=(max_disk_size, k_neighbors),
                dtype=torch.float32,
                device=device,
                disk_path=os.path.join(disk_path, 'edge_success_rate'),
                hot_capacity=graph_cache,
                flush_interval=5.0,
                enable_async=True
            )
            
            # Metadata - ALSO disk-backed for 200K+ scale!
            self.cluster_ids = DiskBackedTensor(
                shape=(max_disk_size,),
                dtype=torch.int64,
                device=device,
                disk_path=os.path.join(disk_path, 'cluster_ids'),
                hot_capacity=graph_cache,
                flush_interval=5.0,
                enable_async=True
            )
            
            self.depths = DiskBackedTensor(
                shape=(max_disk_size,),
                dtype=torch.float32,
                device=device,
                disk_path=os.path.join(disk_path, 'depths'),
                hot_capacity=graph_cache,
                flush_interval=5.0,
                enable_async=True
            )
            
            # TYPE SYSTEM: Continuous type embeddings (learned from context)
            if use_types:
                self.type_embeddings = DiskBackedTensor(
                    shape=(max_disk_size, type_dim),
                    dtype=torch.float32,
                    device=device,
                    disk_path=os.path.join(disk_path, 'type_embeddings'),
                    hot_capacity=graph_cache,
                    flush_interval=5.0,
                    enable_async=True
                )
            
            print(f"💾 Disk-backed tier: {disk_path}")
            print(f"   Max size: {max_disk_size} memories")
            print(f"   Hot cache: {graph_cache} in RAM per tensor")
            print(f"   🔥 GRAPH STRUCTURE ON DISK (scales to millions!)")
            print(f"   Strategy: Transparent write-back cache with async I/O")
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
        
        # Metadata
        self.register_buffer('rewards', torch.zeros(0, device=device))
        self.register_buffer('age', torch.zeros(0, device=device))
        self.register_buffer('access', torch.zeros(0, device=device))
        
        # Current size - always use a buffer, just access it differently
        self.register_buffer('_size', torch.tensor(0, dtype=torch.long, device=device))
        
        # 🛣️ HIGHWAY TRACKING: Monitor which edges are being strengthened
        # Keep top-K most strengthened edges for debugging
        self.highway_log = []  # List of (source_idx, target_idx, old_weight, new_weight, strengthening_amount)
        self.highway_log_max = 100  # Keep top 100 highways
    
    @property
    def size(self):
        """Get current size - compatible with both disk-backed and in-memory."""
        if self.use_disk:
            # For disk-backed: use _actual_size (not .shape which is max capacity)
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
            # For disk-backed, size is managed by DiskBackedTensor (read-only)
            pass
        else:
            if isinstance(value, torch.Tensor):
                self._size = value.to(self.device)
            else:
                self._size = torch.tensor(value, dtype=torch.long, device=self.device)
    
    def add_node_dynamic(self, embedding: torch.Tensor, poincare, cluster_id: int = -1, skip_disk_search: bool = False) -> int:
        """
        Add a single node and WIRE it into existing graph dynamically.
        
        This makes the graph GROW like a city - new buildings connect to nearby roads!
        
        Args:
            embedding: Memory vector
            poincare: Poincare manifold for hyperbolic distance
            cluster_id: Cluster assignment (-1 to infer from neighbors)
            skip_disk_search: If True, only search hot RAM (faster for bulk preload)
        
        Returns: index of new node
        """
        # CRITICAL: Ensure embedding is on the correct device
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
            dists_hot = poincare.distance(
                embedding.unsqueeze(0),  # [1, D]
                self.embeddings  # [N_hot, D]
            ).squeeze(-1).squeeze(0)  # [N_hot] - remove keepdim dimensions
            
            # Step 2: Approximate search on disk using previews (if disk enabled)
            disk_candidates = []
            dists_disk = []
            if self.use_disk and len(self.disk_index) > 0:
                # OPTIMIZATION: Search disk with preview for top candidates
                # Use 2x k to get a good pool, then refine with exact distances
                preview_candidates = self.search_disk_with_preview(embedding, k=min(self.k_neighbors * 2, len(self.disk_index)))
                
                # OPTIMIZATION: Cluster-aware loading - find dominant cluster in preview
                cluster_counts = {}
                for disk_idx in preview_candidates[:self.k_neighbors]:
                    cluster_id = self.disk_index[disk_idx].get('cluster_id', -1)
                    if cluster_id >= 0:
                        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
                
                # Prioritize candidates from the dominant cluster (hyperbolic locality)
                dominant_cluster = max(cluster_counts, key=cluster_counts.get) if cluster_counts else -1
                
                # Load and compute exact distances
                for disk_idx in preview_candidates[:self.k_neighbors]:
                    entry = self.disk_index[disk_idx]
                    filepath = entry['filepath']
                    
                    # Load from cache or disk
                    if filepath in self.hot_cache:
                        disk_emb = self.hot_cache[filepath]['embedding']
                    else:
                        data = torch.load(filepath, map_location=self.device)
                        self.hot_cache[filepath] = data
                        disk_emb = data['embedding']
                        
                        # OPTIMIZATION: Prefetch neighbors from same cluster
                        # If this memory is from dominant cluster, prefetch its neighbors too
                        if entry.get('cluster_id', -1) == dominant_cluster and 'adjacency' in data:
                            neighbor_adj = data['adjacency']
                            for neighbor_hot_idx in neighbor_adj:
                                if neighbor_hot_idx < 0:
                                    continue
                                # Find this neighbor in disk_index
                                for prefetch_idx, prefetch_entry in enumerate(self.disk_index):
                                    if prefetch_entry.get('hot_idx') == neighbor_hot_idx.item():
                                        prefetch_path = prefetch_entry['filepath']
                                        # Add to cache if not already present
                                        if prefetch_path not in self.hot_cache and len(self.hot_cache) < self.cache_capacity:
                                            prefetch_data = torch.load(prefetch_path, map_location=self.device)
                                            self.hot_cache[prefetch_path] = prefetch_data
                                        break
                        
                        # LRU eviction - keep cache size bounded
                        while len(self.hot_cache) > self.cache_capacity:
                            oldest_key = list(self.hot_cache.keys())[0]
                            del self.hot_cache[oldest_key]
                    
                    # Compute exact distance in hyperbolic space
                    dist = poincare.distance(
                        embedding.unsqueeze(0),
                        disk_emb.unsqueeze(0).to(self.device)
                    ).squeeze()
                    
                    disk_candidates.append(disk_idx)
                    dists_disk.append(dist)
            
            # Step 3: Merge hot and disk candidates, select true k-nearest
            all_dists = []
            all_sources = []  # 'hot' or 'disk'
            all_indices = []  # index into hot RAM or disk_index
            
            # Add hot memories
            if dists_hot.numel() > 0:
                if dists_hot.ndim == 0:  # 0-d tensor
                    all_dists.append(dists_hot.item())
                    all_sources.append('hot')
                    all_indices.append(0)
                else:
                    for i in range(dists_hot.size(0)):
                        all_dists.append(dists_hot[i].item())
                        all_sources.append('hot')
                        all_indices.append(i)
            
            # Add disk memories
            for i, dist in enumerate(dists_disk):
                all_dists.append(dist.item() if torch.is_tensor(dist) else dist)
                all_sources.append('disk')
                all_indices.append(disk_candidates[i])
            
            # Sort by distance and take top k
            if len(all_dists) > 0:
                sorted_pairs = sorted(zip(all_dists, all_sources, all_indices), key=lambda x: x[0])
                k_actual = min(self.k_neighbors, len(sorted_pairs))
                
                topk_dists = torch.tensor([x[0] for x in sorted_pairs[:k_actual]], device=self.device)
                topk_sources = [x[1] for x in sorted_pairs[:k_actual]]
                topk_indices = torch.tensor([x[2] for x in sorted_pairs[:k_actual]], dtype=torch.long, device=self.device)
            else:
                # No existing memories
                k_actual = 0
                topk_dists = torch.tensor([], device=self.device)
                topk_indices = torch.tensor([], dtype=torch.long, device=self.device)
                topk_sources = []
            
            # Ensure we can get neighbors
            num_existing = self.size.item()
            if num_existing > 0:
                k_actual = min(self.k_neighbors, len(topk_dists))
            else:
                k_actual = 0
            
            # Ensure tensors are at least 1-d (topk returns 0-d when k=1)
            if k_actual == 1:
                topk_dists = topk_dists.unsqueeze(0)
                topk_indices = topk_indices.unsqueeze(0)
            
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
                # CRITICAL: Only use HOT neighbors for type inference (disk neighbors not in type_embeddings)
                hot_neighbor_mask = torch.tensor([s == 'hot' for s in topk_sources[:k_actual]], device=self.device)
                hot_neighbor_indices = topk_indices[:k_actual][hot_neighbor_mask]
                
                if len(hot_neighbor_indices) > 0 and hot_neighbor_indices.max() < self.type_embeddings.size(0):
                    my_type = torch.zeros(self.type_dim, device=self.device)  # Will be inferred below
                    neighbor_types = self.type_embeddings[hot_neighbor_indices]  # [n_hot, type_dim]
                    # Cosine similarity of type vectors
                    type_sim = F.cosine_similarity(
                        my_type.unsqueeze(0).expand(len(hot_neighbor_indices), -1),
                        neighbor_types,
                        dim=-1
                    )  # [n_hot]
                    # High type similarity → same syntactic role (only mark hot neighbors)
                    hot_idx_in_topk = 0
                    for i in range(k_actual):
                        if topk_sources[i] == 'hot' and hot_idx_in_topk < len(type_sim):
                            new_edge_types[0, i, 1] = (type_sim[hot_idx_in_topk] > 0.7).float()
                            hot_idx_in_topk += 1
            
            # Type 2: SEMANTIC - infer from depth similarity (same level in hierarchy)
            if k_actual > 0:
                # Only compute for HOT neighbors (have access to embeddings)
                my_depth = embedding.norm()
                for i in range(k_actual):
                    if topk_sources[i] == 'hot':
                        neighbor_idx = topk_indices[i]
                        if neighbor_idx < self.embeddings.size(0):
                            neighbor_depth = self.embeddings[neighbor_idx].norm()
                            depth_diff = torch.abs(my_depth - neighbor_depth)
                            # Similar depth → semantic peers (synonyms, not hypernyms)
                            new_edge_types[0, i, 2] = (depth_diff < 0.1).float()
            
            # Type 3: SEQUENCE - will be learned from co-occurrence (initially 0)
            
            # CRITICAL: Bidirectional wiring - neighbors point back to new node!
            # Only wire to HOT neighbors (in RAM), not disk neighbors
            new_idx = self.size.item()
            for i, (neighbor_idx, dist, source) in enumerate(zip(topk_indices, topk_dists, topk_sources)):
                if source != 'hot':
                    # Neighbor is on disk - skip bidirectional wiring
                    # (it will be rewired when loaded back from disk)
                    continue
                
                # Check if new node is closer than neighbor's current furthest neighbor
                valid_mask = self.adjacency[neighbor_idx] >= 0
                if valid_mask.sum() < self.k_neighbors:
                    # Neighbor has free slots
                    free_slot = (~valid_mask).nonzero(as_tuple=True)[0][0]
                    self.adjacency[neighbor_idx, free_slot] = new_idx
                    self.edge_weights[neighbor_idx, free_slot] = dist
                elif dist < self.edge_weights[neighbor_idx].max():
                    # Replace furthest neighbor with new node
                    furthest_slot = self.edge_weights[neighbor_idx].argmax()
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
        
        # Add to tier - CYBERNETIC: Buffers must NOT be part of computation graph
        with torch.no_grad():
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
            
            # cluster_ids might be DiskBackedTensor
            from disk_backed_tensor import DiskBackedTensor
            if isinstance(self.cluster_ids, DiskBackedTensor):
                old_size = self.cluster_ids._actual_size
                self.cluster_ids[old_size] = new_cluster[0]
            else:
                self.cluster_ids = torch.cat([self.cluster_ids, new_cluster], dim=0)
            
            # rewards, age, access are always regular tensors
            self.rewards = torch.cat([self.rewards, torch.zeros(1, device=self.device)], dim=0)
            self.age = torch.cat([self.age, torch.zeros(1, device=self.device)], dim=0)
            self.access = torch.cat([self.access, torch.zeros(1, device=self.device)], dim=0)
            self.edge_traversal_count = torch.cat([self.edge_traversal_count, 
                                                   torch.zeros(1, self.k_neighbors, device=self.device)], dim=0)
            self.edge_success_rate = torch.cat([self.edge_success_rate, 
                                               torch.zeros(1, self.k_neighbors, device=self.device)], dim=0)
            
            # Infer type embedding from neighbors if available
            if self.use_types and self.size > 0 and k_actual > 0:
                # CRITICAL: Bounds check for type_embeddings
                valid_indices = topk_indices[:k_actual][topk_indices[:k_actual] < self.type_embeddings.size(0)]
                if len(valid_indices) > 0:
                    neighbor_type_embs = self.type_embeddings[valid_indices]
                    # Average neighbor types (inherit from community)
                    inferred_type = neighbor_type_embs.mean(dim=0)
                    if isinstance(self.type_embeddings, DiskBackedTensor):
                        old_size = self.type_embeddings._actual_size
                        self.type_embeddings[old_size] = inferred_type
                    else:
                        self.type_embeddings = torch.cat([self.type_embeddings, inferred_type.unsqueeze(0)], dim=0)
                else:
                    # No valid neighbors - use random type
                    random_type = torch.randn(1, self.type_dim, device=self.device) * 0.1
                    if isinstance(self.type_embeddings, DiskBackedTensor):
                        old_size = self.type_embeddings._actual_size
                        self.type_embeddings[old_size] = random_type[0]
                    else:
                        self.type_embeddings = torch.cat([self.type_embeddings, random_type], dim=0)
            elif self.use_types:
                # First node - random type
                random_type = torch.randn(1, self.type_dim, device=self.device) * 0.1
                if isinstance(self.type_embeddings, DiskBackedTensor):
                    old_size = self.type_embeddings._actual_size
                    self.type_embeddings[old_size] = random_type[0]
                else:
                    self.type_embeddings = torch.cat([self.type_embeddings, random_type], dim=0)
        
        new_idx = self.size.item()
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
        # BUT skip DiskBackedTensors (they manage their own device placement)
        from disk_backed_tensor import DiskBackedTensor
        
        if self.device == 'cpu':
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
        elif self.size > 0:
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
                    existing_embs = self.embeddings[indices].to(compute_device)
                    index_mapping = indices
                else:
                    # Use all existing
                    existing_embs = self.embeddings[:self.size].to(compute_device)
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
                if isinstance(self.embeddings, DiskBackedTensor):
                    # Use setitem interface for DiskBackedTensor
                    start_idx = self.size.item() if hasattr(self, 'size') else self.embeddings._actual_size
                    for local_i, emb in enumerate(batch_cpu):
                        self.embeddings[start_idx + local_i] = emb
                else:
                    # Regular concatenation for normal tensors
                    self.embeddings = torch.cat([self.embeddings, batch_cpu], dim=0)
                
                # Handle adjacency and edge_weights (might be DiskBackedTensor)
                if isinstance(self.adjacency, DiskBackedTensor):
                    start_idx = self.size.item() if hasattr(self, 'size') else self.adjacency._actual_size
                    for local_i in range(chunk_size):
                        self.adjacency[start_idx + local_i] = batch_adjacency_cpu[local_i]
                else:
                    self.adjacency = torch.cat([self.adjacency, batch_adjacency_cpu], dim=0)
                
                if isinstance(self.edge_weights, DiskBackedTensor):
                    start_idx = self.size.item() if hasattr(self, 'size') else self.edge_weights._actual_size
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
        if isinstance(self.embeddings, DiskBackedTensor):
            self.embeddings.flush()
        
        # Synchronize if we used GPU
        if torch.cuda.is_available() and any(compute_device == 'cuda' for _ in range(1)):
            torch.cuda.synchronize()
        
        return added_count
    
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
        
        # Find edge in adjacency list
        neighbors = self.adjacency[source_idx]
        edge_mask = neighbors == target_idx
        
        if edge_mask.any():
            edge_slot = edge_mask.nonzero(as_tuple=True)[0][0]
            
            # 🔥 CRITICAL: For DiskBackedTensor compatibility, use explicit assignment!
            # In-place ops like += and *= don't trigger __setitem__, so changes
            # would stay in RAM cache and never reach disk. We must READ then WRITE.
            
            # Update traversal count (read-modify-write)
            old_count = self.edge_traversal_count[source_idx, edge_slot]
            self.edge_traversal_count[source_idx, edge_slot] = old_count + 1
            
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
            radius = torch.norm(source_embedding, p=2).item()  # L2 norm = hyperbolic radius
            
            # Scale factor: sinh(r) with clamping to prevent explosion/collapse
            # Clamp to [0.1, 10.0] ensures:
            # - Central nodes: min 10% learning (don't freeze completely)
            # - Leaf nodes: max 10× boost (don't explode)
            hyperbolic_scale = torch.sinh(torch.tensor(radius)).item()
            hyperbolic_scale = max(0.1, min(10.0, hyperbolic_scale))
            
            # Apply scaled Hebbian update
            # Effective LR: learning_rate * hyperbolic_scale
            # - Generic paths (r<1): slow learning, stable
            # - Specific paths (r>2): fast learning, cementing rare connections
            scaled_lr = learning_rate * hyperbolic_scale
            
            old_success = self.edge_success_rate[source_idx, edge_slot]
            self.edge_success_rate[source_idx, edge_slot] = (
                (1.0 - scaled_lr) * old_success + scaled_lr * reward
            )
            
            # Mark edge as CO_RETRIEVED (type 4)
            # Strength proportional to traversal count
            new_count = self.edge_traversal_count[source_idx, edge_slot].item()
            co_retrieval_strength = min(1.0, new_count / 10.0)
            old_edge_type = self.edge_types[source_idx, edge_slot].clone()
            old_edge_type[4] = co_retrieval_strength
            self.edge_types[source_idx, edge_slot] = old_edge_type
            
            # 🔥 STRENGTHEN HIGHWAY: Reduce hyperbolic distance proportional to reward
            # Higher reward → smaller distance → stronger connection!
            # Use configurable learning_rate for fast highway formation
            old_weight = self.edge_weights[source_idx, edge_slot]
            
            # 🔍 DEBUG: Check what edge_weights contains
            if not hasattr(self, '_edge_weight_debug_count'):
                self._edge_weight_debug_count = 0
            self._edge_weight_debug_count += 1
            if self._edge_weight_debug_count <= 3:
                print(f"[EDGE DEBUG #{self._edge_weight_debug_count}] Accessing edge_weights[{source_idx}, {edge_slot}]")
                print(f"  old_weight type: {type(old_weight)}, value: {old_weight}")
                print(f"  edge_weights type: {type(self.edge_weights)}")
                print(f"  edge_weights shape: {self.edge_weights.shape if hasattr(self.edge_weights, 'shape') else 'no shape'}")
                print(f"  learning_rate: {learning_rate}, reward: {reward}")
                print(f"  hyperbolic_scale: {hyperbolic_scale:.4f}, radius: {radius:.4f}")
                # 🔍 NEW: Show which tier this is!
                tier_name = "UNKNOWN"
                if hasattr(self, 'capacity'):
                    if self.capacity == 10:
                        tier_name = "WORKING"
                    elif self.capacity == 100:
                        tier_name = "BUFFER"
                    elif self.capacity >= 1000:
                        tier_name = "LONGTERM"
                print(f"  TIER: {tier_name} (capacity={self.capacity if hasattr(self, 'capacity') else 'N/A'})")
            
            # 🔥 HYPERBOLIC WEIGHT UPDATE with normalization
            # Weight reduction: higher reward → bigger reduction (shorter path)
            # Apply same hyperbolic scaling to prevent collapse
            # BUT clamp the effective change to maintain DEQ stability (Lipschitz bound)
            weight_change = scaled_lr * reward  # How much to strengthen
            weight_change = min(weight_change, 0.5)  # 🔥 MAX 50% change per update (DEQ stability)
            
            new_weight = old_weight * (1.0 - weight_change)
            
            # 🔥 CRITICAL: Normalize to prevent graph from exploding/collapsing
            # Ensure weights stay in reasonable range [min_edge_weight, original_distance]
            # This preserves DEQ Lipschitz constant
            min_edge_weight = 0.01  # Minimum distance (don't collapse to zero)
            new_weight = torch.clamp(new_weight, min=min_edge_weight, max=old_weight)
            
            self.edge_weights[source_idx, edge_slot] = new_weight
            
            # 🛣️ LOG HIGHWAY FORMATION
            # Track how much this edge was strengthened
            strengthening = (old_weight - new_weight).item()
            if strengthening > 0:
                # Add to highway log
                self.highway_log.append({
                    'source_idx': source_idx,
                    'target_idx': target_idx,
                    'old_weight': old_weight.item(),
                    'new_weight': new_weight.item(),
                    'strengthening': strengthening,
                    'traversal_count': new_count,
                    'success_rate': self.edge_success_rate[source_idx, edge_slot].item()
                })
                
                # DEBUG: Log first few highway formations
                if not hasattr(self, '_highway_debug_count'):
                    self._highway_debug_count = 0
                self._highway_debug_count += 1
                if self._highway_debug_count <= 10:
                    print(f"🛣️ [HIGHWAY #{self._highway_debug_count}] {source_idx}→{target_idx}: "
                          f"weight {old_weight.item():.4f}→{new_weight.item():.4f} "
                          f"(Δ={strengthening:.4f}, r={radius:.2f}, h_scale={hyperbolic_scale:.2f}x, reward={reward:.4f})")
                
                # Keep only top-K by strengthening amount
                if len(self.highway_log) > self.highway_log_max * 2:
                    # Sort by strengthening and keep top half
                    self.highway_log.sort(key=lambda x: x['strengthening'], reverse=True)
                    self.highway_log = self.highway_log[:self.highway_log_max]
            else:
                # DEBUG: Why no strengthening?
                if not hasattr(self, '_no_strengthen_debug_count'):
                    self._no_strengthen_debug_count = 0
                self._no_strengthen_debug_count += 1
                if self._no_strengthen_debug_count <= 5:
                    print(f"[NO STRENGTHEN #{self._no_strengthen_debug_count}] {source_idx}→{target_idx}: "
                          f"reward={reward:.6f}, old_weight={old_weight.item():.6f}, "
                          f"strengthening={strengthening:.6f}")
    
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
                  rewards: torch.Tensor) -> None:
        """Add new nodes with their graph structure."""
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
            if self.adjacency.device != target_device:
                self.adjacency = self.adjacency.to(target_device)
            if self.edge_weights.device != target_device:
                self.edge_weights = self.edge_weights.to(target_device)
            if self.edge_types.device != target_device:
                self.edge_types = self.edge_types.to(target_device)
            if self.cluster_ids.device != target_device:
                self.cluster_ids = self.cluster_ids.to(target_device)
            if self.rewards.device != target_device:
                self.rewards = self.rewards.to(target_device)
            if self.age.device != target_device:
                self.age = self.age.to(target_device)
            if self.access.device != target_device:
                self.access = self.access.to(target_device)
            if self.depths.device != target_device:
                self.depths = self.depths.to(target_device)
            if self.use_types and self.type_embeddings.device != target_device:
                self.type_embeddings = self.type_embeddings.to(target_device)
            if self.edge_traversal_count.device != target_device:
                self.edge_traversal_count = self.edge_traversal_count.to(target_device)
            if self.edge_success_rate.device != target_device:
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
            
            if isinstance(self.embeddings, torch.Tensor) and not hasattr(self.embeddings, 'is_disk_backed'):
                # Regular tensor: use concatenation
                self.embeddings = torch.cat([self.embeddings, embeddings_device], dim=0)
            else:
                # DiskBackedTensor: use setitem interface
                for i in range(batch_size):
                    self.embeddings[old_size + i] = embeddings_device[i]
            
            # Handle adjacency (might be DiskBackedTensor)
            if not isinstance(self.adjacency, DiskBackedTensor):
                self.adjacency = torch.cat([self.adjacency, adjacency_device], dim=0)
            else:
                for i in range(batch_size):
                    self.adjacency[old_size + i] = adjacency_device[i]
            
            # Handle edge_weights (might be DiskBackedTensor)
            if not isinstance(self.edge_weights, DiskBackedTensor):
                self.edge_weights = torch.cat([self.edge_weights, edge_weights_device], dim=0)
            else:
                for i in range(batch_size):
                    self.edge_weights[old_size + i] = edge_weights_device[i]
            
            # Initialize edge_types for new nodes (default: all zeros except PROXIMITY)
            new_edge_types = torch.zeros(batch_size, adjacency.shape[1], 8, device=self.device)
            new_edge_types[:, :, 0] = 1.0  # All edges are PROXIMITY by default
            if not isinstance(self.edge_types, DiskBackedTensor):
                self.edge_types = torch.cat([self.edge_types, new_edge_types], dim=0)
            else:
                for i in range(batch_size):
                    self.edge_types[old_size + i] = new_edge_types[i]
            
            # cluster_ids might be DiskBackedTensor
            if isinstance(self.cluster_ids, DiskBackedTensor):
                for i in range(batch_size):
                    self.cluster_ids[old_size + i] = cluster_ids_device[i]
            else:
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
                # CRITICAL: Ensure self.depths is on correct device (might have been moved during retrieval)
                if str(self.depths.device).split(':')[0] != str(self.device).split(':')[0]:
                    self.depths = self.depths.to(self.device)
                self.depths = torch.cat([self.depths, new_depths], dim=0)
            
            # type_embeddings might be DiskBackedTensor
            if self.use_types:
                new_type_embeddings = torch.randn(batch_size, self.type_dim, device=self.device) * 0.1
                if isinstance(self.type_embeddings, DiskBackedTensor):
                    for i in range(batch_size):
                        self.type_embeddings[old_size + i] = new_type_embeddings[i]
                else:
                    self.type_embeddings = torch.cat([self.type_embeddings, new_type_embeddings], dim=0)
            
            # Initialize edge tracking for new nodes (might be DiskBackedTensor)
            new_edge_traversal = torch.zeros(batch_size, adjacency.shape[1], device=self.device)
            new_edge_success = torch.zeros(batch_size, adjacency.shape[1], device=self.device)
            if not isinstance(self.edge_traversal_count, DiskBackedTensor):
                self.edge_traversal_count = torch.cat([self.edge_traversal_count, new_edge_traversal], dim=0)
            else:
                for i in range(batch_size):
                    self.edge_traversal_count[old_size + i] = new_edge_traversal[i]
            
            if not isinstance(self.edge_success_rate, DiskBackedTensor):
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
    
    def update_access(self, indices: torch.Tensor) -> None:
        """Mark memories as accessed (reset their access counter)."""
        self.access[indices] = 0
    
    def step(self) -> None:
        """Increment age and access counters."""
        self.age += 1
        self.access += 1
    
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
        
        # CRITICAL: Device-aware neighbor gathering (CPU long-term → GPU processing)
        if node_embeddings.device != x.device:
            valid_mask = adjacency >= 0
            adjacency_cpu = adjacency.cpu()
            safe_adjacency_cpu = torch.where(valid_mask.cpu(), adjacency_cpu, torch.tensor(0, dtype=torch.long))
            unique_idx = torch.unique(safe_adjacency_cpu)
            
            # Transfer ONLY needed neighbors to GPU (not all 200K!)
            needed_embeddings = node_embeddings[unique_idx].to(x.device)
            
            # Build local index map
            idx_map = torch.zeros(node_embeddings.shape[0], dtype=torch.long)
            idx_map[unique_idx] = torch.arange(len(unique_idx))
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
    
    def forward(self, query: torch.Tensor, memory_tier: GraphMemoryTier, 
                k: int = 20) -> dict:
        """
        Args:
            query: [B, T, query_dim] reflex embeddings
            memory_tier: GraphMemoryTier to query
            k: number of memories to retrieve
        
        Returns:
            bundle: dict with STRUCTURE for predictor to navigate:
                'embeddings': [B, T, k, memory_dim] - content
                'indices': [B, T, k] - which memories
                'adjacency': [B, T, k, k_neighbors] - local graph
                'edge_weights': [B, T, k, k_neighbors] - edge distances
                'depths': [B, T, k] - distance from origin (abstract→concrete)
                'cluster_ids': [B, T, k] - community membership
                'type_embeddings': [B, T, k, type_dim] - emergent types (if enabled)
        """
        B, T, _ = query.shape
        
        if memory_tier.size == 0:
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
        
        # Project query to memory space
        # HANDLE DEVICE MISMATCH: query_proj is on CUDA, but query might be on CPU (for longterm tier)
        query_device = query.device
        proj_device = next(self.query_proj.parameters()).device
        if query_device != proj_device:
            query = query.to(proj_device)
        
        query_emb = self.query_proj(query)  # [B, T, memory_dim]
        
        # Move back to original device if needed
        if query_device != proj_device:
            query_emb = query_emb.to(query_device)
        
        # Map to hyperbolic space (memory efficient)
        query_flat = query_emb.view(-1, self.memory_dim)  # [B*T, memory_dim]
        query_hyp = self.poincare.exponential_map(
            torch.zeros_like(query_flat), 
            query_flat
        ).view(B, T, self.memory_dim)  # [B, T, memory_dim]
        
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
        query_flat = query_hyp.view(-1, self.memory_dim)  # [B*T, memory_dim]
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
        batch_size = min(1000, M)
        
        all_dists = []
        for batch_start in range(0, M, batch_size):
            batch_end = min(batch_start + batch_size, M)
            batch_emb = memory_tier.embeddings[batch_start:batch_end]  # Load batch from disk
            
            # Euclidean distance: ||q - m||^2
            # Ensure batch_emb is float32 too
            batch_dists = torch.cdist(
                query_flat.unsqueeze(0),  # [1, BT, D]
                batch_emb.float().unsqueeze(0)    # [1, batch_size, D]
            ).squeeze(0)  # [BT, batch_size]
            all_dists.append(batch_dists)
        
        # Concatenate and get top-k
        dists = torch.cat(all_dists, dim=1)  # [BT, M]
        topk_dists, topk_indices = torch.topk(dists, min(k_actual, M), largest=False, dim=-1)
        
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
        
        retrieved_emb_flat = memory_tier.embeddings[flat_indices]  # [B*T*k, memory_dim]
        retrieved_adj_flat = memory_tier.adjacency[flat_indices]  # [B*T*k, k_neighbors]
        retrieved_edge_weights_flat = memory_tier.edge_weights[flat_indices]  # [B*T*k, k_neighbors]
        retrieved_edge_types_flat = memory_tier.edge_types[flat_indices]  # [B*T*k, k_neighbors, num_edge_types]
        retrieved_clusters_flat = memory_tier.cluster_ids[flat_indices]  # [B*T*k]
        
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
            # Use flattened indices for DiskBackedTensor compatibility
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
            # Device handling for node_embeddings is done inside graph_conv._forward_impl
            refined_flat = self.graph_conv(x, adj, weights, memory_tier.embeddings, self.poincare)
            refined = refined_flat.view(B, T, k, self.memory_dim)
        else:
            # Just use direct retrieval without GNN refinement
            refined = retrieved_emb.view(B, T, k, self.memory_dim)
        
        # Reshape everything to [B, T, k, ...]
        bundle = {
            'embeddings': refined,  # [B, T, k, memory_dim]
            'indices': topk_indices.view(B, T, k),  # [B, T, k]
            'adjacency': retrieved_adj.view(B, T, k, self.k_neighbors),  # [B, T, k, k_neighbors]
            'edge_weights': retrieved_edge_weights.view(B, T, k, self.k_neighbors),  # [B, T, k, k_neighbors]
            'edge_types': retrieved_edge_types.view(B, T, k, self.k_neighbors, self.num_edge_types),  # [B, T, k, k_neighbors, num_edge_types]
            'depths': retrieved_depths.view(B, T, k),  # [B, T, k]
            'cluster_ids': retrieved_clusters.view(B, T, k),  # [B, T, k]
            'type_embeddings': retrieved_types.view(B, T, k, memory_tier.type_dim) if retrieved_types is not None else None,  # [B, T, k, type_dim]
        }
        
        # Update access times
        valid_mask = topk_indices >= 0
        memory_tier.update_access(topk_indices[valid_mask])
        
        return bundle


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
        
        # Gating: how much to trust memory
        self.gate_mlp = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
            nn.Sigmoid()
        )
        
        self.ln1 = nn.LayerNorm(context_dim)
        self.ln2 = nn.LayerNorm(context_dim)
    
    def forward(self, context: torch.Tensor, retrieved_memories: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [B, T, context_dim] current reflex embeddings
            retrieved_memories: [B, T, num_mems, memory_dim] retrieved graph memories
        
        Returns:
            [B, T, context_dim] context enhanced with memory
        """
        B, T, C = context.shape
        _, _, num_mems, M = retrieved_memories.shape
        
        # Cross-attention
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
                 working_capacity: int = 20,
                 buffer_capacity: int = 100,
                 longterm_capacity: int = 20000,
                 longterm_disk_path: str = None,  # NEW: disk backing for longterm
                 longterm_max_disk_size: int = 100000,  # Maximum total memories on disk
                 k_neighbors: int = 20,
                 gnn_hidden_dim: int = 512,
                 n_head: int = 4,
                 enable_gnn: bool = False,  # DISABLED by default until we optimize hyperbolic ops for VRAM
                 highway_learning_rate: float = 0.3,  # 🔥 NEW: How fast highways strengthen
                 use_full_hyperbolic_gnn: bool = False):  # 🌀 NEW: Full Möbius vs hybrid
        super().__init__()
        
        self.memory_dim = memory_dim
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.k_neighbors = k_neighbors
        self.highway_learning_rate = highway_learning_rate  # Store for Hebbian updates
        
        # Hyperbolic geometry
        self.poincare = PoincareManifold(dim=memory_dim)
        
        # Memory tiers
        self.working = GraphMemoryTier(working_capacity, memory_dim, k_neighbors, device='cuda')
        self.buffer = GraphMemoryTier(buffer_capacity, memory_dim, k_neighbors, device='cuda')
        self.longterm = GraphMemoryTier(longterm_capacity, memory_dim, k_neighbors, device='cpu',
                                       disk_path=longterm_disk_path,  # Pass disk path!
                                       max_disk_size=longterm_max_disk_size)  # Pass max size!
        
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
    
    def store_memory_dynamic(self, embedding: torch.Tensor, reward: float = 1.0) -> int:
        """
        Store new memory with DYNAMIC graph integration.
        
        The graph GROWS as new memories form, like a city expanding!
        
        🔥 FLASHBULB MEMORY: High-reward memories (reward > 1.5) bypass
        working/buffer and go straight to long-term storage!
        Like how emotionally salient events get immediately consolidated.
        
        Args:
            embedding: [D] memory vector
            reward: utility score (>1.5 = immediate longterm storage)
        
        Returns:
            index of stored memory in working tier
        """
        # 🔥 FLASHBULB MEMORY PATHWAY - Immediate consolidation for salient events
        # Like remembering where you were during important moments
        if reward > 1.5:
            # Skip working/buffer - go straight to long-term!
            longterm_idx = self.longterm.add_node_dynamic(
                embedding.cpu(),  # Longterm is on CPU
                self.poincare,
                cluster_id=-1
            )
            # Also add to buffer for immediate availability (dual-storage)
            buffer_idx = self.buffer.add_node_dynamic(
                embedding.to('cuda'),
                self.poincare,
                cluster_id=-1
            )
            self.buffer.rewards[buffer_idx] = reward
            # Only print occasionally during preload
            if longterm_idx % 1000 == 0:
                print(f"🔥 Flashbulb memory! High reward ({reward:.2f}) → immediate longterm storage (#{longterm_idx})")
            return buffer_idx  # Return buffer index for fast retrieval
        
        # Normal pathway - add to working memory
        new_idx = self.working.add_node_dynamic(
            embedding.to('cuda'),
            self.poincare,
            cluster_id=-1  # Infer from neighbors
        )
        
        # Set reward
        self.working.rewards[new_idx] = reward
        
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
                    
                    # Strengthen working memory highways (fast adaptation)
                    working_indices = paths['working'][b, t]
                    if working_indices.numel() > 1 and working_indices[0] >= 0:
                        self.record_retrieval_success(
                            query_token,
                            working_indices,
                            reward=reward,
                            tier='working',
                            learning_rate=self.highway_learning_rate  # Fast learning for working memory
                        )
                    
                    # Strengthen buffer memory highways (medium-term)
                    buffer_indices = paths['buffer'][b, t]
                    if buffer_indices.numel() > 1 and buffer_indices[0] >= 0:
                        self.record_retrieval_success(
                            query_token,
                            buffer_indices,
                            reward=reward * 0.7,  # Slightly scaled down
                            tier='buffer',
                            learning_rate=self.highway_learning_rate * 0.7  # Medium learning
                        )
                    
                    # Strengthen long-term memory highways (slow, stable learning)
                    longterm_indices = paths['longterm'][b, t]
                    if longterm_indices.numel() > 1 and longterm_indices[0] >= 0:
                        self.record_retrieval_success(
                            query_token,
                            longterm_indices,
                            reward=reward * 0.3,  # More conservative reward
                            tier='longterm',
                            learning_rate=self.highway_learning_rate * 0.2  # Slow, stable learning
                        )
        
        # Clear stored paths
        self._last_retrieval_paths = None
    
    def record_retrieval_success(self, query_embedding: torch.Tensor, 
                                 retrieved_indices: torch.Tensor, 
                                 reward: float,
                                 tier: str = 'working',
                                 learning_rate: float = 0.3):
        """
        Record which memories were retrieved together successfully.
        
        Strengthens edges along the traversal path - creates "highways"!
        
        🔥 HEBBIAN LEARNING IN LONG-TERM MEMORY:
        When the DEQ navigates through memories and gets rewarded (low loss),
        we strengthen the edges it traversed. Over time, this creates "highways"
        through the 200K memory graph - fast paths to useful knowledge!
        
        Args:
            query_embedding: The query that triggered retrieval
            retrieved_indices: Which memories were retrieved (in order)
            reward: How helpful this retrieval was (higher = strengthen more)
            tier: Which tier to strengthen ('working', 'buffer', or 'longterm')
            learning_rate: How fast to update edge weights (0.1=slow, 0.5=fast)
        """
        # Skip debug logging - system is working correctly!
        
        if len(retrieved_indices) < 2:
            return
        
        # Select tier to update
        if tier == 'working':
            memory_tier = self.working
        elif tier == 'buffer':
            memory_tier = self.buffer
        elif tier == 'longterm':
            memory_tier = self.longterm
        else:
            return
        
        # Safety: skip if tier is empty
        if memory_tier.size == 0:
            return
        
        # Strengthen edges between consecutively retrieved memories
        for i in range(len(retrieved_indices) - 1):
            idx_a = retrieved_indices[i].item()
            idx_b = retrieved_indices[i+1].item()
            
            # Only strengthen if both are valid indices in this tier
            # Check: >= 0 (not placeholder) and < size (within bounds)
            if (idx_a >= 0 and idx_a < memory_tier.size.item() and 
                idx_b >= 0 and idx_b < memory_tier.size.item()):
                memory_tier.strengthen_edge(idx_a, idx_b, reward=reward, learning_rate=learning_rate)
                memory_tier.strengthen_edge(idx_b, idx_a, reward=reward, learning_rate=learning_rate)  # Bidirectional
    
    def evolve_graph(self, enable_sleep_replay: bool = False):
        """
        Periodic graph evolution: prune weak edges, optimize structure.
        
        Args:
            enable_sleep_replay: If True, performs sleep-like memory reconsolidation
                                 (hippocampal replay + systems consolidation)
        
        Call this during consolidation to maintain graph health!
        """
        # Prune weak edges in working memory
        if self.working.size > 10:  # Only if enough nodes
            self.working.prune_weak_edges(self.poincare, threshold=0.1)
        
        # Prune weak edges in buffer
        if self.buffer.size > 20:
            self.buffer.prune_weak_edges(self.poincare, threshold=0.15)
        
        # Long-term gets less aggressive pruning (more stable)
        if self.longterm.size > 100:
            self.longterm.prune_weak_edges(self.poincare, threshold=0.05)
        
        # SLEEP-STAGE RECONSOLIDATION
        # During "sleep" (graph evolution), highly-accessed long-term memories
        # are temporarily brought back to working/buffer for strengthening
        if enable_sleep_replay and self.longterm.size > 0:
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
        """Add memories to working tier with graph structure."""
        # For now, just build edges within the new batch
        # TODO: Rebuild full graph when adding to existing tier
        if embeddings.shape[0] == 0:
            return
        
        # DIMENSION SAFETY: Check if embedding dimension matches
        # This can happen when resuming from checkpoint with different n_embd
        if embeddings.shape[-1] != self.memory_dim:
            print(f"[Memory] WARNING: Embedding dim mismatch: got {embeddings.shape[-1]}, expected {self.memory_dim}")
            print(f"[Memory] Updating memory_dim from {self.memory_dim} → {embeddings.shape[-1]}")
            print(f"[Memory] Clearing memory tiers due to architecture change")
            
            # UPDATE DIMENSION to match new architecture
            old_memory_dim = self.memory_dim
            self.memory_dim = embeddings.shape[-1]
            
            # REBUILD QUERY NETWORK with new dimensions
            print(f"[Memory] Rebuilding query network: {self.query_dim} → {self.memory_dim}")
            self.query_network = GraphMemoryQueryNetwork(
                self.query_dim, 
                self.memory_dim, 
                self.query_network.hidden_dim,  # Preserve hidden_dim
                self.k_neighbors,
                enable_gnn=self.query_network.enable_gnn,
                num_edge_types=8
            ).cuda()
            
            # REBUILD INTEGRATION NETWORK with new dimensions
            print(f"[Memory] Rebuilding integration network: {self.context_dim} ← {self.memory_dim}")
            self.integration_network = GraphMemoryIntegrationNetwork(
                self.context_dim,
                self.memory_dim,
                self.integration_network.n_head  # Preserve n_head
            ).cuda()
            
            # REBUILD POINCARE MANIFOLD with new dimension
            print(f"[Memory] Rebuilding Poincaré manifold: dim={self.memory_dim}")
            self.poincare = PoincareManifold(dim=self.memory_dim)
            
            # Clear all tiers - can't mix different embedding dimensions
            self.working.embeddings = torch.zeros(0, self.memory_dim, device='cuda')
            self.working.adjacency = torch.full((0, self.k_neighbors), -1, dtype=torch.long, device='cuda')
            self.working.edge_weights = torch.zeros(0, self.k_neighbors, device='cuda')
            self.working.edge_types = torch.zeros(0, self.k_neighbors, 8, device='cuda')
            self.working.cluster_ids = torch.full((0,), -1, dtype=torch.long, device='cuda')
            self.working.rewards = torch.zeros(0, device='cuda')
            self.working.age = torch.zeros(0, device='cuda')
            self.working.access = torch.zeros(0, device='cuda')
            self.working.size = torch.tensor(0, dtype=torch.long, device='cuda')
            self.working.depths = torch.zeros(0, device='cuda')
            self.working.type_embeddings = torch.zeros(0, self.working.type_dim, device='cuda')
            
            self.buffer.embeddings = torch.zeros(0, self.memory_dim, device='cuda')
            self.buffer.adjacency = torch.full((0, self.k_neighbors), -1, dtype=torch.long, device='cuda')
            self.buffer.edge_weights = torch.zeros(0, self.k_neighbors, device='cuda')
            self.buffer.edge_types = torch.zeros(0, self.k_neighbors, 8, device='cuda')
            self.buffer.cluster_ids = torch.full((0,), -1, dtype=torch.long, device='cuda')
            self.buffer.rewards = torch.zeros(0, device='cuda')
            self.buffer.age = torch.zeros(0, device='cuda')
            self.buffer.access = torch.zeros(0, device='cuda')
            self.buffer.size = torch.tensor(0, dtype=torch.long, device='cuda')
            self.buffer.depths = torch.zeros(0, device='cuda')
            self.buffer.type_embeddings = torch.zeros(0, self.buffer.type_dim, device='cuda')
            
            self.longterm.embeddings = torch.zeros(0, self.memory_dim, device='cpu')
            self.longterm.adjacency = torch.full((0, self.k_neighbors), -1, dtype=torch.long, device='cpu')
            self.longterm.edge_weights = torch.zeros(0, self.k_neighbors, device='cpu')
            self.longterm.edge_types = torch.zeros(0, self.k_neighbors, 8, device='cpu')
            self.longterm.cluster_ids = torch.full((0,), -1, dtype=torch.long, device='cpu')
            self.longterm.rewards = torch.zeros(0, device='cpu')
            self.longterm.age = torch.zeros(0, device='cpu')
            self.longterm.access = torch.zeros(0, device='cpu')
            self.longterm.size = torch.tensor(0, dtype=torch.long, device='cpu')
            self.longterm.depths = torch.zeros(0, device='cpu')
            self.longterm.type_embeddings = torch.zeros(0, self.longterm.type_dim, device='cpu')
            
            print(f"[Memory] ✓ Architecture updated successfully: {old_memory_dim} → {self.memory_dim}")
            return
        
        # Build edges within new memories
        adjacency, edge_weights = self.build_graph_edges(embeddings)
        cluster_ids = torch.zeros(embeddings.shape[0], dtype=torch.long, device=embeddings.device)
        
        self.working.add_nodes(embeddings, adjacency, edge_weights, cluster_ids, rewards)
    
    def consolidate_to_buffer(self) -> None:
        """Move working memories to buffer, preserving graph structure."""
        if self.working.size == 0:
            return
        
        # Transfer all working memories to buffer
        self.buffer.add_nodes(
            self.working.embeddings,
            self.working.adjacency,
            self.working.edge_weights,
            self.working.cluster_ids,
            self.working.rewards
        )
        
        # Clear working memory
        self.working.embeddings = torch.zeros(0, self.memory_dim, device='cuda')
        self.working.adjacency = torch.full((0, self.k_neighbors), -1, dtype=torch.long, device='cuda')
        self.working.edge_weights = torch.zeros(0, self.k_neighbors, device='cuda')
        self.working.cluster_ids = torch.full((0,), -1, dtype=torch.long, device='cuda')
        self.working.rewards = torch.zeros(0, device='cuda')
        self.working.age = torch.zeros(0, device='cuda')
        self.working.access = torch.zeros(0, device='cuda')
        self.working.size = torch.tensor(0, dtype=torch.long, device='cuda')
    
    def consolidate_to_longterm(self) -> None:
        """Move buffer memories to long-term, preserving graph structure."""
        if self.buffer.size == 0:
            return
        
        # Transfer to CPU
        self.longterm.add_nodes(
            self.buffer.embeddings.cpu(),
            self.buffer.adjacency.cpu(),
            self.buffer.edge_weights.cpu(),
            self.buffer.cluster_ids.cpu(),
            self.buffer.rewards.cpu()
        )
        
        # Clear buffer
        self.buffer.embeddings = torch.zeros(0, self.memory_dim, device='cuda')
        self.buffer.adjacency = torch.full((0, self.k_neighbors), -1, dtype=torch.long, device='cuda')
        self.buffer.edge_weights = torch.zeros(0, self.k_neighbors, device='cuda')
        self.buffer.cluster_ids = torch.full((0,), -1, dtype=torch.long, device='cuda')
        self.buffer.rewards = torch.zeros(0, device='cuda')
        self.buffer.age = torch.zeros(0, device='cuda')
        self.buffer.access = torch.zeros(0, device='cuda')
        self.buffer.size = torch.tensor(0, dtype=torch.long, device='cuda')
    
    def retrieve(self, query: torch.Tensor, k: int = None) -> dict:
        """
        Retrieve memories with FULL STRUCTURE for navigation.
        Also stores new memories during training AND records retrieval patterns!
        
        Args:
            query: [B, T, query_dim] reflex embeddings
            k: number of memories to retrieve per tier (default: use self.k_neighbors from config)
        
        Returns:
            dict with:
                'enhanced_context': [B, T, context_dim] - integrated output
                'bundle': dict with structure (embeddings, adjacency, depths, types, etc.)
        """
        # Use configured k_neighbors if not specified
        if k is None:
            k = self.k_neighbors
        
        B, T, C = query.shape
        
        # FORMATION: Add to working memory during training
        if self.training:
            # Take mean over sequence as representative embedding
            representative = query.mean(dim=[0, 1])  # [C]
            
            # Compute reward as negative perplexity of current state
            # This will be refined with actual loss later via apply_dopamine
            reward = torch.ones(1, device=query.device)
            
            # Add to working memory
            self.add_to_working(representative.unsqueeze(0), reward)
        
        # Query each tier - NOW RETURNS STRUCTURED BUNDLES!
        working_bundle = self.query_network(query, self.working, k=k)
        buffer_bundle = self.query_network(query, self.buffer, k=k)
        
        # Long-term on CPU, need to move query temporarily
        query_cpu = query.cpu()
        longterm_bundle = self.query_network(query_cpu, self.longterm, k=k)
        
        # NOTE: Disk backing is now handled transparently via DiskBackedTensor in longterm tier!
        # No need for separate disk_index - embeddings[i] automatically loads from disk if needed
        
        # Move ALL longterm bundle components back to GPU (clone to avoid modifying CPU originals!)
        # CRITICAL: Must clone() to avoid modifying self.longterm.depths etc in-place
        for key in longterm_bundle:
            if isinstance(longterm_bundle[key], torch.Tensor):
                longterm_bundle[key] = longterm_bundle[key].cuda().clone()
        
        # DYNAMIC LEARNING: Record successful retrievals to strengthen edges!
        # Skip if dimension mismatch or query network projection layer is incompatible
        # (Can happen after architecture change - query network not rebuilt yet)
        skip_dynamic_learning = (C != self.memory_dim or 
                                 self.query_network.query_proj.in_features != C or
                                 self.query_network.query_proj.out_features != self.memory_dim)
        
        if self.training and not skip_dynamic_learning:
            # Compute retrieval quality (simple: based on attention entropy)
            # Lower entropy = more confident retrieval = better reward
            with torch.no_grad():
                query_flat = query.view(-1, C)  # [B*T, C]
                
                # Compute attention scores from combined memories (use longterm as primary signal)
                # This is where most of the knowledge is!
                if longterm_bundle['embeddings'].numel() > 0:
                    longterm_flat = longterm_bundle['embeddings'].view(-1, k, self.memory_dim)  # [B*T, k, D]
                    
                    # Compute attention scores
                    scores = torch.bmm(longterm_flat, query_flat.unsqueeze(-1)).squeeze(-1)  # [B*T, k]
                    probs = torch.softmax(scores, dim=-1)  # [B*T, k]
                    
                    # Entropy (lower = more confident)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [B*T]
                    max_entropy = torch.log(torch.tensor(k, dtype=torch.float, device=entropy.device))
                    
                    # 🔥 FIX: Entropy-based reward was too harsh (always 0 for uniform attention)
                    # Old: reward = 1.0 - entropy / max_entropy (0 when uniform)
                    # New: Use confidence (max prob) as reward - simpler & more direct!
                    max_prob = probs.max(dim=-1)[0].mean().item()  # Average max attention weight
                    
                    # 🔥 DEFERRED HEBBIAN LEARNING: Store retrieval paths for ALL tokens (per-token strengthening!)
                    # We'll strengthen AFTER we know the PER-TOKEN prediction loss (true reward signal)
                    # Store full [B, T] structure so we can strengthen individually
                    B, T, _ = query.shape
                    self._last_retrieval_paths = {
                        'working': working_bundle['indices'],  # [B, T, k]
                        'buffer': buffer_bundle['indices'],    # [B, T, k]
                        'longterm': longterm_bundle['indices'], # [B, T, k]
                        'query': query,  # [B, T, D]
                        'shape': (B, T),
                        'attention_confidence': max_prob  # Optional: can use as bonus signal
                    }
        
        # Combine all retrieved memories (flatten k dimension)
        bundles_to_combine = [working_bundle, buffer_bundle, longterm_bundle]
        
        all_memories = torch.cat([b['embeddings'] for b in bundles_to_combine], dim=2)  # [B, T, 3*k, memory_dim]
        
        # Integrate into context
        enhanced = self.integration_network(query, all_memories)
        
        # Return BOTH enhanced context AND full structure bundle
        # Predictor can use structure for navigation without memorization!
        combined_bundle = {
            'embeddings': all_memories,  # [B, T, (3+disk)*k, D]
            'depths': torch.cat([b['depths'] for b in bundles_to_combine], dim=2),
            'edge_weights': torch.cat([b['edge_weights'] for b in bundles_to_combine], dim=2),
            'edge_types': torch.cat([b['edge_types'] for b in bundles_to_combine], dim=2),
            'cluster_ids': torch.cat([b['cluster_ids'] for b in bundles_to_combine], dim=2),
        }
        
        if working_bundle.get('type_embeddings') is not None:
            type_embs = [b['type_embeddings'] for b in bundles_to_combine if b.get('type_embeddings') is not None]
            if type_embs:
                combined_bundle['type_embeddings'] = torch.cat(type_embs, dim=2)
        
        return {
            'enhanced_context': enhanced,
            'bundle': combined_bundle
        }
    
    def retrieve_hierarchical(self, query: torch.Tensor, k: int = 20, k_clusters: int = 5) -> dict:
        """
        TWO-STAGE HIERARCHICAL RETRIEVAL for better scaling.
        
        Stage 1: Find k_clusters most relevant semantic clusters
        Stage 2: Within those clusters, find k most relevant nodes
        
        This provides better coverage at scale (800+ nodes) by focusing search
        on semantically coherent regions of the graph.
        
        Args:
            query: [B, T, query_dim] reflex embeddings
            k: number of memories to retrieve total
            k_clusters: number of clusters to search within
        
        Returns:
            Same format as retrieve() - enhanced context + structure bundle
        """
        B, T, C = query.shape
        device = query.device
        
        # Still add to working memory during training (same as retrieve)
        if self.training:
            representative = query.mean(dim=[0, 1])
            reward = torch.ones(1, device=device)
            self.add_to_working(representative.unsqueeze(0), reward)
        
        # === STAGE 1: CLUSTER-LEVEL RETRIEVAL ===
        
        # Build cluster centroids for long-term memory (largest tier)
        # Move to CPU for longterm processing
        query_cpu = query.cpu()
        
        if self.longterm.num_nodes > 0:
            unique_clusters = torch.unique(self.longterm.cluster_ids[:self.longterm.num_nodes])
            
            if len(unique_clusters) > 1:
                # Compute centroid for each cluster
                cluster_centroids = []
                cluster_sizes = []
                
                for cluster_id in unique_clusters:
                    mask = self.longterm.cluster_ids[:self.longterm.num_nodes] == cluster_id
                    if mask.sum() > 0:
                        centroid = self.longterm.embeddings[mask].mean(dim=0)
                        cluster_centroids.append(centroid)
                        cluster_sizes.append(mask.sum().item())
                
                cluster_centroids = torch.stack(cluster_centroids)  # [num_clusters, memory_dim]
                
                # Query against cluster centroids
                query_flat = query_cpu.view(-1, C)  # [B*T, C]
                
                # Project query to memory space if needed
                if C != self.memory_dim:
                    query_projected = self.query_network.query_proj(query_flat)  # [B*T, memory_dim]
                else:
                    query_projected = query_flat
                
                # Cosine similarity between query and cluster centroids
                query_norm = F.normalize(query_projected, dim=-1)
                centroid_norm = F.normalize(cluster_centroids, dim=-1)
                cluster_scores = torch.matmul(query_norm, centroid_norm.T)  # [B*T, num_clusters]
                
                # Get top-k_clusters
                k_clusters_actual = min(k_clusters, len(unique_clusters))
                top_cluster_scores, top_cluster_indices = torch.topk(cluster_scores, k_clusters_actual, dim=-1)
                
                # === STAGE 2: NODE-LEVEL RETRIEVAL WITHIN CLUSTERS ===
                
                # For each query position, retrieve from selected clusters
                # We'll do this per-batch for simplicity
                longterm_bundles = []
                
                for b in range(B):
                    for t in range(T):
                        query_idx = b * T + t
                        selected_clusters = unique_clusters[top_cluster_indices[query_idx]]
                        
                        # Get nodes from these clusters
                        cluster_mask = torch.isin(
                            self.longterm.cluster_ids[:self.longterm.num_nodes],
                            selected_clusters
                        )
                        
                        if cluster_mask.sum() == 0:
                            # Fallback: use all nodes
                            cluster_mask = torch.ones(self.longterm.num_nodes, dtype=torch.bool, device='cpu')
                        
                        # Temporarily create a filtered tier for query_network
                        # (This is a bit hacky but avoids rewriting query_network)
                        candidate_indices = torch.where(cluster_mask)[0]
                        
                        # Use standard query on filtered set - query_network handles the GNN
                        # We pass single query [1, 1, C]
                        single_query = query_cpu[b:b+1, t:t+1, :]
                        
                        # Query longterm tier (which will use GNN on full graph)
                        # Note: We rely on GNN to focus attention on relevant clusters
                        bundle = self.query_network(single_query, self.longterm, k=k)
                        longterm_bundles.append(bundle)
                
                # Stack bundles from all positions
                longterm_bundle = {
                    'embeddings': torch.cat([b['embeddings'] for b in longterm_bundles], dim=0).view(B, T, k, self.memory_dim),
                    'depths': torch.cat([b['depths'] for b in longterm_bundles], dim=0).view(B, T, k),
                    'edge_weights': torch.cat([b['edge_weights'] for b in longterm_bundles], dim=0).view(B, T, k, self.k_neighbors),
                    'edge_types': torch.cat([b['edge_types'] for b in longterm_bundles], dim=0).view(B, T, k, self.k_neighbors, self.num_edge_types),
                    'cluster_ids': torch.cat([b['cluster_ids'] for b in longterm_bundles], dim=0).view(B, T, k),
                    'type_embeddings': torch.cat([b['type_embeddings'] for b in longterm_bundles], dim=0).view(B, T, k, self.type_dim) if longterm_bundles[0]['type_embeddings'] is not None else None,
                }
            else:
                # Only one cluster or no clusters - fall back to standard retrieval
                longterm_bundle = self.query_network(query_cpu, self.longterm, k=k)
        else:
            # No longterm memories - return empty
            longterm_bundle = self.query_network(query_cpu, self.longterm, k=k)
        
        # Move longterm bundle back to GPU
        for key in longterm_bundle:
            if isinstance(longterm_bundle[key], torch.Tensor):
                longterm_bundle[key] = longterm_bundle[key].to(device)
        
        # For working and buffer, use standard retrieval (small enough)
        working_bundle = self.query_network(query, self.working, k=k)
        buffer_bundle = self.query_network(query, self.buffer, k=k)
        
        # Combine and integrate (same as retrieve)
        all_memories = torch.cat([
            working_bundle['embeddings'], 
            buffer_bundle['embeddings'], 
            longterm_bundle['embeddings']
        ], dim=2)  # [B, T, 3*k, memory_dim]
        
        enhanced = self.integration_network(query, all_memories)
        
        combined_bundle = {
            'embeddings': all_memories,
            'depths': torch.cat([working_bundle['depths'], buffer_bundle['depths'], longterm_bundle['depths']], dim=2),
            'edge_weights': torch.cat([working_bundle['edge_weights'], buffer_bundle['edge_weights'], longterm_bundle['edge_weights']], dim=2),
            'edge_types': torch.cat([working_bundle['edge_types'], buffer_bundle['edge_types'], longterm_bundle['edge_types']], dim=2),
            'cluster_ids': torch.cat([working_bundle['cluster_ids'], buffer_bundle['cluster_ids'], longterm_bundle['cluster_ids']], dim=2),
        }
        
        if working_bundle['type_embeddings'] is not None:
            combined_bundle['type_embeddings'] = torch.cat([
                working_bundle['type_embeddings'],
                buffer_bundle['type_embeddings'],
                longterm_bundle['type_embeddings']
            ], dim=2)
        
        return {
            'enhanced_context': enhanced,
            'bundle': combined_bundle
        }
    
    def step(self) -> None:
        """Increment age/access counters for all tiers."""
        self.working.step()
        self.buffer.step()
        self.longterm.step()
    
    def apply_dopamine(self, loss: float):
        """
        Apply dopaminergic reward signal to recent memories.
        
        HEBBIAN LEARNING: Memories formed during LOW LOSS get HIGHER REWARDS.
        This strengthens useful memory patterns and weakens noise.
        
        CYBERNETIC FEEDBACK: This is a control signal, not part of differentiable flow.
        Modifications happen outside gradient tracking.
        
        Args:
            loss: scalar prediction loss (lower = better = higher reward)
        """
        if self.working.size == 0:
            return
        
        # Convert loss to reward: lower loss → higher reward
        # reward ∈ [0.1, 2.0] for loss ∈ [0, 10]
        reward = 1.0 / (1.0 + loss * 0.5)  # loss=0 → reward=1.0, loss=10 → reward=0.17
        reward = max(0.1, min(2.0, reward))  # Clamp to reasonable range
        
        # CRITICAL: Use no_grad() to prevent in-place modifications from
        # accumulating computation graphs across gradient accumulation steps
        with torch.no_grad():
            # Update MOST RECENT memory (just added this step)
            recent_idx = self.working.size - 1
            if recent_idx >= 0:
                # Exponential moving average: blend new reward with existing
                alpha = 0.3  # Learning rate for reward update
                old_reward = self.working.rewards[recent_idx].item()
                new_reward = alpha * reward + (1 - alpha) * old_reward
                self.working.rewards[recent_idx] = new_reward
                
                # 🛣️ HIGHWAY FORMATION: Strengthen edges along successful retrieval paths!
                # This is the key to forming "highways" - Hebbian learning on query paths
                if hasattr(self, '_last_retrieval_paths') and self._last_retrieval_paths is not None:
                    paths = self._last_retrieval_paths
                    
                    # DEBUG: Log highway formation attempt
                    if not hasattr(self, '_highway_attempt_count'):
                        self._highway_attempt_count = 0
                    self._highway_attempt_count += 1
                    if self._highway_attempt_count <= 3:
                        print(f"[Highway Attempt #{self._highway_attempt_count}] reward={reward:.3f}, paths available: "
                              f"working={paths.get('working') is not None}, "
                              f"buffer={paths.get('buffer') is not None}, "
                              f"longterm={paths.get('longterm') is not None}")
                    
                    # Strengthen highways in ALL tiers based on retrieval success
                    # reward is high when loss is low → strengthen paths that led to good predictions
                    for tier_name in ['working', 'buffer', 'longterm']:
                        indices = paths.get(tier_name)  # [B, T, k]
                        if indices is None or indices.numel() == 0:
                            continue
                        
                        tier = getattr(self, tier_name)
                        if tier.size < 2:
                            continue
                        
                        # Flatten to get all retrieved memory pairs [B*T*k]
                        B, T, k = indices.shape
                        indices_flat = indices.view(-1)  # [B*T*k]
                        
                        # Strengthen consecutive retrievals (build highways along query path)
                        for i in range(len(indices_flat) - 1):
                            src_idx = indices_flat[i].item()
                            tgt_idx = indices_flat[i + 1].item()
                            
                            if src_idx >= 0 and tgt_idx >= 0 and src_idx != tgt_idx:
                                # Strengthen edge directly with hyperbolic Hebbian learning
                                tier.strengthen_edge(src_idx, tgt_idx, reward=reward)
                    
                    # Clear after use to avoid strengthening same paths multiple times
                    self._last_retrieval_paths = None
                else:
                    # DEBUG: Why no paths?
                    if not hasattr(self, '_no_paths_count'):
                        self._no_paths_count = 0
                    self._no_paths_count += 1
                    if self._no_paths_count <= 3:
                        print(f"[No Paths #{self._no_paths_count}] apply_dopamine called but no retrieval paths stored")
    
    def update_balancer_feedback(self, sigma_memory: float):
        """
        Homeostatic feedback from balancer.
        
        When σ_memory is low (balancer confident memory is useful),
        we should consolidate more aggressively.
        
        Args:
            sigma_memory: balancer's uncertainty for memory loss component
        """
        # Store for conditional consolidation
        self.sigma_memory = sigma_memory
        
        # Consolidate working → buffer with HYBRID trigger:
        # 1. CAPACITY: Always consolidate when working memory is full (prevent overflow)
        # 2. HOMEOSTATIC: Consolidate earlier when balancer trusts memory (sigma < 0.8)
        capacity_full = self.working.size >= self.working.capacity
        homeostatic_trigger = sigma_memory < 0.8 and self.working.size >= self.working.capacity * 0.7
        
        if capacity_full or homeostatic_trigger:
            self.consolidate_to_buffer()
            print(f"[Memory] Consolidated working→buffer: W:{self.working.size}→{self.buffer.size}, σ={sigma_memory:.2f}")
        
        # Consolidate buffer → longterm when buffer is full
        # TRIGGER SLEEP REPLAY after longterm consolidation!
        if self.buffer.size >= self.buffer.capacity * 0.9:
            self.consolidate_to_longterm()
            print(f"[Memory] Consolidated buffer→longterm: B:{self.buffer.size}→LT:{self.longterm.size}")
            # Sleep replay: Bring important long-term memories back to working
            # This happens right after consolidation, simulating sleep after learning
            if self.longterm.size > 10:  # Only if we have memories to replay
                print(f"[Memory] 💤 Sleep-stage reconsolidation triggered...")
                self._sleep_replay_reconsolidation(replay_fraction=0.1)
        
        # DYNAMIC EVOLUTION: Periodic graph optimization during consolidation!
        self.consolidation_counter += 1
        if self.consolidation_counter % 100 == 0:  # Every 100 consolidations (~100 iters)
            print(f"[Graph Evolution] Pruning weak edges and optimizing structure...")
            self.evolve_graph(enable_sleep_replay=False)  # Sleep replay now happens at consolidation
            print(f"  Working: {self.working.size} nodes, Buffer: {self.buffer.size}, Longterm: {self.longterm.size}")
            
            # 🛣️ HIGHWAY REPORT: Show top strengthened edges
            highway_stats = self.longterm.get_highway_stats(top_k=5)
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
        SLEEP-STAGE MEMORY REPLAY (Hippocampal Replay + Systems Consolidation)
        
        Biologically inspired: During sleep, highly-accessed long-term memories
        are replayed and strengthened by temporarily bringing them back to
        working memory. This:
        
        1. Strengthens important connections (consolidation)
        2. Updates old memories with new context (reconsolidation)
        3. Transfers semantic structure back from long-term → working
        
        This is the INVERSE of wake consolidation (working → long-term)!
        
        PRIORITY: Replays memories with HIGH REWARD (good predictions) + HIGH ACCESS
        
        Args:
            replay_fraction: What fraction of long-term to replay (default 10%)
        """
        if self.longterm.size == 0:
            return
        
        # Find important long-term memories worth replaying
        # IMPORTANCE = REWARD × ACCESS (both matter!)
        num_replay = max(1, int(self.longterm.size.item() * replay_fraction))
        num_replay = min(num_replay, 5)  # Max 5 per sleep cycle to avoid disruption
        
        # COMBINE REWARD + ACCESS for importance score
        # Reward: How good were predictions when this memory was used?
        # Access: How often was this memory retrieved?
        access_scores = self.longterm.access.cpu()
        reward_scores = self.longterm.rewards.cpu()
        
        if access_scores.max() > 0 and reward_scores.max() > 0:
            # Normalize both to [0, 1]
            access_norm = access_scores / (access_scores.max() + 1e-8)
            reward_norm = reward_scores / (reward_scores.max() + 1e-8)
            
            # Combined importance: 60% reward + 40% access
            # (Prioritize QUALITY over frequency)
            importance = 0.6 * reward_norm + 0.4 * access_norm
            
            _, top_indices = torch.topk(importance, min(num_replay, len(importance)))
            
            # Move these memories from long-term BACK to working (reconsolidation)
            # Keep original graph structure (adjacency, edge weights, etc.)
            with torch.no_grad():
                replayed_embeddings = self.longterm.embeddings[top_indices].cuda()
                replayed_adjacency = self.longterm.adjacency[top_indices].cuda()
                replayed_edge_weights = self.longterm.edge_weights[top_indices].cuda()
                replayed_clusters = self.longterm.cluster_ids[top_indices].cuda()
                replayed_rewards = self.longterm.rewards[top_indices].cuda()
                
                # Add to working memory (will be re-experienced during wake)
                # If working is full, this triggers LRU eviction (natural forgetting)
                self.working.add_nodes(
                    replayed_embeddings,
                    replayed_adjacency,
                    replayed_edge_weights,
                    replayed_clusters,
                    replayed_rewards
                )
                
                # DON'T remove from long-term (memories stay consolidated)
                # This is REPLAY, not transfer - same memory in both places temporarily
                # During next wake cycle, working version may be refined, then
                # re-consolidated back to long-term with updated structure
                
                print(f"[Sleep Replay] Replayed {len(top_indices)} important memories from long-term → working")
    
    def get_memory_stats(self) -> dict:
        """Get statistics about memory usage."""
        # Get highway stats from longterm (where most learning happens)
        longterm_highways = self.longterm.get_highway_stats(top_k=5)
        
        stats = {
            'num_working': self.working.size.item(),
            'num_buffer': self.buffer.size.item(),
            'num_longterm': self.longterm.size.item(),
            'working_size': self.working.size.item(),  # Legacy compatibility
            'buffer_size': self.buffer.size.item(),
            'longterm_size': self.longterm.size.item(),
            'total_size': self.working.size.item() + self.buffer.size.item() + self.longterm.size.item(),
            # 🛣️ HIGHWAY STATS
            'highways_formed': longterm_highways['total_highways'],
            'max_highway_strength': longterm_highways['max_strengthening'],
            'avg_highway_strength': longterm_highways['avg_strengthening']
        }
        
        return stats
    
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
        
        checkpoint = {
            'working': {
                'embeddings': self.working.embeddings.cpu() if hasattr(self.working.embeddings, 'cpu') else self.working.embeddings,
                'adjacency': self.working.adjacency.cpu(),
                'edge_weights': self.working.edge_weights.cpu(),
                'edge_types': self.working.edge_types.cpu(),
                'edge_traversal_count': self.working.edge_traversal_count.cpu(),
                'edge_success_rate': self.working.edge_success_rate.cpu(),
                'cluster_ids': self.working.cluster_ids.cpu(),
                'rewards': self.working.rewards.cpu(),
                'access': self.working.access.cpu(),
                'age': self.working.age.cpu(),
                'size': self.working.size.item(),
                'capacity': self.working.capacity,
            },
            'buffer': {
                'embeddings': self.buffer.embeddings.cpu() if hasattr(self.buffer.embeddings, 'cpu') else self.buffer.embeddings,
                'adjacency': self.buffer.adjacency.cpu(),
                'edge_weights': self.buffer.edge_weights.cpu(),
                'edge_types': self.buffer.edge_types.cpu(),
                'edge_traversal_count': self.buffer.edge_traversal_count.cpu(),
                'edge_success_rate': self.buffer.edge_success_rate.cpu(),
                'cluster_ids': self.buffer.cluster_ids.cpu(),
                'rewards': self.buffer.rewards.cpu(),
                'access': self.buffer.access.cpu(),
                'age': self.buffer.age.cpu(),
                'size': self.buffer.size.item(),
                'capacity': self.buffer.capacity,
            },
            'longterm': {
                # DiskBackedTensor handles embeddings separately - DON'T include them in pickle!
                # Only save regular tensors that can be pickled
                'adjacency': self.longterm.adjacency.cpu() if isinstance(self.longterm.adjacency, torch.Tensor) else None,
                'edge_weights': self.longterm.edge_weights.cpu() if isinstance(self.longterm.edge_weights, torch.Tensor) else None,
                'edge_types': self.longterm.edge_types.cpu() if isinstance(self.longterm.edge_types, torch.Tensor) else None,
                'edge_traversal_count': self.longterm.edge_traversal_count.cpu() if isinstance(self.longterm.edge_traversal_count, torch.Tensor) else None,
                'edge_success_rate': self.longterm.edge_success_rate.cpu() if isinstance(self.longterm.edge_success_rate, torch.Tensor) else None,
                'cluster_ids': self.longterm.cluster_ids.cpu() if isinstance(self.longterm.cluster_ids, torch.Tensor) else None,
                'rewards': self.longterm.rewards.cpu() if isinstance(self.longterm.rewards, torch.Tensor) else None,
                'access': self.longterm.access.cpu() if isinstance(self.longterm.access, torch.Tensor) else None,
                'age': self.longterm.age.cpu() if isinstance(self.longterm.age, torch.Tensor) else None,
                'size': self.longterm.size.item(),
                'capacity': self.longterm.capacity,
                # Note: embeddings/depths/type_embeddings are DiskBackedTensor - loaded automatically from disk
            },
            'config': {
                'k_neighbors': self.k_neighbors,
                'memory_dim': self.memory_dim,
                'context_dim': self.context_dim,
                'query_dim': self.query_dim,
                'consolidation_counter': self.consolidation_counter,
            },
            'version': '2.0',  # Graph structure version
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        size_mb = os.path.getsize(filepath) / (1024**2)
        print(f"💾 Saved graph checkpoint: {filepath} ({size_mb:.1f} MB)")
        print(f"   W:{checkpoint['working']['size']}, B:{checkpoint['buffer']['size']}, LT:{checkpoint['longterm']['size']}")
        print(f"   🧠 Preserved {checkpoint['longterm']['size'] * self.k_neighbors} edges with Hebbian weights!")
    
    def load_checkpoint(self, filepath: str):
        """
        📂 LOAD COMPLETE GRAPH STATE
        
        Restores ALL graph structure including Hebbian learning state.
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore working tier
        self.working.embeddings = checkpoint['working']['embeddings'].to(self.working.device)
        self.working.adjacency = checkpoint['working']['adjacency'].to(self.working.device)
        self.working.edge_weights = checkpoint['working']['edge_weights'].to(self.working.device)
        self.working.edge_types = checkpoint['working']['edge_types'].to(self.working.device)
        self.working.edge_traversal_count = checkpoint['working']['edge_traversal_count'].to(self.working.device)
        self.working.edge_success_rate = checkpoint['working']['edge_success_rate'].to(self.working.device)
        self.working.cluster_ids = checkpoint['working']['cluster_ids'].to(self.working.device)
        self.working.rewards = checkpoint['working']['rewards'].to(self.working.device)
        self.working.access = checkpoint['working']['access'].to(self.working.device)
        self.working.age = checkpoint['working']['age'].to(self.working.device)
        self.working.size = torch.tensor(checkpoint['working']['size'], device=self.working.device)
        self.working.capacity = checkpoint['working']['capacity']
        
        # Restore buffer tier
        self.buffer.embeddings = checkpoint['buffer']['embeddings'].to(self.buffer.device)
        self.buffer.adjacency = checkpoint['buffer']['adjacency'].to(self.buffer.device)
        self.buffer.edge_weights = checkpoint['buffer']['edge_weights'].to(self.buffer.device)
        self.buffer.edge_types = checkpoint['buffer']['edge_types'].to(self.buffer.device)
        self.buffer.edge_traversal_count = checkpoint['buffer']['edge_traversal_count'].to(self.buffer.device)
        self.buffer.edge_success_rate = checkpoint['buffer']['edge_success_rate'].to(self.buffer.device)
        self.buffer.cluster_ids = checkpoint['buffer']['cluster_ids'].to(self.buffer.device)
        self.buffer.rewards = checkpoint['buffer']['rewards'].to(self.buffer.device)
        self.buffer.access = checkpoint['buffer']['access'].to(self.buffer.device)
        self.buffer.age = checkpoint['buffer']['age'].to(self.buffer.device)
        self.buffer.size = torch.tensor(checkpoint['buffer']['size'], device=self.buffer.device)
        self.buffer.capacity = checkpoint['buffer']['capacity']
        
        # Restore longterm tier (embeddings loaded from DiskBackedTensor separately)
        self.longterm.adjacency = checkpoint['longterm']['adjacency'].to(self.longterm.device)
        self.longterm.edge_weights = checkpoint['longterm']['edge_weights'].to(self.longterm.device)
        self.longterm.edge_types = checkpoint['longterm']['edge_types'].to(self.longterm.device)
        self.longterm.edge_traversal_count = checkpoint['longterm']['edge_traversal_count'].to(self.longterm.device)
        self.longterm.edge_success_rate = checkpoint['longterm']['edge_success_rate'].to(self.longterm.device)
        self.longterm.cluster_ids = checkpoint['longterm']['cluster_ids'].to(self.longterm.device)
        self.longterm.rewards = checkpoint['longterm']['rewards'].to(self.longterm.device)
        self.longterm.access = checkpoint['longterm']['access'].to(self.longterm.device)
        self.longterm.age = checkpoint['longterm']['age'].to(self.longterm.device)
        self.longterm.size = torch.tensor(checkpoint['longterm']['size'], device=self.longterm.device)
        self.longterm.capacity = checkpoint['longterm']['capacity']
        
        # Restore config
        self.consolidation_counter = checkpoint['config']['consolidation_counter']
        
        print(f"✅ Loaded graph checkpoint: {filepath}")
        print(f"   W:{checkpoint['working']['size']}, B:{checkpoint['buffer']['size']}, LT:{checkpoint['longterm']['size']}")
        print(f"   🧠 Restored {checkpoint['longterm']['size'] * self.k_neighbors} edges with Hebbian learning state!")

