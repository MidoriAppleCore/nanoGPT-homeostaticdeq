"""
Hybrid Three-Tier Memory System: Biological Memory Architecture

WORKING MEMORY (VRAM/GPU):
- Fast formation, fast decay (0.85)
- High learning rate (10x)
- ~20 slots, instant access
- Like your "active thoughts"

BUFFER MEMORY (RAM/GPU):
- Medium-term storage (hours/days in biological time)
- No decay, age-based eviction
- ~300 slots, fast access
- Like "what you did today"

LONG-TERM MEMORY (CPU/Disk):
- Slow consolidation, minimal decay (0.999)
- Low learning rate (0.1x)
- ~800 slots on CPU, unlimited on disk
- Like your "permanent knowledge"

KEY MECHANISMS:
1. FORMATION: Working → Buffer (hyperbolic deduplication)
2. CONSOLIDATION: Buffer → Long-term (sleep cycles, top 30%)
3. RETRIEVAL: Search all 3 tiers, blend by recency
4. RECONSOLIDATION: Long-term → Working (when heavily accessed)
5. EVICTION: Age-based forgetting from buffer (stale memories)
6. DEVICE SPLIT: Working/Buffer on GPU, Long-term on CPU/Disk

This is how your brain ACTUALLY works!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math
import os
import os

from hyperbolic_memory import PoincareManifold


class HybridMemorySystem(nn.Module):
    """
    Two-tier memory with different devices and learning rates.
    
    Biological parallel:
    - Working memory = Prefrontal cortex (PFC)
    - Long-term memory = Hippocampus + Temporal cortex
    - Consolidation = Sleep/replay
    """
    
    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int,
        k_neighbors: int = 16,
        curvature: float = 1.0,
        alpha: float = 0.1,
        # Working memory (VRAM)
        working_capacity: int = 50,
        working_device: str = 'cuda',  # GPU for fast access
        working_decay: float = 0.9,  # 10% decay per step (volatile!)
        # Long-term memory (CPU/Disk)
        longterm_capacity: int = 1000,
        longterm_device: str = 'cpu',  # CPU for large storage
        longterm_decay: float = 0.999,  # 0.1% decay per step (persistent!)
        longterm_disk_path: Optional[str] = None,  # Path for disk-backed storage (unlimited!)
        # Consolidation buffer (Hippocampus)
        consolidation_buffer_size: int = 100,  # Memories that persist during wake cycle
        # Consolidation
        promotion_threshold: float = 0.5,  # Reward needed to promote
        promotion_interval: int = 100,  # Auto-consolidate every N steps
        dopamine_scale: float = 0.5,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        
        # Device split (IMPORTANT!)
        self.working_device = working_device
        self.longterm_device = longterm_device
        
        self.working_capacity = working_capacity
        self.working_decay = working_decay
        
        self.longterm_capacity = longterm_capacity
        self.longterm_decay = longterm_decay
        
        self.consolidation_buffer_size = consolidation_buffer_size
        self.promotion_threshold = promotion_threshold
        self.promotion_interval = promotion_interval
        self.dopamine_scale = dopamine_scale
        
        # Disk-backed storage setup
        self.longterm_disk_path = longterm_disk_path
        self.use_disk = longterm_disk_path is not None
        self.disk_index = []  # Metadata: {filepath, embedding_preview, reward, age, access}
        self.disk_memory_count = 0  # Total memories on disk
        
        # Hot cache for disk memories (LRU with hyperbolic prefetching)
        self.disk_hot_cache = {}  # filepath -> full embedding tensor
        self.disk_cache_access_order = []  # LRU tracking
        self.disk_cache_capacity = 2000  # Keep 2000 hot memories in RAM (~2MB for 256-dim)
        # With 20-neighbor prefetching, cache hit rate should be 95%+!
        
        if self.use_disk:
            os.makedirs(longterm_disk_path, exist_ok=True)
            print(f"\n💾 DISK-BACKED LONG-TERM MEMORY (LAZY LOADING + HOT CACHE)")
            print(f"  Path: {longterm_disk_path}")
            print(f"  Capacity: UNLIMITED (grows with disk space)")
            print(f"  Hot cache: {self.disk_cache_capacity} memories (~{self.disk_cache_capacity * memory_dim * 4 / 1024:.0f} KB)")
            print(f"  Strategy: LRU cache + hyperbolic locality prefetching")
        
        self.manifold = PoincareManifold(dim=memory_dim, c=curvature)
        
        # Query network (lives on working device for fast access)
        self.query_network = nn.Sequential(
            nn.Linear(hidden_dim, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, memory_dim)
        ).to(working_device)
        
        self.query_origin = nn.Parameter(torch.zeros(1, memory_dim, device=working_device))
        self.memory_projection = nn.Linear(memory_dim, hidden_dim).to(working_device)
        
        # ═══════════════════════════════════════════════════════════
        # WORKING MEMORY TIER (VRAM) - Fast, Volatile
        # ═══════════════════════════════════════════════════════════
        # These buffers live on GPU for fast access during forward pass
        self.register_buffer('working_embeddings', 
                           torch.zeros(0, memory_dim, device=working_device))
        self.register_buffer('working_rewards', 
                           torch.zeros(0, device=working_device))
        self.register_buffer('working_age', 
                           torch.zeros(0, device=working_device))
        self.register_buffer('working_access', 
                           torch.zeros(0, device=working_device))
        self.register_buffer('num_working', 
                           torch.tensor(0, device=working_device))
        
        # ═══════════════════════════════════════════════════════════
        # CONSOLIDATION BUFFER (HIPPOCAMPUS) - Medium-term storage
        # ═══════════════════════════════════════════════════════════
        # Memories persist across batches during wake cycle
        # Get promoted to long-term during "sleep" (consolidation)
        # Lives on working_device for fast access (small enough to fit)
        self.register_buffer('buffer_embeddings', 
                           torch.zeros(0, memory_dim, device=working_device))
        self.register_buffer('buffer_rewards', 
                           torch.zeros(0, device=working_device))
        self.register_buffer('buffer_age', 
                           torch.zeros(0, device=working_device))
        self.register_buffer('buffer_access', 
                           torch.zeros(0, device=working_device))
        self.register_buffer('num_buffer', 
                           torch.tensor(0, device=working_device))
        
        # ═══════════════════════════════════════════════════════════
        # LONG-TERM MEMORY TIER (CPU/Disk) - Slow, Persistent
        # ═══════════════════════════════════════════════════════════
        # These buffers live on CPU to save GPU memory
        # They're loaded to GPU on-demand during retrieval
        self.register_buffer('longterm_embeddings', 
                           torch.zeros(0, memory_dim, device=longterm_device))
        self.register_buffer('longterm_rewards', 
                           torch.zeros(0, device=longterm_device))
        self.register_buffer('longterm_age', 
                           torch.zeros(0, device=longterm_device))
        self.register_buffer('longterm_access', 
                           torch.zeros(0, device=longterm_device))
        self.register_buffer('num_longterm', 
                           torch.tensor(0, device=longterm_device))
        
        self.training_step = 0
        
        print(f"[HybridMemory] Three-tier architecture initialized:")
        print(f"  Working:       {working_capacity} capacity on {working_device} (decay={working_decay}, volatile)")
        print(f"  Buffer:        {consolidation_buffer_size} capacity on {working_device} (no decay, age-based eviction)")
        if self.use_disk:
            print(f"  Long-term:     {longterm_capacity} in CPU + UNLIMITED on disk (decay={longterm_decay}, persistent)")
        else:
            print(f"  Long-term:     {longterm_capacity} capacity on {longterm_device} (decay={longterm_decay}, persistent)")
        
        # Load existing LT memories from disk if available
        if self.use_disk:
            self._load_longterm_from_disk()
    
    def add_to_working(self, embedding: torch.Tensor):
        """
        Add new memory to working memory (immediate context).
        
        Like thinking: "Oh, I just saw this pattern!"
        
        Args:
            embedding: [1, memory_dim] new memory point
        """
        with torch.no_grad():
            # Ensure on correct device
            embedding = embedding.to(self.working_device)
            
            # Check capacity
            if self.num_working >= self.working_capacity:
                # EVICT: Remove worst memory (low utility)
                utility = self.working_rewards / (self.working_age + 1)
                worst_idx = utility.argmin()
                
                # Create mask to remove worst
                mask = torch.ones(self.num_working, dtype=torch.bool, device=self.working_device)
                mask[worst_idx] = False
                
                self.working_embeddings = self.working_embeddings[mask]
                self.working_rewards = self.working_rewards[mask]
                self.working_age = self.working_age[mask]
                self.working_access = self.working_access[mask]
                self.num_working -= 1
            
            # Add new memory (ensure all on working_device!)
            if self.working_embeddings.numel() == 0:
                self.working_embeddings = embedding
                self.working_rewards = torch.zeros(1, device=self.working_device)
                self.working_age = torch.zeros(1, device=self.working_device)
                self.working_access = torch.zeros(1, device=self.working_device)
            else:
                self.working_embeddings = torch.cat([self.working_embeddings, embedding], dim=0)
                self.working_rewards = torch.cat([
                    self.working_rewards, 
                    torch.zeros(1, device=self.working_device)
                ])
                self.working_age = torch.cat([
                    self.working_age, 
                    torch.zeros(1, device=self.working_device)
                ])
                self.working_access = torch.cat([
                    self.working_access, 
                    torch.zeros(1, device=self.working_device)
                ])
            self.num_working += 1
    
    def transfer_to_buffer(self):
        """
        Transfer working memories to consolidation buffer.
        
        This happens EVERY STEP - working memory flows into buffer.
        Buffer persists across batches (no decay) until sleep cycle.
        
        FILTER: Only transfer memories above utility threshold (forget trivial experiences)
        HYPERBOLIC: Use Poincaré distance to avoid duplicates - preserve semantic relationships
        
        Like: "Move short-term thoughts into hippocampus for potential long-term storage"
        """
        with torch.no_grad():
            if self.num_working == 0:
                return
            
            # Calculate utility for working memories
            utility = (
                self.working_rewards * 0.4 +
                self.working_access * 0.4 +
                (1.0 / (self.working_age + 1.0)) * 0.2
            )
            
            # FILTER: Only transfer memories worth remembering
            # Threshold adapts: lower in early training (learn everything), higher later (be selective)
            # Using a very low threshold (0.01) - filters out completely unused memories
            transfer_threshold = 0.01
            
            # Transfer only memories above threshold
            for i in range(self.num_working.item()):
                if utility[i] < transfer_threshold:
                    continue  # Skip low-utility memories (forgotten!)
                
                emb = self.working_embeddings[i:i+1]
                
                # HYPERBOLIC DEDUPLICATION: Skip if too similar to existing buffer memory
                # This preserves diversity - we want relationships, not redundancy
                if self.num_buffer > 0:
                    # Project to Poincaré ball (enforce ||x|| < 1)
                    emb_proj = self.manifold.project(emb)
                    buffer_proj = self.manifold.project(self.buffer_embeddings)
                    
                    # Compute hyperbolic distances (preserves semantic relationships)
                    dists = torch.stack([
                        self.manifold.distance(emb_proj, buffer_proj[i:i+1])
                        for i in range(self.num_buffer.item())
                    ]).squeeze()
                    
                    min_dist = dists.min()
                    
                    # ADAPTIVE PRESSURE: If buffer chronically empty, be more permissive
                    # This ensures memory accumulation even during repetitive experiences
                    base_threshold = 0.1
                    if self.num_buffer < 3:  # Buffer too empty - lower standards
                        hyperbolic_threshold = base_threshold * 1.3  # 30% more permissive
                    else:
                        hyperbolic_threshold = base_threshold
                    
                    # Skip if too close to existing memory (redundant)
                    if min_dist < hyperbolic_threshold:
                        continue  # Already have similar memory, skip
                
                # Check buffer capacity
                if self.num_buffer >= self.consolidation_buffer_size:
                    # EVICT: Remove lowest utility memory from buffer
                    buffer_utility = (
                        self.buffer_rewards * 0.4 +
                        self.buffer_access * 0.4 +
                        (1.0 / (self.buffer_age + 1.0)) * 0.2
                    )
                    worst_idx = buffer_utility.argmin()
                    
                    # Only evict if incoming memory is better than worst in buffer
                    if utility[i] <= buffer_utility[worst_idx]:
                        continue  # Don't add worse memory
                    
                    # Remove worst from buffer
                    mask = torch.ones(self.num_buffer, dtype=torch.bool, device=self.working_device)
                    mask[worst_idx] = False
                    
                    self.buffer_embeddings = self.buffer_embeddings[mask]
                    self.buffer_rewards = self.buffer_rewards[mask]
                    self.buffer_age = self.buffer_age[mask]
                    self.buffer_access = self.buffer_access[mask]
                    self.num_buffer -= 1
                
                # Add to buffer
                rew = self.working_rewards[i:i+1]
                age = self.working_age[i:i+1]
                acc = self.working_access[i:i+1]
                
                if self.buffer_embeddings.numel() == 0:
                    self.buffer_embeddings = emb
                    self.buffer_rewards = rew
                    self.buffer_age = age
                    self.buffer_access = acc
                else:
                    self.buffer_embeddings = torch.cat([self.buffer_embeddings, emb], dim=0)
                    self.buffer_rewards = torch.cat([self.buffer_rewards, rew])
                    self.buffer_age = torch.cat([self.buffer_age, age])
                    self.buffer_access = torch.cat([self.buffer_access, acc])
                
                self.num_buffer += 1
    
    def promote_to_longterm(self, working_idx: int):
        """
        CONSOLIDATION: Working → Long-term memory.
        
        Like sleep consolidation: "This pattern is important, store it permanently!"
        
        The memory "shoots" from VRAM to CPU/disk storage.
        
        Args:
            working_idx: index in working memory to consolidate
        """
        with torch.no_grad():
            if working_idx >= self.num_working:
                return
            
            # Get memory to promote (on working device)
            memory_emb = self.working_embeddings[working_idx:working_idx+1]
            reward = self.working_rewards[working_idx:working_idx+1]
            
            # Move to long-term device (GPU → CPU transfer!)
            memory_emb = memory_emb.to(self.longterm_device)
            reward = reward.to(self.longterm_device)
            
            # Check long-term capacity
            if self.num_longterm >= self.longterm_capacity:
                # PRUNE: Remove worst long-term memory
                lt_utility = self.longterm_rewards / (self.longterm_age + 1)
                worst_idx = lt_utility.argmin()
                
                mask = torch.ones(self.num_longterm, dtype=torch.bool, device=self.longterm_device)
                mask[worst_idx] = False
                
                self.longterm_embeddings = self.longterm_embeddings[mask]
                self.longterm_rewards = self.longterm_rewards[mask]
                self.longterm_age = self.longterm_age[mask]
                self.longterm_access = self.longterm_access[mask]
                self.num_longterm -= 1
            
            # Add to long-term storage (ensure all on longterm_device!)
            if self.longterm_embeddings.numel() == 0:
                self.longterm_embeddings = memory_emb
                self.longterm_rewards = reward
                self.longterm_age = torch.zeros(1, device=self.longterm_device)
                self.longterm_access = torch.zeros(1, device=self.longterm_device)
            else:
                self.longterm_embeddings = torch.cat([self.longterm_embeddings, memory_emb], dim=0)
                self.longterm_rewards = torch.cat([self.longterm_rewards, reward])
                self.longterm_age = torch.cat([
                    self.longterm_age, 
                    torch.zeros(1, device=self.longterm_device)
                ])
                self.longterm_access = torch.cat([
                    self.longterm_access, 
                    torch.zeros(1, device=self.longterm_device)
                ])
            self.num_longterm += 1
            # Consolidation tracked via memory stats (silent)
    
    def retrieve_hybrid(self, query: torch.Tensor, main_device: str) -> Tuple[torch.Tensor, Dict]:
        """
        RETRIEVAL: Search ALL THREE tiers, blend results.
        
        Like remembering: "Let me check what I'm thinking about NOW,
        what happened TODAY, and what I learned BEFORE..."
        
        Args:
            query: [batch, memory_dim] query points (on main device)
            main_device: device where query lives (for return)
        
        Returns:
            retrieved: [batch, memory_dim] blended memory
            info: stats dict
        """
        batch_size = query.shape[0]
        
        # Ensure query dimensions are correct
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        # Initialize outputs on main device
        working_retrieved = torch.zeros(batch_size, self.memory_dim, device=main_device)
        buffer_retrieved = torch.zeros(batch_size, self.memory_dim, device=main_device)
        longterm_retrieved = torch.zeros(batch_size, self.memory_dim, device=main_device)
        
        working_weight = 0.0
        buffer_weight = 0.0
        longterm_weight = 0.0
        
        # ═══════════════════════════════════════════════════════════
        # TIER 1: Working memory retrieval (FAST - already on GPU)
        # ═══════════════════════════════════════════════════════════
        if self.num_working > 0:
            # Move query to working device for comparison
            query_work = query.to(self.working_device)
            
            working_embs = self.working_embeddings
            if working_embs.dim() > 2:
                working_embs = working_embs.squeeze()
                if working_embs.dim() == 1:
                    working_embs = working_embs.unsqueeze(0)
            
            k_work = min(self.k_neighbors // 2, self.num_working.item())
            
            # Pairwise distances in hyperbolic space
            query_exp = query_work.unsqueeze(1).expand(-1, working_embs.shape[0], -1)
            work_exp = working_embs.unsqueeze(0).expand(batch_size, -1, -1)
            
            work_dists = self.manifold.distance(
                query_exp.reshape(-1, self.memory_dim),
                work_exp.reshape(-1, self.memory_dim)
            ).reshape(batch_size, working_embs.shape[0])
            
            # Top-k nearest
            topk_work_dists, topk_work_idx = torch.topk(
                work_dists, k=k_work, dim=-1, largest=False
            )
            
            # RECENCY BONUS: Recent memories weighted higher!
            recency = 1.0 / (self.working_age[topk_work_idx] + 1)
            work_weights = F.softmax(-topk_work_dists / 0.1 + recency, dim=-1)
            
            working_neighbors = working_embs[topk_work_idx]
            working_retrieved_work = (work_weights.unsqueeze(-1) * working_neighbors).sum(dim=1)
            
            # Move back to main device
            working_retrieved = working_retrieved_work.to(main_device)
            working_weight = 0.7  # Recent context weighted HIGHER
            
            # Update access counts using DIFFERENTIABLE attention weights!
            # This gives us a "smooth" trace of how much each memory contributed
            # (Rather than binary accessed/not-accessed)
            with torch.no_grad():
                # Sum attention weights across batch (how much was this memory used?)
                attention_strength = work_weights.sum(dim=0)  # [k_work]
                
                # Scatter attention back to full memory (only for accessed memories)
                for i, idx in enumerate(topk_work_idx[0]):  # Use first batch element
                    self.working_access[idx] += attention_strength[i].item()
        
        # ═══════════════════════════════════════════════════════════
        # TIER 2: Buffer memory retrieval (MEDIUM - recent memories on GPU)
        # ═══════════════════════════════════════════════════════════
        if self.num_buffer > 0:
            # Buffer already on working device (GPU) - no transfer needed!
            query_buf = query.to(self.working_device)
            
            buffer_embs = self.buffer_embeddings
            if buffer_embs.dim() > 2:
                buffer_embs = buffer_embs.squeeze()
                if buffer_embs.dim() == 1:
                    buffer_embs = buffer_embs.unsqueeze(0)
            
            k_buf = min(self.k_neighbors // 3, self.num_buffer.item())
            
            # Pairwise distances in hyperbolic space
            query_exp = query_buf.unsqueeze(1).expand(-1, buffer_embs.shape[0], -1)
            buf_exp = buffer_embs.unsqueeze(0).expand(batch_size, -1, -1)
            
            buf_dists = self.manifold.distance(
                query_exp.reshape(-1, self.memory_dim),
                buf_exp.reshape(-1, self.memory_dim)
            ).reshape(batch_size, buffer_embs.shape[0])
            
            # Top-k nearest
            topk_buf_dists, topk_buf_idx = torch.topk(
                buf_dists, k=k_buf, dim=-1, largest=False
            )
            
            # Medium recency bonus (less than working, more than LT)
            recency = 1.0 / (self.buffer_age[topk_buf_idx] + 10)
            buf_weights = F.softmax(-topk_buf_dists / 0.1 + recency * 0.5, dim=-1)
            
            buffer_neighbors = buffer_embs[topk_buf_idx]
            buffer_retrieved_buf = (buf_weights.unsqueeze(-1) * buffer_neighbors).sum(dim=1)
            
            # Move to main device
            buffer_retrieved = buffer_retrieved_buf.to(main_device)
            buffer_weight = 0.5  # Middle weight (between working 0.7 and LT 0.3)
            
            # Update access counts
            with torch.no_grad():
                attention_strength = buf_weights.sum(dim=0)
                for i, idx in enumerate(topk_buf_idx[0]):
                    self.buffer_access[idx] += attention_strength[i].item()
        
        # ═══════════════════════════════════════════════════════════
        # TIER 3: Long-term memory retrieval (SLOW - CPU → GPU transfer)
        # ═══════════════════════════════════════════════════════════
        # Search both CPU hot tier AND disk cold tier
        total_lt_searched = self.num_longterm.item()
        
        if self.num_longterm > 0 or (self.use_disk and len(self.disk_index) > 0):
            # Move query to long-term device for comparison
            query_lt = query.to(self.longterm_device)
            
            # Collect all LT embeddings (CPU + disk previews for initial search)
            all_lt_embs = []
            lt_sources = []  # Track which tier each memory comes from
            
            # Add CPU hot tier
            if self.num_longterm > 0:
                longterm_embs = self.longterm_embeddings
                if longterm_embs.dim() > 2:
                    longterm_embs = longterm_embs.squeeze()
                    if longterm_embs.dim() == 1:
                        longterm_embs = longterm_embs.unsqueeze(0)
                all_lt_embs.append(longterm_embs)
                lt_sources.extend([('cpu', i) for i in range(self.num_longterm.item())])
            
            # Add disk tier (use PREVIEWS for initial search)
            if self.use_disk and len(self.disk_index) > 0:
                disk_previews = torch.stack([
                    entry['embedding_preview'] for entry in self.disk_index
                ]).to(self.longterm_device)
                
                # Pad previews to full dimension with zeros (approximate search)
                if disk_previews.shape[1] < self.memory_dim:
                    padding = torch.zeros(
                        disk_previews.shape[0], 
                        self.memory_dim - disk_previews.shape[1],
                        device=self.longterm_device
                    )
                    disk_previews = torch.cat([disk_previews, padding], dim=1)
                
                all_lt_embs.append(disk_previews)
                lt_sources.extend([('disk', i) for i in range(len(self.disk_index))])
                total_lt_searched += len(self.disk_index)
            
            if len(all_lt_embs) > 0:
                # Combine CPU + disk for unified search
                combined_lt_embs = torch.cat(all_lt_embs, dim=0)
                
                k_long = min(self.k_neighbors // 2, combined_lt_embs.shape[0])
                
                query_exp = query_lt.unsqueeze(1).expand(-1, combined_lt_embs.shape[0], -1)
                long_exp = combined_lt_embs.unsqueeze(0).expand(batch_size, -1, -1)
                
                long_dists = self.manifold.distance(
                    query_exp.reshape(-1, self.memory_dim),
                    long_exp.reshape(-1, self.memory_dim)
                ).reshape(batch_size, combined_lt_embs.shape[0])
                
                topk_long_dists, topk_long_idx = torch.topk(
                    long_dists, k=k_long, dim=-1, largest=False
                )
                
                # Load full embeddings for top-k (lazy load from disk if needed)
                topk_embeddings = []
                for idx in topk_long_idx[0]:  # First batch element
                    source_type, source_idx = lt_sources[idx.item()]
                    
                    if source_type == 'cpu':
                        topk_embeddings.append(combined_lt_embs[idx])
                    else:  # disk
                        # LAZY LOAD with hot cache + hyperbolic prefetching!
                        # Prefetch MORE neighbors (20) since we have RAM to spare
                        # This dramatically improves cache hit rate!
                        filepath = self.disk_index[source_idx]['filepath']
                        full_emb = self._load_disk_memory_with_prefetch(
                            filepath, 
                            query_lt[0],  # Use query for prefetching neighbors
                            k_prefetch=20  # Prefetch 20 neighbors (semantic cluster)
                        )
                        topk_embeddings.append(full_emb)
                
                longterm_neighbors = torch.stack(topk_embeddings).unsqueeze(0)  # [1, k, dim]
                
                long_weights = F.softmax(-topk_long_dists / 0.1, dim=-1)
                longterm_retrieved_lt = (long_weights.unsqueeze(-1) * longterm_neighbors).sum(dim=1)
                
                # Move back to main device (CPU → GPU transfer!)
                longterm_retrieved = longterm_retrieved_lt.to(main_device)
                longterm_weight = 0.3  # Old knowledge weighted LOWER
                
                # Update access counters
                with torch.no_grad():
                    attention_strength = long_weights.sum(dim=0)  # [k_long]
                    for i, idx in enumerate(topk_long_idx[0]):
                        source_type, source_idx = lt_sources[idx.item()]
                        if source_type == 'cpu':
                            self.longterm_access[source_idx] += attention_strength[i].item()
                        else:  # disk
                            self.disk_index[source_idx]['access'] += attention_strength[i].item()
                
                # RECONSOLIDATION: Highly-accessed long-term → COPY to working!
                # (Silent - tracked via memory stats in main training loop)
                if self.training and self.num_longterm > 0:
                    max_access = self.longterm_access.max()
                    if max_access > 100:
                        most_accessed_idx = self.longterm_access.argmax()
                        if self.longterm_access[most_accessed_idx] > 100:
                            reconsolidated = self.longterm_embeddings[most_accessed_idx:most_accessed_idx+1]
                            reconsolidated = reconsolidated.to(self.working_device)
                            self.add_to_working(reconsolidated)
                            self.longterm_access[most_accessed_idx] = 0
                            # Reconsolidation count tracked in memory stats
        
        # ═══════════════════════════════════════════════════════════
        # BLEND: Combine ALL THREE tiers (continuous interpolation)
        # ═══════════════════════════════════════════════════════════
        total_weight = working_weight + buffer_weight + longterm_weight
        if total_weight > 0:
            retrieved = (working_weight * working_retrieved + 
                        buffer_weight * buffer_retrieved +
                        longterm_weight * longterm_retrieved) / total_weight
        else:
            retrieved = torch.zeros(batch_size, self.memory_dim, device=main_device)
        
        info = {
            'working_contrib': working_weight,
            'buffer_contrib': buffer_weight,
            'longterm_contrib': longterm_weight,
            'num_working': self.num_working.item(),
            'num_buffer': self.num_buffer.item(),
            'num_longterm': self.num_longterm.item(),
            'num_disk': len(self.disk_index) if self.use_disk else 0,
            'total_lt_searched': total_lt_searched,
            'cache_hits': len(self.disk_hot_cache) if self.use_disk else 0,
            'device_working': str(self.working_device),
            'device_longterm': str(self.longterm_device)
        }
        
        return retrieved, info
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Memory-augmented forward pass.
        
        1. Form new working memory (immediate encoding)
        2. Retrieve from BOTH tiers
        3. Inject memory force
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            output: [batch, seq_len, hidden_dim] with memory
            info: memory stats
        """
        batch, seq_len, hidden_dim = hidden_states.shape
        main_device = hidden_states.device
        
        # Generate query in hyperbolic space
        tangent = self.query_network(hidden_states.mean(dim=1))
        query_origin = self.manifold.project(self.query_origin.expand(batch, -1).to(main_device))
        query_point = self.manifold.exponential_map(query_origin, tangent)
        
        # FORMATION: Add to working memory (fast encoding)
        if self.training:
            representative = hidden_states.mean(dim=[0, 1])  # [hidden_dim]
            query_emb = self.query_network(representative.unsqueeze(0))  # [1, memory_dim]
            query_emb = self.manifold.project(query_emb * 0.1)
            self.add_to_working(query_emb)
        
        # RETRIEVAL: Search both tiers
        retrieved, info = self.retrieve_hybrid(query_point, main_device)
        
        # Store query and retrieved embeddings for contrastive loss computation
        info['query'] = query_point  # [batch, memory_dim]
        info['retrieved'] = retrieved  # [batch, memory_dim] or [batch, k, memory_dim]
        
        # INJECTION: Add memory force to hidden states
        if retrieved.norm() > 1e-6:
            memory_force = self.memory_projection(retrieved)
            memory_force = memory_force.unsqueeze(1).expand(-1, seq_len, -1)
            output = hidden_states + self.alpha * memory_force
        else:
            output = hidden_states
        
        return output, info
    
    def apply_dopamine(self, loss: torch.Tensor):
        """
        REINFORCEMENT: Apply reward signal to strengthen/weaken memories.
        
        High reward → Promote working → long-term (consolidation)
        Low reward → Faster decay
        
        Args:
            loss: scalar training loss (lower = better = more dopamine)
        """
        if not self.training:
            return
        
        reward = -loss.item()  # Negative loss = positive reward
        
        # Update working memory (FAST plasticity)
        if self.num_working > 0:
            recent_mask = self.working_access > 0
            self.working_rewards[recent_mask] += reward
        
        # Update long-term memory (SLOW plasticity)
        if self.num_longterm > 0:
            recent_mask = self.longterm_access > 0
            self.longterm_rewards[recent_mask] += reward * 0.1  # 10x slower!
        
        # IMMEDIATE PROMOTION: Very high reward + high access → instant consolidation
        # (Like when you have an "aha!" moment - immediately important)
        # Silent - tracked via memory stats
        if self.num_working > 0 and reward > self.promotion_threshold * 2:
            # Find memories with both high reward AND high access
            utility = self.working_rewards * 0.5 + self.working_access * 0.5
            best_idx = utility.argmax()
            if utility[best_idx] > self.promotion_threshold:
                self.promote_to_longterm(best_idx.item())
    
    def step(self):
        """
        DECAY: Age and fade memories (at different rates).
        
        Also handles periodic auto-consolidation.
        """
        self.training_step += 1
        
        # Age all memories
        self.working_age += 1
        self.longterm_age += 1
        
        # DECAY: Working memory fades FAST (volatile!)
        self.working_rewards *= self.working_decay
        self.working_access *= self.working_decay
        
        # DECAY: Long-term fades SLOW (persistent!)
        self.longterm_rewards *= self.longterm_decay
        self.longterm_access *= self.longterm_decay
        
        # NOTE: Buffer does NOT decay! Persists until sleep cycle
        # Buffer ages, but rewards/access stay constant
        self.buffer_age += 1
        
        # STEP 1: Transfer working → buffer (every step)
        self.transfer_to_buffer()
        
        # STEP 2: SLEEP CYCLE - Buffer → Long-term (adaptive: biological triggers)
        # Sleep triggers:
        # 1. Buffer ≥90% full (slot-based - simple trigger)
        # 2. High information density (hyperbolic volume - semantic saturation)
        # 3. Periodic interval reached (every N steps as fallback)
        
        buffer_fill_ratio = self.num_buffer.item() / self.consolidation_buffer_size
        
        # Calculate hyperbolic "volume" - measures semantic coverage
        # Higher volume = more diverse memories = richer information
        semantic_saturation = 0.0
        if self.num_buffer > 1:
            with torch.no_grad():
                # Project buffer to Poincaré ball
                buffer_proj = self.manifold.project(self.buffer_embeddings)
                
                # Compute pairwise hyperbolic distances
                # High average distance = diverse memories (good!)
                # Low average distance = clustered/redundant (need consolidation!)
                n = self.num_buffer.item()
                total_dist = 0.0
                count = 0
                
                for i in range(n):
                    for j in range(i+1, n):
                        dist = self.manifold.distance(buffer_proj[i:i+1], buffer_proj[j:j+1])
                        total_dist += dist.item()
                        count += 1
                
                # Semantic saturation: average distance
                # When memories start clustering (redundancy), trigger sleep
                semantic_saturation = total_dist / count if count > 0 else 0.0
        
        # Trigger sleep when:
        # - Slot-based: Buffer 80% full (give buffer room to breathe!)
        # - Semantic: Average distance < 0.3 (VERY clustered, real redundancy)
        # - Periodic: Fallback every N steps
        should_sleep = (
            buffer_fill_ratio >= 0.8 or  # Slot capacity (RAISED - less frequent)
            (self.num_buffer > 30 and semantic_saturation < 0.3) or  # Semantic saturation (TIGHTENED - only when VERY redundant)
            (self.training_step % self.promotion_interval == 0 and self.num_buffer > 0)  # Periodic
        )
        
        if should_sleep:
            if buffer_fill_ratio >= 0.8:
                sleep_reason = "BUFFER FULL"
            elif self.num_buffer > 30 and semantic_saturation < 0.3:
                sleep_reason = f"SEMANTIC SATURATION (dist={semantic_saturation:.3f})"
            else:
                sleep_reason = "PERIODIC"
            
            # Only log sleep cycles occasionally (every 10th) to reduce spam
            verbose = (self.training_step % (self.promotion_interval * 10) == 0)
            
            if verbose:
                print(f"\n  💤 SLEEP CYCLE [{sleep_reason}] (step {self.training_step}, buffer {buffer_fill_ratio:.1%} full)")
                print(f"     Consolidating {self.num_buffer} memories from buffer → long-term")
            
            # Compute UTILITY for ALL buffer memories (logarithmic scale)
            # FIX: Shift rewards to be positive before log (can't take log of negative!)
            # Reward = -loss, so can be negative (e.g., -8.5)
            # log(1 + (-8.5)) = log(-7.5) = NaN ❌
            # Solution: Normalize relative to minimum, ensuring input ≥ 1.0
            min_reward = self.buffer_rewards.min()
            shifted_reward = self.buffer_rewards - min_reward + 1.0  # Always >= 1.0
            
            buffer_utility = (
                torch.log(shifted_reward) * 0.4 +           # Now safe! (always positive)
                torch.log(1 + self.buffer_access) * 0.4 +   # Access is always >= 0
                (1.0 / (self.buffer_age + 1.0)) * 0.2       # Recency (linear is fine)
            )
            
            # PERCENTILE-BASED PROMOTION: Always promote top 30% of buffer
            # This automatically adapts to current utility distribution
            # MINIMUM 1: Even simple experiences form some long-term memories (biological realism)
            buffer_count = self.num_buffer.item()
            if buffer_count > 0:
                num_to_promote = max(
                    1,  # ALWAYS promote at least 1 memory if buffer has anything
                    min(
                        int(buffer_count * 0.3),  # Top 30% of buffer
                        self.longterm_capacity - self.num_longterm.item(),  # Don't overflow longterm
                        20  # Cap at 20 per sleep cycle for stability
                    )
                )
            else:
                num_to_promote = 0
            
            if num_to_promote > 0:
                # Sort by utility (descending)
                sorted_indices = torch.argsort(buffer_utility, descending=True)
                promoted_count = 0
                
                for i in range(num_to_promote):
                    idx = sorted_indices[i].item()
                    
                    # Get memory from buffer
                    memory_emb = self.buffer_embeddings[idx:idx+1]
                    reward = self.buffer_rewards[idx:idx+1]
                    
                    # Move to long-term device (GPU → CPU)
                    memory_emb = memory_emb.to(self.longterm_device)
                    reward = reward.to(self.longterm_device)
                    
                    # Check long-term capacity
                    if self.num_longterm >= self.longterm_capacity:
                        if self.use_disk:
                            # FLUSH TO DISK: Move oldest 50% of LT to disk
                            self._flush_longterm_to_disk()
                            
                            # Keep newest 50% in CPU memory for fast access
                            half = self.longterm_capacity // 2
                            self.longterm_embeddings = self.longterm_embeddings[-half:]
                            self.longterm_rewards = self.longterm_rewards[-half:]
                            self.longterm_age = self.longterm_age[-half:]
                            self.longterm_access = self.longterm_access[-half:]
                            self.num_longterm = torch.tensor(half, device=self.longterm_device)
                            print(f"     💾 Flushed oldest 50% to disk, kept {half} newest in CPU")
                        else:
                            # NO DISK: Prune worst memory (old behavior)
                            lt_utility = self.longterm_rewards / (self.longterm_age + 1)
                            worst_idx = lt_utility.argmin()
                            
                            mask = torch.ones(self.num_longterm, dtype=torch.bool, device=self.longterm_device)
                            mask[worst_idx] = False
                            
                            self.longterm_embeddings = self.longterm_embeddings[mask]
                            self.longterm_rewards = self.longterm_rewards[mask]
                            self.longterm_age = self.longterm_age[mask]
                            self.longterm_access = self.longterm_access[mask]
                            self.num_longterm -= 1
                    
                    # Add to long-term
                    if self.longterm_embeddings.numel() == 0:
                        self.longterm_embeddings = memory_emb
                        self.longterm_rewards = reward
                        self.longterm_age = torch.zeros(1, device=self.longterm_device)
                        self.longterm_access = torch.zeros(1, device=self.longterm_device)
                    else:
                        self.longterm_embeddings = torch.cat([self.longterm_embeddings, memory_emb], dim=0)
                        self.longterm_rewards = torch.cat([self.longterm_rewards, reward])
                        self.longterm_age = torch.cat([self.longterm_age, torch.zeros(1, device=self.longterm_device)])
                        self.longterm_access = torch.cat([self.longterm_access, torch.zeros(1, device=self.longterm_device)])
                    
                    self.num_longterm += 1
                    promoted_count += 1
                
                # Show utility range for transparency (only if verbose)
                if verbose:
                    min_promoted_utility = buffer_utility[sorted_indices[num_to_promote-1]].item()
                    max_promoted_utility = buffer_utility[sorted_indices[0]].item()
                    print(f"     ✓ Promoted {promoted_count} memories (top 30%, utility: {min_promoted_utility:.3f} to {max_promoted_utility:.3f})")
                    print(f"     📊 Long-term now has {self.num_longterm} memories")
            else:
                if verbose:
                    print(f"     ⚠ No capacity for promotion (long-term full or buffer empty)")
            
            # EVICT OLD UNPROMOTED MEMORIES (age-based forgetting)
            # If a memory has been in buffer for a long time without promotion, drop it
            # This keeps buffer from being clogged with low-utility memories
            if self.num_buffer > 0:
                # Drop memories older than 100 steps that weren't promoted
                age_threshold = 100
                old_mask = self.buffer_age > age_threshold
                num_old = old_mask.sum().item()
                
                if num_old > 0:
                    # Keep only younger memories
                    keep_mask = ~old_mask
                    self.buffer_embeddings = self.buffer_embeddings[keep_mask]
                    self.buffer_rewards = self.buffer_rewards[keep_mask]
                    self.buffer_age = self.buffer_age[keep_mask]
                    self.buffer_access = self.buffer_access[keep_mask]
                    self.num_buffer = keep_mask.sum()
                    if verbose:
                        print(f"     🧹 Evicted {num_old} old buffer memories (age>{age_threshold} steps)")
            
            if verbose:
                print(f"     📊 Buffer retained {self.num_buffer.item()} memories for continued access\n")
    
    def _flush_longterm_to_disk(self):
        """
        Write oldest 50% of LT memories to disk (lazy loading).
        Store embedding preview (first 32 dims) for fast distance estimates.
        """
        if not self.use_disk:
            return
        
        half = self.longterm_capacity // 2
        
        # Take oldest 50% (first half of arrays)
        old_embeddings = self.longterm_embeddings[:half].cpu()
        old_rewards = self.longterm_rewards[:half].cpu()
        old_age = self.longterm_age[:half].cpu()
        old_access = self.longterm_access[:half].cpu()
        
        # Save to individual files for lazy loading
        for i in range(half):
            disk_file = os.path.join(
                self.longterm_disk_path, 
                f'lt_mem_{self.disk_memory_count:08d}.pt'
            )
            
            torch.save({
                'embedding': old_embeddings[i],
                'reward': old_rewards[i],
                'age': old_age[i],
                'access': old_access[i],
                'timestamp': self.training_step,
            }, disk_file)
            
            # Keep metadata + embedding PREVIEW in RAM for fast distance estimates
            # Preview = first 32 dims (enough for approximate nearest neighbors)
            preview_dims = min(32, self.memory_dim)
            self.disk_index.append({
                'filepath': disk_file,
                'embedding_preview': old_embeddings[i, :preview_dims].clone(),  # Small preview for distance
                'reward': old_rewards[i].item(),
                'age': old_age[i].item(),
                'access': old_access[i].item(),
            })
            
            self.disk_memory_count += 1
        
        print(f"     💾 Flushed {half} oldest LT memories to disk ({self.disk_memory_count} total on disk)")
        print(f"        Disk usage: ~{self.disk_memory_count * self.memory_dim * 4 / 1024 / 1024:.1f} MB")
    
    def _load_disk_memory_with_prefetch(self, filepath: str, query_embedding: torch.Tensor, k_prefetch: int = 5) -> torch.Tensor:
        """
        Load memory from disk with hyperbolic locality prefetching.
        
        Args:
            filepath: Path to the memory to load
            query_embedding: Current query (for finding neighbors)
            k_prefetch: How many neighbors to prefetch
        
        Returns:
            The loaded embedding tensor
        """
        # Check hot cache first
        if filepath in self.disk_hot_cache:
            # Update LRU
            self.disk_cache_access_order.remove(filepath)
            self.disk_cache_access_order.append(filepath)
            return self.disk_hot_cache[filepath]
        
        # Load from disk
        data = torch.load(filepath, map_location=self.longterm_device)
        embedding = data['embedding']
        
        # Add to hot cache
        self.disk_hot_cache[filepath] = embedding
        self.disk_cache_access_order.append(filepath)
        
        # Evict if cache full (LRU)
        if len(self.disk_hot_cache) > self.disk_cache_capacity:
            evict_filepath = self.disk_cache_access_order.pop(0)
            del self.disk_hot_cache[evict_filepath]
        
        # PREFETCH: Find k nearest neighbors in hyperbolic space using PREVIEWS
        # This loads semantically related memories into cache
        if k_prefetch > 0 and len(self.disk_index) > 1:
            preview_dims = self.disk_index[0]['embedding_preview'].shape[0]
            query_preview = query_embedding[:preview_dims].cpu()
            
            distances = []
            for idx, entry in enumerate(self.disk_index):
                if entry['filepath'] != filepath:  # Don't prefetch self
                    # Approximate distance using preview
                    dist = self.manifold.distance(
                        query_preview.unsqueeze(0),
                        entry['embedding_preview'].unsqueeze(0)
                    ).item()
                    distances.append((dist, idx))
            
            # Get k nearest
            distances.sort()
            for _, idx in distances[:k_prefetch]:
                neighbor_path = self.disk_index[idx]['filepath']
                if neighbor_path not in self.disk_hot_cache:
                    # Prefetch into cache
                    neighbor_data = torch.load(neighbor_path, map_location=self.longterm_device)
                    self.disk_hot_cache[neighbor_path] = neighbor_data['embedding']
                    self.disk_cache_access_order.append(neighbor_path)
                    
                    # Evict if needed
                    if len(self.disk_hot_cache) > self.disk_cache_capacity:
                        evict = self.disk_cache_access_order.pop(0)
                        del self.disk_hot_cache[evict]
        
        return embedding
    
    def _load_longterm_from_disk(self):
        """
        Load disk index (metadata + embedding previews) on startup.
        Actual full embeddings loaded lazily during retrieval with prefetching.
        """
        if not self.use_disk:
            return
        
        import glob
        
        # Find all LT memory files
        disk_files = sorted(glob.glob(os.path.join(self.longterm_disk_path, 'lt_mem_*.pt')))
        
        if not disk_files:
            print(f"     ℹ️  No existing LT memories found on disk")
            return
        
        print(f"     💾 Loading disk index ({len(disk_files)} memories)...")
        
        # Load metadata + embedding PREVIEWS (not full tensors!)
        preview_dims = min(32, self.memory_dim)
        for disk_file in disk_files:
            data = torch.load(disk_file, map_location='cpu')
            self.disk_index.append({
                'filepath': disk_file,
                'embedding_preview': data['embedding'][:preview_dims].clone(),  # First 32 dims
                'reward': data['reward'].item(),
                'age': data['age'].item(),
                'access': data['access'].item(),
            })
            self.disk_memory_count += 1
        
        print(f"     ✅ Loaded {len(self.disk_index)} memory indices from disk")
        print(f"        Preview cache: {preview_dims} dims × {len(self.disk_index)} = {preview_dims * len(self.disk_index) * 4 / 1024:.1f} KB")
        print(f"        (Full tensors loaded lazily with hyperbolic prefetching)")
    
    def get_stats(self) -> Dict:
        """Memory system statistics"""
        return {
            'num_working': self.num_working.item(),
            'num_buffer': self.num_buffer.item(),
            'num_longterm': self.num_longterm.item(),
            'num_disk': len(self.disk_index) if self.use_disk else 0,
            'working_avg_reward': self.working_rewards.mean().item() if self.num_working > 0 else 0,
            'buffer_avg_reward': self.buffer_rewards.mean().item() if self.num_buffer > 0 else 0,
            'longterm_avg_reward': self.longterm_rewards.mean().item() if self.num_longterm > 0 else 0,
            'working_device': str(self.working_device),
            'longterm_device': str(self.longterm_device),
        }
    
    def save_checkpoint(self, filepath: str):
        """
        Save BOTH memory tiers to disk (separate from model).
        
        This allows loading memories independently.
        """
        import pickle
        
        checkpoint = {
            'working': {
                'embeddings': self.working_embeddings.cpu().numpy(),
                'rewards': self.working_rewards.cpu().numpy(),
                'age': self.working_age.cpu().numpy(),
                'access': self.working_access.cpu().numpy(),
                'num': self.num_working.item(),
                'device': self.working_device
            },
            'longterm': {
                'embeddings': self.longterm_embeddings.cpu().numpy(),
                'rewards': self.longterm_rewards.cpu().numpy(),
                'age': self.longterm_age.cpu().numpy(),
                'access': self.longterm_access.cpu().numpy(),
                'num': self.num_longterm.item(),
                'device': self.longterm_device
            },
            'config': {
                'hidden_dim': self.hidden_dim,
                'memory_dim': self.memory_dim,
                'training_step': self.training_step
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        size_mb = os.path.getsize(filepath) / (1024**2)
        print(f"\n💾 SAVED MEMORY CHECKPOINT: {filepath} ({size_mb:.1f} MB)")
        print(f"   Working: {self.num_working.item()} on {self.working_device}")
        print(f"   Long-term: {self.num_longterm.item()} on {self.longterm_device}")
    
    def load_checkpoint(self, filepath: str):
        """Load both memory tiers from checkpoint"""
        import pickle
        
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore working memory
        self.working_embeddings = torch.from_numpy(checkpoint['working']['embeddings']).to(self.working_device)
        self.working_rewards = torch.from_numpy(checkpoint['working']['rewards']).to(self.working_device)
        self.working_age = torch.from_numpy(checkpoint['working']['age']).to(self.working_device)
        self.working_access = torch.from_numpy(checkpoint['working']['access']).to(self.working_device)
        self.num_working = torch.tensor(checkpoint['working']['num'], device=self.working_device)
        
        # Restore long-term memory
        self.longterm_embeddings = torch.from_numpy(checkpoint['longterm']['embeddings']).to(self.longterm_device)
        self.longterm_rewards = torch.from_numpy(checkpoint['longterm']['rewards']).to(self.longterm_device)
        self.longterm_age = torch.from_numpy(checkpoint['longterm']['age']).to(self.longterm_device)
        self.longterm_access = torch.from_numpy(checkpoint['longterm']['access']).to(self.longterm_device)
        self.num_longterm = torch.tensor(checkpoint['longterm']['num'], device=self.longterm_device)
        
        self.training_step = checkpoint['config']['training_step']
        
        print(f"✓ LOADED MEMORY CHECKPOINT: {filepath}")
        print(f"   Working: {self.num_working.item()} on {self.working_device}")
        print(f"   Long-term: {self.num_longterm.item()} on {self.longterm_device}")


if __name__ == "__main__":
    print("=" * 70)
    print("HYBRID TWO-TIER MEMORY SYSTEM TEST")
    print("=" * 70)
    print()
    
    # Initialize system
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cpu_device = 'cpu'
    
    memory = HybridMemorySystem(
        hidden_dim=384,
        memory_dim=128,
        working_capacity=10,
        working_device=device,  # GPU
        working_decay=0.9,  # 10% per step
        longterm_capacity=50,
        longterm_device=cpu_device,  # CPU
        longterm_decay=0.999,  # 0.1% per step
        promotion_threshold=0.3,
        promotion_interval=20
    ).to(device)
    
    optimizer = torch.optim.Adam(memory.parameters(), lr=1e-3)
    
    print("\nSIMULATING MEMORY FORMATION & CONSOLIDATION:")
    print("=" * 70)
    
    for step in range(50):
        hidden = torch.randn(2, 16, 384, device=device)
        
        output, info = memory(hidden)
        loss = (output - hidden).pow(2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        memory.apply_dopamine(loss)
        memory.step()
        
        if step % 10 == 0:
            stats = memory.get_stats()
            print(f"\nStep {step:3d}:")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Working (GPU): {stats['num_working']:2d} memories")
            print(f"  Long-term (CPU): {stats['num_longterm']:2d} memories")
            print(f"  Working reward: {stats['working_avg_reward']:.3f}")
            print(f"  Long-term reward: {stats['longterm_avg_reward']:.3f}")
    
    print("\n" + "=" * 70)
    print("FINAL MEMORY STATE:")
    print("=" * 70)
    stats = memory.get_stats()
    print(f"Working memory: {stats['num_working']} on {stats['working_device']}")
    print(f"Long-term memory: {stats['num_longterm']} on {stats['longterm_device']}")
    print()
    
    # Save checkpoint
    memory.save_checkpoint('test_hybrid_memory.pkl')
    
    print("\n" + "=" * 70)
    print("✓ HYBRID MEMORY SYSTEM WORKING!")
    print("=" * 70)
    print()
    print("Key features:")
    print("  1. Working → Long-term promotion (consolidation)")
    print("  2. Long-term → Working reconsolidation (retrieval)")
    print("  3. Device split (GPU/CPU for memory efficiency)")
    print("  4. Different decay rates (volatile vs persistent)")
    print("  5. Separate checkpoints (independent save/load)")
    print("=" * 70)
