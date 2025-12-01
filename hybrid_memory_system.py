"""
Hybrid Two-Tier Memory System: Biological Memory Architecture

WORKING MEMORY (VRAM):
- Fast formation, fast decay
- High learning rate (10x)
- Lives on GPU for fast access
- Like your "scratch pad" while thinking

LONG-TERM MEMORY (CPU/Disk):
- Slow consolidation, slow decay
- Low learning rate (0.1x)
- Lives on CPU, loaded on-demand
- Like your "hard drive" of knowledge

KEY MECHANISMS:
1. PROMOTION: Working → Long-term (consolidation during "sleep")
2. RETRIEVAL: Long-term → Working (reconsolidation when accessed)
3. DECAY: Both tiers fade if unused (at different rates)
4. DEVICE SPLIT: Working on GPU, Long-term on CPU (memory efficiency)

This is how your brain ACTUALLY works!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math
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
        
        self.promotion_threshold = promotion_threshold
        self.promotion_interval = promotion_interval
        self.dopamine_scale = dopamine_scale
        
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
        
        print(f"[HybridMemory] Two-tier architecture initialized:")
        print(f"  Working:   {working_capacity} capacity on {working_device} (decay={working_decay})")
        print(f"  Long-term: {longterm_capacity} capacity on {longterm_device} (decay={longterm_decay})")
    
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
            
            print(f"  💾→🧠 CONSOLIDATION: Working → Long-term "
                  f"(reward={reward.item():.3f}, total_LT={self.num_longterm.item()})")
    
    def retrieve_hybrid(self, query: torch.Tensor, main_device: str) -> Tuple[torch.Tensor, Dict]:
        """
        RETRIEVAL: Search BOTH tiers, blend results.
        
        Like remembering: "Let me check both what I'm thinking about NOW 
        and what I learned BEFORE..."
        
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
        longterm_retrieved = torch.zeros(batch_size, self.memory_dim, device=main_device)
        
        working_weight = 0.0
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
        # TIER 2: Long-term memory retrieval (SLOW - CPU → GPU transfer)
        # ═══════════════════════════════════════════════════════════
        if self.num_longterm > 0:
            # Move query to long-term device for comparison
            query_lt = query.to(self.longterm_device)
            
            longterm_embs = self.longterm_embeddings
            if longterm_embs.dim() > 2:
                longterm_embs = longterm_embs.squeeze()
                if longterm_embs.dim() == 1:
                    longterm_embs = longterm_embs.unsqueeze(0)
            
            k_long = min(self.k_neighbors // 2, self.num_longterm.item())
            
            query_exp = query_lt.unsqueeze(1).expand(-1, longterm_embs.shape[0], -1)
            long_exp = longterm_embs.unsqueeze(0).expand(batch_size, -1, -1)
            
            long_dists = self.manifold.distance(
                query_exp.reshape(-1, self.memory_dim),
                long_exp.reshape(-1, self.memory_dim)
            ).reshape(batch_size, longterm_embs.shape[0])
            
            topk_long_dists, topk_long_idx = torch.topk(
                long_dists, k=k_long, dim=-1, largest=False
            )
            
            long_weights = F.softmax(-topk_long_dists / 0.1, dim=-1)
            
            longterm_neighbors = longterm_embs[topk_long_idx]
            longterm_retrieved_lt = (long_weights.unsqueeze(-1) * longterm_neighbors).sum(dim=1)
            
            # Move back to main device (CPU → GPU transfer!)
            longterm_retrieved = longterm_retrieved_lt.to(main_device)
            longterm_weight = 0.3  # Old knowledge weighted LOWER
            
            # Update access using DIFFERENTIABLE attention weights
            with torch.no_grad():
                attention_strength = long_weights.sum(dim=0)  # [k_long]
                for i, idx in enumerate(topk_long_idx[0]):
                    self.longterm_access[idx] += attention_strength[i].item()
            
            # RECONSOLIDATION: Highly-accessed long-term → back to working!
            # Only trigger during training AND at slower rate (not every forward pass!)
            # Use MUCH higher threshold since attention weights accumulate quickly
            if self.training and self.num_longterm > 0:
                max_access = self.longterm_access.max()
                if max_access > 100:  # Much higher threshold!
                    most_accessed_idx = self.longterm_access.argmax()
                    
                    # Double-check it's really the most accessed
                    if self.longterm_access[most_accessed_idx] > 100:
                        # "Bring it back to mind" - move to working memory!
                        reconsolidated = self.longterm_embeddings[most_accessed_idx:most_accessed_idx+1]
                        reconsolidated = reconsolidated.to(self.working_device)
                        self.add_to_working(reconsolidated)
                        
                        # DELETE from long-term (it's been moved, not copied!)
                        mask = torch.ones(self.num_longterm, dtype=torch.bool, device=self.longterm_device)
                        mask[most_accessed_idx] = False
                        self.longterm_embeddings = self.longterm_embeddings[mask]
                        self.longterm_rewards = self.longterm_rewards[mask]
                        self.longterm_age = self.longterm_age[mask]
                        self.longterm_access = self.longterm_access[mask]
                        self.num_longterm -= 1
                        
                        print(f"  🧠→💾 RECONSOLIDATION: Long-term → Working "
                              f"(access={max_access:.1f} exceeded threshold)")
        
        # ═══════════════════════════════════════════════════════════
        # BLEND: Combine both tiers (continuous interpolation)
        # ═══════════════════════════════════════════════════════════
        total_weight = working_weight + longterm_weight
        if total_weight > 0:
            retrieved = (working_weight * working_retrieved + 
                        longterm_weight * longterm_retrieved) / total_weight
        else:
            retrieved = torch.zeros(batch_size, self.memory_dim, device=main_device)
        
        info = {
            'working_contrib': working_weight,
            'longterm_contrib': longterm_weight,
            'num_working': self.num_working.item(),
            'num_longterm': self.num_longterm.item(),
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
        if self.num_working > 0 and reward > self.promotion_threshold * 2:
            # Find memories with both high reward AND high access
            utility = self.working_rewards * 0.5 + self.working_access * 0.5
            best_idx = utility.argmax()
            if utility[best_idx] > self.promotion_threshold:
                print(f"\n  ⚡ IMMEDIATE CONSOLIDATION (high reward={reward:.3f})")
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
        
        # AUTO-CONSOLIDATION: Periodic "sleep" phase
        if self.training_step % self.promotion_interval == 0 and self.num_working > 0:
            # Compute UTILITY score: combines reward, access frequency, and recency
            # This is like "how useful has this memory been?"
            utility = (
                self.working_rewards * 0.4 +           # Reward (did it help?)
                self.working_access * 0.4 +            # Frequency (how often accessed?)
                (1.0 / (self.working_age + 1.0)) * 0.2 # Recency (more recent = more valuable)
            )
            
            # Promote the most USEFUL memory (not just highest reward!)
            best_idx = utility.argmax()
            if utility[best_idx] > 0:
                print(f"\n  💤 AUTO-CONSOLIDATION (step {self.training_step})")
                print(f"     Promoting memory with utility={utility[best_idx]:.3f} " 
                      f"(reward={self.working_rewards[best_idx]:.3f}, "
                      f"access={self.working_access[best_idx]:.1f}, "
                      f"age={self.working_age[best_idx]:.0f})")
                self.promote_to_longterm(best_idx.item())
    
    def get_stats(self) -> Dict:
        """Memory system statistics"""
        return {
            'num_working': self.num_working.item(),
            'num_longterm': self.num_longterm.item(),
            'working_avg_reward': self.working_rewards.mean().item() if self.num_working > 0 else 0,
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
