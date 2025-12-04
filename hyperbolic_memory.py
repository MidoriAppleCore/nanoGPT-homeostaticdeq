"""
Hyperbolic Memory Manifold - Poincaré Ball Model

Memory embeddings live in hyperbolic space (Poincaré ball).
- Center = general/abstract concepts
- Boundary = specific instances
- Distance = semantic dissimilarity
- Navigation = geodesic paths

The reflex module learns to navigate this curved geometry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math


class PoincareManifold:
    """
    Poincaré ball model of hyperbolic space.
    
    M = {x ∈ ℝⁿ : ||x|| < 1}
    
    Distance formula:
    d(u,v) = arccosh(1 + 2||u-v||²/((1-||u||²)(1-||v||²)))
    """
    
    def __init__(self, dim: int, c: float = 1.0, eps: float = 1e-5):
        """
        Args:
            dim: embedding dimension
            c: curvature (c=1 is unit hyperbolic, c→0 is Euclidean)
            eps: numerical stability constant
        """
        self.dim = dim
        self.c = c
        self.eps = eps
        self.max_norm = 1.0 - eps
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project points onto Poincaré ball (enforce ||x|| < 1)"""
        norm = x.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return x / torch.maximum(norm / self.max_norm, torch.ones_like(norm))
    
    def distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Poincaré distance between points u and v.
        
        d(u,v) = arccosh(1 + 2||u-v||²/((1-||u||²)(1-||v||²)))
        """
        sqrt_c = math.sqrt(self.c)
        
        # Squared norms
        norm_u_sq = torch.sum(u * u, dim=-1, keepdim=True).clamp(0, self.max_norm**2)
        norm_v_sq = torch.sum(v * v, dim=-1, keepdim=True).clamp(0, self.max_norm**2)
        
        # Squared distance in ambient space
        diff = u - v
        dist_sq = torch.sum(diff * diff, dim=-1, keepdim=True)
        
        # Hyperbolic distance formula
        numerator = 2 * dist_sq
        denominator = (1 - norm_u_sq) * (1 - norm_v_sq)
        
        # arccosh(1 + x) for numerical stability
        x = numerator / denominator.clamp_min(self.eps)
        return torch.acosh(1 + x) / sqrt_c
    
    def exponential_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map: move from x along tangent vector v.
        This is how we navigate in hyperbolic space!
        
        exp_x(v) = x ⊕ tanh(||v||/2) * v/||v||
        where ⊕ is Möbius addition
        """
        sqrt_c = math.sqrt(self.c)
        
        v_norm = v.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        
        # Scaling factor
        second_term = torch.tanh(sqrt_c * v_norm / 2) * v / (sqrt_c * v_norm)
        
        # Möbius addition
        return self.mobius_add(x, second_term)
    
    def logarithmic_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map: find tangent vector at x pointing toward y.
        This is the inverse of exponential_map.
        
        log_x(y) = (2/√c) * artanh(√c||(-x)⊕y||) * ((-x)⊕y)/||(-x)⊕y||
        
        Used for:
        - Path compression (find direction from path_summary to current_node)
        - Gradient descent in hyperbolic space
        """
        sqrt_c = math.sqrt(self.c)
        
        # Compute (-x) ⊕ y (Möbius addition from -x to y)
        neg_x = -x
        diff = self.mobius_add(neg_x, y)
        
        diff_norm = diff.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        
        # artanh(√c * ||diff||)
        # Use safe artanh to avoid numerical issues at boundary
        scaled_norm = (sqrt_c * diff_norm).clamp(-self.max_norm, self.max_norm)
        artanh_term = torch.atanh(scaled_norm)
        
        # Scale and normalize
        tangent = (2.0 / sqrt_c) * artanh_term * diff / diff_norm
        
        return tangent
    
    def mobius_add(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Möbius addition (non-commutative!)
        
        u ⊕ v = ((1+2c⟨u,v⟩+c||v||²)u + (1-c||u||²)v) / (1+2c⟨u,v⟩+c²||u||²||v||²)
        """
        dot_uv = torch.sum(u * v, dim=-1, keepdim=True)
        norm_u_sq = torch.sum(u * u, dim=-1, keepdim=True).clamp(0, self.max_norm**2)
        norm_v_sq = torch.sum(v * v, dim=-1, keepdim=True).clamp(0, self.max_norm**2)
        
        numerator = (1 + 2*self.c*dot_uv + self.c*norm_v_sq) * u + (1 - self.c*norm_u_sq) * v
        denominator = 1 + 2*self.c*dot_uv + self.c*self.c*norm_u_sq*norm_v_sq
        
        return self.project(numerator / denominator.clamp_min(self.eps))
    
    def parallel_transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport vector v from point x to point y.
        Needed for proper gradient descent in curved space.
        """
        # Simplified formula for Poincaré ball
        return v  # TODO: Implement proper parallel transport if needed


class HyperbolicEncoder(nn.Module):
    """
    Encoder that maps text → hyperbolic embeddings.
    
    Architecture:
    - Token embedding (Euclidean)
    - Transformer layers (Euclidean)
    - Exponential map (project to hyperbolic)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_layers: int = 4,
        num_heads: int = 6,
        curvature: float = 1.0
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.manifold = PoincareManifold(dim=embedding_dim, c=curvature)
        
        # Euclidean encoder (standard transformer)
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, embedding_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection to hyperbolic space
        self.to_tangent = nn.Linear(embedding_dim, embedding_dim)
        self.ln = nn.LayerNorm(embedding_dim)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [batch, seq_len] token indices
        
        Returns:
            hyperbolic_embedding: [batch, embedding_dim] point on Poincaré ball
        """
        # Euclidean encoding
        x = self.token_embedding(tokens)
        seq_len = x.shape[1]
        x = x + self.pos_embedding[:, :seq_len, :]
        
        x = self.transformer(x)
        x = x.mean(dim=1)  # Pool over sequence
        
        # Map to hyperbolic space
        x = self.ln(x)
        tangent_vector = self.to_tangent(x)
        
        # Exponential map from origin
        origin = torch.zeros_like(tangent_vector)
        hyperbolic_embedding = self.manifold.exponential_map(origin, tangent_vector)
        
        return hyperbolic_embedding


class HyperbolicContrastiveLoss(nn.Module):
    """
    Contrastive loss in hyperbolic space.
    
    Pulls positive pairs closer via geodesics.
    Pushes negative pairs apart via geodesics.
    """
    
    def __init__(self, manifold: PoincareManifold, temperature: float = 0.1):
        super().__init__()
        self.manifold = manifold
        self.temperature = temperature
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor: [batch, dim] anchor points
            positive: [batch, dim] positive pairs
            negatives: [batch, num_neg, dim] negative samples
        
        Returns:
            loss: InfoNCE loss in hyperbolic space
        """
        # Hyperbolic distances
        pos_dist = self.manifold.distance(anchor, positive).squeeze(-1)
        
        # Distance to each negative
        batch_size, num_neg, dim = negatives.shape
        anchor_expanded = anchor.unsqueeze(1).expand(-1, num_neg, -1)
        neg_dist = self.manifold.distance(
            anchor_expanded.reshape(-1, dim),
            negatives.reshape(-1, dim)
        ).reshape(batch_size, num_neg)
        
        # InfoNCE with negative distances (smaller dist = more similar)
        logits = torch.cat([
            -pos_dist.unsqueeze(1),  # Negative because smaller is better
            -neg_dist
        ], dim=1) / self.temperature
        
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss


class HyperbolicMemoryRetrieval(nn.Module):
    """
    Memory retrieval in hyperbolic space with DOPAMINERGIC PLASTICITY.
    
    The reflex module navigates the Poincaré ball via learned queries.
    Memories that help prediction get strengthened (dopamine-modulated LTP).
    
    Learning modes:
    - frozen: Memory embeddings fixed (baseline)
    - adaptive: Memory embeddings are parameters (co-trained)
    - dopaminergic: Adaptive + reward-modulated updates (biological)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int,
        k_neighbors: int = 16,
        curvature: float = 1.0,
        alpha: float = 0.1,
        learning_mode: str = 'dopaminergic',  # frozen | adaptive | dopaminergic
        dopamine_scale: float = 1.0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.learning_mode = learning_mode
        self.dopamine_scale = dopamine_scale
        
        self.manifold = PoincareManifold(dim=memory_dim, c=curvature)
        
        # Query network: hidden state → tangent vector (direction to move)
        self.query_network = nn.Sequential(
            nn.Linear(hidden_dim, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, memory_dim)
        )
        
        # Current position on manifold (learnable starting point)
        self.query_origin = nn.Parameter(torch.zeros(1, memory_dim))
        
        # Project retrieved hyperbolic points back to hidden space
        self.memory_projection = nn.Linear(memory_dim, hidden_dim)
        
        # Memory storage - will be converted to Parameter if adaptive
        self.register_buffer('memory_embeddings', torch.zeros(1, memory_dim))
        
        # Track retrieval history for dopamine modulation
        self.register_buffer('retrieval_counts', torch.zeros(1))
        self.register_buffer('reward_history', torch.zeros(1))
    
    def load_hyperbolic_manifold(self, embeddings: torch.Tensor):
        """
        Load pre-trained hyperbolic memory embeddings.
        
        Args:
            embeddings: [num_memories, memory_dim] points on Poincaré ball
        """
        # Project to ensure all points are in valid ball
        embeddings = self.manifold.project(embeddings)
        
        # Convert to parameter if adaptive learning is enabled
        if self.learning_mode in ['adaptive', 'dopaminergic']:
            # Make memories learnable!
            self.memory_embeddings = nn.Parameter(embeddings)
            print(f"✓ Loaded {len(embeddings)} hyperbolic memories (ADAPTIVE)")
        else:
            # Keep frozen
            self.memory_embeddings = embeddings
            print(f"✓ Loaded {len(embeddings)} hyperbolic memories (FROZEN)")
        
        # Initialize tracking buffers
        num_memories = len(embeddings)
        self.retrieval_counts = torch.zeros(num_memories, device=embeddings.device)
        self.reward_history = torch.zeros(num_memories, device=embeddings.device)
    
    def retrieve_geodesic_neighbors(
        self,
        query: torch.Tensor,
        k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve k nearest neighbors via geodesic distance.
        
        Args:
            query: [batch, memory_dim] query points on Poincaré ball
            k: number of neighbors (default: self.k_neighbors)
        
        Returns:
            neighbors: [batch, k, memory_dim] retrieved memory points
            distances: [batch, k] geodesic distances
        """
        k = k or self.k_neighbors
        batch_size = query.shape[0]
        num_memories = self.memory_embeddings.shape[0]
        
        # Compute all pairwise geodesic distances
        query_expanded = query.unsqueeze(1).expand(-1, num_memories, -1)
        memory_expanded = self.memory_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        distances = self.manifold.distance(
            query_expanded.reshape(-1, self.memory_dim),
            memory_expanded.reshape(-1, self.memory_dim)
        ).reshape(batch_size, num_memories)
        
        # Get top-k nearest (smallest distance)
        topk_distances, topk_indices = torch.topk(
            distances, k=k, dim=-1, largest=False
        )
        
        # Gather neighbor embeddings
        neighbors = self.memory_embeddings[topk_indices]
        
        return neighbors, topk_distances
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Navigate hyperbolic memory and inject into hidden states.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            injected: [batch, seq_len, hidden_dim] with memory force
        """
        batch, seq_len, hidden_dim = hidden_states.shape
        
        # Generate tangent vector (direction to move)
        tangent = self.query_network(hidden_states.mean(dim=1))  # [batch, memory_dim]
        
        # Navigate: move from origin along tangent vector
        query_origin = self.manifold.project(self.query_origin.expand(batch, -1))
        query_point = self.manifold.exponential_map(query_origin, tangent)
        
        # Retrieve geodesic neighbors
        neighbors, distances = self.retrieve_geodesic_neighbors(query_point)
        
        # Weighted combination (inverse distance weighting in hyperbolic space)
        weights = F.softmax(-distances / 0.1, dim=-1)  # Smaller distance = higher weight
        retrieved = torch.sum(weights.unsqueeze(-1) * neighbors, dim=1)  # [batch, memory_dim]
        
        # Project back to hidden space
        memory_force = self.memory_projection(retrieved)  # [batch, hidden_dim]
        
        # Inject into hidden states
        memory_force = memory_force.unsqueeze(1).expand(-1, seq_len, -1)
        injected = hidden_states + self.alpha * memory_force
        
        return injected
    
    def apply_dopamine_modulation(self, loss: torch.Tensor, retrieved_indices: torch.Tensor):
        """
        Apply dopaminergic learning: modulate memory gradients based on reward.
        
        Biological inspiration:
        - Dopamine = prediction error signal
        - LTP/LTD = synaptic plasticity
        - Reward prediction error modulates learning rate
        
        Args:
            loss: scalar loss value (lower = better = more reward)
            retrieved_indices: [batch, k] indices of retrieved memories
        """
        if self.learning_mode != 'dopaminergic' or not self.training:
            return
        
        # Compute reward signal (negative loss = positive reward)
        reward = -loss.detach()
        
        # Update retrieval statistics
        unique_indices = torch.unique(retrieved_indices.flatten())
        self.retrieval_counts[unique_indices] += 1
        self.reward_history[unique_indices] += reward.item()
        
        # Modulate gradients of retrieved memories
        # This happens automatically via backprop, but we can add explicit scaling
        if self.memory_embeddings.grad is not None:
            # Scale gradients by dopamine signal
            # Positive reward → stronger updates (like dopamine-enhanced LTP)
            dopamine_multiplier = 1.0 + self.dopamine_scale * torch.tanh(reward)
            
            # Only modulate gradients of retrieved memories (sparse updates)
            grad_mask = torch.zeros_like(self.memory_embeddings)
            grad_mask[unique_indices] = dopamine_multiplier
            
            # Apply modulation
            self.memory_embeddings.grad *= (1.0 + grad_mask.mean(dim=-1, keepdim=True))
    
    def get_memory_statistics(self):
        """
        Get statistics about memory usage and learning.
        
        Returns:
            dict with memory metrics
        """
        if self.retrieval_counts.sum() == 0:
            return {}
        
        avg_reward = self.reward_history / (self.retrieval_counts + 1e-8)
        
        return {
            'total_retrievals': self.retrieval_counts.sum().item(),
            'active_memories': (self.retrieval_counts > 0).sum().item(),
            'avg_reward': avg_reward.mean().item(),
            'best_memory_reward': avg_reward.max().item(),
            'worst_memory_reward': avg_reward.min().item(),
            'memory_utilization': (self.retrieval_counts > 0).float().mean().item()
        }


if __name__ == "__main__":
    print("Testing Hyperbolic Memory Manifold with Dopaminergic Plasticity...")
    print("=" * 70)
    
    # Test Poincaré distance
    manifold = PoincareManifold(dim=128)
    
    # Points near origin (general concepts)
    u = torch.randn(4, 128) * 0.1
    v = torch.randn(4, 128) * 0.1
    u = manifold.project(u)
    v = manifold.project(v)
    
    dist = manifold.distance(u, v)
    print(f"Distance between general concepts: {dist.mean():.4f}")
    
    # Points near boundary (specific instances)
    u_specific = torch.randn(4, 128) * 0.9
    v_specific = torch.randn(4, 128) * 0.9
    u_specific = manifold.project(u_specific)
    v_specific = manifold.project(v_specific)
    
    dist_specific = manifold.distance(u_specific, v_specific)
    print(f"Distance between specific concepts: {dist_specific.mean():.4f}")
    
    print("\n" + "=" * 70)
    print("Testing Dopaminergic Memory Learning")
    print("=" * 70)
    
    # Create retrieval module in dopaminergic mode
    retrieval = HyperbolicMemoryRetrieval(
        hidden_dim=384,
        memory_dim=128,
        k_neighbors=8,
        learning_mode='dopaminergic',
        dopamine_scale=0.5
    )
    
    # Fake memory embeddings
    fake_memories = torch.randn(100, 128) * 0.5
    fake_memories = manifold.project(fake_memories)
    retrieval.load_hyperbolic_manifold(fake_memories)
    
    # Simulate training loop
    optimizer = torch.optim.Adam(retrieval.parameters(), lr=1e-3)
    
    print("\nSimulating training with dopamine modulation:")
    for step in range(5):
        # Forward pass
        hidden = torch.randn(2, 16, 384)
        output = retrieval(hidden)
        
        # Fake loss (memories should learn to minimize this)
        loss = (output - hidden).pow(2).mean()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Apply dopamine modulation before optimizer step
        # (In real training, this would use actual prediction error)
        fake_indices = torch.randint(0, 100, (2, 8))  # Simulate retrieval
        retrieval.apply_dopamine_modulation(loss, fake_indices)
        
        optimizer.step()
        
        # Project memories back to Poincaré ball
        with torch.no_grad():
            retrieval.memory_embeddings.data = manifold.project(retrieval.memory_embeddings.data)
        
        print(f"  Step {step}: Loss = {loss.item():.4f}")
    
    # Check memory statistics
    stats = retrieval.get_memory_statistics()
    print("\nMemory Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "=" * 70)
    print(f"✓ Input shape: {hidden.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Memory embeddings are Parameters: {isinstance(retrieval.memory_embeddings, nn.Parameter)}")
    print(f"✓ Hyperbolic dopaminergic memory retrieval working!")
    print("=" * 70)
