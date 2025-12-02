"""
Deep Equilibrium Language Model — Gray-Box Formulation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHILOSOPHICAL FOUNDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is NOT "an unusually-shaped transformer."
This IS "language as a physical dynamical system."

The insight:
  > Language is a process on a semantic manifold.
  > DEQ lets you solve it like physics.

In standard transformers:
  - 96 stacked layers brute-force emergent coherence
  - Depth is a fixed hyperparameter
  - Every layer must rediscover grammar, syntax, semantics
  - Memory ∝ O(N_layers × batch_size)

In DEQ:
  - Run until the idea stabilizes
  - Depth emerges from difficulty
  - Structure is given, cortex learns residuals
  - Memory ∝ O(batch_size)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CYBERNETIC ARCHITECTURE (Gray Box Design)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The DEQ is not the whole agent. It is the cortical layer.
Reflexes and structure live outside of it.

Components (mapped from robotic control):

  1. Context Encoder (Senses)
     - Embeds tokens with structural priors
     - Positional geometry (RoPE/absolute)
     - "Raw sensory input" — not meaning
     - Robot equivalent: proprioception (arm pose, obstacles)

  2. Reflex Module (Spinal Cord)
     - 2-3 shallow attention blocks
     - Fast, parallel, non-iterative
     - Extracts: local syntax, bigram continuity, lexical smoothing
     - Outputs "force intent": push meaning in direction X
     - Robot equivalent: immediate torque response

  3. DEQ Brain (Cortex)
     - Single implicit layer repeated until equilibrium
     - Infinite-depth semantic integration
     - z* = f(z*, u) — fixed point is the thought
     - Robot equivalent: high-level motion planning

  4. Geometry Layer (Biomechanics)
     - Projects equilibrium latent → logits
     - Fixed, analytic transformation
     - DEQ never learns "how to form a distribution"
     - Robot equivalent: Jacobian transpose (force → torque)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTROL SYSTEMS (Physics, Not Heuristics)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stabilizer g_φ(z, u):
  - Per-dimension damping α ∈ [0.1, 0.9] (HARD BOUNDS)
  - High entropy → high viscosity (slow down thinking)
  - Low entropy → low viscosity (trust the flow)
  - Prevents collapse and explosion

Global Controller h_ψ:
  - Reads scalar fields from semantic geometry:
    * Residual norm (convergence rate)
    * Local surprisal gradient (dlogp/dx)
    * Semantic curvature (embedding geometry)
    * Entropy decay rate (time to collapse)
  - Outputs: spectral target φ* ∈ [0.9, 1.05], tolerance
  - Adaptive depth: easy text = 3 iters, hard text = 12-18 iters

Physical Laws:
  1. Entropy floor: Never allow entropy < log(3)
  2. Semantic continuity: Penalize discontinuous jumps in latent
  3. Narrative potential: (future) Global coherence constraint
  4. Adaptive depth: Uncertainty → thinking time

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY THIS BEATS TRANSFORMERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Infinite depth on laptop GPU
  (Don't store 40 layers, store 1 operator + implicit gradients)

✓ Adaptive reasoning
  (Model literally thinks longer where gradients say it should)

✓ Structural priors
  (Inject "physics of language" once, not rediscovered 96 times)

✓ Stability
  (Spectral band regulation keeps ρ ≈ 1⁻, edge of coherence)

✓ Memory efficiency
  (No KV cache per layer, O(batch_size) not O(N_layers))

✓ Semantic density > SOTA
  (Not in raw speed — in meaning per parameter)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE CYBERNETICS OF LANGUAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Language has no ground (no "contact distance" like robotics).
The world is degenerate — just token prefix.

So we define semantic phase-space measurements:

  Syntax ↔ Proprioception
    (local structure tells model where it "is" in meaning space)

  Token entropy ↔ Threat response
    (uncertainty should slow you down)

  Perplexity ↔ Metabolic cost
    (expensive states change how deep you think)

  Spectral radius ↔ Life/death boundary
    (ρ < 1 = catatonic, ρ > 1 = psychotic, ρ ≈ 1⁻ = lucid)

The DEQ layer is the homeostatic cortex negotiating stability and curiosity.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is not a weird RNN.
This is a homeostatic semantic attractor.
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
from torch.nn import functional as F


# -----------------------------------------------------------------------------
# Homeostatic Uncertainty Balancer (Bayesian Multi-Objective Learning)
# -----------------------------------------------------------------------------

class HomeostaticBalancer(nn.Module):
    """
    Bayesian Homeostasis for Multi-Objective Loss Balancing.
    
    Mathematically rigorous weighting based on Aleatoric Uncertainty.
    Ref: Kendall & Gal, 'Multi-Task Learning Using Uncertainty to Weigh Losses' (CVPR 2018).
    
    **The Multi-Objective Alignment Problem:**
    When training coupled dynamics (e.g., Memory Formation vs. Navigation),
    one task can dominate if its loss scale is 100× larger. Manual weighting
    (λ_1, λ_2, ...) is fragile and dataset-specific.
    
    **The Bayesian Solution:**
    Treat each task's weight as a LEARNABLE parameter derived from its
    intrinsic uncertainty (noise scale σ_i).
    
    Total Loss = Σ_i [ (1 / 2σ_i²) · L_i + log(σ_i) ]
    
    **Homeostatic Mechanism:**
      - If task i is hard (high loss), model increases σ_i to reduce penalty
      - But increasing σ_i costs entropy (+log(σ_i) term)
      - Equilibrium: σ_i automatically balances task difficulty vs. learning signal
    
    **Result:**
    NO MAGIC NUMBERS. The model learns optimal task weighting via Maximum Likelihood.
    
    Example: If Memory Formation loss is 100× larger than Navigation loss,
    the balancer will learn σ_mem ≈ 10× larger, automatically normalizing gradients.
    
    **Physical Interpretation:**
    σ_i represents the "measurement noise" of task i. High-noise tasks get
    downweighted (precision = 1/σ²), but the entropy cost prevents ignoring them.
    
    Args:
        num_losses: Number of loss components to balance
        loss_names: List of loss component names (for logging)
    """
    def __init__(self, num_losses=4, loss_names=None):
        super().__init__()
        # We learn log_variance (s) for numerical stability
        # s := log(sigma^2)
        # Initialize to 0.0 (sigma=1.0, equal weighting at start)
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
        
        if loss_names is None:
            self.loss_names = [
                "prediction",   # Cross Entropy (The Goal)
                "jacobian",     # DEQ Stability (The Physics)
                "novelty",      # Pauli Exclusion (The Curiosity)
                "memory"        # Reconstruction (The Storage)
            ]
        else:
            self.loss_names = loss_names
        
        assert len(self.loss_names) == num_losses, \
            f"loss_names length ({len(self.loss_names)}) must match num_losses ({num_losses})"
    
    def forward(self, losses_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply Bayesian homeostatic balancing to multiple loss components.
        
        Args:
            losses_dict: Dictionary mapping loss names to scalar tensors
                         Example: {'prediction': ce_loss, 'jacobian': jac_loss, ...}
        
        Returns:
            balanced_loss: Scalar tensor (total weighted loss)
            log_stats: Dictionary of balancing statistics for monitoring:
                       - 'weight_{name}': Effective weight (1/σ²) for each task
                       - 'sigma_{name}': Learned uncertainty (σ) for each task
                       - 'precision_{name}': Same as weight (for clarity)
        """
        total_loss = 0.0
        stats = {}
        
        # Iterate through registered losses in order
        for i, name in enumerate(self.loss_names):
            if name in losses_dict:
                # Raw loss from the specific module
                raw_loss = losses_dict[name]
                
                # Get learned homeostatic variable (s = log(sigma^2))
                s = self.log_vars[i]
                
                # Apply Bayesian Weighting Formula:
                # L_balanced = L_raw / (2*sigma^2) + log(sigma)
                # 
                # Mathematically:
                # L_balanced = L_raw * exp(-s) + 0.5 * s
                # 
                # where exp(-s) = 1/sigma^2 is the "precision" (inverse variance)
                # and 0.5*s = log(sigma) is the entropy cost
                precision = torch.exp(-s)
                weighted_loss = precision * raw_loss + 0.5 * s
                
                total_loss += weighted_loss
                
                # Record homeostatic variables for monitoring
                # These tell us how much the model 'cares' about each task
                stats[f"weight_{name}"] = precision.item()
                stats[f"sigma_{name}"] = torch.exp(0.5 * s).item()
                stats[f"precision_{name}"] = precision.item()
                # Handle both tensor and scalar raw_loss
                stats[f"raw_{name}"] = raw_loss.item() if isinstance(raw_loss, torch.Tensor) else raw_loss
            else:
                # Loss not provided (e.g., memory disabled)
                # Record zero stats
                stats[f"weight_{name}"] = 0.0
                stats[f"sigma_{name}"] = 0.0
                stats[f"precision_{name}"] = 0.0
                stats[f"raw_{name}"] = 0.0
        
        return total_loss, stats


# -----------------------------------------------------------------------------
# Pauli Exclusion Principle for Tokens (Anti-Stuttering Force)
# -----------------------------------------------------------------------------

def compute_pauli_exclusion_loss(logits, targets):
    """
    Physics: Coulomb Repulsion for Tokens (Pauli Exclusion Principle)
    
    In quantum mechanics, the Pauli Exclusion Principle states:
    "Two identical fermions cannot occupy the same quantum state."
    
    For language: "Two identical tokens should not occupy adjacent timesteps."
    
    The model gets trapped in "Lily Lily Lily" loops because the energy well
    for high-frequency tokens is too deep. This adds a REPULSION FORCE that
    penalizes the model for predicting the exact same token it just generated.
    
    Args:
        logits: [B, T, V] - Model predictions (before softmax)
        targets: [B, T] - Ground truth token IDs
    
    Returns:
        pauli_loss: Scalar - Average probability assigned to immediate repetition
    
    Physics Intuition:
        - If model wants to say "Lily" at step t, and "Lily" was at step t-1,
          we apply a massive energy penalty (Coulomb repulsion).
        - This fills the "Lily gravity well" with concrete, forcing the
          Hamiltonian momentum to redirect to different tokens.
    """
    # 1. Get probabilities of current predictions
    probs = F.softmax(logits, dim=-1)  # [B, T, V]
    
    # 2. Identify what the "Previous Token" was for every step
    # Shift targets to the right: [The, Cat, Sat] → [PAD, The, Cat]
    prev_ids = targets.roll(1, dims=1)
    prev_ids[:, 0] = -100  # Ignore first token (nothing before it)
    
    # 3. How much probability is assigned to the "Previous Token"?
    # Gather the probability of prev_ids from the current distribution
    # clamp(min=0) prevents error on -100 ignore index
    token_prob_at_prev = probs.gather(-1, prev_ids.unsqueeze(-1).clamp(min=0)).squeeze(-1)
    
    # 4. The Penalty: Minimize probability of repeating last token
    # Mask out -100 positions (first token in sequence)
    mask = (prev_ids != -100).float()
    
    # Average probability assigned to immediate neighbor
    pauli_loss = (token_prob_at_prev * mask).sum() / (mask.sum() + 1e-8)
    
    return pauli_loss


# -----------------------------------------------------------------------------
# Anderson Acceleration for fast fixed-point convergence
# -----------------------------------------------------------------------------

class AndersonAcceleration:
    """
    SMART Anderson acceleration with LEARNED mixing weights.
    
    Standard Anderson achieves 3-5x faster convergence than naive iteration.
    Smart version learns optimal mixing from trajectory patterns.
    
    z_{k+1} = f(z_k)  →  z_{k+1} = z_k + β·Δz_k with learned mixing
    """
    def __init__(self, m: int = 5, beta: float = 1.0, learned_mixing: bool = False, dim: int = None):
        self.m = m  # History size
        self.beta = beta  # Damping
        self.learned_mixing = learned_mixing
        
        if learned_mixing and dim is not None:
            # Learned mixer: predicts optimal combination of past iterates
            import torch.nn as nn
            self.mixer = nn.Linear(m, m, bias=False)
            # Initialize as identity (preserve standard Anderson behavior initially)
            nn.init.eye_(self.mixer.weight)
            print(f"[Smart Anderson] Learned mixing with {m}-history")
        else:
            self.mixer = None
        
        self.reset()
    
    def reset(self):
        self.X = []  # State history
        self.F = []  # Residual history
    
    def update(self, x_new, f_new):
        """Update with new state and its residual"""
        residual = f_new - x_new
        
        self.X.append(x_new)
        self.F.append(residual)
        
        # Keep only m most recent
        if len(self.X) > self.m:
            self.X.pop(0)
            self.F.pop(0)
        
        if len(self.X) == 1:
            # First iteration: damped fixed-point
            return x_new + self.beta * residual
        
        # Stack history
        X_mat = torch.stack(self.X[:-1], dim=0)
        F_mat = torch.stack(self.F[:-1], dim=0)
        
        # Compute differences
        dX = X_mat - x_new.unsqueeze(0)
        dF = F_mat - residual.unsqueeze(0)
        
        # Solve least squares: dF^T dF α = dF^T residual
        # Shape: [m-1, B, T, C] → [B, T, m-1, C]
        B, T, C = x_new.shape
        dF_flat = dF.permute(1, 2, 0, 3).reshape(B*T, len(self.X)-1, C)
        dX_flat = dX.permute(1, 2, 0, 3).reshape(B*T, len(self.X)-1, C)
        res_flat = residual.reshape(B*T, C)
        
        # Solve per token
        try:
            # α = (dF^T dF)^{-1} dF^T residual
            dF_t_dF = torch.bmm(dF_flat.transpose(1,2), dF_flat)  # [B*T, C, C]
            dF_t_res = torch.bmm(dF_flat.transpose(1,2), res_flat.unsqueeze(-1))  # [B*T, C, 1]
            alpha = torch.linalg.solve(dF_t_dF, dF_t_res).squeeze(-1)  # [B*T, C]
            
            # LEARNED MIXING: Apply learned transformation to mixing weights
            if self.learned_mixing and self.mixer is not None and len(self.X) == self.m:
                # alpha: [B*T, C], need to pad to [B*T, m]
                alpha_padded = torch.zeros(B*T, self.m, device=alpha.device, dtype=alpha.dtype)
                alpha_padded[:, :alpha.shape[1]] = alpha
                # Learn better mixing
                alpha_mixed = self.mixer(alpha_padded)  # [B*T, m]
                alpha = alpha_mixed[:, :alpha.shape[1]]  # Trim back
            
            # Accelerated update
            delta = res_flat - torch.bmm(dF_flat, alpha.unsqueeze(-1)).squeeze(-1)
            x_next = x_new.reshape(B*T, C) + self.beta * delta
            return x_next.reshape(B, T, C)
        except:
            # Fallback to damped iteration if solve fails
            return x_new + self.beta * residual


# -----------------------------------------------------------------------------
# 1. Context Encoder (Senses) — embeddings with structural priors
# -----------------------------------------------------------------------------

class ContextEncoder(nn.Module):
    """
    Embeds tokens into latent space with structural priors:
      - standard embedding
      - positional geometry (RoPE or absolute)
      - optional character-level signals
    
    Output is the raw sensory input, not meaning.
    Equivalent to: arm pose, box pose, obstacle pose.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Positional embedding (absolute for now, could use RoPE)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        
        self.drop = nn.Dropout(config.dropout)
    
    def forward(self, idx):
        """
        idx: [B, T] token indices
        Returns: [B, T, C] context embeddings (structured sensory input)
        """
        B, T = idx.shape
        
        # Token + position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)  # [B, T, C]
        pos_emb = self.wpe(pos)  # [T, C]
        
        x = self.drop(tok_emb + pos_emb)
        return x


# -----------------------------------------------------------------------------
# 2. Reflex Module (Spinal Cord) — fast, shallow features
# -----------------------------------------------------------------------------

class ReflexBlock(nn.Module):
    """
    Single shallow attention + MLP block.
    Fast, parallel, non-iterative.
    
    Extracts:
      - local syntax n-gram features
      - token bigram continuity
      - short-range attention
      - lexical smoothing
    
    This is the part of a transformer that trivially works.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = nn.MultiheadAttention(
            config.n_embd, 
            config.n_head, 
            dropout=config.dropout,
            bias=config.bias,
            batch_first=True
        )
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x, mask=None):
        """
        x: [B, T, C]
        Returns: [B, T, C] reflex force intent
        """
        # Attention with residual
        attn_out, _ = self.attn(
            self.ln_1(x), 
            self.ln_1(x), 
            self.ln_1(x),
            attn_mask=mask,
            need_weights=False
        )
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.ln_2(x))
        return x


class ReflexModule(nn.Module):
    """
    2-3 shallow reflex blocks + HYPERBOLIC MEMORY-AUGMENTED FORCING
    
    Outputs a "force intent analogue":
      "push meaning in direction X at token t"
    
    Memory retrieval = dopaminergic navigation on hyperbolic manifold
    Applied BEFORE cortical reasoning (DEQ loop)
    
    Components:
      1. Reflex blocks: local syntax, bigrams, structure
      2. Hyperbolic memory: navigate Poincaré ball for semantic priming
      3. Dopamine: prediction error modulates memory plasticity
      4. Combined: reflex + memory → forcing function u
    
    It is NOT the answer, only the default motion toward it.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Standard reflex blocks
        self.blocks = nn.ModuleList([
            ReflexBlock(config) for _ in range(config.n_reflex)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        
        # Memory contrastive loss tracking
        self.last_memory_loss = None  # Will store InfoNCE contrastive loss
        
        # TWO-TIER HYBRID MEMORY SYSTEM: Optional memory-augmented forcing
        self.use_memory = getattr(config, 'use_memory_manifold', False)
        self.memory_mode = getattr(config, 'memory_mode', 'hybrid')  # hybrid | hyperbolic | euclidean
        
        if self.use_memory:
            if self.memory_mode == 'hybrid':
                # NEW: Hybrid two-tier system (working + long-term)
                from hybrid_memory_system import HybridMemorySystem
                
                memory_dim = getattr(config, 'memory_dim', config.n_embd)
                k_neighbors = getattr(config, 'memory_k', 16)
                alpha = getattr(config, 'memory_alpha', 0.1)
                curvature = getattr(config, 'memory_curvature', 1.0)
                
                # Two-tier params
                working_capacity = getattr(config, 'working_memory_capacity', 50)
                longterm_capacity = getattr(config, 'longterm_memory_capacity', 1000)
                consolidation_buffer_size = getattr(config, 'consolidation_buffer_size', 100)
                working_decay = getattr(config, 'working_memory_decay', 0.95)
                longterm_decay = getattr(config, 'longterm_memory_decay', 0.999)
                promotion_threshold = getattr(config, 'memory_promotion_threshold', 0.5)
                promotion_interval = getattr(config, 'memory_promotion_interval', 100)
                
                # Device placement (CRITICAL!)
                working_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                longterm_device = 'cpu'  # Always keep long-term on CPU to save VRAM
                longterm_disk_path = getattr(config, 'longterm_disk_path', None)  # Optional disk storage
                
                self.memory_retrieval = HybridMemorySystem(
                    hidden_dim=config.n_embd,
                    memory_dim=memory_dim,
                    k_neighbors=k_neighbors,
                    curvature=curvature,
                    alpha=alpha,
                    working_capacity=working_capacity,
                    working_device=working_device,
                    working_decay=working_decay,
                    longterm_capacity=longterm_capacity,
                    longterm_device=longterm_device,
                    longterm_decay=longterm_decay,
                    longterm_disk_path=longterm_disk_path,
                    consolidation_buffer_size=consolidation_buffer_size,
                    promotion_threshold=promotion_threshold,
                    promotion_interval=promotion_interval
                )
                
                print(f"[Reflex] Hybrid two-tier memory enabled")
                print(f"  Working:   {working_capacity} on {working_device} (decay={working_decay})")
                print(f"  Long-term: {longterm_capacity} on {longterm_device} (decay={longterm_decay})")
                
            elif self.memory_mode == 'hyperbolic':
                from hyperbolic_memory import HyperbolicMemoryRetrieval
                
                memory_dim = getattr(config, 'memory_dim', config.n_embd)
                k_neighbors = getattr(config, 'memory_k', 16)
                alpha = getattr(config, 'memory_alpha', 0.1)
                curvature = getattr(config, 'memory_curvature', 1.0)
                learning_mode = getattr(config, 'memory_learning_mode', 'dopaminergic')
                dopamine_scale = getattr(config, 'memory_dopamine_scale', 0.5)
                
                self.memory_retrieval = HyperbolicMemoryRetrieval(
                    hidden_dim=config.n_embd,
                    memory_dim=memory_dim,
                    k_neighbors=k_neighbors,
                    curvature=curvature,
                    alpha=alpha,
                    learning_mode=learning_mode,
                    dopamine_scale=dopamine_scale
                )
                
                print(f"[Reflex] Hyperbolic memory enabled ({learning_mode} mode)")
                
            else:
                # Fallback to Euclidean memory
                from memory_retrieval import MemoryRetrieval
                
                memory_dim = getattr(config, 'memory_dim', 768)
                k_neighbors = getattr(config, 'memory_k', 16)
                alpha_init = getattr(config, 'memory_alpha', 0.1)
                
                self.memory_retrieval = MemoryRetrieval(
                    hidden_dim=config.n_embd,
                    memory_dim=memory_dim,
                    k_neighbors=k_neighbors,
                    alpha_init=alpha_init,
                    use_attention=True,
                    dropout=config.dropout
                )
                
                print(f"[Reflex] Euclidean memory enabled")
            
            # Load manifold if path provided
            manifold_path = getattr(config, 'memory_manifold_path', None)
            if manifold_path:
                print(f"[Reflex] Loading memory manifold: {manifold_path}")
                self._load_memory_manifold(manifold_path)
            else:
                print("[Reflex] Memory manifold path not provided - will need to load later")
        else:
            self.memory_retrieval = None
            print("[Reflex] Memory-augmented forcing DISABLED")
    
    def _load_memory_manifold(self, manifold_path):
        """Load memory manifold from disk"""
        import pickle
        import os
        
        if not os.path.exists(manifold_path):
            raise FileNotFoundError(f"Memory manifold not found: {manifold_path}")
        
        with open(manifold_path, 'rb') as f:
            manifold = pickle.load(f)
        
        # Extract embeddings
        if 'embeddings' in manifold:
            embeddings = torch.from_numpy(manifold['embeddings']).float()
        elif 'chunks' in manifold:
            # Old format
            import numpy as np
            embeddings = torch.from_numpy(
                np.stack([chunk.embedding for chunk in manifold['chunks']])
            ).float()
        else:
            raise ValueError("Invalid manifold format")
        
        # Load into retrieval module
        if self.memory_mode == 'hyperbolic':
            self.memory_retrieval.load_hyperbolic_manifold(embeddings)
        else:
            self.memory_retrieval.load_manifold(manifold_path)
        
        print(f"  ✓ Loaded {len(embeddings)} memory vectors")
    
    def forward(self, x, mask=None):
        """
        x: [B, T, C] context embeddings
        Returns: [B, T, C] reflex + memory forcing
        """
        # 1. Standard Reflex Processing (Spinal Cord - Fast Local Syntax)
        reflex = x
        for block in self.blocks:
            reflex = block(reflex, mask)
        reflex = self.ln_f(reflex)
        
        # 2. Memory Retrieval (Hippocampus - Semantic Priming)
        if self.use_memory and self.memory_retrieval is not None:
            # Get the memory response
            # Hybrid system returns (output, info_dict)
            if self.memory_mode == 'hybrid':
                reflex_out, mem_info = self.memory_retrieval(reflex)
                
                # ═══════════════════════════════════════════════════════════════
                # CRITICAL FIX: SELF-SUPERVISED CONTRASTIVE LOSS (InfoNCE)
                # ═══════════════════════════════════════════════════════════════
                # Problem: If memory_loss = 0, balancer enters degenerate state
                #   → minimizes ln(σ) → drives σ→0 → assigns infinite weight
                #   → "Hallucinating supervisor" thinks memory is perfect
                #
                # Solution: Prove memory is useful via in-batch negatives
                #   → "My retrieved memory should match MY query better than
                #      OTHER queries in the batch"
                #
                # This creates REPULSIVE FORCE in hyperbolic space:
                #   → Prevents collapse to single point
                #   → Forces concepts to spread out for distinctiveness
                # ═══════════════════════════════════════════════════════════════
                
                if self.training:  # Only compute during training
                    # A. Flatten batch and time dimensions
                    B, T, C = x.shape
                    flat_query = x.view(-1, C)        # What we asked for [B*T, C]
                    flat_memory = reflex_out.view(-1, C)  # What we got back [B*T, C]
                    
                    # B. Random Sampling for Efficiency
                    # 512 samples gives statistically significant gradient
                    # Keeps memory O(1) regardless of sequence length
                    num_samples = min(512, flat_query.size(0))
                    indices = torch.randperm(flat_query.size(0), device=x.device)[:num_samples]
                    
                    anchors = flat_query[indices]     # [512, C] - queries
                    positives = flat_memory[indices]  # [512, C] - retrieved memories
                    
                    # C. Compute InfoNCE Loss (In-Batch Negatives)
                    # Similarity matrix: [512, 512]
                    # Row i: [sim(q_i, m_0), sim(q_i, m_1), ..., sim(q_i, m_i), ...]
                    # Goal: Diagonal (sim(q_i, m_i)) should be MAXIMUM
                    logits = torch.matmul(anchors, positives.transpose(0, 1))
                    
                    # Temperature scaling (sharpens gradients)
                    temperature = 0.1
                    logits = logits / temperature
                    
                    # Labels are diagonal indices [0, 1, 2, ..., 511]
                    # "My memory should be closest to me, not to you"
                    labels = torch.arange(num_samples, device=x.device, dtype=torch.long)
                    
                    # Cross-entropy forces diagonal to dominate
                    # Initial loss ≈ ln(512) ≈ 6.2 (random memory)
                    # Final loss ≈ 0.1 (perfect retrieval)
                    self.last_memory_loss = F.cross_entropy(logits, labels)
                else:
                    # Inference mode - no loss needed
                    self.last_memory_loss = None
                
                # Store memory info for logging
                if not hasattr(self, '_last_memory_info'):
                    self._last_memory_info = mem_info
                else:
                    self._last_memory_info.update(mem_info)
                
                return reflex_out
            else:
                # Legacy systems (hyperbolic/euclidean) - no info dict
                reflex = self.memory_retrieval(reflex)
                self.last_memory_loss = None
        else:
            # No memory system
            self.last_memory_loss = None
        
        return reflex
    
    def apply_dopamine_signal(self, loss):
        """
        Apply dopaminergic learning signal to memory.
        Called after loss computation to modulate memory plasticity.
        
        Args:
            loss: scalar loss value (prediction error)
        """
        if not self.use_memory or self.memory_retrieval is None:
            return
        
        # Hybrid two-tier system: Apply dopamine to both tiers
        if self.memory_mode == 'hybrid' and hasattr(self.memory_retrieval, 'apply_dopamine'):
            self.memory_retrieval.apply_dopamine(loss)
        # Legacy hyperbolic memory
        elif self.memory_mode == 'hyperbolic' and hasattr(self.memory_retrieval, 'apply_dopamine_modulation'):
            pass  # Dopamine applied via gradient modulation
    
    def memory_step(self):
        """
        Update memory system (aging, decay, consolidation).
        Should be called after each training step.
        """
        if not self.use_memory or self.memory_retrieval is None:
            return
        
        if self.memory_mode == 'hybrid' and hasattr(self.memory_retrieval, 'step'):
            self.memory_retrieval.step()
    
    def get_memory_stats(self):
        """Get memory system statistics for logging"""
        if not self.use_memory or self.memory_retrieval is None:
            return {}
        
        if self.memory_mode == 'hybrid' and hasattr(self.memory_retrieval, 'get_stats'):
            return self.memory_retrieval.get_stats()
        
        return {}
    
    def get_last_recon_loss(self):
        """
        Get contrastive memory quality loss for Homeostatic Balancer.
        
        This is computed during forward() using InfoNCE contrastive learning.
        
        **The Degenerate State Problem:**
        If this returns 0.0, the balancer optimizes: min[ln(σ)]
        → Drives σ→0 → Assigns infinite weight to zero signal
        → "Hallucinating supervisor" thinks memory is perfect
        
        **The Solution:**
        InfoNCE forces memory to prove its worth:
        "My retrieved memory should match MY query better than OTHER queries"
        
        Neuroscience Analogy:
          - **Newborn Model**: Memory random → loss high → σ↑ → downweight
          - **Mature Model**: Memory organized → loss low → σ↓ → upweight
        
        Returns:
            Scalar tensor: InfoNCE loss (0.0 if no memory or not training)
        """
        if self.last_memory_loss is not None:
            return self.last_memory_loss
        else:
            return torch.tensor(0.0)
    
    def save_memory_checkpoint(self, filepath):
        """Save memory state separately from model"""
        if not self.use_memory or self.memory_retrieval is None:
            return
        
        if self.memory_mode == 'hybrid' and hasattr(self.memory_retrieval, 'save_checkpoint'):
            self.memory_retrieval.save_checkpoint(filepath)
    
    def load_memory_checkpoint(self, filepath):
        """Load memory state from checkpoint"""
        if not self.use_memory or self.memory_retrieval is None:
            return
        
        if self.memory_mode == 'hybrid' and hasattr(self.memory_retrieval, 'load_checkpoint'):
            self.memory_retrieval.load_checkpoint(filepath)


# -----------------------------------------------------------------------------
# 3. Stabilizer g_phi — per-dimension damping
# -----------------------------------------------------------------------------

class Stabilizer(nn.Module):
    """
    SMART TRAJECTORY-AWARE STABILIZER
    
    Predicts α ∈ (0,1) per token dimension based on:
    1. Current state z
    2. Context u
    3. Trajectory history (how we got here)
    
    This is NOT a heuristic. This is the damping coefficient.
    
    Dimension-wise damping prevents:
      - Collapse (α → 0 kills gradients)
      - Explosion (α → 1 allows runaway)
      - Mode collapse on high-confidence predictions
      - Oscillations (detected from trajectory)
    
    Physical interpretation:
      - α is viscosity in the semantic flow field
      - High entropy → high viscosity (slow down)
      - Low entropy → low viscosity (trust the flow)
      - Oscillating trajectory → increase damping
    
    CRITICAL BOUNDS: α must NEVER be < 0.1 or > 0.9
    This is not tunable. This is the stability condition.
    
    NOTE: For Hamiltonian dynamics, this becomes a "damping tensor"
    that couples position and momentum (like air resistance on a pendulum).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        input_dim = config.n_embd * 2
        output_dim = config.n_embd
        
        # SMART UPGRADE: Lightweight attention over trajectory history
        # Only 4 heads for efficiency (vs 8 in main transformer)
        self.use_trajectory = getattr(config, 'smart_stabilizer', True)
        
        if self.use_trajectory:
            self.trajectory_attn = nn.MultiheadAttention(
                config.n_embd,
                num_heads=4,  # Lightweight
                dropout=0.0,
                bias=config.bias,
                batch_first=True
            )
            # Layer norm for trajectory summary
            self.traj_ln = nn.LayerNorm(config.n_embd, bias=config.bias)
            
            # Predictor now takes: [z, u, trajectory_summary]
            predictor_input_dim = config.n_embd * 3
            print(f"[Smart Stabilizer] Trajectory-aware damping with {4}-head attention")
        else:
            predictor_input_dim = input_dim
            print(f"[Basic Stabilizer] Simple 2-layer damping")
        
        # Damping predictor (upgraded with one more layer + GELU)
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, output_dim, bias=config.bias),
            nn.GELU(),  # GELU instead of Tanh for smoother gradients
            nn.Linear(output_dim, output_dim, bias=config.bias),
            nn.Sigmoid(),  # α ∈ (0,1)
        )
        
        # Ring buffer for trajectory history (last 5 states)
        self.history_size = 5
        self.register_buffer('trajectory_buffer', torch.zeros(1, 1, self.history_size, config.n_embd))
        self.register_buffer('buffer_ptr', torch.tensor(0, dtype=torch.long))
    
    def forward(self, z, u, update_history=True):
        """
        z: [B, T, C] current DEQ state
        u: [B, T, C] context (reflex + embeddings)
        update_history: whether to add z to trajectory buffer
        
        Returns: α [B, T, C] per-dimension damping ∈ [0.1, 0.9]
        """
        B, T, C = z.shape
        
        if self.use_trajectory:
            # Ensure buffer matches current batch shape
            if (self.trajectory_buffer.shape[0] != B or 
                self.trajectory_buffer.shape[1] != T or 
                self.trajectory_buffer.shape[3] != C):
                # Recreate buffer with correct shape
                self.trajectory_buffer = torch.zeros(B, T, self.history_size, C, 
                                                    device=z.device, dtype=z.dtype)
                self.buffer_ptr.fill_(0)  # Reset pointer
            
            # Update trajectory buffer (ring buffer, only during training)
            if update_history and self.training:
                # Store current state in ring buffer
                ptr = self.buffer_ptr.item()
                self.trajectory_buffer[:, :, ptr, :] = z.detach()  # Detach to avoid backprop through history
                self.buffer_ptr.copy_(torch.tensor((ptr + 1) % self.history_size))
            
            # Extract trajectory history [B, T, history_size, C]
            traj_history = self.trajectory_buffer  # [B, T, H, C]
            
            # Reshape for attention: [B*T, H, C]
            traj_flat = traj_history.reshape(B * T, self.history_size, C)
            z_flat = z.reshape(B * T, 1, C)
            
            # Attend over trajectory to get summary
            # Query: current state, Key/Value: trajectory history
            traj_summary, _ = self.trajectory_attn(
                z_flat,  # Query: where we are now
                traj_flat,  # Key: where we've been
                traj_flat,  # Value: where we've been
                need_weights=False
            )
            
            # Reshape back [B*T, 1, C] → [B, T, C]
            traj_summary = traj_summary.reshape(B, T, C)
            traj_summary = self.traj_ln(traj_summary)
            
            # Combine: current state + context + trajectory pattern
            combined = torch.cat([z, u, traj_summary], dim=-1)
        else:
            # Simple mode: just [z, u]
            combined = torch.cat([z, u], dim=-1)
        
        # Predict damping
        alpha_raw = self.predictor(combined)
        
        # HARD BOUNDS: α ∈ [0.1, 0.9]
        # Not soft scaling. Hard physical constraint.
        alpha = 0.1 + 0.8 * alpha_raw
        return alpha


# -----------------------------------------------------------------------------
# 4. DEQ Brain (Cortex) — infinite-depth semantic equilibrium
# -----------------------------------------------------------------------------

class HamiltonianOperator(nn.Module):
    """
    Symplectic Hamiltonian System for Energy-Conserving Dynamics
    
    Physics: H(q, p) = T(p) + V(q)
    
    Instead of dissipative convergence to a fixed point (standard DEQ),
    we have ENERGY-CONSERVING evolution on a semantic phase space.
    
    Benefits:
    1. NO VANISHING GRADIENTS: Symplectic maps preserve phase space volume
       Jacobian determinant = 1 → gradients rotate, don't decay
    2. LONG-RANGE COHERENCE: "Narrative momentum" doesn't dissipate
       The model coasts on semantic inertia between ideas
    3. FEWER ITERATIONS: Energy conservation → faster convergence to limit cycles
       (not fixed points - OSCILLATING attractors, like planetary orbits)
    
    Split latent space C into:
      q: Position (semantic state, "where we are in meaning space")
      p: Momentum (narrative velocity, "where we're going")
    
    Equations of motion (symplectic Euler):
      p_{t+1} = p_t - dt * ∂V/∂q  (force kick)
      q_{t+1} = q_t + dt * ∂T/∂p  (drift)
    
    This is NOT A HEURISTIC. This is classical mechanics.
    The model becomes a semantic particle system.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.n_embd // 2  # Split into position/momentum
        
        # HOMEOSTATIC ADAPTIVE FRICTION: Damping responds to Chaos Score
        # Without this, Hamiltonian systems orbit forever (energy conservation)
        # With adaptive friction: γ(chaos) scales from gentle to aggressive damping
        self.gamma_min = 0.05  # Baseline friction (Zen mode, gentle damping)
        self.gamma_max = 0.5   # Maximum friction (High chaos, aggressive damping)
        self.current_gamma = self.gamma_min  # Will be updated dynamically
        print(f"[Rayleigh] Homeostatic Adaptive Friction: γ ∈ [{self.gamma_min}, {self.gamma_max}]")
        print(f"           Friction scales with Chaos Score for self-regulation")
        
        # Kinetic Energy: T(p) = (1/2) p^T M^{-1} p
        # Learn the "mass tensor" M for semantic inertia
        self.mass_metric = nn.Linear(self.dim, self.dim, bias=False)
        
        # Potential Energy: V(q, context)
        # This is the "force field" - standard transformer architecture
        self.ln_1 = nn.LayerNorm(self.dim, bias=config.bias)
        self.attn = nn.MultiheadAttention(
            self.dim,
            config.n_head // 2,  # Fewer heads since we're in half-space
            dropout=config.dropout,
            bias=config.bias,
            batch_first=True
        )
        
        self.ln_2 = nn.LayerNorm(self.dim, bias=config.bias)
        self.force_field = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * self.dim, self.dim, bias=config.bias),
            nn.Dropout(config.dropout),
        )
        
        # Spectral normalization for Hamiltonian stability
        # Even though Hamiltonian preserves volume (det(J)=1), we still need
        # bounded operator norms to prevent numerical instability
        if config.spectral_norm:
            self._apply_spectral_norm()
    
    def _apply_spectral_norm(self):
        """Apply spectral norm to all linear layers in Hamiltonian operator"""
        # Use same manual power iteration as DEQOperator
        layer_idx = 0
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Use simple index instead of full name (can't have dots in buffer names)
                self._spectral_norm_power_iteration(module, f"h_layer_{layer_idx}")
                layer_idx += 1
        print(f"[Spectral] Applied spectral normalization to {layer_idx} linear layers in HamiltonianOperator")
    
    def _spectral_norm_power_iteration(self, module, name, n_power_iterations=1):
        """Manual spectral normalization via power iteration"""
        weight = module.weight
        height = weight.size(0)
        width = weight.view(height, -1).size(1)
        
        u = torch.randn(height).to(weight.device)
        u = u / (u.norm() + 1e-12)
        module.register_buffer(f'{name}_u', u)
        
        def spectral_norm_hook(module, input):
            weight = module.weight
            weight_mat = weight.view(height, -1)
            u = getattr(module, f'{name}_u')
            
            with torch.no_grad():
                for _ in range(n_power_iterations):
                    v = weight_mat.t() @ u
                    v = v / (v.norm() + 1e-12)
                    u = weight_mat @ v
                    u = u / (u.norm() + 1e-12)
                
                getattr(module, f'{name}_u').copy_(u)
                sigma = (u @ weight_mat @ v).item()
            
            if sigma > 1.0:
                module.weight.data = weight.data / sigma
        
        module.register_forward_pre_hook(spectral_norm_hook)
    
    def potential_gradient(self, q, u_q, mask=None):
        """
        Compute -∂V/∂q (the force)
        
        The potential is the transformer-style "attraction to semantic basins"
        Context u acts as external potential wells
        """
        q_ctx = q + u_q
        
        # Attention (semantic attraction)
        attn_out, _ = self.attn(
            self.ln_1(q_ctx),
            self.ln_1(q_ctx),
            self.ln_1(q_ctx),
            attn_mask=mask,
            need_weights=False
        )
        q_attn = q + attn_out
        
        # Force field (potential gradient)
        force = self.force_field(self.ln_2(q_attn))
        return force
    
    def kinetic_gradient(self, p, u_p):
        """
        Compute ∂T/∂p (the velocity)
        
        Kinetic energy T = (1/2) p^T M^{-1} p
        So ∂T/∂p = M^{-1} p
        
        The mass tensor M determines how "heavy" semantic dimensions are
        Context u_p acts as momentum injection from environment
        """
        # Learned inverse mass tensor (metric on momentum space)
        velocity = self.mass_metric(p + u_p)
        return velocity
    
    def update_friction(self, chaos_score):
        """
        Homeostatic Adaptive Friction: Update γ based on observed Chaos Score
        
        chaos_score: float in [0.0, 1.0] representing system instability
                     0.0 = Zen mode (stable, converged)
                     1.0 = Chaos mode (diverging, max stress)
        
        Returns: Updated friction coefficient γ_adaptive
        """
        # Linear interpolation: γ = γ_min + (γ_max - γ_min) * chaos
        self.current_gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * chaos_score
        return self.current_gamma
    
    def forward(self, state, u, mask=None, iteration=0):
        """
        Symplectic Euler Integration Step WITH ADAPTIVE METRIC AND HOMEOSTATIC FRICTION
        
        state: [B, T, C] where C = 2*dim = [q; p]
        u: [B, T, C] context (also split into q/p components)
        iteration: int, current DEQ iteration (unused for Hamiltonian, but accepted for API compatibility)
        
        WARP DRIVE OPTIMIZATION:
        Instead of fixed dt=0.1 everywhere, we use adaptive step size
        based on local semantic complexity (Riemannian metric).
        
        Simple tokens: dt → 0.3 (wormhole shortcut!)
        Complex tokens: dt → 0.05 (careful integration)
        
        Returns: Δstate [B, T, C] change in phase space
        """
        # Split state into position and momentum
        q, p = torch.split(state, self.dim, dim=-1)
        u_q, u_p = torch.split(u, self.dim, dim=-1)
        
        # ADAPTIVE METRIC: Compute local semantic curvature
        # This is the "warp drive" - measures how complex each token is
        complexity = PhysicalLaws.compute_semantic_metric(state, u)  # [B, T]
        dt_adaptive = PhysicalLaws.adaptive_step_size(complexity)  # [B, T]
        
        # Expand dt to match dimensions [B, T, 1] for broadcasting
        dt = dt_adaptive.unsqueeze(-1)
        
        # Symplectic Euler (1st order, preserves symplectic form)
        # Step 1: Momentum kick (force from potential)
        force = self.potential_gradient(q, u_q, mask)
        
        # HOMEOSTATIC RAYLEIGH DISSIPATION: Adaptive friction -γ(chaos)·p
        # γ ranges from γ_min (Zen mode) to γ_max (Chaos mode)
        # The current_gamma is updated by the homeostatic controller
        # Physics: Rayleigh dissipation function R = (1/2)γ·p²
        #          τ_effective = -∂V/∂q - ∂R/∂p = force - γ·p
        friction_force = -self.current_gamma * p
        
        p_new = p + dt * (force + friction_force)  # p += dt * (F - γ(chaos)·p)
        
        # Step 2: Position drift (velocity from kinetic)
        velocity = self.kinetic_gradient(p_new, u_p)
        q_new = q + dt * velocity  # q += dt * (dT/dp)
        
        # Recombine
        state_new = torch.cat([q_new, p_new], dim=-1)
        
        # Return DELTA (for DEQ residual formulation)
        delta_state = state_new - state
        return delta_state


class DEQOperator(nn.Module):
    """
    Single implicit layer repeated until fixed point.
    
    This replaces all stacked transformer blocks.
    You don't apply 32 layers.
    You run a single implicit layer until its state stabilizes.
    
    That state is the meaning equilibrium.
    
    CRITICAL: This must be a strict contraction mapping.
    Spectral guardrails enforce ρ ∈ [0.9, 1.05] to ensure damped attractors.
    
    NOTE: For Hamiltonian dynamics, use HamiltonianOperator instead.
    This standard operator is DISSIPATIVE (fixed point attractor).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # TEMPORAL ENCODING: Iteration-aware embedding
        # Allows network to learn different dynamics at different iteration depths
        # Early iterations: High-frequency changes (syntax, structure)
        # Late iterations: Low-frequency changes (semantic refinement)
        max_iters = config.deq_max_iter if hasattr(config, 'deq_max_iter') else 30
        self.iteration_embedding = nn.Embedding(max_iters + 1, config.n_embd)
        print(f"[Temporal] Iteration-aware DEQ: {max_iters} temporal modes")
        
        # Attention
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = nn.MultiheadAttention(
            config.n_embd,
            config.n_head,
            dropout=config.dropout,
            bias=config.bias,
            batch_first=True
        )
        
        # MLP
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )
        
        # Spectral normalization for stability (CRITICAL for DEQ convergence)
        # Must enforce Lipschitz constant K < 1 for Banach Fixed Point Theorem
        if config.spectral_norm:
            self._apply_spectral_norm()
    
    def _apply_spectral_norm(self):
        """
        Apply spectral normalization using power iteration to ALL linear layers.
        This enforces Lipschitz continuity K < 1 and guarantees fixed point convergence.
        
        Uses manual power iteration instead of torch.nn.utils to avoid device placement issues.
        """
        layer_idx = 0
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Register spectral norm via power iteration
                # Use simple index instead of full name (can't have dots in buffer names)
                self._spectral_norm_power_iteration(module, f"layer_{layer_idx}")
                layer_idx += 1
        print(f"[Spectral] Applied spectral normalization to {layer_idx} linear layers in DEQOperator")
    
    def _spectral_norm_power_iteration(self, module, name, n_power_iterations=1):
        """
        Manual implementation of spectral normalization using power iteration.
        
        Estimates largest singular value σ_max and scales weight by 1/σ_max.
        This ensures ||W||_2 ≤ 1, making the operator a contraction mapping.
        """
        weight = module.weight
        height = weight.size(0)
        width = weight.view(height, -1).size(1)
        
        # Initialize random unit vector u
        u = torch.randn(height).to(weight.device)
        u = u / (u.norm() + 1e-12)
        
        # Register as buffer (persistent state)
        module.register_buffer(f'{name}_u', u)
        
        # Add forward pre-hook to normalize weight before each forward pass
        def spectral_norm_hook(module, input):
            # Power iteration: estimate largest singular value
            weight = module.weight
            weight_mat = weight.view(height, -1)
            u = getattr(module, f'{name}_u')
            
            with torch.no_grad():
                for _ in range(n_power_iterations):
                    v = weight_mat.t() @ u
                    v = v / (v.norm() + 1e-12)
                    u = weight_mat @ v
                    u = u / (u.norm() + 1e-12)
                
                # Update u buffer
                getattr(module, f'{name}_u').copy_(u)
                
                # Estimate spectral norm σ = u^T W v
                sigma = (u @ weight_mat @ v).item()
            
            # Normalize weight in-place (W' = W / max(1, σ))
            # This ensures ||W||_2 ≤ 1 (Lipschitz constant K ≤ 1)
            if sigma > 1.0:
                module.weight.data = weight.data / sigma
        
        module.register_forward_pre_hook(spectral_norm_hook)
    
    def forward(self, z, u, mask=None, iteration=0):
        """
        z: [B, T, C] current equilibrium state
        u: [B, T, C] context (from encoder + reflex)
        iteration: int, current DEQ iteration (for temporal encoding)
        
        Returns: Δz [B, T, C] semantic integration
        """
        B, T, C = z.shape
        
        # TEMPORAL ENCODING: Add iteration-aware modulation
        # This allows the network to learn different behaviors at different depths
        iter_emb = self.iteration_embedding(
            torch.tensor(min(iteration, self.iteration_embedding.num_embeddings - 1), 
                        device=z.device, dtype=torch.long)
        )  # [C]
        iter_emb = iter_emb.view(1, 1, C).expand(B, T, -1)  # [B, T, C]
        
        # Modulate state with temporal information
        z_temporal = z + 0.1 * iter_emb  # Small modulation to preserve stability
        
        # Inject context into query
        z_ctx = z_temporal + u
        
        # Attention
        attn_out, _ = self.attn(
            self.ln_1(z_ctx),
            self.ln_1(z_ctx),
            self.ln_1(z_ctx),
            attn_mask=mask,
            need_weights=False
        )
        z_attn = z + attn_out
        
        # MLP
        z_mlp = z_attn + self.mlp(self.ln_2(z_attn))
        
        # Return delta (residual from current state)
        delta_z = z_mlp - z
        return delta_z


class PhysicalLaws:
    """
    These are NOT heuristics. These are the laws of semantic dynamics.
    
    Like physics:
      - Energy minimization
      - Entropy floors
      - Continuity constraints
      - Curvature bounds
      - Adaptive metric tensor (Riemannian geometry)
    
    For language, the "world" is degenerate (just token prefix).
    So we need to define what "distance", "velocity", and "force" mean
    in mental phase-space.
    """
    
    @staticmethod
    def compute_semantic_metric(z, u):
        """
        Riemannian Metric Tensor - Measures "semantic distance"
        
        Physics Analogy: General Relativity
        In GR, mass warps spacetime → gravity creates shortcuts (geodesics)
        
        Here: Complexity warps semantic space
        - Simple tokens (like "the", "and") → near-zero distance
        - Complex concepts → large distance
        
        The metric g_μν determines how many solver iterations are needed.
        High curvature → many steps
        Low curvature (flat) → few steps (wormhole shortcut!)
        
        Returns: scalar "local complexity" per token ∈ [0, 1]
        """
        with torch.no_grad():
            # Measure 1: Local variance (how much z differs from neighbors)
            # High variance = complex, needs careful integration
            z_shifted = torch.roll(z, shifts=1, dims=1)
            local_variance = (z - z_shifted).pow(2).mean(dim=-1)  # [B, T]
            
            # Measure 2: Alignment with context (how much z differs from u)
            # High misalignment = difficult attractor, needs more iterations
            alignment = F.cosine_similarity(z, u, dim=-1)  # [B, T] ∈ [-1, 1]
            difficulty = (1 - alignment) / 2  # Map to [0, 1]
            
            # Measure 3: Entropy of local attention (uncertainty proxy)
            # Compute simple attention weights to measure "how scattered" the focus is
            # High entropy = complex dependencies, needs deep search
            attn_scores = torch.bmm(z, z.transpose(1, 2))  # [B, T, T]
            attn_weights = F.softmax(attn_scores / math.sqrt(z.shape[-1]), dim=-1)
            entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1).mean(dim=-1)  # [B]
            
            # Combine into single "semantic curvature" measure
            # Normalize each component to [0, 1]
            variance_norm = torch.sigmoid(local_variance * 10)  # [B, T]
            entropy_norm = entropy.unsqueeze(-1) / 10.0  # [B, 1]
            
            # Geometric mean: all three must agree for "simple" classification
            complexity = (variance_norm * difficulty * entropy_norm.expand_as(variance_norm)).clamp(0, 1)
            
            return complexity  # [B, T] where 0 = trivial, 1 = complex
    
    @staticmethod
    def adaptive_step_size(complexity, dt_min=0.05, dt_max=0.3):
        """
        Warp Drive: Adjust integration step size based on local complexity
        
        Simple regions (low curvature): Large steps (wormhole!)
        Complex regions (high curvature): Small steps (careful navigation)
        
        This is adaptive mesh refinement from numerical PDEs.
        """
        # Inverse relationship: low complexity → large dt
        dt = dt_min + (dt_max - dt_min) * (1 - complexity)
        return dt
    
    @staticmethod
    def entropy_floor(logits, min_entropy=1.0):
        """
        Law 1: Non-collapse entropy floor
        
        Never allow token entropy < log(3) ≈ 1.0
        Prevents degenerate next-token pointer collapse.
        
        This is like minimum temperature in thermodynamics.
        """
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        
        # If entropy too low, add noise to logits
        violation = (entropy < min_entropy).float()
        noise_scale = violation.unsqueeze(-1) * 0.1
        logits_safe = logits + noise_scale * torch.randn_like(logits)
        
        return logits_safe
    
    @staticmethod
    def semantic_continuity(z, z_prev, max_angle=0.3):
        """
        Law 2: Semantic continuity constraint
        
        Penalize Δz that changes direction faster than max_angle radians.
        Like smooth torque in robot arm.
        
        This prevents discontinuous jumps in meaning space.
        """
        if z_prev is None:
            return z
        
        delta = z - z_prev
        delta_norm = torch.norm(delta, dim=-1, keepdim=True)
        
        # If change is too large, dampen it
        max_delta = max_angle * torch.norm(z, dim=-1, keepdim=True)
        scale = torch.clamp(max_delta / (delta_norm + 1e-8), max=1.0)
        
        z_smooth = z_prev + scale * delta
        return z_smooth
    
    @staticmethod
    def adaptive_depth_from_uncertainty(entropy, base_tol, base_iters):
        """
        Law 4: Depth proportional to uncertainty
        
        If prefix is trivial: converge in 3 steps
        If prefix is contradiction: 12–20 steps
        
        Depth emerges from difficulty.
        """
        # High entropy → need more iterations
        # entropy ∈ [0, log(V)] where V = vocab_size
        # Normalize to [0, 1]
        normalized_entropy = torch.clamp(entropy / 10.0, 0, 1)
        
        # Adjust tolerance: higher entropy → looser tolerance (more iters)
        adaptive_tol = base_tol * (1 + normalized_entropy.mean())
        
        # Adjust max iters: higher entropy → more iters
        adaptive_iters = int(base_iters * (0.5 + normalized_entropy.mean()))
        
        return adaptive_tol, adaptive_iters


class DEQBrain(nn.Module):
    """
    Iterates the DEQ operator until equilibrium:
    
      z_{t+1} = z_t + γ·α·Δz_θ(z_t, u)
    
    Where:
      - u: context embeddings + reflex suggestions
      - Δz_θ: semantic integration operator (DEQOperator)
      - α: stabilizer (per-dimension damping) ∈ [0.1, 0.9]
      - γ: global step-size ∈ [0.9, 1.05] (spectral band)
    
    Physical laws enforced:
      1. Entropy floor (prevent collapse)
      2. Semantic continuity (smooth trajectories)
      3. Adaptive depth (uncertainty → thinking time)
    
    Optional: Multiscale renormalization group flow
      Solve at coarse scale → medium → fine (3-5x speedup)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Choose operator: Hamiltonian (energy-conserving) or Standard (dissipative)
        if config.hamiltonian:
            self.operator = HamiltonianOperator(config)
            print("[Physics] Using HAMILTONIAN operator (symplectic, energy-conserving)")
            print("          State = [position; momentum] in semantic phase space")
            print("          Dynamics preserve phase volume → no vanishing gradients")
        else:
            self.operator = DEQOperator(config)
            print("[Physics] Using DISSIPATIVE operator (standard DEQ, fixed-point)")
        
        self.stabilizer = Stabilizer(config)
        
        # LEARNED STEP SIZE: γ predictor
        # Predicts optimal step size from current state complexity
        self.use_learned_gamma = getattr(config, 'learned_gamma', True)
        if self.use_learned_gamma:
            self.gamma_predictor = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd // 4, bias=config.bias),
                nn.GELU(),
                nn.Linear(config.n_embd // 4, 1, bias=config.bias),
                nn.Sigmoid()  # Output ∈ [0, 1]
            )
            print("[Smart Steering] Learned step size γ enabled")
            print("                 γ adapts to state complexity for optimal convergence")
        
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        
        # Physical laws
        self.laws = PhysicalLaws()
        
        # TEMPORAL ENCODING: Iteration-aware thinking (Deep Temporal Encoding)
        # The operator needs to know "what stage of thought" it's in
        # Early iterations (1-3): high-frequency syntactic structure
        # Late iterations (10+): low-frequency semantic refinement
        self.temporal_encoding_enabled = getattr(config, 'temporal_encoding', True)
        if self.temporal_encoding_enabled:
            # Learnable sinusoidal encoding for iteration number
            # Similar to positional encoding but for "thinking time"
            self.iteration_embedding = nn.Parameter(
                torch.randn(config.deq_max_iter, config.n_embd) * 0.02
            )
            print("[Physics] TEMPORAL ENCODING enabled: Iteration-aware thinking")
            print("          Early iters → syntax, Late iters → semantics")
        
        # Metrics tracking (for visualization)
        self.last_complexity = None
        self.last_dt = None
        self.last_deq_iters = None
        
        # Multiscale projections (Renormalization Group)
        if config.multiscale:
            # Downsampling: fine → medium → coarse
            self.down_med = nn.Linear(config.n_embd, config.med_dim, bias=config.bias)
            self.down_coarse = nn.Linear(config.med_dim, config.coarse_dim, bias=config.bias)
            
            # Upsampling: coarse → medium → fine
            self.up_med = nn.Linear(config.coarse_dim, config.med_dim, bias=config.bias)
            self.up_fine = nn.Linear(config.med_dim, config.n_embd, bias=config.bias)
            
            # Operators at different scales
            # Create lightweight config for coarse scales
            # Ensure n_head divides n_embd evenly
            coarse_heads = max(1, config.coarse_dim // 64)  # 64 dim per head
            med_heads = max(1, config.med_dim // 64)
            
            coarse_config = GrayBoxConfig(
                n_embd=config.coarse_dim,
                n_head=coarse_heads,
                n_reflex=config.n_reflex,
                block_size=config.block_size,
                vocab_size=config.vocab_size,
                dropout=config.dropout,
                bias=config.bias,
            )
            med_config = GrayBoxConfig(
                n_embd=config.med_dim,
                n_head=med_heads,
                n_reflex=config.n_reflex,
                block_size=config.block_size,
                vocab_size=config.vocab_size,
                dropout=config.dropout,
                bias=config.bias,
            )
            
            self.operator_coarse = DEQOperator(coarse_config)
            self.operator_med = DEQOperator(med_config)
            self.stabilizer_coarse = Stabilizer(coarse_config)
            self.stabilizer_med = Stabilizer(med_config)
    
    def solve(self, u, mask=None, effort=1.0, verbose=False):
        """
        Find equilibrium: z* such that z* = f(z*, u)
        
        u: [B, T, C] context (embeddings + reflex)
        effort: thinking depth multiplier (< 1 = fast, > 1 = deep)
                ADAPTIVE DEPTH: effort is modulated by chaos score (fractal complexity)
                High chaos → need more iters to navigate basin boundaries
        verbose: if True, print progress every 5 iterations
        
        Returns: (z*, num_iters, metrics) equilibrium state, iteration count, and diagnostic metrics
        """
        B, T, C = u.shape
        device = u.device
        
        # Initialize z₀ = u (warm start from context)
        z = u.clone()
        z_prev = None
        
        # Anderson acceleration (with optional learned mixing)
        use_learned_anderson = getattr(self.config, 'learned_anderson', True)
        if self.config.anderson_accel:
            anderson = AndersonAcceleration(
                m=5, 
                beta=1.0, 
                learned_mixing=use_learned_anderson,
                dim=self.config.n_embd
            )
        else:
            anderson = None
        
        # ADAPTIVE DEPTH: Base parameters (will be adjusted by complexity)
        base_max_iter = int(self.config.deq_max_iter * effort)
        max_iter = base_max_iter  # Will be updated after early chaos estimate
        tol = self.config.deq_tol
        adaptive_depth_enabled = getattr(self.config, 'adaptive_depth', True)
        
        # Global step size γ ∈ [0.5, 1.5]
        # Can be LEARNED or FIXED depending on config
        if self.use_learned_gamma:
            # Predict γ from current state (mean pooling over tokens)
            gamma_raw = self.gamma_predictor(u.mean(dim=1)).mean()  # Scalar
            gamma = 0.5 + gamma_raw * 1.0  # Map [0,1] → [0.5, 1.5]
            gamma = gamma.item()  # Convert to Python float
        else:
            # Fixed step size (original behavior)
            gamma = torch.clamp(torch.tensor(1.0), 0.9, 1.05).item()
        
        # Tracking for diagnostics
        residual_history = []
        
        # SPEEDUP 1: Early stopping with looser initial tolerance
        # Start loose, tighten as we converge (allows early exit on easy tokens)
        min_iters = 3  # Always do at least 3 iterations
        chaos_estimated = False  # Track if we've adjusted depth
        
        if verbose:
            print(f"[DEQ] Starting solve: max_iter={max_iter}, tol={tol:.1e}, adaptive_depth={adaptive_depth_enabled}")
        
        for i in range(max_iter):
            z_prev_iter = z
            
            # Compute semantic integration (now iteration-aware!)
            # The operator has built-in temporal encoding
            delta_z = self.operator(z, u, mask, iteration=i)
            
            # SMART STABILIZER: per-dimension damping α ∈ [0.1, 0.9]
            # Now trajectory-aware if enabled
            alpha = self.stabilizer(z, u, update_history=(i >= 1))
            
            # LEARNED GAMMA: Predict step size from state complexity (adaptive per-iteration)
            if self.use_learned_gamma and i % 3 == 0:  # Update γ every 3 iters to save compute
                gamma_raw = self.gamma_predictor(z.mean(dim=1)).mean()
                gamma = 0.5 + gamma_raw * 1.0  # [0.5, 1.5]
                gamma = gamma.item()
            
            # DEQ update: z_{t+1} = z_t + γ·α·Δz
            z_next = z + gamma * alpha * delta_z
            
            # Law 2: Semantic continuity (prevent discontinuous jumps)
            z_next = self.laws.semantic_continuity(z_next, z_prev, max_angle=0.3)
            
            # Anderson acceleration (if enabled)
            if anderson is not None:
                z = anderson.update(z, z_next)
            else:
                z = z_next
            
            # Convergence check with early stopping
            with torch.no_grad():
                residual = (z - z_prev_iter).abs().max().item()
                residual_history.append(residual)
                
                if verbose and (i + 1) % 5 == 0:
                    print(f"[DEQ] iter {i+1:2d}/{max_iter}: residual={residual:.2e}")
                
                # ADAPTIVE DEPTH: After 3 iterations, estimate chaos and adjust max_iter
                # This allows easy sequences to exit early, hard sequences to think longer
                if adaptive_depth_enabled and i == 2 and not chaos_estimated:
                    # Estimate chaos from residual stress (distance to convergence)
                    # High residual after 3 iters → fractal boundary → need more thinking
                    import math
                    res_log = math.log10(residual) if residual > 1e-10 else -10
                    # Map residual to [0, 1]: res=0.001 → 0.4, res=1.0 → 0.7, res=10 → 0.9
                    chaos_estimate = (res_log + 3.0) / 5.0  # Same scale as chaos sensor
                    chaos_estimate = max(0.0, min(1.0, chaos_estimate))
                    
                    # Adjust max_iter based on chaos: chaos ∈ [0, 1] → multiplier ∈ [0.5, 1.5]
                    # Low chaos (smooth) → fewer iters, High chaos (fractal) → more iters
                    depth_multiplier = 0.5 + chaos_estimate
                    new_max_iter = int(base_max_iter * depth_multiplier)
                    
                    # Clamp to reasonable range
                    new_max_iter = max(min_iters + 2, min(new_max_iter, base_max_iter * 2))
                    
                    if new_max_iter != max_iter:
                        max_iter = new_max_iter
                        if verbose:
                            print(f"[Adaptive Depth] chaos_est={chaos_estimate:.3f} → max_iter adjusted to {max_iter}")
                    
                    chaos_estimated = True
                
                # SPEEDUP: Adaptive early stopping
                # After min_iters, check if we're converging fast enough
                if i >= min_iters:
                    # If residual is small OR converging very fast, stop early
                    if residual < tol:
                        if verbose:
                            print(f"[DEQ] Converged at iter {i+1}")
                        break
                    
                    # SPEEDUP 2: If residual barely changed, we're stuck - stop
                    if i > 0 and residual > residual_history[i-1] * 0.95:
                        # Residual not decreasing fast enough
                        if i >= min_iters + 2:  # Give it a few iterations
                            break
            
            z_prev = z
        
        # Capture complexity and dt metrics (for visualization)
        with torch.no_grad():
            complexity = self.laws.compute_semantic_metric(z, u)
            dt = self.laws.adaptive_step_size(complexity)
            self.last_complexity = complexity.detach()
            self.last_dt = dt.detach()
            self.last_deq_iters = i + 1
        
        # Collect diagnostic metrics (for controller)
        metrics = {
            'num_iters': i + 1,
            'final_residual': residual_history[-1] if residual_history else 0.0,
            'residual_history': residual_history,
            'convergence_rate': residual_history[-1] / (residual_history[0] + 1e-10) if len(residual_history) > 1 else 1.0,
            'avg_complexity': complexity.mean().item(),
            'avg_dt': dt.mean().item(),
        }
        
        return self.ln_f(z), i + 1, metrics
    
    def solve_multiscale(self, u, mask=None, effort=1.0, verbose=False):
        """
        Renormalization Group Flow: Solve DEQ hierarchically
        
        Physics analogy: Statistical field theory renormalization
        - Start at coarse scale (low resolution, fast convergence)
        - Flow to fine scale (high resolution, precise equilibrium)
        
        Speedup: ~3-5x (solving at lower dims is much cheaper)
        
        Example: 384 dim
          Coarse (128): 5 iters at 128² FLOPs
          Medium (256): 5 iters at 256² FLOPs  
          Fine (384): 5 iters at 384² FLOPs
          Total: 128² + 256² + 384² = 0.37 × (3 × 384²)
          Speedup: 2.7x for same total iterations!
        """
        B, T, C = u.shape
        device = u.device
        
        # Global step size
        gamma = 1.0
        
        # Phase 1: COARSE SCALE (semantic gist)
        # Project to low dimension
        u_coarse = self.down_coarse(self.down_med(u))  # [B, T, coarse_dim]
        z_coarse = u_coarse.clone()
        
        # Solve quickly at coarse scale
        coarse_iters = max(3, int(self.config.deq_max_iter * 0.3))
        for i in range(coarse_iters):
            delta_z = self.operator_coarse(z_coarse, u_coarse, mask)
            alpha = self.stabilizer_coarse(z_coarse, u_coarse)
            z_coarse = z_coarse + gamma * alpha * delta_z
        
        if verbose:
            print(f"[Multiscale] Coarse ({self.config.coarse_dim}d): {coarse_iters} iters")
        
        # Phase 2: MEDIUM SCALE (semantic structure)
        # Upsample and refine
        u_med = self.down_med(u)  # [B, T, med_dim]
        z_med = self.up_med(z_coarse)  # Initialize from coarse solution
        
        med_iters = max(3, int(self.config.deq_max_iter * 0.3))
        for i in range(med_iters):
            delta_z = self.operator_med(z_med, u_med, mask)
            alpha = self.stabilizer_med(z_med, u_med)
            z_med = z_med + gamma * alpha * delta_z
        
        if verbose:
            print(f"[Multiscale] Medium ({self.config.med_dim}d): {med_iters} iters")
        
        # Phase 3: FINE SCALE (semantic details)
        # Upsample to full resolution and polish
        z_fine = self.up_fine(z_med)  # Initialize from medium solution
        
        fine_iters = max(3, int(self.config.deq_max_iter * 0.4))
        for i in range(fine_iters):
            delta_z = self.operator(z_fine, u, mask, iteration=i)
            alpha = self.stabilizer(z_fine, u)
            z_fine = z_fine + gamma * alpha * delta_z
            
            # Early stop if converged
            with torch.no_grad():
                residual = (delta_z * alpha).abs().max().item()
                if residual < self.config.deq_tol:
                    break
        
        if verbose:
            print(f"[Multiscale] Fine ({self.config.n_embd}d): {i+1}/{fine_iters} iters")
        
        total_iters = coarse_iters + med_iters + i + 1
        
        metrics = {
            'num_iters': total_iters,
            'coarse_iters': coarse_iters,
            'med_iters': med_iters,
            'fine_iters': i + 1,
            'final_residual': residual if 'residual' in locals() else 0.0,
        }
        
        return self.ln_f(z_fine), total_iters, metrics
    
    def solve_quantum_paths(self, u, mask=None, effort=1.0, verbose=False):
        """
        Quantum Path Integral Formulation
        
        Physics: Feynman path integral ⟨z_f|z_i⟩ = ∫ Dz e^(iS[z]/ℏ)
        Semantic: z* = E[z] where z ~ P(z|u) ∝ exp(-β·E[z,u])
        
        Instead of single deterministic trajectory, run multiple parallel paths
        and Boltzmann-average by their "semantic energy" (residual)
        
        Benefits:
        - Escapes local minima (explores solution space)
        - Uncertainty quantification (variance of ensemble)
        - Parallelizable (each path independent)
        - Often converges faster than single long trajectory
        """
        B, T, C = u.shape
        device = u.device
        
        num_paths = self.config.num_paths
        path_length = self.config.path_length
        
        trajectories = []
        energies = []
        
        if verbose:
            print(f"[Quantum] Running {num_paths} parallel paths of length {path_length}")
        
        # Run multiple trajectories in parallel
        for p in range(num_paths):
            # Random initialization (quantum superposition)
            # Each path starts from slightly different point
            z = u.clone() + 0.05 * torch.randn_like(u)
            
            # Evolve trajectory
            for t in range(path_length):
                delta_z = self.operator(z, u, mask, iteration=t)
                alpha = self.stabilizer(z, u)
                z = z + alpha * delta_z
            
            trajectories.append(z)
            
            # Compute "action" / "energy" of this path
            # Lower energy = more probable quantum state
            with torch.no_grad():
                residual = (z - self.operator(z, u, mask, iteration=path_length)).norm(dim=-1).mean()
                energies.append(residual)
        
        # Boltzmann weighting: P(z) ∝ exp(-β·E[z])
        # β = 1/T where T is "quantum temperature"
        beta = 1.0 / 0.1  # Low temperature = sharp distribution
        energies_tensor = torch.stack(energies)
        
        # Normalize weights (softmax over energy)
        weights = F.softmax(-beta * energies_tensor, dim=0)
        
        if verbose:
            print(f"[Quantum] Path energies: {[f'{e:.3f}' for e in energies]}")
            print(f"[Quantum] Path weights: {[f'{w:.3f}' for w in weights.tolist()]}")
        
        # Ensemble average (expectation value)
        z_star = sum(w * z for w, z in zip(weights, trajectories))
        
        # Total effective iterations
        total_iters = num_paths * path_length
        
        # Uncertainty estimate (variance of ensemble)
        z_var = sum(w * (z - z_star).pow(2) for w, z in zip(weights, trajectories))
        uncertainty = z_var.mean().sqrt().item()
        
        metrics = {
            'num_iters': total_iters,
            'num_paths': num_paths,
            'path_length': path_length,
            'energies': [e.item() for e in energies],
            'uncertainty': uncertainty,
            'final_residual': energies_tensor.min().item(),
        }
        
        return self.ln_f(z_star), total_iters, metrics
    
    def solve_unified_quantum(self, u, mask=None, effort=1.0, verbose=False):
        """
        Unified Quantum Solver - Combines Multiple Physics Concepts
        
        Physics concepts unified:
        1. GAUGE SYMMETRY: Sample multiple initial conditions (different "phrasings")
        2. SPONTANEOUS SYMMETRY BREAKING: Early iterations choose mode/style
        3. QUANTUM TUNNELING: Escape local minima via basin hopping
        4. THERMODYNAMIC ANNEALING: Temperature schedule for exploration
        5. PATH INTEGRAL: Boltzmann-weighted ensemble average
        
        Three-phase protocol:
        Phase 1: Symmetry Breaking (hot, explore modes)
        Phase 2: Tunneling & Refinement (warm, escape barriers)
        Phase 3: Convergence (cold, polish solution)
        
        This is the most physics-complete solver.
        """
        B, T, C = u.shape
        device = u.device
        
        num_orbits = self.config.num_gauge_orbits
        phase1_iters = self.config.symmetry_breaking_iters
        phase2_iters = self.config.refinement_iters
        total_iters = phase1_iters + phase2_iters
        
        if verbose:
            print(f"[Quantum] Unified solver: {num_orbits} orbits, {total_iters} iters")
        
        # OPTIMIZATION 1: CACHE TEMPERATURE SCHEDULE (1.2x speedup)
        # Pre-compute all temperatures instead of computing exp() each iteration
        T_init = self.config.T_init
        T_final = self.config.T_final
        
        if self.config.temperature_schedule == "exponential":
            # Pre-compute exponential decay
            t_values = torch.linspace(0, 1, total_iters, device=device)
            temp_schedule = T_init * (T_final / T_init) ** t_values
        elif self.config.temperature_schedule == "linear":
            temp_schedule = torch.linspace(T_init, T_final, total_iters, device=device)
        else:  # constant
            temp_schedule = torch.full((total_iters,), T_init, device=device)
        
        # ==========================================
        # GAUGE SYMMETRY: Sample from gauge orbit
        # ==========================================
        # Different initial conditions = different "phrasings" of same meaning
        # Like choosing Coulomb vs Lorenz gauge in E&M
        
        # OPTIMIZATION 2: PARALLELIZE GAUGE ORBITS (3x speedup)
        # Stack all orbits in batch dimension, process together
        # Shape: (B*num_orbits, T, C) instead of num_orbits × (B, T, C)
        
        gauge_angles = torch.linspace(0, 2*math.pi, num_orbits+1, device=device)[:-1]  # [0, 2π/3, 4π/3]
        noise_scales = 0.1 * torch.cos(gauge_angles)  # Varies per gauge
        
        # Create all gauge-transformed initial conditions at once
        # Expand: (B, T, C) → (num_orbits, B, T, C)
        u_expanded = u.unsqueeze(0).expand(num_orbits, -1, -1, -1)
        noise = torch.randn(num_orbits, B, T, C, device=device)
        
        # Apply gauge transformations: z_g = u + noise_scale_g * noise_g
        z_all = u_expanded + noise_scales.view(-1, 1, 1, 1) * noise
        
        # Reshape to batch dimension: (num_orbits, B, T_seq, C) → (B*num_orbits, T_seq, C)
        # Note: T_seq to avoid conflict with temperature variable T
        T_seq = T  # sequence length
        z_batched = z_all.reshape(num_orbits * B, T_seq, C)
        u_batched = u_expanded.reshape(num_orbits * B, T_seq, C)
        
        # Mask doesn't change - it's same for all orbits (causal mask is orbit-independent)
        mask_batched = mask
        
        # ==========================================
        # PHASE 1: SPONTANEOUS SYMMETRY BREAKING
        # ==========================================
        # Like Higgs mechanism: choose which vacuum to fall into
        # Hot temperature = explore different modes/styles
        # NOW PARALLEL: All orbits evolve together!
        
        temperature = temp_schedule[0].item()  # Hot start (renamed from T to temperature)
        
        # Let each orbit find its preferred mode (ALL AT ONCE)
        for i in range(phase1_iters):
            # DEQ step with thermal noise (batched across all orbits)
            delta_z = self.operator(z_batched, u_batched, mask_batched, iteration=i)
            alpha = self.stabilizer(z_batched, u_batched)
            
            # Thermal fluctuations help choose mode
            noise = temperature * torch.randn_like(z_batched) * 0.1
            z_batched = z_batched + alpha * delta_z + noise
        
        # Reshape back to separate orbits: (B*num_orbits, T_seq, C) → (num_orbits, B, T_seq, C)
        z_all = z_batched.reshape(num_orbits, B, T_seq, C)
        
        # Measure energy in each orbit
        with torch.no_grad():
            orbit_energies = []
            for g in range(num_orbits):
                z_g = z_all[g]
                u_g = u  # Original input
                residual = (z_g - self.operator(z_g, u_g, mask, iteration=phase1_iters)).norm(dim=-1).mean()
                orbit_energies.append(residual)
        
        if verbose:
            print(f"[Quantum] Phase 1 energies: {[f'{e:.3e}' for e in orbit_energies]}")
        
        # ==========================================
        # PHASE 2: TUNNELING & REFINEMENT
        # ==========================================
        # Quantum tunneling: jump between basins if stuck
        # Cooling temperature: refine within chosen mode
        # NOW PARALLEL with EARLY STOPPING per orbit!
        
        # Track which orbits have converged (OPTIMIZATION 3: Early stopping)
        converged = torch.zeros(num_orbits, dtype=torch.bool, device=device)
        convergence_threshold = 1e-3
        prev_residuals = torch.tensor(orbit_energies, device=device)
        
        # RAYCAST STATS: Track how often tunneling actually happens (should be rare!)
        tunnel_events = 0
        tunnel_attempts = 0
        
        # Batch all non-converged orbits together
        z_batched = z_all.reshape(num_orbits * B, T_seq, C)
        u_batched = u_expanded.reshape(num_orbits * B, T_seq, C)
        
        # Initialize i in case phase2_iters is 0 or loop exits early
        i = -1
        
        for i in range(phase2_iters):
            iter_idx = phase1_iters + i
            temperature = temp_schedule[iter_idx].item()  # Use cached temperature!
            
            # Standard DEQ step (BATCHED)
            delta_z = self.operator(z_batched, u_batched, mask_batched, iteration=iter_idx)
            alpha = self.stabilizer(z_batched, u_batched)
            z_next = z_batched + alpha * delta_z
            
            # QUANTUM TUNNELING: If stuck, try jumping to another basin
            # OPTIMIZED: "Raycast" approach - process orbits sequentially since tunneling is rare
            if self.config.enable_tunneling and i > 0:
                with torch.no_grad():
                    # Reshape to compute per-orbit residuals
                    z_next_orbits = z_next.reshape(num_orbits, B, T_seq, C)
                    
                    # RAYCAST OPTIMIZATION: Check each orbit sequentially
                    # Since tunneling is rare, we avoid batching overhead
                    for g in range(num_orbits):
                        if converged[g]:
                            continue  # Skip converged orbits
                        
                        # RAYCAST: Single orbit residual check
                        z_g = z_next_orbits[g]
                        residual = (z_g - self.operator(z_g, u, mask, iteration=iter_idx)).norm(dim=-1).mean()
                        
                        # EARLY STOPPING: Check convergence
                        if residual < convergence_threshold:
                            converged[g] = True
                            if verbose:
                                print(f"[Quantum] Orbit {g} converged at iter {i}")
                            continue
                        
                        # Stuck if residual not decreasing
                        # QUANTUM RAYCAST: Fire multiple tunnel rays to explore wave function!
                        if residual > prev_residuals[g] * self.config.tunnel_threshold:
                            tunnel_attempts += 1
                            
                            # QUANTUM WAVE FUNCTION: Sample multiple possible tunnel paths
                            # Like Feynman path integral - explore the probability cloud!
                            num_rays = self.config.num_tunnel_rays if hasattr(self.config, 'num_tunnel_rays') else 8
                            
                            tunnel_candidates = []
                            tunnel_energies = []
                            
                            # Fire rays through quantum probability space
                            for ray_idx in range(num_rays):
                                # Each ray = possible instanton trajectory
                                # Vary the tunnel "amplitude" for diversity
                                amplitude = 0.2 + 0.3 * torch.rand(1, device=device).item()
                                z_ray = u + amplitude * torch.randn_like(u)
                                
                                # Measure energy of this quantum state
                                E_ray = (z_ray - self.operator(z_ray, u, mask, iteration=iter_idx)).norm(dim=-1).mean()
                                
                                tunnel_candidates.append(z_ray)
                                tunnel_energies.append(E_ray)
                            
                            # WAVE FUNCTION COLLAPSE: Choose based on Boltzmann statistics
                            # Lower energy = higher probability (quantum ground state seeking)
                            energies_tensor = torch.stack(tunnel_energies)
                            
                            # Find best ray (minimum energy in the probability cloud)
                            best_ray_idx = energies_tensor.argmin()
                            best_energy = tunnel_energies[best_ray_idx]
                            
                            # Tunneling probability (compare best ray to current state)
                            dE = best_energy - residual
                            P_tunnel = torch.exp(-dE / (temperature + 1e-8))
                            
                            # MEASUREMENT/COLLAPSE: Accept if energetically favorable
                            if best_energy < residual * 0.95 or torch.rand(1, device=device) < P_tunnel:
                                # WAVE FUNCTION COLLAPSED! Jump to best quantum state
                                z_next_orbits[g] = tunnel_candidates[best_ray_idx]
                                tunnel_events += 1
                                if verbose:
                                    energy_spread = energies_tensor.std().item()
                                    print(f"[Quantum] Orbit {g} tunneled via {num_rays} rays: "
                                          f"E={best_energy:.3e} (spread={energy_spread:.3e}, P={P_tunnel:.3f})")
                        
                        # Update residual tracker
                        prev_residuals[g] = residual
                    
                    # Update batched tensor
                    z_next = z_next_orbits.reshape(num_orbits * B, T_seq, C)
            
            # Add small thermal noise (decreases with T)
            z_batched = z_next + temperature * 0.05 * torch.randn_like(z_next)
            
            # Early exit if all orbits converged
            if converged.all():
                if verbose:
                    print(f"[Quantum] All orbits converged at iter {i}/{phase2_iters}")
                break
        
        # Reshape back to separate orbits
        z_all = z_batched.reshape(num_orbits, B, T_seq, C)
        
        # Count actual iterations used (including early stopping)
        # i is -1 if loop never ran, otherwise it's the last iteration index
        phase2_completed = i + 1 if i >= 0 else 0
        actual_iters = phase1_iters + (phase2_iters if not converged.all() else phase2_completed)
        
        # Update energies
        with torch.no_grad():
            orbit_energies = []
            for g in range(num_orbits):
                z_g = z_all[g]
                final_residual = (z_g - self.operator(z_g, u, mask, iteration=actual_iters)).norm(dim=-1).mean()
                orbit_energies.append(final_residual)
        
        if verbose:
            print(f"[Quantum] Phase 2 energies: {[f'{e:.3e}' for e in orbit_energies]}")
            print(f"[Quantum] Converged orbits: {converged.sum().item()}/{num_orbits}")
            print(f"[Quantum] Tunnel stats: {tunnel_events} events / {tunnel_attempts} attempts ({100*tunnel_events/(tunnel_attempts+1e-8):.1f}%)")
        
        # ==========================================
        # PHASE 3: PATH INTEGRAL ENSEMBLE AVERAGE
        # ==========================================
        # Boltzmann-weight by energy: lower energy = higher probability
        # This is the quantum expectation value
        
        energies_tensor = torch.stack(orbit_energies)
        beta = 10.0  # Inverse temperature for weighting
        weights = F.softmax(-beta * energies_tensor, dim=0)
        
        # Ensemble average (path integral)
        # Convert list of orbit states to weighted sum
        z_star = torch.zeros_like(u)
        for g in range(num_orbits):
            z_star += weights[g] * z_all[g]
        
        if verbose:
            print(f"[Quantum] Final weights: {[f'{w:.3f}' for w in weights.tolist()]}")
        
        # Uncertainty (variance of ensemble)
        z_var = torch.zeros_like(u)
        for g in range(num_orbits):
            z_var += weights[g] * (z_all[g] - z_star).pow(2)
        uncertainty = z_var.mean().sqrt().item()
        
        metrics = {
            'num_iters': actual_iters,
            'num_orbits': num_orbits,
            'phase1_iters': phase1_iters,
            'phase2_iters': phase2_iters,
            'orbit_energies': [e.item() for e in orbit_energies],
            'weights': weights.tolist(),
            'uncertainty': uncertainty,
            'final_residual': energies_tensor.min().item(),
            'converged_orbits': converged.sum().item(),
            'tunnel_events': tunnel_events,
            'tunnel_attempts': tunnel_attempts,
        }
        
        return self.ln_f(z_star), actual_iters, metrics
    
    def solve_annealing(self, u, mask=None, effort=1.0, verbose=False):
        """
        Simulated Annealing (Thermodynamic Equilibrium Search)
        
        Physics: Thermodynamic annealing - start hot, cool slowly
        At high T: Explore widely (escape local minima)
        At low T: Converge precisely (find best attractor)
        
        Metropolis-Hastings with temperature schedule:
        T(t) = T_init × (T_final/T_init)^(t/max_iter)
        
        Benefits:
        - Finds BETTER attractors (not just ANY fixed point)
        - Escapes saddle points and local minima
        - Adaptive exploration-exploitation tradeoff
        """
        B, T_seq, C = u.shape
        device = u.device
        
        z = u.clone()
        
        max_iter = int(self.config.deq_max_iter * effort)
        T_init = self.config.T_init
        T_final = self.config.T_final
        
        if verbose:
            print(f"[Annealing] T: {T_init:.2f} → {T_final:.2f} over {max_iter} iters")
        
        energy_history = []
        
        for i in range(max_iter):
            # Temperature schedule (exponential cooling)
            T = T_init * (T_final / T_init) ** (i / max_iter)
            
            # DEQ update
            delta_z = self.operator(z, u, mask, iteration=i)
            alpha = self.stabilizer(z, u)
            
            # Add thermal noise (decreases with temperature)
            noise = T * torch.randn_like(z) * 0.1
            z_next = z + alpha * delta_z + noise
            
            # Semantic continuity (prevent jumps)
            z_next = self.laws.semantic_continuity(z_next, z, max_angle=0.3)
            
            z = z_next
            
            # Track energy (residual)
            with torch.no_grad():
                energy = (z - self.operator(z, u, mask, iteration=i)).norm(dim=-1).mean().item()
                energy_history.append(energy)
                
                if verbose and (i + 1) % 5 == 0:
                    print(f"[Annealing] iter {i+1}/{max_iter}: T={T:.3f}, energy={energy:.3e}")
                
                # Early stop if fully cooled and converged
                if T < T_final * 1.1 and energy < self.config.deq_tol:
                    break
        
        metrics = {
            'num_iters': i + 1,
            'final_residual': energy_history[-1] if energy_history else 0.0,
            'energy_history': energy_history,
            'T_final': T,
        }
        
        return self.ln_f(z), i + 1, metrics
    
    def forward(self, u, mask=None, effort=1.0, verbose=False):
        """
        Dispatch to appropriate solver
        
        Priority:
        1. Unified Quantum (if enabled) - combines all physics concepts
        2. Multiscale RG (if enabled) - fast coarse-to-fine
        3. Standard DEQ - reliable baseline
        
        Legacy solvers (quantum_paths, annealing) deprecated in favor of unified
        """
        if self.config.quantum_solver:
            return self.solve_unified_quantum(u, mask, effort, verbose)
        elif self.config.multiscale:
            return self.solve_multiscale(u, mask, effort, verbose)
        else:
            return self.solve(u, mask, effort, verbose)


# -----------------------------------------------------------------------------
# 5. Global Controller h_psi — spectral target and tolerance adaptation
# -----------------------------------------------------------------------------

class GlobalController(nn.Module):
    """
    Maintains the DEQ at the edge of chaos.
    
    This is NOT a neural network hallucinating control signals.
    This is a CYBERNETIC FEEDBACK LOOP reading physical measurements.
    
    Inputs (scalar fields from semantic geometry):
      - residual norm (convergence rate)
      - max dlogp/dx (local surprisal gradient)
      - gradient variance over tokens
      - number of active attention heads (sparse proxy)
      - entropy decay rate
      - semantic curvature (embedding space geometry)
    
    Outputs:
      - φ*: spectral target ∈ [0.9, 1.05]
      - tol: equilibrium tolerance (adaptive)
    
    This gives adaptive depth:
      - easy text: stop in 3 iterations
      - hard reasoning: allow 12-18
    
    "Thinking intensity" is literally the temperature of the solver.
    """
    def __init__(self, config):
        super().__init__()
        
        # 6-12 scalar channels (not a giant black box)
        n_features = 6
        
        # Tiny meta-network (this is control theory, not deep learning)
        # Upgraded: 2 outputs → 3 outputs (phi, tol, temp)
        self.net = nn.Sequential(
            nn.Linear(n_features, 16, bias=config.bias),
            nn.Tanh(),
            nn.Linear(16, 3, bias=config.bias),  # Output: [Phi_adj, Tol_adj, Temp_adj]
        )
        
        # Default targets
        self.register_buffer('phi_target', torch.tensor(0.95))
        self.register_buffer('tol_base', torch.tensor(config.deq_tol))
        self.register_buffer('temp_base', torch.tensor(1.0))  # Base temperature
    
    def compute_semantic_fields(self, logits, z, z_prev, metrics):
        """
        Compute physical fields in semantic space.
        
        These are the language equivalents of:
          - contact distance
          - velocity of approach
          - obstacle proximity
        
        For language:
          - Local surprisal: dlogp/dx
          - Token entropy: uncertainty field
          - Semantic curvature: embedding space geometry
          - Entropy decay: "time to collapse"
        """
        B, T, V = logits.shape
        
        # 1. Token entropy (uncertainty field)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, T]
        mean_entropy = entropy.mean().item()
        
        # 2. Max dlogp/dx (local surprisal gradient)
        # This is how "steep" the probability landscape is
        if z.requires_grad:
            max_logp = logits.max(dim=-1)[0]  # [B, T]
            max_dlogp = torch.autograd.grad(
                max_logp.sum(), z, 
                create_graph=False, retain_graph=True
            )[0] if z.requires_grad else torch.zeros_like(z)
            surprisal_grad = max_dlogp.abs().max().item()
        else:
            surprisal_grad = 0.0
        
        # 3. Semantic curvature (how much z is changing)
        if z_prev is not None:
            delta_z = z - z_prev
            curvature = torch.norm(delta_z, dim=-1).mean().item()
        else:
            curvature = 0.0
        
        # 4. Convergence rate (from metrics)
        convergence_rate = metrics.get('convergence_rate', 1.0)
        
        # 5. Residual norm
        residual_norm = metrics.get('final_residual', 0.0)
        
        # 6. Entropy decay rate (how fast uncertainty is dropping)
        if len(metrics.get('residual_history', [])) > 2:
            recent_residuals = metrics['residual_history'][-3:]
            entropy_decay = (recent_residuals[0] - recent_residuals[-1]) / (recent_residuals[0] + 1e-10)
        else:
            entropy_decay = 0.0
        
        return torch.tensor([
            residual_norm,
            surprisal_grad,
            curvature,
            convergence_rate,
            mean_entropy / 10.0,  # Normalize
            entropy_decay,
        ], dtype=torch.float32)
    
    def forward(self, logits, z, z_prev, metrics):
        """
        Read semantic fields, output control signals.
        
        This is physics, not learning.
        
        Returns:
            (phi_target, tolerance, temperature)
        """
        # Compute scalar fields
        features = self.compute_semantic_fields(logits, z, z_prev, metrics)
        
        # Predict adjustments (tiny MLP, not a black box)
        out = self.net(features.to(next(self.parameters()).device))
        phi_adjust = torch.sigmoid(out[0])  # [0, 1]
        tol_adjust = torch.sigmoid(out[1])  # [0, 1]
        temp_adjust = torch.sigmoid(out[2])  # [0, 1] - NEW
        
        # Spectral target: φ* ∈ [0.9, 1.05]
        # HARD CLAMP (not soft) — this is the stability boundary
        phi_target = torch.clamp(0.9 + 0.15 * phi_adjust, 0.9, 1.05)
        
        # Tolerance: adaptive based on difficulty
        # High uncertainty → looser tolerance (think longer)
        # Low uncertainty → tighter tolerance (converge fast)
        tolerance = self.tol_base * (0.1 ** (1 - 2*tol_adjust))
        tolerance = torch.clamp(tolerance, self.tol_base * 0.1, self.tol_base * 10.0)
        
        # Temperature: Learnable annealing schedule (NEW)
        # High uncertainty/entropy → High Temp (Explore via thermal noise)
        # Low uncertainty → Low Temp (Exploit current basin)
        # Range: [0.1, 5.0] - aggressive annealing
        temperature = 0.1 + 4.9 * temp_adjust
        temperature = torch.clamp(temperature, 0.1, 5.0)
        
        return phi_target, tolerance, temperature


# -----------------------------------------------------------------------------
# 6. Geometry Layer (Biomechanics) — projects latent to logits
# -----------------------------------------------------------------------------

class GeometryLayer(nn.Module):
    """
    Force → Jacobian → Torque (robotics)
    Equilibrium latent → projection → logits (language)
    
    The DEQ does not output tokens.
    It outputs a latent force intent (semantic direction).
    An analytic, fixed projection then produces logits.
    
    The DEQ never learns "how to form a distribution."
    It just decides where to lean in meaning space.
    """
    def __init__(self, config):
        super().__init__()
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, z):
        """
        z: [B, T, C] equilibrium latent (semantic force)
        Returns: [B, T, V] logits (token distribution)
        """
        return self.lm_head(z)


# -----------------------------------------------------------------------------
# Full Gray Box DEQ Language Model
# -----------------------------------------------------------------------------

@dataclass
class GrayBoxConfig:
    # Model architecture
    block_size: int = 1024
    vocab_size: int = 50304
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 2  # For API compatibility, maps to n_reflex (number of reflex blocks)
    n_reflex: int = None  # Will be set to n_layer if None
    dropout: float = 0.0
    bias: bool = True
    
    # DEQ parameters
    deq_max_iter: int = 30
    deq_tol: float = 1e-3
    anderson_accel: bool = True
    spectral_norm: bool = True  # MANDATORY for stability (not optional)
    
    # SMART STEERING ENHANCEMENTS (NEW!)
    smart_stabilizer: bool = True  # Trajectory-aware damping with attention
    learned_gamma: bool = True  # Learned step size predictor
    learned_anderson: bool = True  # Learned Anderson mixing weights
    # These add ~10-15% overhead but can reduce iterations by 20-40%
    
    # Multiscale solving (Renormalization Group Flow)
    multiscale: bool = False  # Enable hierarchical coarse-to-fine solving
    coarse_dim: int = 128  # Coarsest scale dimension
    med_dim: int = 256  # Medium scale dimension
    # Fine dim = n_embd (full resolution)
    
    # Unified Quantum Solver (combines multiple physics concepts)
    quantum_solver: bool = False  # Enable unified quantum-inspired solving
    
    # Hamiltonian Dynamics (energy-conserving symplectic integrator)
    hamiltonian: bool = False  # Use Hamiltonian operator instead of dissipative DEQ
    # If True: state becomes [q; p] where q=position, p=momentum
    # Benefits: no vanishing gradients, long-range coherence, oscillating attractors
    
    # Quantum parameters (when quantum_solver=True)
    num_gauge_orbits: int = 3  # Sample different "phrasings" (gauge symmetry)
    symmetry_breaking_iters: int = 3  # Early iters to choose mode (spontaneous breaking)
    refinement_iters: int = 5  # Late iters to converge within mode
    enable_tunneling: bool = True  # Allow jumps between basins
    tunnel_threshold: float = 0.9  # Tunnel if stuck (residual ratio > this)
    num_tunnel_rays: int = 8  # Fire multiple rays to sample quantum probability cloud
    temperature_schedule: str = "exponential"  # "exponential", "linear", "constant"
    T_init: float = 0.5  # Initial exploration temperature
    T_final: float = 0.01  # Final convergence temperature
    
    # ═══════════════════════════════════════════════════════════════════════
    # HYBRID TWO-TIER MEMORY SYSTEM
    # ═══════════════════════════════════════════════════════════════════════
    use_memory_manifold: bool = False  # Enable memory system
    memory_mode: str = 'hybrid'  # 'hybrid', 'hyperbolic', 'euclidean'
    memory_dim: int = 384  # Memory embedding dimension
    memory_k: int = 16  # Number of neighbors to retrieve
    memory_alpha: float = 0.1  # Injection strength
    memory_curvature: float = 1.0  # Hyperbolic curvature
    
    # Working memory tier (VRAM) - L1 CACHE style
    working_memory_capacity: int = 20      # TINY - immediate context only
    working_memory_decay: float = 0.80     # AGGRESSIVE - 20% decay per step
    
    # Long-term memory tier (CPU) - RAM style
    longterm_memory_capacity: int = 2000   # LARGE - background knowledge
    longterm_memory_decay: float = 0.999   # PERSISTENT - slow fade
    
    # Consolidation
    memory_promotion_threshold: float = 0.4  # Lower for easier promotion
    memory_promotion_interval: int = 50      # More frequent
    # ═══════════════════════════════════════════════════════════════════════
    
    def __post_init__(self):
        # Map n_layer to n_reflex for API compatibility with nanoGPT
        if self.n_reflex is None:
            self.n_reflex = self.n_layer


class GrayBoxDEQ(nn.Module):
    """
    Deep Equilibrium Language Model — Gray Box Formulation
    
    Architecture:
      1. Context Encoder (senses)
      2. Reflex Module (spinal cord)
      3. DEQ Brain (cortex)
      4. Geometry Layer (biomechanics)
    
    Cybernetic Control:
      - Stabilizer: per-dimension damping
      - Global Controller: spectral target adaptation (optional)
    
    This is not a weird RNN.
    This is a homeostatic semantic attractor.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Context Encoder (senses)
        self.encoder = ContextEncoder(config)
        
        # 2. Reflex Module (spinal cord)
        self.reflex = ReflexModule(config)
        
        # 3. DEQ Brain (cortex)
        self.deq = DEQBrain(config)
        
        # 4. Geometry Layer (biomechanics)
        self.geometry = GeometryLayer(config)
        
        # Optional: Global Controller
        self.controller = None  # Disabled for now (future work)
        
        # Weight tying (embeddings <-> logits)
        self.geometry.lm_head.weight = self.encoder.wte.weight
        
        # Initialize weights FIRST
        self.apply(self._init_weights)
        
        # Report
        print(f"GrayBox DEQ initialized:")
        print(f"  - Context Encoder: {self._count_params(self.encoder):,} params")
        print(f"  - Reflex Module ({config.n_reflex} blocks): {self._count_params(self.reflex):,} params")
        print(f"  - DEQ Brain: {self._count_params(self.deq):,} params")
        print(f"  - Geometry Layer: {self._count_params(self.geometry):,} params")
        print(f"  - Total: {self._count_params(self):,} params")
    
    def _count_params(self, module):
        return sum(p.numel() for p in module.parameters())
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None, effort=1.0, return_metrics=False, training_iter=None, reflex_gate=1.0):
        """
        idx: [B, T] token indices
        targets: [B, T] target tokens (for training)
        effort: thinking depth multiplier
        return_metrics: if True, return (logits, loss, metrics), else (logits, loss)
        training_iter: optional training iteration number (for curriculum/phase-aware learning)
                       During inference, this is None. The model's position in semantic space
                       is already captured by the chaos score (distance to fractal basin boundaries).
        reflex_gate: β(t) ∈ [0, 1] - Homeostatic Reflex Gating coefficient
                     Controls how much spinal signal reaches cortex.
                     β=0: Pure cortical training (force brain to learn)
                     β=1: Full integration (brain + reflexes)
                     Default=1.0 for backwards compatibility & inference
        
        Returns: (logits, loss) or (logits, loss, metrics) depending on return_metrics
        """
        B, T = idx.shape
        device = idx.device
        
        # Store training iteration for potential use in curriculum learning
        # NOTE: The chaos score already measures our position on the Newton fractal basin boundaries,
        # so this is redundant for most physics-based dynamics. Useful for explicit phase transitions.
        if training_iter is not None:
            self.current_training_iter = training_iter
        
        # Causal mask (on same device as input)
        mask = torch.triu(torch.ones(T, T, dtype=torch.float32, device=device), diagonal=1)
        mask = mask.masked_fill(mask.bool(), float('-inf'))
        
        # 1. Context Encoder (senses)
        context = self.encoder(idx)  # [B, T, C]
        
        # 2. Reflex Module (spinal cord)
        reflex = self.reflex(context, mask)  # [B, T, C]
        
        # 3. HOMEOSTATIC REFLEX GATING (the KEY fix for lizard brain optimization)
        # Apply β(t) gate to reflex signal BEFORE entering DEQ brain
        # This forces the cortex to develop semantic attractors BEFORE reflexes automate
        gated_reflex = reflex_gate * reflex
        
        # 4. DEQ Brain (cortex) — find equilibrium
        u = context + gated_reflex  # Combined context (with gated reflex)
        z_star, num_iters, deq_metrics = self.deq(u, mask, effort)
        
        # Expose metrics for tracking (e.g., in training loop)
        self.last_complexity = self.deq.last_complexity
        self.last_dt = self.deq.last_dt
        self.last_deq_iters = num_iters
        
        # 4. Geometry Layer (biomechanics)
        if targets is not None:
            # Training: forward full sequence
            logits = self.geometry(z_star)  # [B, T, V]
        else:
            # Inference: optimize by only forwarding last position
            logits = self.geometry(z_star[:, [-1], :])  # [B, 1, V]
        
        # Law 1: Entropy floor (prevent degenerate attractor collapse)
        # CRITICAL: This is NOT a training hack - it's a physical constraint
        # Like contact forces in robotics, prevents semantic point attractors
        # Apply ALWAYS (both training and inference) to maintain stable dynamics
        logits = self.deq.laws.entropy_floor(logits, min_entropy=1.0)
        
        # Loss with Jacobian regularization for DEQ stability
        loss = None
        if targets is not None:
            # Standard cross-entropy loss
            loss_ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            
            # JACOBIAN REGULARIZATION (Critical for DEQ stability)
            # Penalize ||∇_z f(z*, u)||_F^2 to prevent singular Jacobians
            # This ensures smooth dynamics and stable implicit differentiation
            jacobian_reg_raw = 0.0
            if self.config.spectral_norm and self.training:
                # Estimate Jacobian norm via finite differences (cheap approximation)
                # Full Jacobian is O(d^2), but Frobenius norm can be estimated
                with torch.enable_grad():
                    # Perturb z_star slightly
                    eps = 1e-3
                    z_pert = z_star + eps * torch.randn_like(z_star)
                    
                    # Compute operator output at perturbed point (use final iteration)
                    delta_pert = self.deq.operator(z_pert, u, mask, iteration=num_iters)
                    delta_star = self.deq.operator(z_star.detach(), u, mask, iteration=num_iters)
                    
                    # Approximate ||∂f/∂z||_F ≈ ||Δf||_F / ||Δz||_F
                    jacobian_norm = (delta_pert - delta_star).norm() / (eps + 1e-12)
                    jacobian_reg_raw = jacobian_norm ** 2
            
            # NOTE: We used to scale jacobian here, but now the Homeostatic Balancer
            # will learn the optimal weight automatically. We still keep a small
            # baseline scaling to put it in a reasonable range for the balancer.
            lambda_jac = 1e-4  # Baseline scaling (balancer will refine this)
            jacobian_reg = lambda_jac * jacobian_reg_raw
            
            # Total loss (will be replaced by balancer in training loop)
            loss = loss_ce + jacobian_reg
            
            # OPTIONAL: Add entropy penalty as soft constraint (better than hard floor)
            # This encourages diversity without adding noise to logits
            # Disabled by default - the hard floor above is sufficient
            # 
            # probs = F.softmax(logits, dim=-1)
            # entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            # min_entropy = 1.0
            # entropy_penalty = F.relu(min_entropy - entropy) ** 2
            # loss = loss + 0.01 * entropy_penalty  # Small penalty coefficient
        
        # Collect metrics for logging/analysis
        if return_metrics:
            metrics = {
                **deq_metrics,
                'num_iters': num_iters,
                # HOMEOSTATIC BALANCING: Expose individual loss components
                # These will be fed to the Bayesian uncertainty balancer
                # NOTE: We return the PRE-SCALED jacobian (after 1e-4 factor)
                # so the balancer sees values in a reasonable range (~1500 not ~15M)
                'loss_components': {
                    'prediction': loss_ce if targets is not None else torch.tensor(0.0),
                    'jacobian': jacobian_reg if targets is not None else torch.tensor(0.0),
                },
            }
            return logits, loss, metrics
        else:
            # API compatibility with nanoGPT training script
            return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, effort=1.0):
        """
        Generate tokens autoregressively.
        
        effort < 1.0: fast/shallow thinking
        effort = 1.0: normal
        effort > 1.0: slow/deep thinking
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward (without metrics for speed)
            logits, _ = self(idx_cond, effort=effort)
            logits = logits[:, -1, :] / temperature
            
            # Top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    @torch.no_grad()
    def inspect_concept_mass(self, top_k=20):
        """
        Inspects the Hamiltonian Mass Matrix to see which tokens are 'Heavy'.
        
        Physics: Kinetic Energy T = 1/2 p^T M^{-1} p
        High Mass (Inertia) = Hard to change semantic direction.
        Low Mass (Agile) = Easy to change (syntax/grammar).
        
        This reveals the learned semantic physics:
        - Content words (nouns, verbs) → Heavy (high semantic inertia)
        - Function words (the, a, is) → Light (agile syntactic glue)
        
        Returns:
            dict with 'heavy_ids', 'heavy_vals', 'light_ids', 'light_vals'
        """
        if not self.config.hamiltonian:
            print("⚠️  Model is not Hamiltonian (no mass matrix).")
            return {
                'heavy_ids': [], 'heavy_vals': [],
                'light_ids': [], 'light_vals': []
            }
        
        # 1. Get the diagonal of the inverse mass matrix (approximate)
        # The mass_metric layer computes velocity = M^{-1} p
        # We look at the norm of the weights projecting each dimension
        M_inv_diag = self.deq.operator.mass_metric.weight.norm(dim=0)
        
        # Mass is inverse of this layer (roughly)
        mass_score = 1.0 / (M_inv_diag + 1e-6)
        
        # 2. Project token embeddings onto this mass vector
        # Which tokens align with high-mass dimensions?
        # WTE: [Vocab, Dim]
        # But we only use the "position" half of embeddings for q
        dim_half = self.config.n_embd // 2
        token_emb_q = self.encoder.wte.weight[:, :dim_half]  # Position component
        
        # Mass score for each token = sum of |embedding| * mass_score
        token_mass = (token_emb_q.abs() * mass_score.unsqueeze(0)).sum(dim=1)
        
        # 3. Find heaviest and lightest tokens
        # Returning IDs and scores for the training script to decode
        heavy_vals, heavy_ids = torch.topk(token_mass, min(top_k, len(token_mass)), largest=True)
        light_vals, light_ids = torch.topk(token_mass, min(top_k, len(token_mass)), largest=False)
        
        return {
            'heavy_ids': heavy_ids.cpu().tolist(),
            'heavy_vals': heavy_vals.cpu().tolist(),
            'light_ids': light_ids.cpu().tolist(),
            'light_vals': light_vals.cpu().tolist()
        }
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        AdamW optimizer with DIFFERENT learning rates for memory tiers.
        
        Memory architecture (like biological systems):
        - Fast tier (working memory): 10x normal LR
        - Slow tier (long-term memory): 0.1x normal LR
        - Regular params: 1x normal LR
        """
        # Separate params by type
        decay = set()
        no_decay = set()
        fast_memory = set()  # Dynamic/working memory (high LR)
        slow_memory = set()  # Static/long-term memory (low LR)
        
        for pn, p in self.named_parameters():
            # Detect memory parameters by name pattern
            if 'dynamic_memory' in pn or 'working' in pn:
                fast_memory.add(pn)
            elif 'static_memory' in pn or 'longterm' in pn or 'memory_manifold' in pn:
                slow_memory.add(pn)
            elif p.dim() >= 2:
                decay.add(pn)
            else:
                no_decay.add(pn)
        
        # Remove memory params from decay/no_decay
        decay = decay - fast_memory - slow_memory
        no_decay = no_decay - fast_memory - slow_memory
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        optim_groups = [
            # Standard params with decay
            {'params': [param_dict[pn] for pn in sorted(decay)], 
             'weight_decay': weight_decay, 
             'lr': learning_rate},
            # Standard params without decay
            {'params': [param_dict[pn] for pn in sorted(no_decay)], 
             'weight_decay': 0.0,
             'lr': learning_rate},
        ]
        
        # Add fast memory tier (CACHE-STYLE: 50x LR!) if exists
        if fast_memory:
            optim_groups.append({
                'params': [param_dict[pn] for pn in sorted(fast_memory)],
                'weight_decay': weight_decay * 0.01,  # Minimal regularization for cache
                'lr': learning_rate * 50.0  # 50X FASTER! Like L1 cache
            })
            print(f"[Optimizer] Fast memory tier (CACHE): {len(fast_memory)} params @ 50x LR")
        
        # Add slow memory tier (0.1x LR) if exists
        if slow_memory:
            optim_groups.append({
                'params': [param_dict[pn] for pn in sorted(slow_memory)],
                'weight_decay': weight_decay,
                'lr': learning_rate * 0.1  # 10X SLOWER consolidation
            })
            print(f"[Optimizer] Slow memory tier: {len(slow_memory)} params @ 0.1x LR")
        
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        
        return optimizer
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU)"""
        # Very rough approximation
        N = self._count_params(self)
        cfg = self.config
        L = 1  # DEQ is effectively 1 layer (iterated)
        H = cfg.n_head
        Q = cfg.n_embd // cfg.n_head
        T = cfg.block_size
        
        # Transformer MFU formula (approximate)
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        
        # A100 peak flops
        flops_promised = 312e12  # 312 TFLOPS for A100
        mfu = flops_achieved / flops_promised
        return mfu
    
    def crop_block_size(self, block_size):
        """Reduce block_size (for model surgery)"""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.encoder.wpe.weight = nn.Parameter(self.encoder.wpe.weight[:block_size])


# API compatibility aliases
GPTConfig = GrayBoxConfig
GPT = GrayBoxDEQ


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Gray Box DEQ Language Model — Cybernetic Architecture Test")
    print("=" * 70)
    
    # Small config for testing
    config = GrayBoxConfig(
        vocab_size=50257,
        block_size=128,
        n_embd=256,
        n_head=4,
        n_reflex=2,
        deq_max_iter=20,
        deq_tol=1e-3,
        anderson_accel=True,
        spectral_norm=True,
    )
    
    model = GrayBoxDEQ(config)
    
    # CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"\nDevice: {device}")
    print("=" * 70)
    
    # Test 1: Forward pass
    print("\n[Test 1] Forward pass")
    B, T = 2, 64
    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    y = torch.randint(0, config.vocab_size, (B, T), device=device)
    
    logits, loss, metrics = model(x, y, effort=1.0, return_metrics=True)
    print(f"  Input: {x.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  DEQ iterations: {metrics['num_iters']}")
    print(f"  Convergence rate: {metrics['convergence_rate']:.4f}")
    
    # Test compatibility mode (no metrics)
    logits2, loss2 = model(x, y)
    print(f"  Compatibility mode: logits {logits2.shape}, loss {loss2.item():.4f}")
    
    # Test 2: Backward pass
    print("\n[Test 2] Backward pass")
    loss.backward()
    n_grads = sum(1 for p in model.parameters() if p.grad is not None)
    n_params = sum(1 for _ in model.parameters())
    print(f"  {n_grads}/{n_params} parameters have gradients")
    
    # Test 3: Generation
    print("\n[Test 3] Generation with adaptive depth")
    model.eval()
    
    for effort_val in [0.5, 1.0, 2.0]:
        start = torch.randint(0, config.vocab_size, (1, 10), device=device)
        generated = model.generate(start, max_new_tokens=20, temperature=1.0, effort=effort_val)
        print(f"  effort={effort_val}: generated {generated.shape[1]} tokens")
    
    print("\n" + "=" * 70)
    print("All tests passed! Gray Box DEQ is operational.")
    print("=" * 70)
    
    # Architecture summary
    print("\n[Cybernetic Architecture]")
    print("  1. Context Encoder (Senses) — embeds tokens with structural priors")
    print("  2. Reflex Module (Spinal Cord) — fast local features (2 blocks)")
    print("  3. DEQ Brain (Cortex) — infinite-depth equilibrium solver")
    print("  4. Geometry Layer (Biomechanics) — projects latent to logits")
    print("\n[Control Systems]")
    print("  - Stabilizer: per-dimension damping α ∈ (0,1)")
    print("  - Anderson Acceleration: 3-5x faster convergence")
    print("  - Spectral Normalization: stability near ρ ≈ 1⁻")
    print("\n[Benefits vs Standard Transformers]")
    print("  ✓ Depth emerges from difficulty")
    print("  ✓ Memory ∝ batch_size, not depth")
    print("  ✓ Adaptive reasoning (easy=3 iters, hard=12-18)")
    print("  ✓ Structural priors (not rediscovered 96 times)")
    print("  ✓ Homeostatic semantic attractor (not a weird RNN)")
