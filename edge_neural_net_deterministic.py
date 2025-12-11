import torch
import torch.nn as nn
from typing import Optional

class EdgeNeuralNet(nn.Module):
    """
    The 'Convergent' Trajectory Shooter with FOURIER RESONANCE + GREYBOX CYBERNETICS.
    
    TRUE DEQ BEHAVIOR:
    - Iterates until the physics reaches Equilibrium (or max_steps).
    - The network learns to 'settle' into the answer.
    - Implements the "Auto PDE" concept: The 3-Net drives the system to a fixed point.
    
    FOURIER CARRIER PHYSICS:
    - Carrier operates in frequency domain (oscillator modes)
    - Repeated patterns cause resonance (amplitude buildup)
    - Natural momentum through phase evolution
    - Network learns which frequencies encode context vs. prediction
    
    GREYBOX CYBERNETICS (NEW!):
    - Protected memory registers (Freq 0-19) that never decay
    - Symbolic arithmetic injected via residual force blending
    - Learnable gate decides neural vs symbolic trust
    - Neuro-Symbolic AI for sample-efficient arithmetic learning
    """
    
    def __init__(self, hidden_dim: int, manifold=None, num_heads: int = 4, 
                 max_steps: int = 50, tolerance: float = 1e-3, 
                 use_greybox: bool = False, vocab_size: int = 16, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.manifold = manifold
        self.max_steps = max_steps      # Safety limit (not a fixed target)
        self.tolerance = tolerance      # When to stop (The Equilibrium condition)
        self.use_greybox = use_greybox  # Enable greybox cybernetics
        
        spatial_dim = hidden_dim - 1 
        # Frequency domain: rfft gives (spatial_dim//2 + 1) complex components
        freq_dim = spatial_dim // 2 + 1  # 128 → 65 complex numbers (130 real values)
        self.freq_dim = freq_dim
        
        # Input: [carrier_freq, trajectory_freq, source_pos, target_pos]
        # carrier_freq: freq_dim*2 (real+imag), trajectory_freq: freq_dim*2, positions: spatial_dim*2
        input_dim = freq_dim * 2 + freq_dim * 2 + spatial_dim * 2 
        
        # === 1. FREQUENCY-SPACE OPERATORS (The Resonant Force) ===
        # Operate on Fourier coefficients (complex numbers represented as 2*freq_dim reals)
        # Network learns which frequency modes to amplify/dampen for each edge
        self.carrier_freq_net = nn.Sequential(
            nn.Linear(input_dim, freq_dim * 4),
            nn.LayerNorm(freq_dim * 4),
            nn.GELU(),
            nn.Linear(freq_dim * 4, freq_dim * 2),  # Output: real and imaginary parts
            nn.Tanh()
        )
        
        self.trajectory_freq_net = nn.Sequential(
            nn.Linear(input_dim, freq_dim * 4),
            nn.LayerNorm(freq_dim * 4),
            nn.GELU(),
            nn.Linear(freq_dim * 4, freq_dim * 2),
            nn.Tanh()
        )
        
        # === 2. FREQUENCY-DEPENDENT DAMPING ===
        # Learn which frequencies decay (high freq = fast decay, low freq = memory)
        self.freq_damping = nn.Parameter(torch.linspace(0.95, 0.5, freq_dim))  # Low→high freq decay
        
        # === 3. STABILIZER & CONTROLLER ===
        # Still needed for convergence control
        self.damping_net = nn.Sequential(
            nn.Linear(freq_dim * 2, freq_dim),
            nn.GELU(),
            nn.Linear(freq_dim, 1)
        )
        
        self.steering_net = nn.Sequential(
            nn.Linear(freq_dim * 2, freq_dim // 2),
            nn.GELU(),
            nn.Linear(freq_dim // 2, 1)
        )
        
        # === 4. INITIALIZATION ===
        # Start with chaos (strong weights) so it has to learn to stabilize
        for net in [self.carrier_freq_net, self.trajectory_freq_net]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.5) 
                    if m.bias is not None: nn.init.zeros_(m.bias)

        # Force gates open initially
        nn.init.constant_(self.damping_net[-1].bias, 1.0)
        nn.init.constant_(self.steering_net[-1].bias, 1.0)
        
        self.carrier_scale = nn.Parameter(torch.tensor(0.1))
        self.trajectory_scale = nn.Parameter(torch.tensor(0.1))
        
        # === 5. GREYBOX CYBERNETICS (REMOVED) ===
        # Old built-in greybox completely removed - it was computing answers (CHEATING!)
        # Use arithmetic_greybox.py wrapper instead, which provides TOOLS only (no computation)
    
    def set_vocab_mapping(self, stoi: dict):
        """Removed - old greybox deleted. Use arithmetic_greybox.py wrapper instead."""
        pass
    
    def set_tokens(self, src_token: int, tgt_token: int):
        """Removed - old greybox deleted. Use arithmetic_greybox.py wrapper instead."""
        pass

    def forward(self, carrier_in: torch.Tensor, trajectory_in: torch.Tensor,
                source_pos: torch.Tensor, 
                target_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with FOURIER RESONANCE and positional information.
        
        Args:
            carrier_in: Current carrier state [batch, seq, spatial_dim]
            trajectory_in: Current trajectory state [batch, seq, spatial_dim]
            source_pos: Source token position in hyperbolic space
            target_pos: Target token position in hyperbolic space
        
        Returns:
            (new_carrier, new_trajectory)
        """
        
        # Normalize dimensions
        if carrier_in.dim() == 1: carrier_in = carrier_in.view(1, 1, -1)
        if trajectory_in.dim() == 1: trajectory_in = trajectory_in.view(1, 1, -1)
        
        curr_c = carrier_in
        curr_t = trajectory_in
        
        # Extract spatial part from hyperbolic positions
        spatial_dim = self.hidden_dim - 1
        
        if source_pos.dim() == 1:
            source_pos = source_pos.unsqueeze(0)
        src_spatial = source_pos[..., :spatial_dim]
        
        if target_pos.dim() == 1:
            target_pos = target_pos.unsqueeze(0)
        tgt_spatial = target_pos[..., :spatial_dim]
        
        # Expand positions to match batch dimensions if needed
        if src_spatial.dim() == 2 and curr_c.dim() == 3:
            src_spatial = src_spatial.unsqueeze(1).expand(-1, curr_c.shape[1], -1)
            tgt_spatial = tgt_spatial.unsqueeze(1).expand(-1, curr_c.shape[1], -1)
        
        # === TRANSFORM TO FREQUENCY DOMAIN ===
        # rfft: real FFT, output shape [..., freq_dim] complex
        # We'll work with real representation: [..., freq_dim*2] for [real, imag] interleaved
        curr_c_freq = torch.fft.rfft(curr_c, dim=-1)  # Complex
        curr_t_freq = torch.fft.rfft(curr_t, dim=-1)  # Complex
        
        # Convert complex to real representation for network processing
        curr_c_freq_real = torch.view_as_real(curr_c_freq)  # [..., freq_dim, 2]
        curr_t_freq_real = torch.view_as_real(curr_t_freq)  # [..., freq_dim, 2]
        
        # Flatten real/imag: [..., freq_dim*2]
        curr_c_freq_flat = curr_c_freq_real.flatten(-2, -1)
        curr_t_freq_flat = curr_t_freq_real.flatten(-2, -1)
        
        # === THE CONVERGENCE LOOP (in frequency space!) ===
        for _ in range(self.max_steps):
            prev_c_freq_flat = curr_c_freq_flat
            prev_t_freq_flat = curr_t_freq_flat
            
            # A. Fuse: [carrier_freq, trajectory_freq, source_pos, target_pos]
            # Positions stay in spatial domain, frequencies in freq domain
            combined = torch.cat([curr_c_freq_flat, curr_t_freq_flat, src_spatial, tgt_spatial], dim=-1)
            
            # B. Forces in frequency space
            # Network learns which frequency modes to excite for this geometric transition
            neural_force_c_freq = self.carrier_freq_net(combined)
            neural_force_t_freq = self.trajectory_freq_net(combined)
            
            # === GREYBOX BLENDING ===
            # DISABLED: Old greybox was computing answers (CHEATING!)
            # Use arithmetic_greybox.py wrapper instead (wraps entire EdgeNeuralNet)
            # That wrapper provides TOOLS (digit one-hots, operation flags) NOT ANSWERS
            force_c_freq = neural_force_c_freq
            force_t_freq = neural_force_t_freq
            
            # C. Natural damping (frequency-dependent)
            # Low frequencies = memory (slow decay), high frequencies = details (fast decay)
            freq_dim = curr_c_freq_flat.shape[-1] // 2
            damping_c = self.freq_damping.unsqueeze(0).unsqueeze(0).repeat(1, 1, 2).flatten(-2, -1)  # [..., freq_dim*2]
            damping_c = damping_c.expand_as(curr_c_freq_flat)
            
            # D. Adaptive control
            # Sanitize frequency inputs before passing to damping_net to avoid NaNs/Infs
            safe_c = torch.nan_to_num(curr_c_freq_flat, nan=0.0, posinf=1e6, neginf=-1e6)
            safe_t = torch.nan_to_num(curr_t_freq_flat, nan=0.0, posinf=1e6, neginf=-1e6)

            alpha_c = torch.sigmoid(self.damping_net(safe_c))
            alpha_t = torch.sigmoid(self.damping_net(safe_t))
            gamma = torch.sigmoid(self.steering_net(safe_c)) + 0.5

            # Defensive: if damping_net somehow produces NaNs (shouldn't), replace them and log rarely
            if torch.isnan(alpha_c).any():
                alpha_c = torch.where(torch.isnan(alpha_c), torch.zeros_like(alpha_c), alpha_c)
                self._nan_alpha_counter = getattr(self, '_nan_alpha_counter', 0) + 1
                if self._nan_alpha_counter % 1000 == 1:
                    try:
                        print(f"[DEQ WARN] damping_net produced NaNs for alpha_c; replaced with zeros (occurrence {self._nan_alpha_counter})")
                    except Exception:
                        pass
            if torch.isnan(alpha_t).any():
                alpha_t = torch.where(torch.isnan(alpha_t), torch.zeros_like(alpha_t), alpha_t)
                self._nan_alpha_counter = getattr(self, '_nan_alpha_counter', 0) + 1
                if self._nan_alpha_counter % 1000 == 1:
                    try:
                        print(f"[DEQ WARN] damping_net produced NaNs for alpha_t; replaced with zeros (occurrence {self._nan_alpha_counter})")
                    except Exception:
                        pass
            
            # E. Update in frequency space with natural damping
            update_c_freq = gamma * alpha_c * self.carrier_scale * force_c_freq
            update_t_freq = gamma * alpha_t * self.trajectory_scale * force_t_freq
            
            # Apply frequency-dependent damping to carrier (momentum decay)
            curr_c_freq_flat = damping_c * curr_c_freq_flat + update_c_freq
            curr_t_freq_flat = damping_c * curr_t_freq_flat + update_t_freq
            
            # F. CONVERGENCE CHECK
            diff_c_freq = torch.norm(curr_c_freq_flat - prev_c_freq_flat, dim=-1).mean()
            diff_t_freq = torch.norm(curr_t_freq_flat - prev_t_freq_flat, dim=-1).mean()
            
            if diff_c_freq < self.tolerance and diff_t_freq < self.tolerance:
                break
            
            # Sanity check in frequency space
            if torch.norm(curr_c_freq_flat) > 10.0:
                curr_c_freq_flat = curr_c_freq_flat / torch.norm(curr_c_freq_flat) * 10.0
            if torch.norm(curr_t_freq_flat) > 10.0:
                curr_t_freq_flat = curr_t_freq_flat / torch.norm(curr_t_freq_flat) * 10.0
        
        # === TRANSFORM BACK TO SPATIAL DOMAIN ===
        # Reshape freq_flat back to complex
        freq_dim = curr_c_freq_flat.shape[-1] // 2
        curr_c_freq_real = curr_c_freq_flat.view(*curr_c_freq_flat.shape[:-1], freq_dim, 2)
        curr_t_freq_real = curr_t_freq_flat.view(*curr_t_freq_flat.shape[:-1], freq_dim, 2)
        
        curr_c_freq = torch.view_as_complex(curr_c_freq_real)
        curr_t_freq = torch.view_as_complex(curr_t_freq_real)
        
        # Inverse FFT back to spatial domain
        curr_c = torch.fft.irfft(curr_c_freq, n=spatial_dim, dim=-1)
        curr_t = torch.fft.irfft(curr_t_freq, n=spatial_dim, dim=-1)
        
        return curr_c, curr_t
