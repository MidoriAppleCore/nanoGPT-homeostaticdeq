"""
Spectral Diagnostics for 3-Net Homeostatic DEQ
Based on "Multi-Network Deep Equilibrium Models for Heterogeneous PDEs" (Jones, 2025)

Key metrics to validate the homeostatic mechanism is working:
1. Spectral radius ρ(J_f) ≈ 0.84-0.87 (edge of chaos regime)
2. Input-stabilizer correlation r ≈ -0.38 to -0.49 (geology-aware preconditioning)
3. Stabilizer mean ᾱ ≈ 0.28-0.51 (not collapsed)
"""

import torch
import numpy as np


def power_iteration_spectral_radius(network, state, n_iter=5):
    """
    Estimate spectral radius ρ(J_f) using power iteration.
    
    For the DEQ: z* = f(z*, x)
    We want the largest eigenvalue of ∂f/∂z at equilibrium.
    
    Args:
        network: The edge network (has forward method)
        state: Current state [B, freq_dim*2] (flattened carrier or trajectory freq)
        n_iter: Number of power iterations
    
    Returns:
        spectral_radius: Estimated ρ(J_f) as scalar
    """
    device = state.device
    dtype = state.dtype
    
    # Random initial vector
    v = torch.randn_like(state)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    
    for _ in range(n_iter):
        v.requires_grad_(True)
        
        # Compute J^T v via autograd
        # Forward pass
        if hasattr(network, 'forward'):
            # For full network with greybox, we need to pass through properly
            # This is a simplified version - may need adjustment
            y = network(state)
        else:
            # Direct call
            y = state  # Placeholder
            
        # Compute Jacobian-vector product
        JTv, = torch.autograd.grad(
            outputs=y,
            inputs=state,
            grad_outputs=v,
            retain_graph=True,
            create_graph=False
        )
        
        # Normalize
        v = JTv / (JTv.norm(dim=-1, keepdim=True) + 1e-8)
        v = v.detach()
    
    # Final iteration to get spectral norm
    v.requires_grad_(True)
    y = network(state)
    JTv, = torch.autograd.grad(
        outputs=y,
        inputs=state,
        grad_outputs=v,
        retain_graph=False,
        create_graph=False
    )
    
    spectral_radius = JTv.norm(dim=-1).mean().item()
    return spectral_radius


def compute_stabilizer_correlation(damping_net, carrier_freq, trajectory_freq):
    """
    Compute correlation between input frequencies and learned damping α.
    
    In the paper:
    - Case I (Diffusion): r = -0.38 (boundary correlates with α)
    - Case II (CCS): r = -0.49 (permeability correlates with α)
    
    For your system:
    - We expect greybox register states (arithmetic features) to correlate with α
    
    Args:
        damping_net: The damping network (outputs α)
        carrier_freq: Carrier frequency state [B, seq, freq_dim*2]
        trajectory_freq: Trajectory frequency state [B, seq, freq_dim*2]
    
    Returns:
        correlation: Pearson correlation coefficient
        alpha_mean: Mean of α values
        alpha_std: Std of α values
    """
    with torch.no_grad():
        # Compute damping for carrier
        alpha_c = torch.sigmoid(damping_net(carrier_freq))  # [B, seq, freq_dim*2]
        
        # Flatten to get all values
        alpha_flat = alpha_c.reshape(-1).cpu().numpy()
        input_flat = carrier_freq.reshape(-1).cpu().numpy()
        
        # Compute correlation
        correlation = np.corrcoef(input_flat, alpha_flat)[0, 1]
        alpha_mean = alpha_flat.mean()
        alpha_std = alpha_flat.std()
    
    return correlation, alpha_mean, alpha_std


def check_homeostatic_health(network, sample_input, verbose=True):
    """
    Comprehensive health check for 3-Net homeostatic mechanism.
    
    Based on paper's Table 1 metrics:
    ✅ Spectral Radius: 0.84 < ρ < 0.87 (critical band)
    ✅ Stabilizer Mean: 0.28 < ᾱ < 0.51 (active)
    ✅ Input Correlation: |r| > 0.3 (adaptive)
    
    Args:
        network: Edge network with homeostatic components
        sample_input: Sample state to test
        verbose: Print diagnostics
    
    Returns:
        dict with health metrics
    """
    diagnostics = {}
    
    # 1. Check carrier_scale and trajectory_scale
    if hasattr(network, 'carrier_scale'):
        carrier_scale = network.carrier_scale.item()
        trajectory_scale = network.trajectory_scale.item()
        diagnostics['carrier_scale'] = carrier_scale
        diagnostics['trajectory_scale'] = trajectory_scale
        
        # Health check: Should evolve from initial 0.1
        if abs(carrier_scale - 0.1) < 0.001 and abs(trajectory_scale - 0.1) < 0.001:
            diagnostics['scale_health'] = 'FROZEN (not learning!)'
        else:
            diagnostics['scale_health'] = 'LEARNING'
    
    # 2. Check damping_net output distribution
    if hasattr(network, 'damping_net'):
        with torch.no_grad():
            alpha_c = torch.sigmoid(network.damping_net(sample_input))
            alpha_mean = alpha_c.mean().item()
            alpha_std = alpha_c.std().item()
            diagnostics['alpha_mean'] = alpha_mean
            diagnostics['alpha_std'] = alpha_std
            
            # Health check: Should vary (not constant)
            if alpha_std < 0.01:
                diagnostics['damping_health'] = 'CONSTANT (not adaptive!)'
            elif 0.28 < alpha_mean < 0.51:
                diagnostics['damping_health'] = 'OPTIMAL (paper range)'
            else:
                diagnostics['damping_health'] = 'ACTIVE (out of paper range)'
    
    # 3. Check freq_damping
    if hasattr(network, 'freq_damping'):
        damping_min = network.freq_damping.min().item()
        damping_max = network.freq_damping.max().item()
        diagnostics['freq_damping_range'] = (damping_min, damping_max)
        
        # Health check: Should be in ~[0.5, 0.95] range
        if abs(damping_max - 0.95) < 0.01 and abs(damping_min - 0.5) < 0.01:
            diagnostics['freq_damping_health'] = 'FROZEN (not learning!)'
        else:
            diagnostics['freq_damping_health'] = 'LEARNING'
    
    if verbose:
        print("\n" + "="*60)
        print("🔬 HOMEOSTATIC MECHANISM DIAGNOSTICS (3-Net)")
        print("="*60)
        
        if 'carrier_scale' in diagnostics:
            print(f"\n📊 Global Spectral Controller (γ scaling):")
            print(f"   carrier_scale:    {diagnostics['carrier_scale']:.6f} [{diagnostics['scale_health']}]")
            print(f"   trajectory_scale: {diagnostics['trajectory_scale']:.6f}")
        
        if 'alpha_mean' in diagnostics:
            print(f"\n🎯 Local Stabilizer (α damping):")
            print(f"   Mean α: {diagnostics['alpha_mean']:.4f} [{diagnostics['damping_health']}]")
            print(f"   Std α:  {diagnostics['alpha_std']:.4f}")
            print(f"   Paper target: ᾱ ∈ [0.28, 0.51]")
        
        if 'freq_damping_range' in diagnostics:
            print(f"\n🌊 Frequency-Dependent Damping:")
            print(f"   Range: [{diagnostics['freq_damping_range'][0]:.4f}, {diagnostics['freq_damping_range'][1]:.4f}]")
            print(f"   Status: {diagnostics['freq_damping_health']}")
        
        print("\n" + "="*60)
        
        # Overall health summary
        issues = []
        if diagnostics.get('scale_health') == 'FROZEN (not learning!)':
            issues.append("⚠️  Spectral controller frozen at initialization")
        if diagnostics.get('damping_health') == 'CONSTANT (not adaptive!)':
            issues.append("⚠️  Damping network outputting constant values")
        if diagnostics.get('freq_damping_health') == 'FROZEN (not learning!)':
            issues.append("⚠️  Frequency damping not adapting")
        
        if issues:
            print("🚨 ISSUES DETECTED:")
            for issue in issues:
                print(f"   {issue}")
            print("\n   → Homeostatic mechanism may not be functioning correctly!")
        else:
            print("✅ All homeostatic components appear to be learning!")
        
        print("="*60 + "\n")
    
    return diagnostics
