"""
Gradient Phase Portrait Visualization

Mathematical Tool for Diagnosing Optimization Topology

This module reveals the local geometry of the loss landscape by projecting
the high-dimensional gradient vector field onto its principal components.

Physical Interpretation:
- The gradient field is a VELOCITY FIELD in parameter space
- PCA finds the plane of maximum "action" (variance in flow)
- The resulting phase portrait shows where energy flows

Topological Signatures:
1. SINK (Stable Fixed Point): Arrows converge to origin
   → Healthy training, system descending into minimum
   
2. VORTEX/SPIRAL (Limit Cycle): Arrows circulate
   → Oscillating loss, spectral radius > 1, chaos
   
3. SADDLE (Unstable): Arrows diverge along one axis
   → Local maximum or plateau, optimizer stuck
   
4. ORTHOGONAL FLOWS: Reflex moving, DEQ stationary
   → Gradient bypass problem (the lizard brain failure mode)

Usage:
    from debug_gradient_flow import visualize_gradient_flow
    
    # In training loop, after backward() but before step()
    scaler.scale(loss).backward()
    
    if iter_num % 100 == 0:
        visualize_gradient_flow(raw_model, optimizer, iter_num, out_dir)
    
    scaler.step(optimizer)
"""

import torch
import numpy as np
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️  matplotlib or sklearn not available - gradient flow visualization disabled")


def visualize_gradient_flow(model, optimizer, iter_num, out_dir, max_layers=50):
    """
    Mathematically rigorous visualization of the optimization landscape.
    Projects the high-dimensional gradient vector field onto 2D PCA space.
    
    Args:
        model: The neural network model (raw_model, not DDP wrapper)
        optimizer: The optimizer (used to get learning rates if needed)
        iter_num: Current training iteration
        out_dir: Output directory for saving plots
        max_layers: Maximum number of layers to visualize (for readability)
    
    Reveals:
        - Sinks (Stable Fixed Points) → Convergence
        - Spirals (Limit Cycles) → Oscillating Loss
        - Sources/Saddles (Unstable) → Divergence/Explosion
        - Orthogonal Flows → Gradient Bypass (Reflex vs DEQ)
    """
    if not VISUALIZATION_AVAILABLE:
        return
    
    # Silent mode - data saved to CSV and graphed automatically
    
    # 1. Collect Gradients from all parameters
    # We treat the gradient of each layer as a "particle" in the flow
    grads = []
    names = []
    layer_types = []  # Track which subsystem each gradient belongs to
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Flatten gradient to vector
            g = param.grad.view(-1).cpu().numpy()
            
            # Subsample large layers to keep PCA tractable (maintain distribution)
            if len(g) > 1000:
                indices = np.random.choice(len(g), 1000, replace=False)
                g = g[indices]
            
            grads.append(g)
            names.append(name)
            
            # Classify layer by subsystem
            if 'deq' in name.lower():
                layer_types.append('DEQ_Cortex')
            elif 'reflex' in name.lower():
                layer_types.append('Reflex_Spinal')
            elif 'encoder' in name.lower() or 'tok_emb' in name.lower():
                layer_types.append('Encoder_Sensory')
            elif 'geometry' in name.lower() or 'lm_head' in name.lower():
                layer_types.append('Geometry_Motor')
            else:
                layer_types.append('Other')
    
    if not grads or len(grads) < 3:
        return  # Skip silently - normal in early training
    
    # 2. Compute Gradient Norms (Energy of Each Layer)
    grad_norms = np.array([np.linalg.norm(g) for g in grads])
    
    # 3. Sample layers if too many (keep most important ones)
    if len(grads) > max_layers:
        # Keep layers with largest gradients (most active)
        top_indices = np.argsort(grad_norms)[-max_layers:]
        grads = [grads[i] for i in top_indices]
        names = [names[i] for i in top_indices]
        layer_types = [layer_types[i] for i in top_indices]
        grad_norms = grad_norms[top_indices]
    
    # 4. Pad gradients to same length (required for stacking)
    max_len = max(len(g) for g in grads)
    grads_padded = []
    for g in grads:
        if len(g) < max_len:
            # Zero-pad shorter gradients
            g_padded = np.zeros(max_len)
            g_padded[:len(g)] = g
            grads_padded.append(g_padded)
        else:
            grads_padded.append(g)
    
    # Stack into [N_layers, D_samples] matrix
    # Each row is a layer's gradient vector state
    grad_matrix = np.stack(grads_padded)
    
    # 5. PCA Projection (Find the plane of maximum variance/action)
    # This reveals the "dominant dynamics" of the system
    pca = PCA(n_components=2)
    coords = pca.fit_transform(grad_matrix)
    
    # 6. Compute Flow Vectors (Velocity in Phase Space)
    # In optimization, the gradient *is* the velocity vector (v = -g for descent)
    # We project the velocity onto the same PCA basis
    velocities = pca.transform(-grad_matrix)  # Gradient descent moves opposite to gradient
    
    # 7. Generate Phase Portrait
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plt.style.use('dark_background')
    
    # === LEFT PANEL: Phase Portrait with Vector Field ===
    
    # Color mapping for subsystems
    color_map = {
        'DEQ_Cortex': 'yellow',
        'Reflex_Spinal': 'cyan',
        'Encoder_Sensory': 'green',
        'Geometry_Motor': 'magenta',
        'Other': 'gray'
    }
    colors = [color_map.get(lt, 'gray') for lt in layer_types]
    
    # Quiver plot: Arrows show the flow of the optimizer
    # Arrow length = gradient magnitude (how fast this layer is changing)
    ax1.quiver(coords[:, 0], coords[:, 1], 
               velocities[:, 0], velocities[:, 1],
               angles='xy', scale_units='xy', scale=1,
               color=colors, alpha=0.7, width=0.003, headwidth=4)
    
    # Scatter plot: Points are the layers/parameters
    # Size proportional to gradient norm (larger = more active)
    sizes = 50 + 200 * (grad_norms / (grad_norms.max() + 1e-8))
    for layer_type in set(layer_types):
        mask = np.array([lt == layer_type for lt in layer_types])
        if mask.any():
            ax1.scatter(coords[mask, 0], coords[mask, 1], 
                       c=color_map.get(layer_type, 'gray'),
                       s=sizes[mask], alpha=0.8, label=layer_type,
                       edgecolors='white', linewidths=0.5)
    
    # Draw origin (the attractor we want to reach)
    ax1.scatter([0], [0], c='red', s=300, marker='*', 
               edgecolors='white', linewidths=2, label='Origin (Target)', zorder=10)
    
    # Topological Analysis
    explained_var = pca.explained_variance_ratio_
    ax1.set_title(f'Gradient Phase Portrait (Iter {iter_num})\n' + 
                  f'PC1: {explained_var[0]:.1%} | PC2: {explained_var[1]:.1%}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Principal Component 1 (Dominant Mode)', fontsize=11)
    ax1.set_ylabel('Principal Component 2 (Secondary Mode)', fontsize=11)
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.axhline(y=0, color='white', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='white', linestyle='--', alpha=0.3)
    
    # === RIGHT PANEL: Gradient Magnitude Spectrum ===
    
    # Sort layers by gradient norm
    sorted_indices = np.argsort(grad_norms)[::-1]
    sorted_norms = grad_norms[sorted_indices]
    sorted_types = [layer_types[i] for i in sorted_indices]
    sorted_colors = [color_map.get(lt, 'gray') for lt in sorted_types]
    
    # Bar plot of gradient magnitudes
    bars = ax2.barh(range(len(sorted_norms)), sorted_norms, color=sorted_colors, alpha=0.7)
    
    # Label bars with layer type
    for i, (norm, ltype) in enumerate(zip(sorted_norms, sorted_types)):
        ax2.text(norm, i, f' {ltype}', va='center', fontsize=8, color='white')
    
    ax2.set_xlabel('Gradient Norm ||∇θ||₂', fontsize=11)
    ax2.set_ylabel('Layer Index (Sorted by Activity)', fontsize=11)
    ax2.set_title(f'Gradient Magnitude Spectrum\n(Which subsystem is learning?)', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.2, axis='x')
    ax2.set_ylim(-0.5, len(sorted_norms) - 0.5)
    
    # Compute subsystem statistics (silent - saved to graph)
    subsystem_stats = {}
    for lt in set(layer_types):
        mask = np.array([l == lt for l in layer_types])
        if mask.any():
            subsystem_stats[lt] = {
                'mean_norm': grad_norms[mask].mean(),
                'max_norm': grad_norms[mask].max(),
                'count': mask.sum()
            }
    
    # === TOPOLOGICAL DIAGNOSIS === (silent - saved to graph)
    
    # Check for vortex (limit cycle): velocities perpendicular to position
    dot_products = np.sum(coords * velocities, axis=1)
    alignment = np.mean(dot_products) / (np.linalg.norm(coords.mean(axis=0)) * np.linalg.norm(velocities.mean(axis=0)) + 1e-8)
    
    # All diagnostics saved to graph - no console spam
    
    plt.tight_layout()
    
    # Save (silent mode - check reports folder)
    reports_dir = os.path.join(out_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    save_path = os.path.join(reports_dir, f'gradient_flow_iter_{iter_num:06d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # No console spam - gradient flow saved to graph


def diagnose_topology(coords, velocities):
    """
    Classify the local topology of the optimization landscape.
    
    Returns:
        str: One of 'SINK', 'SOURCE', 'SADDLE', 'VORTEX', 'CENTER'
    """
    # Compute Jacobian of flow at mean position
    # J[i,j] = ∂v_i/∂x_j approximated by finite differences
    
    mean_coord = coords.mean(axis=0)
    mean_velocity = velocities.mean(axis=0)
    
    # Eigenvalue analysis would require more samples
    # For now, use simple divergence/curl heuristics
    
    # Divergence: ∇·v (trace of Jacobian)
    # Positive = source, negative = sink
    div = np.sum(velocities, axis=0) / len(velocities)
    
    # Curl: rotation (perpendicularity of v to r)
    alignment = np.mean(np.sum(coords * velocities, axis=1))
    
    if alignment < -0.5 and np.linalg.norm(div) < 0.1:
        return 'SINK'
    elif alignment > 0.5:
        return 'SOURCE'
    elif abs(alignment) < 0.3:
        return 'VORTEX'
    else:
        return 'SADDLE'
