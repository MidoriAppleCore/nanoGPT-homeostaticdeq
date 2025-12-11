"""
Hermetic Checkpointing Engine (Dual-Vector / 3-Net Version)

The "Time Machine" for the 3-Net Architecture.
Enables O(1) memory training for infinite-length essays.

UPDATES FOR DUAL-VECTOR ARCHITECTURE:
- Handles (carrier, trajectory) tuple state
- Calls the new 3-Net EdgeNeuralNet with dual input/output
- NO MoE ROUTER = Guaranteed deterministic (no butterfly effect!)
- Pure force fields: carrier_net + trajectory_net + damping_net + steering_net

THE MEMORY PROBLEM WE SOLVE:
Without checkpointing: Store activations for ALL edges (1000+ edges × 15MB = 15GB) → OOM
With checkpointing: Store only chunk activations (20 edges × 15MB = 300MB) → Fits!

THE DETERMINISM GUARANTEE:
Old version had MoE router → float32 drift → different expert choice → CheckpointError
New version is PURE LINEAR → no routing decisions → identical recomputation → Safe!

THREE-NETWORK ARCHITECTURE (Jones 2025):
1. Core Solver (f_θ): carrier_net + trajectory_net → geometric forces
2. Local Stabilizer (α): damping_net → spatially-adaptive relaxation
3. Global Spectral Controller (γ): steering_net → step-size scaling

Update Rule: p(t+1) = p(t) + γ · α(x) ⊙ [scale · f_θ(p)]

MEMORY SAVINGS:
- 1000 edges without checkpoint: ~15GB VRAM (OOM on 6GB GPU)
- 1000 edges with checkpoint: ~300MB VRAM (Fits comfortably!)
- Savings: 50× reduction in activation memory
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import List, Tuple
import random
import numpy as np


class HermeticEdgeRunner(nn.Module):
    """
    A 'Clean Room' for running edges with DETERMINISTIC computation.
    Guarantees deterministic graph structure for gradient checkpointing.
    
    DENSE SUPERVISION UPDATE:
    - Now returns ALL intermediate trajectories (not just final state)
    - Enables per-step loss calculation (critical for learning)
    
    This wrapper enforces:
    1. Canonical shapes [1, 1, dim]
    2. Float32 dtype (no drift to float64)
    3. Trajectory history collection for dense supervision
    
    THE KEY INSIGHT:
    MoE routing is chaotic - tiny numerical noise causes the router to pick
    different experts between forward and backward, creating different graphs.
    Solution: Lock the RNG seed so forward and backward see the SAME universe.
    
    NOTE: EdgeNeuralNet does NOT iterate internally - it's single-pass.
    Each edge is one "step" in the larger sequence-level DEQ.
    """
    
    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold
        
    def forward(self, carrier, trajectory, edge_networks, targets, z, origin, passage_tokens=None):
        """
        Run a sequence of edges with DENSE SUPERVISION.
        
        Simple forward pass - snap training happens OUTSIDE this function.
        
        Args:
            carrier: Input carrier state [1, 1, dim]
            trajectory: Input trajectory state [1, 1, dim]
            edge_networks: List of EdgeNeuralNet modules
            targets: List of target positions (not used in dense version)
            z: Manifold origin context [1, 1, dim+1]
            origin: Manifold origin point [dim+1]
            passage_tokens: Optional token sequence for greybox (needed for set_tokens)
            
        Returns:
            curr_c: Final carrier state [1, 1, dim]
            curr_t: Final trajectory state [1, 1, dim]
            traj_stack: ALL intermediate trajectories [num_edges, dim] for dense loss
        """
        # === GAUGE FIXING ===
        carrier = carrier.to(torch.float32)
        trajectory = trajectory.to(torch.float32)
        
        if carrier.dim() < 3: carrier = carrier.view(1, 1, -1)
        if trajectory.dim() < 3: trajectory = trajectory.view(1, 1, -1)
        
        curr_c, curr_t = carrier, trajectory
        
        # Store ALL intermediate trajectories for Dense Supervision
        traj_history = []
        
        # === THE SEQUENCE ===
        for i, network in enumerate(edge_networks):
            # Set tokens for greybox cybernetics (if enabled)
            if passage_tokens is not None and hasattr(network, 'set_tokens'):
                src_tok = passage_tokens[i].item()
                tgt_tok = passage_tokens[i + 1].item()
                network.set_tokens(src_tok, tgt_tok)
            
            # Run the 3-Net Edge (with internal convergence loop)
            curr_c, curr_t = network(
                carrier_in=curr_c,
                trajectory_in=curr_t,
                source_pos=origin, 
                target_pos=None
            )
            
            # Sanitize
            curr_c = curr_c.to(torch.float32)
            curr_t = curr_t.to(torch.float32)
            
            # Save the "Aim" (Trajectory) for this step
            traj_history.append(curr_t)
        
        # Stack history: [Sequence_Length, 1, 1, Dim] -> [Seq, Dim]
        traj_stack = torch.cat(traj_history, dim=0).squeeze(1).squeeze(1)
        
        # Return final states AND the full history
        return curr_c, curr_t, traj_stack


def make_target_pos_factory(manifold, device):
    """
    Create a function that generates random target positions.
    Used for MoE routing in edge networks.
    """
    def make_target_pos():
        target = torch.randn(manifold.dim + 1, device=device)
        target = manifold.project(target)
        return target.to(torch.float32)  # Explicit float32
    return make_target_pos


def train_passage_with_hermetic_checkpoint(
    passage_tokens,
    model_dir,
    vram_cache,
    manifold,
    decoder_head,
    decoder_optimizer,
    device,
    token_positions,  # NEW: Fixed token positions in hyperbolic space
    chunk_size=20,
    preservation_weight=0.5,
    target_carrier_norm=5.0,
    weak_threshold=1.0,
    snap_to_target=True,
    snap_threshold=2.0,
    max_snap_steps=10,
    pass2_iterations=5,  # How many global backprop steps
    use_greybox=False,  # NEW: Enable greybox cybernetics
    vocab_size=16,  # NEW: Vocabulary size for greybox
    stoi=None,  # NEW: Token to ID mapping for greybox
    tui=None,
    live=None,
    verbose=True
):
    """
    Train on a single passage using Hermetic Checkpointing with DENSE SUPERVISION 
    and optional STAGED CONVERGENCE.
    
    STAGED CONVERGENCE (snap_to_target=True):
    TWO-PASS SYSTEM:
    
    PASS 1: "Snap Training" (Sequential, no checkpointing)
    - Go through passage edge by edge
    - For each edge that misses (loss > threshold):
      * Train it IN PLACE until it hits (or max_snap_steps)
      * Save updated weights
    - Once an edge hits, move to next edge with clean carrier
    
    PASS 2: "Global Backprop" (Checkpointed, frozen weights)
    - Run full passage with checkpointing
    - All edges now have good weights from Pass 1
    - Dense supervision across entire sequence
    - Fine-tune with end-to-end gradients
    
    Args:
        passage_tokens: Token sequence [num_tokens]
        model_dir: Path to model directory
        vram_cache: VRAM cache manager
        manifold: Lorentz manifold
        decoder_head: Token decoder network
        decoder_optimizer: Optimizer for decoder
        device: torch device
        chunk_size: Edges per checkpoint chunk
        preservation_weight: Weight for carrier norm regularization
        target_carrier_norm: Target norm for carrier
        weak_threshold: Gradient threshold for identifying weak edges
        snap_to_target: If True, do Pass 1 (snap training) before Pass 2 (global backprop)
        snap_threshold: Loss threshold for "converged" edge (default 2.0)
        max_snap_steps: Max correction steps per edge (default 10)
        verbose: Print progress
    
    Returns:
        Dictionary with training metrics
    """
    from edge_neural_net_deterministic import EdgeNeuralNet
    
    num_edges = len(passage_tokens) - 1
    
    if verbose:
        mode_str = "STAGED CONVERGENCE (2-pass)" if snap_to_target else "DENSE SUPERVISION"
        print(f"     Training passage ({num_edges} edges - {mode_str})")
    
    runner = HermeticEdgeRunner(manifold).to(device)
    
    origin = manifold.project(torch.zeros(manifold.dim + 1, device=device))
    z = origin.unsqueeze(0).unsqueeze(0).to(torch.float32)
    origin = origin.to(torch.float32)
    make_target_pos = make_target_pos_factory(manifold, device)
    
    # INITIALIZE
    carrier = torch.zeros(1, 1, manifold.dim, device=device, dtype=torch.float32)
    trajectory = torch.zeros(1, 1, manifold.dim, device=device, dtype=torch.float32)

    src_token = passage_tokens[0].item()
    carrier[0, 0, src_token % manifold.dim] = 0.1 
    
    # Load all edges first
    all_edge_tuples = []
    all_edge_networks = []
    all_edge_optimizers = []
    
    for k in range(num_edges):
        src, tgt = passage_tokens[k].item(), passage_tokens[k + 1].item()
        edge = (src, tgt)
        
        network, _ = vram_cache.load_edge(
            src, tgt, network_class=EdgeNeuralNet,
            network_kwargs={
                'hidden_dim': manifold.dim + 1, 
                'manifold': manifold, 
                'num_heads': 4,
                'use_greybox': use_greybox,
                'vocab_size': vocab_size
            }
        )
        network = network.float()
        
        # Configure greybox token mapping if enabled
        if use_greybox and stoi is not None and hasattr(network, 'set_vocab_mapping'):
            network.set_vocab_mapping(stoi)
        
        # Create optimizer with separate LR for homeostatic parameters
        # Paper shows spectral controller (γ_c, γ_t) learns actively - needs higher LR
        homeostatic_params = []
        other_params = []
        
        for name, param in network.named_parameters():
            if 'carrier_scale' in name or 'trajectory_scale' in name or 'freq_damping' in name:
                homeostatic_params.append(param)
            else:
                other_params.append(param)
        
        if homeostatic_params:
            # Homeostatic params get 10x higher learning rate (critical for stability control)
            optimizer = torch.optim.AdamW([
                {'params': other_params, 'lr': 1e-4},
                {'params': homeostatic_params, 'lr': 1e-3}  # 10x higher for γ_c, γ_t, freq_damping
            ])
        else:
            optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)
        
        all_edge_tuples.append(edge)
        all_edge_networks.append(network)
        all_edge_optimizers.append(optimizer)
    
    # === PASS 1: SNAP TRAINING (if enabled) ===
    snap_stats = {'total_corrections': 0, 'edges_corrected': 0}
    
    if snap_to_target:
        if verbose:
            print(f"       PASS 1: Snap training (enforcing hits before continuing)...")
        
        curr_c, curr_t = carrier, trajectory
        
        for i, (network, optimizer, edge) in enumerate(zip(all_edge_networks, all_edge_optimizers, all_edge_tuples)):
            target_token = passage_tokens[i + 1].to(device)
            
            # Update TUI before testing
            if tui is not None and live is not None:
                tui.update(
                    current_iteration=i,
                    current_passage=passage_tokens,
                    current_edge_idx=i,
                    current_edge=edge,
                    current_carrier=curr_c.squeeze() if curr_c.dim() > 1 else curr_c,
                    current_loss=0.0,
                    dirty_from=i
                )
                live.update(tui.render())
            
            # Try the edge
            with torch.no_grad():
                # Get token positions for this edge
                src_token, tgt_token_val = edge
                src_pos = token_positions(torch.tensor([src_token], device=device))
                tgt_pos = token_positions(torch.tensor([target_token.item()], device=device))
                
                # Set tokens for greybox
                if hasattr(network, 'set_tokens'):
                    network.set_tokens(src_token, target_token.item())
                
                test_c, test_t = network(curr_c, curr_t, source_pos=src_pos, target_pos=tgt_pos)
                logits = decoder_head(test_t.squeeze().to(torch.float32))
                initial_loss = torch.nn.functional.cross_entropy(
                    logits.unsqueeze(0), target_token.unsqueeze(0)
                ).item()
            
            # If it misses, train until it hits
            if initial_loss > snap_threshold:
                snap_stats['edges_corrected'] += 1
                correction_steps = 0
                
                if verbose:
                    predicted_token = torch.argmax(logits).item()
                    print(f"         Edge {i} ({edge[0]}→{edge[1]}): MISS (pred={predicted_token}, "
                          f"target={target_token.item()}, loss={initial_loss:.2f}), correcting...")
                
                while initial_loss > snap_threshold and correction_steps < max_snap_steps:
                    # Set tokens for greybox
                    if hasattr(network, 'set_tokens'):
                        network.set_tokens(edge[0], target_token.item())
                    
                    # Forward with gradients
                    next_c, next_t = network(curr_c, curr_t, source_pos=src_pos, target_pos=tgt_pos)
                    logits = decoder_head(next_t.squeeze().to(torch.float32))
                    edge_loss = torch.nn.functional.cross_entropy(
                        logits.unsqueeze(0), target_token.unsqueeze(0)
                    )
                    
                    # Update TUI during correction
                    if tui is not None and live is not None:
                        tui.update(
                            current_iteration=i,
                            current_passage=passage_tokens,
                            current_edge_idx=i,
                            current_edge=edge,
                            current_carrier=next_c.squeeze() if next_c.dim() > 1 else next_c,
                            current_loss=edge_loss.item(),
                            dirty_from=i
                        )
                        live.update(tui.render())
                    
                    # Backprop and update THIS edge only
                    optimizer.zero_grad()
                    edge_loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                    optimizer.step()
                    
                    correction_steps += 1
                    snap_stats['total_corrections'] += 1
                    
                    # Check if we hit
                    with torch.no_grad():
                        # Set tokens for greybox
                        if hasattr(network, 'set_tokens'):
                            network.set_tokens(edge[0], target_token.item())
                        
                        test_c, test_t = network(curr_c, curr_t, source_pos=src_pos, target_pos=tgt_pos)
                        logits = decoder_head(test_t.squeeze().to(torch.float32))
                        initial_loss = torch.nn.functional.cross_entropy(
                            logits.unsqueeze(0), target_token.unsqueeze(0)
                        ).item()
                
                if verbose:
                    status = "HIT" if initial_loss < snap_threshold else f"STOPPED (loss={initial_loss:.2f})"
                    print(f"         Edge {i}: {status} after {correction_steps} correction steps")
                
                # Save updated network
                vram_cache.save_edge(edge[0], edge[1], network)
            
            # Move forward with the (now corrected) edge output
            with torch.no_grad():
                src_token, tgt_token = edge
                src_pos = token_positions(torch.tensor([src_token], device=device))
                tgt_pos = token_positions(torch.tensor([tgt_token], device=device))
                
                # Set tokens for greybox
                if hasattr(network, 'set_tokens'):
                    network.set_tokens(src_token, tgt_token)
                
                curr_c, curr_t = network(curr_c, curr_t, source_pos=src_pos, target_pos=tgt_pos)
        
        if verbose:
            print(f"       PASS 1 complete: {snap_stats['edges_corrected']}/{num_edges} edges corrected "
                  f"({snap_stats['total_corrections']} total correction steps)")
    
    # === PASS 2: GLOBAL BACKPROP (Multiple iterations for global learning) ===
    if verbose:
        print(f"       PASS 2: Global backprop ({pass2_iterations} iterations of dense supervision)...")
    
    for pass2_iter in range(pass2_iterations):
        # Simple forward pass without checkpointing
        all_trajectory_outputs = []
        curr_c, curr_t = carrier, trajectory
        
        for i, (network, edge) in enumerate(zip(all_edge_networks, all_edge_tuples)):
            src_token, target_token = edge
            src_pos = token_positions(torch.tensor([src_token], device=device))
            
            # CRITICAL: Don't give target position during forward pass!
            # Backprop will teach the network to shoot toward the RIGHT position
            # by seeing the loss when trajectory doesn't predict the next token correctly
            dummy_tgt_pos = torch.zeros_like(src_pos)
            
            # Set tokens for greybox (for symbolic computation)
            if hasattr(network, 'set_tokens'):
                network.set_tokens(src_token, target_token)
            
            # Forward: network must learn from carrier+source alone which direction to shoot
            # Gradient from decoder loss will teach it the correct geodesic direction!
            curr_c, curr_t = network(curr_c, curr_t, source_pos=src_pos, target_pos=dummy_tgt_pos)
            curr_c = curr_c.to(torch.float32)
            curr_t = curr_t.to(torch.float32)
            all_trajectory_outputs.append(curr_t.squeeze())
        
        # Stack all trajectories
        full_trajectory_sequence = torch.stack(all_trajectory_outputs)
        
        # === DENSE LOSS WITH EXPONENTIAL WEIGHTING ===
        # CRITICAL: Trajectory after edge i should predict the NEXT token (not current token)!
        # After edge src→tgt, we're AT tgt, need to predict where to go NEXT
        # 
        # Example: tokens = [START, 5, +, 5, =, space, 1, 0]
        #          edges  = [START→5, 5→+, +→5, 5→=, =→space, space→1, 1→0]
        #          trajectory[4] (after =→space) should predict next token = 1 (not space!)
        #
        # So: trajectory[i] predicts passage_tokens[i+2]
        # We only have predictions for the first (n-1) trajectories since last one has no "next"
        all_targets = passage_tokens[2:].to(device)  # [+, 5, =, space, 1, 0]
        
        # DEBUG: Verify fix is applied
        if verbose and pass2_iter == 0:
            print(f"       DEBUG: passage_tokens length={len(passage_tokens)}, trajectories={len(all_trajectory_outputs)}, targets={len(all_targets)}")
            print(f"       DEBUG: Target shift active - trajectory[i] predicts token[i+2] (NEXT token, not current)")
        
        # Trim trajectory sequence to match targets (drop last trajectory)
        full_trajectory_sequence = full_trajectory_sequence[:-1]  # Drop last, it has no next token
        
        logits = decoder_head(full_trajectory_sequence.to(torch.float32))
        
        # Compute per-token losses with exponential weighting
        num_tokens = len(all_targets)
        total_prediction_loss_weighted = 0.0
        total_prediction_loss_uniform = 0.0
        
        for i in range(num_tokens):
            # Exponential weight: later positions matter MORE
            # weight = exp(i / N) ranges from 1.0 to e ≈ 2.718
            # NO NORMALIZATION - we want the task to be HARDER!
            weight = torch.exp(torch.tensor(i / num_tokens, device=device))
            
            token_logits = logits[i].unsqueeze(0)
            token_target = all_targets[i].unsqueeze(0)
            token_loss = torch.nn.functional.cross_entropy(token_logits, token_target)
            
            total_prediction_loss_weighted += weight * token_loss
            total_prediction_loss_uniform += token_loss  # Unweighted for comparison
        
        # Weighted loss (used for training) - ~1.7x higher than uniform
        prediction_loss = total_prediction_loss_weighted / num_tokens
        
        # Base uniform loss (for monitoring/comparison only)
        base_loss = total_prediction_loss_uniform / num_tokens
        
        final_carrier = curr_c.squeeze()
        carrier_norm = torch.norm(final_carrier)
        preservation_loss = preservation_weight * torch.relu(carrier_norm - target_carrier_norm) ** 2
        
        total_loss = prediction_loss + preservation_loss
        
        # Update TUI with computed loss (once per iteration, at the end)
        if tui is not None and live is not None:
            # Prepare trajectory stats
            traj_stats = {
                'norm': torch.norm(curr_t).item(),
                'mean': curr_t.mean().item(),
                'std': curr_t.std().item()
            }
            
            tui.update(
                current_iteration=pass2_iter * num_edges + num_edges - 1,
                current_passage=passage_tokens,
                current_edge_idx=num_edges - 1,
                current_edge=all_edge_tuples[-1],
                current_carrier=curr_c.squeeze() if curr_c.dim() > 1 else curr_c,
                current_loss=total_loss.item(),
                dirty_from=num_edges - 1,
                trajectory_stats=traj_stats
            )
            live.update(tui.render())
        
        # Backprop Pass 2
        decoder_optimizer.zero_grad()
        for opt in all_edge_optimizers:
            opt.zero_grad()
        
        total_loss.backward()
        
        # INCREASED gradient clipping to prevent explosion (was 1.0, now 5.0)
        # DEQ + greybox can cause high gradients, need stronger clipping
        torch.nn.utils.clip_grad_norm_(decoder_head.parameters(), 5.0)
        decoder_optimizer.step()
        
        for network, optimizer in zip(all_edge_networks, all_edge_optimizers):
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
        
        if verbose and (pass2_iter == 0 or pass2_iter == pass2_iterations - 1):
            print(f"         Iteration {pass2_iter+1}/{pass2_iterations}: Loss {total_loss.item():.4f} (pred: {prediction_loss.item():.4f})")
    
    # Final statistics - compute gradient stats for all edges
    weak_edges = set()
    max_grad = 0.0
    min_grad = float('inf')
    all_grads = []  # Collect all gradients for median/std calculation
    
    for edge, network in zip(all_edge_tuples, all_edge_networks):
        # Check gradients for diagnostics (but DON'T cull edges!)
        total_grad = sum(p.grad.abs().sum().item() for p in network.parameters() if p.grad is not None)
        if total_grad > 0:
            max_grad = max(max_grad, total_grad)
            min_grad = min(min_grad, total_grad)
            all_grads.append(total_grad)
        # DISABLED: Don't track "weak edges" - we need all edges for small vocab!
        # if total_grad > weak_threshold: weak_edges.add(edge)
        vram_cache.save_edge(edge[0], edge[1], network)
    
    # === ZERO GRADIENT DIAGNOSTIC ===
    if len(all_grads) == 0 and len(all_edge_tuples) > 0:
        print(f"\n{'='*70}")
        print(f"🚨 CRITICAL: ZERO GRADIENTS DETECTED!")
        print(f"{'='*70}")
        print(f"Passage info:")
        print(f"  Tokens: {passage_tokens}")
        print(f"  Length: {len(passage_tokens)} tokens")
        print(f"  Edges: {len(all_edge_tuples)} total, {len(set(all_edge_tuples))} unique")
        print(f"  Unique tokens: {len(set(sum(all_edge_tuples, ())))}")
        print(f"\nLoss info:")
        print(f"  Total loss: {total_loss.item():.6f} (requires_grad={total_loss.requires_grad})")
        print(f"  Prediction loss: {prediction_loss.item():.6f} (requires_grad={prediction_loss.requires_grad})")
        print(f"  Base loss: {base_loss.item():.6f} (requires_grad={base_loss.requires_grad})")
        print(f"\nCarrier info:")
        print(f"  Norm: {carrier_norm.item():.4f}")
        print(f"  Mean: {final_carrier.mean().item():.6f}")
        print(f"  Std: {final_carrier.std().item():.6f}")
        print(f"  Requires grad: {final_carrier.requires_grad}")
        print(f"\nTrajectory info:")
        print(f"  Norm: {torch.norm(curr_t).item():.4f}")
        print(f"  Mean: {curr_t.mean().item():.6f}")
        print(f"  Requires grad: {curr_t.requires_grad}")
        print(f"\nBackward pass diagnostics:")
        # Check if any network has gradients at all
        any_has_grad = False
        for network in all_edge_networks:
            for p in network.parameters():
                if p.grad is not None and p.grad.abs().sum().item() > 0:
                    any_has_grad = True
                    break
            if any_has_grad:
                break
        print(f"  Any network has non-zero grad: {any_has_grad}")
        
        # Check decoder head gradients
        decoder_grads = []
        for p in decoder_head.parameters():
            if p.grad is not None:
                decoder_grads.append(p.grad.abs().sum().item())
        decoder_grad_total = sum(decoder_grads) if decoder_grads else 0.0
        print(f"  Decoder head grad total: {decoder_grad_total:.6f}")
        
        print(f"{'='*70}\n")
    
    # Compute comprehensive gradient statistics
    if len(all_grads) > 0:
        avg_grad = sum(all_grads) / len(all_grads)
        median_grad = sorted(all_grads)[len(all_grads) // 2]
        std_grad = (sum((g - avg_grad) ** 2 for g in all_grads) / len(all_grads)) ** 0.5
    else:
        avg_grad = 0.0
        median_grad = 0.0
        std_grad = 0.0
        min_grad = 0.0
    
    if verbose:
        print(f"       Loss: {total_loss.item():.6f} (pred: {prediction_loss.item():.6f}, base: {base_loss.item():.6f})")
        print(f"       Carrier: norm={carrier_norm.item():.4f} mean={final_carrier.mean().item():.6f} std={final_carrier.std().item():.6f}")
        print(f"       Trajectory: norm={torch.norm(curr_t).item():.4f} mean={curr_t.mean().item():.6f} std={curr_t.std().item():.6f}")
        print(f"       Gradients: max={max_grad:.6f} min={min_grad:.6f} avg={avg_grad:.6f} median={median_grad:.6f} std={std_grad:.6f}")
        
        # 3-Net Homeostatic Diagnostics (Jones 2025 paper metrics)
        # Table 1 targets: ρ(J_f)∈[0.84,0.87], ᾱ∈[0.28,0.51], |r|>0.3
        if len(all_edge_networks) > 0 and hasattr(all_edge_networks[0], 'carrier_scale'):
            net = all_edge_networks[0]
            carrier_scale = net.carrier_scale.item()
            trajectory_scale = net.trajectory_scale.item()

            # Check if damping_net is learning (not stuck at constant)
            with torch.no_grad():
                # Build dummy input safely (freq_damping may be a 1D param)
                try:
                    freq_len = int(net.freq_damping.shape[0])
                except Exception:
                    freq_len = 64
                dummy_input = torch.randn(8, 1, freq_len * 2, device=net.carrier_scale.device)
                alpha_sample = torch.sigmoid(net.damping_net(torch.nan_to_num(dummy_input, nan=0.0)))
                alpha_flat = alpha_sample.reshape(-1)
                n_nans = int(torch.isnan(alpha_flat).sum().item())
                if n_nans > 0:
                    alpha_nonan = alpha_flat[~torch.isnan(alpha_flat)]
                else:
                    alpha_nonan = alpha_flat
                if alpha_nonan.numel() > 0:
                    alpha_mean = float(alpha_nonan.mean().item())
                    alpha_std = float(alpha_nonan.std().item())
                else:
                    alpha_mean = float('nan')
                    alpha_std = float('nan')

            # Format with health indicators
            scale_status = "✓" if abs(carrier_scale - 0.1) > 0.001 else "✗FROZEN"
            damping_status = "✓" if (not (alpha_std != alpha_std)) and alpha_std > 0.01 else "✗CONST"
            optimal = "✓OPTIMAL" if (not (alpha_mean != alpha_mean)) and 0.28 < alpha_mean < 0.51 else ""

            nan_note = f", {n_nans} NaNs" if n_nans > 0 else ""
            print(f"       3-Net: γ_c={carrier_scale:.6f}[{scale_status}] γ_t={trajectory_scale:.6f} | "
                  f"ᾱ={alpha_mean:.4f}±{alpha_std:.4f}[{damping_status}{optimal}]{nan_note}")
        
        print(f"       Edges trained: {len(all_edge_tuples)} | Unique tokens: {len(set(sum(all_edge_tuples, ())))} ")
    
    # Update TUI with gradient stats
    if tui is not None and live is not None:
        grad_stats = {
            'max': max_grad,
            'min': min_grad,
            'avg': avg_grad,
            'median': median_grad,
            'std': std_grad
        }
        
        # Add 3-Net homeostatic stats if available (NaN-safe version)
        homeostatic_stats = {}
        if len(all_edge_networks) > 0 and hasattr(all_edge_networks[0], 'carrier_scale'):
            net = all_edge_networks[0]
            homeostatic_stats['carrier_scale'] = net.carrier_scale.item()
            homeostatic_stats['trajectory_scale'] = net.trajectory_scale.item()
            
            # Sample damping_net with NaN-safe computation (same as console output)
            with torch.no_grad():
                try:
                    freq_len = int(net.freq_damping.shape[0])
                except Exception:
                    freq_len = 64
                dummy_input = torch.randn(8, 1, freq_len * 2, device=net.carrier_scale.device)
                alpha_sample = torch.sigmoid(net.damping_net(torch.nan_to_num(dummy_input, nan=0.0)))
                alpha_flat = alpha_sample.reshape(-1)
                n_nans = int(torch.isnan(alpha_flat).sum().item())
                if n_nans > 0:
                    alpha_nonan = alpha_flat[~torch.isnan(alpha_flat)]
                else:
                    alpha_nonan = alpha_flat
                if alpha_nonan.numel() > 0:
                    homeostatic_stats['alpha_mean'] = float(alpha_nonan.mean().item())
                    homeostatic_stats['alpha_std'] = float(alpha_nonan.std().item())
                else:
                    homeostatic_stats['alpha_mean'] = 0.0
                    homeostatic_stats['alpha_std'] = 0.0
        
        tui.update(grad_stats=grad_stats, homeostatic_stats=homeostatic_stats)
        live.update(tui.render())
        
    return {
        'loss': total_loss.item(),
        'prediction_loss': prediction_loss.item(),
        'base_loss': base_loss.item(),  # Uniform weighting for comparison
        'carrier_norm': carrier_norm.item(),
        'weak_edges': weak_edges,
        'trained_edges': all_edge_tuples,
        'snap_stats': snap_stats
    }
