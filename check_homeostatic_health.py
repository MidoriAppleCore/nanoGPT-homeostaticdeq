#!/usr/bin/env python3
"""
Homeostatic Health Check for 3-Net DEQ
Validates that your implementation matches the paper's architecture.

Based on: "Multi-Network Deep Equilibrium Models for Heterogeneous PDEs" (Jones, 2025)

Run this on your current checkpoint to verify:
1. Spectral Controller (carrier_scale, trajectory_scale) is learning
2. Local Stabilizer (damping_net) is producing adaptive α values
3. Frequency Damping is active
"""

import torch
import sys
import os

# Import your edge network
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_neural_net_deterministic import PureFourierDEQOperator
from disk_backed_tensor import VRAMCache


def load_sample_edge_network(checkpoint_dir="hermetic_checkpoints"):
    """Load a sample edge network from checkpoint to inspect."""
    vram = VRAMCache(checkpoint_dir, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Try to load any edge
    edges = vram.list_edges()
    if not edges:
        print(f"❌ No edges found in {checkpoint_dir}")
        print("   Run training first to create checkpoints!")
        return None
    
    print(f"📁 Found {len(edges)} edge networks in checkpoint")
    
    # Load first edge
    src, dst = edges[0]
    network = vram.load_edge(src, dst)
    
    if network is None:
        print(f"❌ Failed to load edge ({src}, {dst})")
        return None
    
    print(f"✅ Loaded edge network: ({src}, {dst})")
    return network


def diagnose_3net_health(network):
    """
    Comprehensive diagnostic following the paper's metrics.
    
    Paper's Table 1 (Case I & II):
    - Spectral Radius ρ̂: 0.846 - 0.865
    - Stabilizer Mean ᾱ: 0.282 - 0.511
    - Input-Stabilizer Correlation: -0.38 to -0.49 (negative = adaptive)
    """
    
    print("\n" + "="*70)
    print("🔬 3-NET HOMEOSTATIC MECHANISM HEALTH CHECK")
    print("="*70)
    print("\nPaper: 'Multi-Network Deep Equilibrium Models for Heterogeneous PDEs'")
    print("Architecture: Core Solver + Local Stabilizer + Global Spectral Controller")
    print("="*70)
    
    issues = []
    warnings = []
    
    # ============================================================
    # 1. GLOBAL SPECTRAL CONTROLLER (γ scaling)
    # ============================================================
    print("\n📊 [1/3] GLOBAL SPECTRAL CONTROLLER")
    print("-" * 70)
    
    if hasattr(network, 'carrier_scale') and hasattr(network, 'trajectory_scale'):
        carrier_scale = network.carrier_scale.item()
        trajectory_scale = network.trajectory_scale.item()
        
        print(f"   carrier_scale:    {carrier_scale:.6f}")
        print(f"   trajectory_scale: {trajectory_scale:.6f}")
        
        # Check if learning
        if abs(carrier_scale - 0.1) < 0.0001 and abs(trajectory_scale - 0.1) < 0.0001:
            print("   ❌ FROZEN at initialization (0.1, 0.1)")
            print("      → These parameters are NOT being optimized!")
            issues.append("Spectral controller frozen")
        elif abs(carrier_scale - 0.1) < 0.001 and abs(trajectory_scale - 0.1) < 0.001:
            print("   ⚠️  Barely changed from initialization")
            print("      → May need more training or higher learning rate")
            warnings.append("Spectral controller barely learning")
        else:
            print("   ✅ LEARNING (evolved from initial 0.1)")
            
            # Check if in reasonable range
            if carrier_scale > 1.0 or trajectory_scale > 1.0:
                print(f"   ⚠️  Very large scale (may cause instability)")
                warnings.append("Large spectral scale")
            elif carrier_scale < 0.01 or trajectory_scale < 0.01:
                print(f"   ⚠️  Very small scale (may slow convergence)")
                warnings.append("Small spectral scale")
    else:
        print("   ❌ carrier_scale/trajectory_scale NOT FOUND!")
        print("      → Network is missing homeostatic components!")
        issues.append("Missing spectral controller")
    
    # ============================================================
    # 2. LOCAL STABILIZER (α adaptive damping)
    # ============================================================
    print("\n🎯 [2/3] LOCAL STABILIZER (Damping Network)")
    print("-" * 70)
    
    if hasattr(network, 'damping_net'):
        # Test with random input
        device = network.carrier_scale.device if hasattr(network, 'carrier_scale') else 'cpu'
        freq_dim = network.freq_damping.shape[0] if hasattr(network, 'freq_damping') else 128
        
        with torch.no_grad():
            # Test with multiple random inputs to see variation
            n_samples = 100
            dummy_input = torch.randn(n_samples, 1, freq_dim * 2, device=device)
            alpha_output = torch.sigmoid(network.damping_net(dummy_input))
            
            alpha_mean = alpha_output.mean().item()
            alpha_std = alpha_output.std().item()
            alpha_min = alpha_output.min().item()
            alpha_max = alpha_output.max().item()
        
        print(f"   α distribution (100 random samples):")
        print(f"      Mean:  {alpha_mean:.4f}")
        print(f"      Std:   {alpha_std:.4f}")
        print(f"      Range: [{alpha_min:.4f}, {alpha_max:.4f}]")
        print(f"\n   Paper target: ᾱ ∈ [0.28, 0.51]")
        
        # Health checks
        if alpha_std < 0.001:
            print("   ❌ CONSTANT OUTPUT (std < 0.001)")
            print("      → Damping network is not adaptive!")
            issues.append("Damping network outputs constant")
        elif alpha_std < 0.01:
            print("   ⚠️  Very low variance (std < 0.01)")
            print("      → Network may be collapsing to single value")
            warnings.append("Low damping variance")
        else:
            print("   ✅ ADAPTIVE (varying outputs)")
        
        # Check if in paper's range
        if 0.28 <= alpha_mean <= 0.51:
            print(f"   ✅ OPTIMAL RANGE (matches paper Case I & II)")
        elif alpha_mean < 0.1:
            print(f"   ⚠️  Very low mean (strong damping, may slow convergence)")
            warnings.append("Very low alpha mean")
        elif alpha_mean > 0.9:
            print(f"   ⚠️  Very high mean (weak damping, may cause instability)")
            warnings.append("Very high alpha mean")
        else:
            print(f"   ✓  Active (outside paper range but varying)")
        
        # Check if saturated
        if alpha_max > 0.99:
            print(f"   ⚠️  Saturated at high end (sigmoid at 1.0)")
            warnings.append("Alpha saturation high")
        if alpha_min < 0.01:
            print(f"   ⚠️  Saturated at low end (sigmoid at 0.0)")
            warnings.append("Alpha saturation low")
            
    else:
        print("   ❌ damping_net NOT FOUND!")
        print("      → Network is missing local stabilizer!")
        issues.append("Missing damping network")
    
    # ============================================================
    # 3. FREQUENCY-DEPENDENT DAMPING
    # ============================================================
    print("\n🌊 [3/3] FREQUENCY-DEPENDENT DAMPING")
    print("-" * 70)
    
    if hasattr(network, 'freq_damping'):
        damping_min = network.freq_damping.min().item()
        damping_max = network.freq_damping.max().item()
        damping_mean = network.freq_damping.mean().item()
        
        print(f"   Range: [{damping_min:.4f}, {damping_max:.4f}]")
        print(f"   Mean:  {damping_mean:.4f}")
        print(f"\n   Expected: Low freq (memory) → 0.95, High freq (details) → 0.5")
        
        # Check if learning
        if abs(damping_max - 0.95) < 0.0001 and abs(damping_min - 0.5) < 0.0001:
            print("   ❌ FROZEN at initialization [0.5, 0.95]")
            print("      → Frequency damping is NOT being optimized!")
            issues.append("Frequency damping frozen")
        elif abs(damping_max - 0.95) < 0.001 and abs(damping_min - 0.5) < 0.001:
            print("   ⚠️  Barely changed from initialization")
            warnings.append("Frequency damping barely learning")
        else:
            print("   ✅ LEARNING (evolved from initial [0.5, 0.95])")
        
        # Check if reasonable
        if damping_max > 1.0 or damping_min > 1.0:
            print("   ⚠️  Damping > 1.0 (amplification instead of decay!)")
            warnings.append("Damping amplification")
        elif damping_max < 0.5:
            print("   ⚠️  Very aggressive damping (may erase memory)")
            warnings.append("Aggressive damping")
    else:
        print("   ❌ freq_damping NOT FOUND!")
        print("      → Network is missing frequency-dependent damping!")
        issues.append("Missing frequency damping")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("📋 DIAGNOSTIC SUMMARY")
    print("="*70)
    
    if not issues and not warnings:
        print("✅ ALL SYSTEMS OPERATIONAL")
        print("\nYour 3-Net homeostatic mechanism matches the paper's architecture!")
        print("The network should be learning stable, adaptive dynamics.")
    else:
        if issues:
            print(f"🚨 CRITICAL ISSUES ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
            print("\n   → Homeostatic mechanism is NOT functioning correctly!")
            print("   → Training may be unstable or ineffective!")
        
        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for i, warning in enumerate(warnings, 1):
                print(f"   {i}. {warning}")
            print("\n   → Homeostatic mechanism is active but suboptimal")
            print("   → Consider adjusting hyperparameters or training longer")
    
    print("="*70 + "\n")
    
    return len(issues) == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check homeostatic mechanism health")
    parser.add_argument("--checkpoint-dir", default="hermetic_checkpoints",
                       help="Directory containing edge network checkpoints")
    args = parser.parse_args()
    
    print("="*70)
    print("HOMEOSTATIC MECHANISM HEALTH CHECK")
    print("="*70)
    
    network = load_sample_edge_network(args.checkpoint_dir)
    
    if network is None:
        print("\n❌ Could not load network. Exiting.")
        sys.exit(1)
    
    healthy = diagnose_3net_health(network)
    
    if healthy:
        print("✅ Homeostatic mechanism is healthy!")
        sys.exit(0)
    else:
        print("❌ Homeostatic mechanism has issues. See diagnostic above.")
        sys.exit(1)
