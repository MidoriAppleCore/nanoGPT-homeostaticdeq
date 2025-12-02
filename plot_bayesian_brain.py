"""
Bayesian Brain Diagnostics: Visualize Precision-Weighted Predictive Coding

This plots the evolution of learned uncertainties (σ) and weights (1/σ²) over training,
revealing what the model finds "difficult" vs "easy" at different developmental stages.

Biological Interpretation:
- σ (uncertainty): How noisy/unreliable the model thinks a signal is
- 1/σ² (precision/weight): How much the model "trusts" that signal (like dopamine)
- High σ: "This task is hard/noisy, lower its influence" (low dopamine)
- Low σ: "This task is reliable, increase its influence" (high dopamine)

Neuroscience Analogies:
1. Ventriloquist Effect: Vision (low σ) dominates hearing (high σ)
2. Dopamine Precision: High dopamine = low σ = "pay attention to this error"
3. Synaptic Scaling: ln(σ) term prevents runaway learning (sleep homeostasis)
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path


def load_balancer_history(reports_dir):
    """
    Load balancer statistics from homeostatic monitor CSV files.
    
    Returns:
        dict: {
            'iters': [...],
            'weights': {'prediction': [...], 'jacobian': [...], ...},
            'sigmas': {'prediction': [...], 'jacobian': [...], ...},
            'loss_components': {'prediction': [...], 'jacobian': [...], ...}
        }
    """
    # Look for balancer stats in homeostatic monitor CSV
    # The monitor logs: iter, weight_X, sigma_X for each task
    
    history = {
        'iters': [],
        'weights': {},
        'sigmas': {},
        'loss_components': {},
        'reflex_gate': []
    }
    
    # Try to find CSV files with balancer data
    csv_files = [
        os.path.join(reports_dir, 'homeostasis.csv'),
        os.path.join(reports_dir, 'metrics.csv')
    ]
    
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            continue
            
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    iter_num = int(row.get('iter', row.get('iteration', 0)))
                    
                    # Extract balancer weights and sigmas
                    # Format: weight_prediction, sigma_prediction, etc.
                    for key, val in row.items():
                        if key.startswith('weight_'):
                            task_name = key.replace('weight_', '')
                            if task_name not in history['weights']:
                                history['weights'][task_name] = []
                            history['weights'][task_name].append(float(val))
                            
                        elif key.startswith('sigma_'):
                            task_name = key.replace('sigma_', '')
                            if task_name not in history['sigmas']:
                                history['sigmas'][task_name] = []
                            history['sigmas'][task_name].append(float(val))
                            
                        elif key.startswith('loss_') and not key.startswith('loss_total'):
                            task_name = key.replace('loss_', '')
                            if task_name not in history['loss_components']:
                                history['loss_components'][task_name] = []
                            history['loss_components'][task_name].append(float(val))
                    
                    # Extract reflex gate
                    if 'reflex_gate' in row:
                        history['reflex_gate'].append(float(row['reflex_gate']))
                    
                    if iter_num not in history['iters']:
                        history['iters'].append(iter_num)
                        
                except (ValueError, KeyError):
                    continue
    
    return history


def plot_bayesian_brain(reports_dir, output_path=None):
    """
    Create comprehensive Bayesian Brain diagnostic visualization.
    
    Shows:
    1. Learned Uncertainties (σ) over time - what the model finds difficult
    2. Learned Weights (1/σ²) over time - what the model pays attention to
    3. Raw Loss Components - the actual signal magnitudes
    4. Reflex Gate Schedule - developmental phases
    5. Precision-Weighted Integration - combined view
    
    This reveals the "Dopamine Dynamics" of your model.
    """
    history = load_balancer_history(reports_dir)
    
    if len(history['iters']) == 0:
        print("⚠️  No balancer data found in reports directory")
        return
    
    # Convert to numpy for easier plotting
    iters = np.array(history['iters'])
    
    # Setup figure with 6 subplots
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Color scheme for tasks (neuroscience-inspired)
    task_colors = {
        'prediction': '#e74c3c',  # Red - Primary cortical function
        'jacobian': '#3498db',    # Blue - Stability/physics
        'novelty': '#f39c12',     # Orange - Curiosity/dopamine
        'memory': '#9b59b6'       # Purple - Hippocampal function
    }
    
    # ════════════════════════════════════════════════════════════════
    # PANEL 1: Learned Uncertainties (σ) - "What is Hard?"
    # ════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#f8f9fa')
    
    for task_name, sigmas in history['sigmas'].items():
        if len(sigmas) > 0:
            color = task_colors.get(task_name, '#95a5a6')
            ax1.plot(iters[:len(sigmas)], sigmas, 
                    color=color, linewidth=2.5, alpha=0.9,
                    label=f'{task_name.capitalize()}', marker='o', markersize=3)
    
    ax1.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Uncertainty σ (Noise Level)', fontsize=12, fontweight='bold')
    ax1.set_title('🧠 Learned Task Uncertainties\n"What Does the Model Find Difficult?"',
                 fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')
    
    # Add interpretation box
    ax1.text(0.02, 0.98, 
            '💡 High σ = Task is noisy/hard\n   Low σ = Task is reliable/easy',
            transform=ax1.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # ════════════════════════════════════════════════════════════════
    # PANEL 2: Learned Weights (1/σ²) - "Dopamine Signal"
    # ════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#f8f9fa')
    
    for task_name, weights in history['weights'].items():
        if len(weights) > 0:
            color = task_colors.get(task_name, '#95a5a6')
            ax2.plot(iters[:len(weights)], weights,
                    color=color, linewidth=2.5, alpha=0.9,
                    label=f'{task_name.capitalize()}', marker='s', markersize=3)
    
    ax2.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Weight 1/σ² (Precision/Attention)', fontsize=12, fontweight='bold')
    ax2.set_title('🎯 Learned Task Weights (Dopamine-Like Precision)\n"What Does the Model Pay Attention To?"',
                 fontsize=13, fontweight='bold', pad=15)
    ax2.legend(loc='best', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_yscale('log')
    
    # Add interpretation
    ax2.text(0.02, 0.98,
            '💡 High weight = High dopamine = "Learn this!"\n   Low weight = Low dopamine = "Ignore noise"',
            transform=ax2.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    # ════════════════════════════════════════════════════════════════
    # PANEL 3: Raw Loss Components - "The Signal"
    # ════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#f8f9fa')
    
    for task_name, losses in history['loss_components'].items():
        if len(losses) > 0:
            color = task_colors.get(task_name, '#95a5a6')
            ax3.plot(iters[:len(losses)], losses,
                    color=color, linewidth=2, alpha=0.8,
                    label=f'{task_name.capitalize()}', linestyle='-')
    
    ax3.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Raw Loss Magnitude', fontsize=12, fontweight='bold')
    ax3.set_title('📊 Unweighted Loss Components\n"The Raw Signal Magnitudes"',
                 fontsize=13, fontweight='bold', pad=15)
    ax3.legend(loc='best', fontsize=10, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_yscale('log')
    
    # Add interpretation
    ax3.text(0.02, 0.98,
            '⚠️ Without balancing, these would fight!\n   Different scales = gradient warfare',
            transform=ax3.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#ffe6e6', alpha=0.8))
    
    # ════════════════════════════════════════════════════════════════
    # PANEL 4: Reflex Gate Schedule - "Developmental Phases"
    # ════════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#f8f9fa')
    
    if len(history['reflex_gate']) > 0:
        gates = history['reflex_gate']
        ax4.plot(iters[:len(gates)], gates, 
                color='#2ecc71', linewidth=3, alpha=0.9, label='β(t) Gate')
        ax4.fill_between(iters[:len(gates)], 0, gates, 
                        color='#2ecc71', alpha=0.2)
        
        # Mark developmental phases
        phase1_end = 500
        phase2_end = 1000
        
        ax4.axvspan(0, phase1_end, alpha=0.15, color='red', label='Phase 1: Pure Cortex')
        ax4.axvspan(phase1_end, phase2_end, alpha=0.15, color='orange', label='Phase 2: Reconnection')
        if iters[-1] > phase2_end:
            ax4.axvspan(phase2_end, iters[-1], alpha=0.15, color='green', label='Phase 3: Integration')
    
    ax4.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Reflex Gate β(t)', fontsize=12, fontweight='bold')
    ax4.set_title('🧬 Homeostatic Reflex Gating\n"Forcing Cortical Development Before Spinal Automation"',
                 fontsize=13, fontweight='bold', pad=15)
    ax4.legend(loc='center right', fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_ylim(-0.05, 1.05)
    
    # Add biological interpretation
    ax4.text(0.02, 0.5,
            '🧠 β=0: Spinal cord severed\n      (Brain learns alone)\n\n'
            '🔗 β=0.5: Partial innervation\n      (Gradual reconnection)\n\n'
            '⚡ β=1.0: Full integration\n      (Brain + Spine working)',
            transform=ax4.transAxes, fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # ════════════════════════════════════════════════════════════════
    # PANEL 5: Weight × Loss (Precision-Weighted Contributions)
    # ════════════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_facecolor('#f8f9fa')
    
    # Calculate weighted contributions: weight × loss
    for task_name in history['weights'].keys():
        if task_name in history['loss_components']:
            weights = np.array(history['weights'][task_name])
            losses = np.array(history['loss_components'][task_name])
            
            # Align lengths
            min_len = min(len(weights), len(losses))
            weighted = weights[:min_len] * losses[:min_len]
            
            color = task_colors.get(task_name, '#95a5a6')
            ax5.plot(iters[:min_len], weighted,
                    color=color, linewidth=2.5, alpha=0.9,
                    label=f'{task_name.capitalize()}', marker='d', markersize=3)
    
    ax5.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Weighted Loss (1/σ² × L)', fontsize=12, fontweight='bold')
    ax5.set_title('⚖️ Precision-Weighted Loss Components\n"Actual Gradient Contributions (Homeostatic Equilibrium)"',
                 fontsize=13, fontweight='bold', pad=15)
    ax5.legend(loc='best', fontsize=10, framealpha=0.9)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.set_yscale('log')
    
    # Add interpretation
    ax5.text(0.02, 0.98,
            '✨ The balancer normalizes these!\n   All tasks speak same gradient language',
            transform=ax5.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#e8f8f5', alpha=0.8))
    
    # ════════════════════════════════════════════════════════════════
    # PANEL 6: Task Dominance Ratio (Relative Attention)
    # ════════════════════════════════════════════════════════════════
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_facecolor('#f8f9fa')
    
    # Calculate relative weights (percentage of total attention)
    if len(history['weights']) > 1:
        # Stack all weights into matrix
        task_names = list(history['weights'].keys())
        
        # Find minimum length across all weight series
        min_weight_len = min(len(history['weights'][name]) for name in task_names)
        
        # Truncate all series to minimum length
        weight_matrix = []
        for task_name in task_names:
            weight_matrix.append(history['weights'][task_name][:min_weight_len])
        
        weight_matrix = np.array(weight_matrix)  # (num_tasks, min_weight_len)
        
        # Normalize to percentages
        total_weight = weight_matrix.sum(axis=0)
        weight_percentages = (weight_matrix / total_weight[None, :]) * 100
        
        # Plot stacked area (ensure iters matches weight_percentages length)
        plot_iters = iters[:min_weight_len]
        ax6.stackplot(plot_iters, 
                     *weight_percentages,
                     labels=[name.capitalize() for name in task_names],
                     colors=[task_colors.get(name, '#95a5a6') for name in task_names],
                     alpha=0.8)
    
    ax6.set_xlabel('Training Iteration', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Attention Share (%)', fontsize=12, fontweight='bold')
    ax6.set_title('🎭 Task Dominance Evolution\n"How Does the Model Allocate Attention?"',
                 fontsize=13, fontweight='bold', pad=15)
    ax6.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax6.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax6.set_ylim(0, 100)
    
    # Add interpretation
    ax6.text(0.02, 0.5,
            '🧠 Like multisensory integration:\n'
            '   • Vision usually dominates\n'
            '   • But in dark, hearing wins\n'
            '   • Brain adapts automatically',
            transform=ax6.transAxes, fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
    
    # ════════════════════════════════════════════════════════════════
    # Super Title
    # ════════════════════════════════════════════════════════════════
    fig.suptitle('🧠 BAYESIAN BRAIN DIAGNOSTICS: Precision-Weighted Predictive Coding\n'
                 'Visualizing Learned Task Uncertainties and Dopamine-Like Precision Signals',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    if output_path is None:
        output_path = os.path.join(reports_dir, 'bayesian_brain_diagnostics.png')
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Bayesian Brain diagnostics saved: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("🧠 BAYESIAN BRAIN SUMMARY")
    print("="*80)
    
    if len(history['sigmas']) > 0:
        print("\n📊 Final Learned Uncertainties (σ):")
        for task_name, sigmas in history['sigmas'].items():
            if len(sigmas) > 0:
                final_sigma = sigmas[-1]
                print(f"   {task_name.capitalize():<15} σ = {final_sigma:.4f}")
        
        print("\n🎯 Final Learned Weights (1/σ²):")
        for task_name, weights in history['weights'].items():
            if len(weights) > 0:
                final_weight = weights[-1]
                print(f"   {task_name.capitalize():<15} w = {final_weight:.4f}")
        
        print("\n💡 Interpretation:")
        print("   • Higher σ = Model thinks task is noisy/difficult")
        print("   • Lower σ = Model thinks task is reliable/easy")
        print("   • Higher w (1/σ²) = More attention/dopamine allocated")
        print("   • Lower w = Less gradient flow from this task")
        
        print("\n🧬 Neuroscience Analogy:")
        print("   This is exactly how your brain combines vision + hearing:")
        print("   • Vision usually has low σ (precise) → high weight")
        print("   • Hearing usually has high σ (noisy location) → low weight")
        print("   • In the dark, brain flips the weights automatically!")
        
    print("="*80 + "\n")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        reports_dir = sys.argv[1]
    else:
        # Default: look in latest run directory
        import glob
        run_dirs = glob.glob('out-*')
        if run_dirs:
            latest_run = max(run_dirs, key=os.path.getmtime)
            reports_dir = os.path.join(latest_run, 'reports')
        else:
            print("❌ No run directories found. Usage: python plot_bayesian_brain.py <reports_dir>")
            sys.exit(1)
    
    print(f"📊 Analyzing Bayesian Brain from: {reports_dir}")
    plot_bayesian_brain(reports_dir)
