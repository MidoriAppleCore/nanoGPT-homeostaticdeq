"""
Homeostatic Monitoring System for DEQ Training

This module tracks and visualizes all physiological metrics of the
Hamiltonian DEQ system during training:

- Loss curves (train/val)
- Chaos Score (Lyapunov-guided LR stress signal)
- DEQ Iterations (computational effort)
- Learning Rate (chaos-aware adaptation)
- Phase Space Energy (Hamiltonian conservation)
- Mass Matrix Statistics (semantic physics)
- Temperature Evolution (if learnable)

All metrics are logged to CSV and visualized in the reports/ folder.
"""

import os
import csv
import time
from collections import defaultdict
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available - visualization disabled")


class HomeostaticMonitor:
    """
    Real-time monitoring and visualization of homeostatic DEQ training.
    """
    
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.reports_dir = os.path.join(out_dir, 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # CSV files for each metric
        self.csv_files = {
            'loss': os.path.join(self.reports_dir, 'loss.csv'),
            'chaos': os.path.join(self.reports_dir, 'chaos.csv'),
            'deq': os.path.join(self.reports_dir, 'deq_stats.csv'),
            'lr': os.path.join(self.reports_dir, 'learning_rate.csv'),
            'energy': os.path.join(self.reports_dir, 'energy.csv'),
            'mass': os.path.join(self.reports_dir, 'mass_stats.csv'),
            'memory': os.path.join(self.reports_dir, 'memory_stats.csv'),
            'memory_quality': os.path.join(self.reports_dir, 'memory_quality.csv'),  # NEW: Contrastive loss
            'loss_components': os.path.join(self.reports_dir, 'loss_components.csv'),
            'gamma': os.path.join(self.reports_dir, 'gamma_friction.csv'),
            'reflex_gate': os.path.join(self.reports_dir, 'reflex_gate.csv'),
            'balancer': os.path.join(self.reports_dir, 'homeostasis.csv'),  # Bayesian balancer stats
        }
        
        # Initialize CSV files with headers
        self._init_csv('loss', ['iter', 'train_loss', 'val_loss', 'timestamp'])
        self._init_csv('chaos', ['iter', 'chaos_score', 'stress_iters', 'stress_residual', 'throttle', 'timestamp'])
        self._init_csv('deq', ['iter', 'num_iters', 'final_residual', 'time_ms', 'timestamp'])
        self._init_csv('lr', ['iter', 'learning_rate', 'base_lr', 'timestamp'])
        self._init_csv('energy', ['iter', 'energy', 'timestamp'])
        self._init_csv('mass', ['iter', 'heavy_mean', 'heavy_std', 'light_mean', 'light_std', 'timestamp'])
        self._init_csv('memory', ['iter', 'num_working', 'num_buffer', 'num_longterm', 'reconsolidations', 'timestamp'])
        self._init_csv('memory_quality', ['iter', 'contrastive_loss', 'avg_similarity', 'retrieval_diversity', 'timestamp'])
        self._init_csv('loss_components', ['iter', 'ce_jacobian', 'pauli', 'efficiency', 'total', 'timestamp'])
        self._init_csv('gamma', ['iter', 'gamma', 'chaos_input', 'timestamp'])
        self._init_csv('reflex_gate', ['iter', 'beta', 'phase', 'timestamp'])
        
        # Dynamic header for balancer (we don't know task names yet)
        # Will be initialized on first log_balancer call
        self.balancer_initialized = False
        
        # In-memory buffer for plotting
        self.history = defaultdict(list)
    
    def _init_csv(self, key, header):
        """Initialize CSV file with header if it doesn't exist"""
        if not os.path.exists(self.csv_files[key]):
            with open(self.csv_files[key], 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
    
    def log_loss(self, iter_num, train_loss, val_loss=None):
        """Log training and validation loss"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.csv_files['loss'], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iter_num, f"{train_loss:.6f}", 
                           f"{val_loss:.6f}" if val_loss is not None else '', 
                           timestamp])
        
        self.history['iter'].append(iter_num)
        self.history['train_loss'].append(train_loss)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
    
    def log_chaos(self, iter_num, chaos_score, stress_iters, stress_residual, throttle):
        """Log chaos metrics (Lyapunov-guided LR signals)"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.csv_files['chaos'], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iter_num, f"{chaos_score:.6f}", 
                           f"{stress_iters:.6f}", f"{stress_residual:.6f}",
                           f"{throttle:.6f}", timestamp])
        
        self.history['chaos_score'].append(chaos_score)
        self.history['stress_iters'].append(stress_iters)
        self.history['stress_residual'].append(stress_residual)
    
    def log_deq(self, iter_num, num_iters, final_residual, time_ms):
        """Log DEQ solver statistics"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.csv_files['deq'], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iter_num, num_iters, f"{final_residual:.6e}", 
                           f"{time_ms:.3f}", timestamp])
        
        self.history['deq_iters'].append(num_iters)
        self.history['deq_residual'].append(final_residual)
        self.history['time_ms'].append(time_ms)
    
    def log_lr(self, iter_num, lr, base_lr):
        """Log learning rate (chaos-aware)"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.csv_files['lr'], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iter_num, f"{lr:.6e}", f"{base_lr:.6e}", timestamp])
        
        self.history['lr'].append(lr)
        self.history['base_lr'].append(base_lr)
    
    def log_energy(self, iter_num, energy):
        """Log Hamiltonian energy (should be conserved)"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.csv_files['energy'], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iter_num, f"{energy:.6f}", timestamp])
        
        self.history['energy'].append(energy)
    
    def log_mass_stats(self, iter_num, heavy_vals, light_vals):
        """Log mass matrix statistics"""
        heavy_mean = np.mean(heavy_vals)
        heavy_std = np.std(heavy_vals)
        light_mean = np.mean(light_vals)
        light_std = np.std(light_vals)
        
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.csv_files['mass'], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iter_num, f"{heavy_mean:.6f}", f"{heavy_std:.6f}",
                           f"{light_mean:.6f}", f"{light_std:.6f}", timestamp])
        
        self.history['heavy_mean'].append(heavy_mean)
        self.history['light_mean'].append(light_mean)
    
    def log_memory_stats(self, iter_num, num_working, num_buffer, num_longterm, reconsolidations=0):
        """Log memory system statistics"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.csv_files['memory'], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iter_num, num_working, num_buffer, num_longterm, reconsolidations, timestamp])
        
        self.history['num_working'].append(num_working)
        self.history['num_buffer'].append(num_buffer)
        self.history['num_longterm'].append(num_longterm)
    
    def log_memory_quality(self, iter_num, contrastive_loss, avg_similarity=0.0, retrieval_diversity=0.0):
        """
        Log memory retrieval quality metrics.
        
        Args:
            iter_num: Current iteration
            contrastive_loss: InfoNCE loss (are retrieved memories relevant?)
            avg_similarity: Average cosine similarity of query to retrieved
            retrieval_diversity: Diversity of retrieved neighbors (entropy)
        
        Interpretation:
            - contrastive_loss: Low = good retrieval, High = poor retrieval
            - avg_similarity: How close query is to retrieved (0-1)
            - retrieval_diversity: Are we retrieving diverse or redundant memories?
        """
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.csv_files['memory_quality'], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iter_num, f"{contrastive_loss:.6f}", 
                           f"{avg_similarity:.6f}", f"{retrieval_diversity:.6f}", timestamp])
        
        if 'memory_contrastive_loss' not in self.history:
            self.history['memory_contrastive_loss'] = []
            self.history['memory_avg_similarity'] = []
            self.history['memory_retrieval_diversity'] = []
        
        self.history['memory_contrastive_loss'].append(contrastive_loss)
        self.history['memory_avg_similarity'].append(avg_similarity)
        self.history['memory_retrieval_diversity'].append(retrieval_diversity)
    
    def log_loss_components(self, iter_num, ce_jacobian, pauli, efficiency, total):
        """Log breakdown of loss components"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.csv_files['loss_components'], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iter_num, f"{ce_jacobian:.6f}", f"{pauli:.6f}", 
                           f"{efficiency:.6f}", f"{total:.6f}", timestamp])
        
        self.history['loss_ce_jac'].append(ce_jacobian)
        self.history['loss_pauli'].append(pauli)
        self.history['loss_total'].append(total)
    
    def log_gamma(self, iter_num, gamma, chaos_input):
        """Log gamma friction parameter"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.csv_files['gamma'], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iter_num, f"{gamma:.6f}", f"{chaos_input:.6f}", timestamp])
        
        self.history['gamma'].append(gamma)
        self.history['gamma_chaos'].append(chaos_input)
    
    def log_reflex_gate(self, iter_num, beta):
        """Log homeostatic reflex gate β(t)"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Determine developmental phase
        if beta < 0.1:
            phase = "CORTEX_ONLY"
        elif beta < 0.99:
            phase = "RECONNECTING"
        else:
            phase = "FULL_INTEGRATION"
        
        with open(self.csv_files['reflex_gate'], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iter_num, f"{beta:.6f}", phase, timestamp])
        
        self.history['reflex_gate'].append(beta)
        self.history['gate_phase'].append(phase)
    
    def log_balancer(self, iter_num, balance_stats, loss_components_raw):
        """
        Log Bayesian Balancer statistics (the Dopamine Dynamics).
        
        Args:
            iter_num: Training iteration
            balance_stats: Dict with keys like 'weight_prediction', 'sigma_prediction', etc.
            loss_components_raw: Dict with raw loss values before balancing
        """
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Initialize CSV header on first call (now we know the task names)
        if not self.balancer_initialized:
            # Extract task names from balance_stats keys
            # Format: weight_<task>, sigma_<task>
            task_names = []
            for key in balance_stats.keys():
                if key.startswith('weight_'):
                    task_names.append(key.replace('weight_', ''))
            
            # Build header: iter, weight_X, sigma_X, loss_X for each task, timestamp
            header = ['iter']
            for task in task_names:
                header.extend([f'weight_{task}', f'sigma_{task}', f'loss_{task}'])
            header.append('timestamp')
            
            self._init_csv('balancer', header)
            self.balancer_initialized = True
            self.balancer_task_names = task_names
        
        # Build row
        row = [iter_num]
        for task in self.balancer_task_names:
            weight = balance_stats.get(f'weight_{task}', 0.0)
            sigma = balance_stats.get(f'sigma_{task}', 1.0)
            raw_loss = loss_components_raw.get(task, 0.0)
            row.extend([f"{weight:.6f}", f"{sigma:.6f}", f"{raw_loss:.6f}"])
        row.append(timestamp)
        
        # Write to CSV
        with open(self.csv_files['balancer'], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # Update history for in-memory access
        for task in self.balancer_task_names:
            weight_key = f'weight_{task}'
            sigma_key = f'sigma_{task}'
            loss_key = f'loss_{task}'
            
            if weight_key not in self.history:
                self.history[weight_key] = []
            if sigma_key not in self.history:
                self.history[sigma_key] = []
            if loss_key not in self.history:
                self.history[loss_key] = []
            
            self.history[weight_key].append(balance_stats.get(f'weight_{task}', 0.0))
            self.history[sigma_key].append(balance_stats.get(f'sigma_{task}', 1.0))
            self.history[loss_key].append(loss_components_raw.get(task, 0.0))
    
    
    def plot_homeostasis(self, iter_num):
        """Generate comprehensive homeostatic dashboard with ALL variables"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        # Use CSV files for plotting (source of truth)
        try:
            import pandas as pd
            loss_df = pd.read_csv(self.csv_files['loss'])
            chaos_df = pd.read_csv(self.csv_files['chaos'])
            deq_df = pd.read_csv(self.csv_files['deq'])
            lr_df = pd.read_csv(self.csv_files['lr'])
            
            # Optional: Load new metrics if available
            memory_df = None
            components_df = None
            gamma_df = None
            
            if os.path.exists(self.csv_files['memory']):
                memory_df = pd.read_csv(self.csv_files['memory'])
            if os.path.exists(self.csv_files['loss_components']):
                components_df = pd.read_csv(self.csv_files['loss_components'])
            if os.path.exists(self.csv_files['gamma']):
                gamma_df = pd.read_csv(self.csv_files['gamma'])
            
            # CRITICAL FIX: Sort by iteration to prevent line jumping
            loss_df = loss_df.sort_values('iter').drop_duplicates('iter', keep='last')
            chaos_df = chaos_df.sort_values('iter').drop_duplicates('iter', keep='last')
            deq_df = deq_df.sort_values('iter').drop_duplicates('iter', keep='last')
            lr_df = lr_df.sort_values('iter').drop_duplicates('iter', keep='last')
            if memory_df is not None:
                memory_df = memory_df.sort_values('iter').drop_duplicates('iter', keep='last')
            if components_df is not None:
                components_df = components_df.sort_values('iter').drop_duplicates('iter', keep='last')
            if gamma_df is not None:
                gamma_df = gamma_df.sort_values('iter').drop_duplicates('iter', keep='last')
        except Exception as e:
            print(f"  ⚠️  Could not load CSV data for plotting: {e}")
            return
        
        if len(loss_df) < 2:
            return
        
        # Set professional style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create MEGA DASHBOARD with 4x3 grid = 12 panels
        fig = plt.figure(figsize=(20, 16), facecolor='white')
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'🧠 Homeostatic DEQ - Complete System Monitor @ Iteration {iter_num}', 
                     fontsize=18, fontweight='bold', color='black')
        
        iters = loss_df['iter'].values
        
        # Color scheme - high contrast, colorblind-friendly
        COLORS = {
            'train': '#1f77b4',  # Blue
            'val': '#ff7f0e',    # Orange
            'chaos': '#d62728',  # Red
            'deq': '#2ca02c',    # Green
            'lr': '#9467bd',     # Purple
            'time': '#e377c2',   # Pink
            'heavy': '#8c564b',  # Brown
            'light': '#17becf',  # Cyan
            'memory': '#bcbd22', # Olive
            'ce': '#e74c3c',     # Bright red
            'pauli': '#f39c12',  # Bright orange
        }
        
        # ============ ROW 1: LOSS METRICS ============
        
        # 1. Total Loss
        ax = fig.add_subplot(gs[0, 0])
        ax.set_facecolor('white')
        ax.plot(iters, loss_df['train_loss'], color=COLORS['train'], 
                label='Train Loss', linewidth=2.5, alpha=0.9)
        if 'val_loss' in loss_df.columns and loss_df['val_loss'].notna().any():
            val_data = loss_df[loss_df['val_loss'].notna()]
            ax.plot(val_data['iter'], val_data['val_loss'], color=COLORS['val'], 
                    label='Val Loss', linewidth=2.5, linestyle='--', alpha=0.9)
        ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax.set_ylabel('Total Loss', fontsize=11, fontweight='bold')
        ax.set_title('1. Total Loss (Lower = Better)', fontsize=12, fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        ax.set_yscale('log')
        
        # 2. Loss Components Breakdown
        ax = fig.add_subplot(gs[0, 1])
        ax.set_facecolor('white')
        if components_df is not None and len(components_df) > 0:
            ax.plot(components_df['iter'], components_df['ce_jacobian'], 
                   color=COLORS['ce'], label='CE + Jacobian', linewidth=2.5, alpha=0.9)
            ax.plot(components_df['iter'], components_df['pauli'], 
                   color=COLORS['pauli'], label='Pauli (×2.0)', linewidth=2.5, alpha=0.9)
            ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
            ax.set_ylabel('Loss Component', fontsize=11, fontweight='bold')
            ax.set_title('2. Loss Breakdown', fontsize=12, fontweight='bold')
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, 'Loss Components\n(Enabled after update)', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=11, color='gray')
        
        # 3. Loss Gradient (rate of change)
        ax = fig.add_subplot(gs[0, 2])
        ax.set_facecolor('white')
        if len(loss_df) >= 10:
            import pandas as pd
            loss_gradient = pd.Series(loss_df['train_loss']).diff().rolling(window=10, min_periods=1).mean()
            ax.plot(iters, loss_gradient, color=COLORS['train'], linewidth=2.5, alpha=0.9)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
            ax.fill_between(iters, 0, loss_gradient, where=(loss_gradient < 0), 
                           color='green', alpha=0.2, label='Improving')
            ax.fill_between(iters, 0, loss_gradient, where=(loss_gradient > 0), 
                           color='red', alpha=0.2, label='Worsening')
            ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
            ax.set_ylabel('Δ Loss / Δ Iter', fontsize=11, fontweight='bold')
            ax.set_title('3. Loss Gradient (10-iter avg)', fontsize=12, fontweight='bold')
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        
        # ============ ROW 2: CHAOS & HOMEOSTASIS ============
        
        # 4. Chaos Score
        ax = fig.add_subplot(gs[1, 0])
        ax.set_facecolor('white')
        if len(chaos_df) > 0:
            ax.plot(chaos_df['iter'], chaos_df['chaos_score'], color=COLORS['chaos'], 
                    label='Chaos Score', linewidth=2.5, alpha=0.9)
            ax.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, 
                      alpha=0.6, label='High Stress')
            ax.axhline(y=0.2, color='green', linestyle='--', linewidth=2, 
                      alpha=0.6, label='Low Stress')
            ax.fill_between(chaos_df['iter'], 0.8, 1.0, color='orange', alpha=0.1)
            ax.fill_between(chaos_df['iter'], 0, 0.2, color='green', alpha=0.1)
            ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
            ax.set_ylabel('Chaos Score', fontsize=11, fontweight='bold')
            ax.set_title('4. System Stress (0=Zen, 1=Panic)', fontsize=12, fontweight='bold')
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
            ax.set_ylim(-0.05, 1.1)
        
        # 5. Stress Components
        ax = fig.add_subplot(gs[1, 1])
        ax.set_facecolor('white')
        if len(chaos_df) > 0:
            ax.plot(chaos_df['iter'], chaos_df['stress_iters'], 
                   color=COLORS['deq'], label='Iteration Stress', linewidth=2.5, alpha=0.9)
            ax.plot(chaos_df['iter'], chaos_df['stress_residual'], 
                   color=COLORS['chaos'], label='Residual Stress', linewidth=2.5, alpha=0.9)
            ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
            ax.set_ylabel('Stress Component', fontsize=11, fontweight='bold')
            ax.set_title('5. Chaos Breakdown', fontsize=12, fontweight='bold')
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
            ax.set_ylim(-0.05, 1.1)
        
        # 6. LR Throttle
        ax = fig.add_subplot(gs[1, 2])
        ax.set_facecolor('white')
        if len(chaos_df) > 0 and 'throttle' in chaos_df.columns:
            ax.plot(chaos_df['iter'], chaos_df['throttle'], 
                   color=COLORS['lr'], linewidth=2.5, alpha=0.9)
            ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Full Speed')
            ax.axhline(y=0.1, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Max Brake')
            ax.fill_between(chaos_df['iter'], 0.1, chaos_df['throttle'], 
                           color=COLORS['lr'], alpha=0.2)
            ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
            ax.set_ylabel('LR Multiplier', fontsize=11, fontweight='bold')
            ax.set_title('6. Chaos Throttle (Brake)', fontsize=12, fontweight='bold')
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
            ax.set_ylim(0, 1.1)
        
        # ============ ROW 3: DEQ DYNAMICS ============
        
        # 7. DEQ Iterations
        ax = fig.add_subplot(gs[2, 0])
        ax.set_facecolor('white')
        if len(deq_df) > 0:
            ax.plot(deq_df['iter'], deq_df['num_iters'], color=COLORS['deq'], 
                    linewidth=2.5, alpha=0.9)
            ax.fill_between(deq_df['iter'], 0, deq_df['num_iters'], 
                           color=COLORS['deq'], alpha=0.2)
            ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
            ax.set_ylabel('DEQ Iterations', fontsize=11, fontweight='bold')
            ax.set_title('7. Solver Effort', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        
        # 8. Final Residual
        ax = fig.add_subplot(gs[2, 1])
        ax.set_facecolor('white')
        if len(deq_df) > 0 and 'final_residual' in deq_df.columns:
            ax.plot(deq_df['iter'], deq_df['final_residual'], color=COLORS['heavy'], 
                    linewidth=2.5, alpha=0.9)
            ax.axhline(y=1e-3, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='Target (1e-3)')
            ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='Good (1.0)')
            ax.axhline(y=10.0, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Stuck (10.0)')
            ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
            ax.set_ylabel('Final Residual', fontsize=11, fontweight='bold')
            ax.set_title('8. Convergence Quality', fontsize=12, fontweight='bold')
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
            ax.set_yscale('log')
        
        # 9. Iteration Time
        ax = fig.add_subplot(gs[2, 2])
        ax.set_facecolor('white')
        if len(deq_df) > 0 and 'time_ms' in deq_df.columns:
            ax.plot(deq_df['iter'], deq_df['time_ms'], color=COLORS['time'], 
                    linewidth=1.5, alpha=0.5, label='Per-iter')
            # Add running average
            if len(deq_df) >= 10:
                import pandas as pd
                rolling_mean = pd.Series(deq_df['time_ms']).rolling(window=10, min_periods=1).mean()
                ax.plot(deq_df['iter'], rolling_mean, color='black', 
                       linewidth=2.5, alpha=0.9, label='10-iter avg')
            ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
            ax.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
            ax.set_title('9. Runtime', fontsize=12, fontweight='bold')
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        
        # ============ ROW 4: MEMORY & LR ============
        
        # 10. Memory Growth
        ax = fig.add_subplot(gs[3, 0])
        ax.set_facecolor('white')
        if memory_df is not None and len(memory_df) > 0:
            ax.plot(memory_df['iter'], memory_df['num_longterm'], 
                   color=COLORS['memory'], label='Long-term', linewidth=2.5, marker='o', 
                   markersize=4, alpha=0.9)
            ax.plot(memory_df['iter'], memory_df['num_working'], 
                   color=COLORS['deq'], label='Working', linewidth=2.5, marker='s', 
                   markersize=4, alpha=0.9)
            ax.plot(memory_df['iter'], memory_df['num_buffer'], 
                   color=COLORS['pauli'], label='Buffer', linewidth=2.5, marker='^', 
                   markersize=4, alpha=0.9)
            ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
            ax.set_ylabel('Memory Count', fontsize=11, fontweight='bold')
            ax.set_title('10. Memory System', fontsize=12, fontweight='bold')
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        else:
            ax.text(0.5, 0.5, 'Memory Stats\n(Available after update)', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=11, color='gray')
        
        # 11. Learning Rate
        ax = fig.add_subplot(gs[3, 1])
        ax.set_facecolor('white')
        if len(lr_df) > 0:
            ax.plot(lr_df['iter'], lr_df['learning_rate'], color=COLORS['lr'], 
                    label='Chaos-Aware LR', linewidth=2.5, alpha=0.9)
            if 'base_lr' in lr_df.columns:
                ax.plot(lr_df['iter'], lr_df['base_lr'], color='gray', 
                       label='Base LR', linewidth=2, linestyle=':', alpha=0.7)
            ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
            ax.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
            ax.set_title('11. Adaptive LR', fontsize=12, fontweight='bold')
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
            ax.set_yscale('log')
        
        # 12. Gamma Friction (if available)
        ax = fig.add_subplot(gs[3, 2])
        ax.set_facecolor('white')
        if gamma_df is not None and len(gamma_df) > 0:
            ax.plot(gamma_df['iter'], gamma_df['gamma'], 
                   color=COLORS['heavy'], linewidth=2.5, alpha=0.9)
            ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
            ax.set_ylabel('γ (Damping)', fontsize=11, fontweight='bold')
            ax.set_title('12. Hamiltonian Friction', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        else:
            # Fallback: show reconsolidations if available
            if memory_df is not None and len(memory_df) > 0 and 'reconsolidations' in memory_df.columns:
                ax.bar(memory_df['iter'], memory_df['reconsolidations'], 
                      color=COLORS['memory'], alpha=0.6, width=max(1, len(memory_df) / 50))
                ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
                ax.set_ylabel('Count', fontsize=11, fontweight='bold')
                ax.set_title('12. Memory Reconsolidations', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
            else:
                ax.text(0.5, 0.5, 'Gamma Friction\n(Hamiltonian mode)', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=11, color='gray')
        
        plt.tight_layout()
        
        # Save to reports
        output_path = os.path.join(self.reports_dir, f'homeostasis_iter_{iter_num:06d}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        # Also create an "all-in-one" combined metrics plot
        self._plot_combined_metrics(iter_num, loss_df, chaos_df, deq_df, lr_df)
        
        print(f"  📊 Homeostatic dashboard (12 panels) saved: {output_path}")
    
    def _plot_combined_metrics(self, iter_num, loss_df, chaos_df, deq_df, lr_df):
        """Create a single plot with all key metrics on same axes (normalized)"""
        if len(loss_df) < 2:
            return
        
        import pandas as pd
        
        # Create combined plot with all metrics normalized to [0, 1]
        fig, ax = plt.subplots(1, 1, figsize=(14, 7), facecolor='white')
        ax.set_facecolor('white')
        
        iters = loss_df['iter'].values
        
        # Normalize each metric to [0, 1] for comparison
        def normalize(series):
            s = pd.Series(series)
            if s.max() == s.min():
                return s * 0
            return (s - s.min()) / (s.max() - s.min())
        
        # High contrast colors
        COLORS = {
            'loss': '#e74c3c',      # Red
            'chaos': '#f39c12',     # Orange
            'deq': '#27ae60',       # Green
            'lr': '#3498db',        # Blue
            'residual': '#9b59b6',  # Purple
        }
        
        # Plot normalized metrics
        ax.plot(iters, normalize(loss_df['train_loss']), 
               color=COLORS['loss'], linewidth=2.5, label='Loss', alpha=0.85)
        
        if len(chaos_df) > 0 and len(chaos_df) == len(iters):
            ax.plot(chaos_df['iter'], normalize(chaos_df['chaos_score']), 
                   color=COLORS['chaos'], linewidth=2.5, label='Chaos Score', alpha=0.85)
        
        if len(deq_df) > 0 and len(deq_df) == len(iters):
            ax.plot(deq_df['iter'], normalize(deq_df['num_iters']), 
                   color=COLORS['deq'], linewidth=2.5, label='DEQ Iters', alpha=0.85)
        
        if len(lr_df) > 0 and len(lr_df) == len(iters):
            ax.plot(lr_df['iter'], normalize(lr_df['learning_rate']), 
                   color=COLORS['lr'], linewidth=2.5, label='Learning Rate', alpha=0.85)
        
        if len(deq_df) > 0 and len(deq_df) == len(iters) and 'final_residual' in deq_df.columns:
            ax.plot(deq_df['iter'], normalize(deq_df['final_residual']), 
                   color=COLORS['residual'], linewidth=2.5, label='Residual', 
                   alpha=0.85, linestyle='--')
        
        ax.set_xlabel('Iteration', fontsize=13, fontweight='bold')
        ax.set_ylabel('Normalized Value [0, 1]', fontsize=13, fontweight='bold')
        ax.set_title(f'All Metrics Combined (Normalized) @ Iteration {iter_num}', 
                    fontsize=14, fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        ax.set_ylim(-0.05, 1.05)
        
        # Add reference lines
        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.4)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        ax.axhline(y=1, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.reports_dir, f'combined_metrics_iter_{iter_num:06d}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"  📈 Combined metrics plot saved: {output_path}")
    
    def create_reflex_animation(self, iter_num, reflex_activations, token_ids, decode_fn):
        """
        Create GIF animation of reflex network activations over sequence.
        
        Args:
            iter_num: Current iteration
            reflex_activations: List of activation tensors [seq_len, hidden_dim]
            token_ids: Token IDs for labeling [seq_len]
            decode_fn: Function to decode token IDs to strings
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        try:
            from PIL import Image
            import io
        except ImportError:
            print("  ⚠️  PIL not available - reflex GIF skipped")
            return
        
        frames = []
        seq_len = min(len(reflex_activations), 20)  # Limit to 20 tokens for manageable GIF
        
        for t in range(seq_len):
            fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor='white')
            
            # Get activation at this timestep
            activation = reflex_activations[t]  # Shape: [hidden_dim]
            
            # Plot as heatmap
            ax.imshow(activation.reshape(1, -1), cmap='RdBu_r', aspect='auto', 
                     vmin=-3, vmax=3, interpolation='nearest')
            
            # Token label
            token_str = decode_fn([token_ids[t]])[0] if t < len(token_ids) else "?"
            ax.set_title(f'Reflex Network @ t={t} | Token: "{token_str}" | Iter {iter_num}',
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('Activation', fontsize=12)
            ax.set_xlabel(f'Hidden Dimension (0-{len(activation)-1})', fontsize=12)
            ax.set_yticks([])
            
            # Add colorbar
            cbar = plt.colorbar(ax.images[0], ax=ax, orientation='horizontal', 
                               pad=0.1, fraction=0.05)
            cbar.set_label('Activation Strength', fontsize=11)
            
            plt.tight_layout()
            
            # Save frame to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, facecolor='white')
            buf.seek(0)
            frames.append(Image.open(buf))
            plt.close()
        
        # Save as GIF
        gif_path = os.path.join(self.reports_dir, f'reflex_animation_iter_{iter_num:06d}.gif')
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], 
                      duration=200, loop=0, optimize=False)
        
        print(f"  🎬 Reflex animation GIF saved: {gif_path} ({len(frames)} frames)")
    
    def save_checkpoint_summary(self, iter_num, checkpoint_path):
        """Save a summary of the current state alongside checkpoint"""
        summary_path = checkpoint_path.replace('ckpt.pt', f'summary_{iter_num:06d}.txt')
        
        with open(summary_path, 'w') as f:
            f.write(f"="*70 + "\n")
            f.write(f"HOMEOSTATIC DEQ CHECKPOINT SUMMARY\n")
            f.write(f"Iteration: {iter_num}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"="*70 + "\n\n")
            
            # Latest metrics
            if len(self.history['iter']) > 0:
                f.write("Latest Metrics:\n")
                f.write(f"  Train Loss: {self.history['train_loss'][-1]:.6f}\n")
                if 'val_loss' in self.history and len(self.history['val_loss']) > 0:
                    f.write(f"  Val Loss: {self.history['val_loss'][-1]:.6f}\n")
                if 'chaos_score' in self.history and len(self.history['chaos_score']) > 0:
                    f.write(f"  Chaos Score: {self.history['chaos_score'][-1]:.4f}\n")
                if 'deq_iters' in self.history and len(self.history['deq_iters']) > 0:
                    f.write(f"  DEQ Iterations: {self.history['deq_iters'][-1]}\n")
                if 'lr' in self.history and len(self.history['lr']) > 0:
                    f.write(f"  Learning Rate: {self.history['lr'][-1]:.6e}\n")
                f.write("\n")
            
            # Statistics
            if len(self.history['train_loss']) > 10:
                recent_loss = self.history['train_loss'][-10:]
                f.write("Recent Performance (last 10 iters):\n")
                f.write(f"  Mean Loss: {np.mean(recent_loss):.6f}\n")
                f.write(f"  Loss Std: {np.std(recent_loss):.6f}\n")
                f.write(f"  Loss Trend: {np.polyfit(range(10), recent_loss, 1)[0]:.6f} /iter\n")
                f.write("\n")
            
            f.write(f"Checkpoint saved to: {checkpoint_path}\n")
            f.write(f"="*70 + "\n")
        
        print(f"  📄 Checkpoint summary: {summary_path}")
