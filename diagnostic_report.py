"""
Comprehensive Diagnostic Report for Homeostatic DEQ Training

This module generates detailed debugging information to help understand
what the model is learning, where it's stuck, and how to fix it.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    # Silent failure - optional visualization dependencies


class DiagnosticReporter:
    """Generate comprehensive diagnostic reports for debugging training"""
    
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.reports_dir = os.path.join(out_dir, 'reports', 'diagnostics')
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Track history
        self.token_history = []
        self.confidence_history = []
        self.diversity_history = []
        
        # Silent mode flag
        self._silent = False
    
    def _print(self, *args, **kwargs):
        """Conditional print - only prints if not in silent mode"""
        if not self._silent:
            self._print(*args, **kwargs)
    
    @torch.no_grad()
    def generate_full_report(self, model, data_loader, meta, iter_num, device='cuda', silent=False):
        """Generate comprehensive diagnostic report
        
        Args:
            silent: If True, suppress console output (data still saved to files)
        """
        
        self._silent = silent  # Set silent mode
        
        self._print("\n" + "="*80)
        self._print(f"🔬 DIAGNOSTIC REPORT @ Iteration {iter_num}")
        self._print("="*80)
        
        model.eval()
        
        # Get decode function
        if 'itos' in meta:
            decode = lambda l: ''.join([meta['itos'][i] for i in l])
        else:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            decode = lambda l: enc.decode(l)
        
        # 1. Token Distribution Analysis
        self._print("\n📊 1. TOKEN DISTRIBUTION ANALYSIS")
        token_dist = self._analyze_token_distribution(model, data_loader, meta, device)
        
        # 2. Semantic Mass Analysis
        self._print("\n⚛️  2. SEMANTIC MASS ANALYSIS")
        mass_analysis = self._analyze_semantic_mass(model, meta)
        
        # 3. Prediction Confidence
        self._print("\n🎯 3. PREDICTION CONFIDENCE")
        confidence_stats = self._analyze_prediction_confidence(model, data_loader, device)
        
        # 4. Memory System Health
        self._print("\n🧠 4. MEMORY SYSTEM HEALTH")
        memory_stats = self._analyze_memory_system(model)
        
        # 5. DEQ Convergence Quality
        self._print("\n🔄 5. DEQ CONVERGENCE QUALITY")
        deq_stats = self._analyze_deq_convergence(model, data_loader, device)
        
        # 6. Generate Visualizations
        if VISUALIZATION_AVAILABLE:
            self._print("\n📈 6. GENERATING VISUALIZATIONS...")
            self._create_diagnostic_visualizations(
                iter_num, token_dist, mass_analysis, 
                confidence_stats, memory_stats, decode
            )
        
        self._print("\n" + "="*80 + "\n")
        
        model.train()
        
        return {
            'token_dist': token_dist,
            'mass_analysis': mass_analysis,
            'confidence_stats': confidence_stats,
            'memory_stats': memory_stats,
            'deq_stats': deq_stats
        }
    
    def _analyze_token_distribution(self, model, data_loader, meta, device):
        """Analyze what tokens the model is predicting"""
        
        # Sample predictions from model
        predicted_tokens = []
        target_tokens = []
        
        for i in range(10):  # Sample 10 batches
            X, Y = data_loader('train')
            logits, _ = model(X, Y)
            
            # Get predictions
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            predicted_tokens.extend(preds.view(-1).cpu().tolist())
            target_tokens.extend(Y.view(-1).cpu().tolist())
        
        # Count frequency
        pred_counter = Counter(predicted_tokens)
        target_counter = Counter(target_tokens)
        
        # Top predictions
        top_predicted = pred_counter.most_common(20)
        top_targets = target_counter.most_common(20)
        
        # Decode and display
        if 'itos' in meta:
            itos = meta['itos']
            self._print(f"  Top Predicted Tokens:")
            for token_id, count in top_predicted[:10]:
                token = itos.get(token_id, f"<{token_id}>")
                self._print(f"    {repr(token):20s}: {count:5d} ({100*count/len(predicted_tokens):.2f}%)")
        else:
            self._print(f"  Top Predicted Token IDs:")
            for token_id, count in top_predicted[:10]:
                self._print(f"    {token_id:6d}: {count:5d} ({100*count/len(predicted_tokens):.2f}%)")
        
        # Calculate diversity
        unique_predicted = len(set(predicted_tokens))
        unique_targets = len(set(target_tokens))
        total_vocab = len(pred_counter)
        
        self._print(f"\n  Diversity Metrics:")
        self._print(f"    Unique tokens predicted: {unique_predicted}")
        self._print(f"    Unique tokens in targets: {unique_targets}")
        self._print(f"    Coverage: {100*unique_predicted/max(1, unique_targets):.2f}%")
        self._print(f"    Entropy: {self._calculate_entropy(pred_counter):.3f} bits")
        
        return {
            'predicted': pred_counter,
            'targets': target_counter,
            'diversity': unique_predicted,
            'entropy': self._calculate_entropy(pred_counter)
        }
    
    def _analyze_semantic_mass(self, model, meta):
        """Analyze semantic mass matrix (heavy vs light concepts)"""
        
        if not hasattr(model, 'inspect_concept_mass'):
            self._print("  ⚠️  Model doesn't have semantic mass - skipping")
            return {}
        
        mass_data = model.inspect_concept_mass(top_k=20)
        
        if 'itos' in meta:
            itos = meta['itos']
            
            self._print(f"  Heavy Concepts (High Inertia - Content Words):")
            for idx, val in zip(mass_data['heavy_ids'][:10], mass_data['heavy_vals'][:10]):
                token = itos.get(idx, f"<{idx}>")
                self._print(f"    {repr(token):20s}: mass={val:.4f}")
            
            self._print(f"\n  Light Concepts (Agile - Function Words):")
            for idx, val in zip(mass_data['light_ids'][:10], mass_data['light_vals'][:10]):
                token = itos.get(idx, f"<{idx}>")
                self._print(f"    {repr(token):20s}: mass={val:.4f}")
        
        # Statistics
        heavy_mean = np.mean(mass_data['heavy_vals'])
        light_mean = np.mean(mass_data['light_vals'])
        separation = heavy_mean / (light_mean + 1e-10)
        
        self._print(f"\n  Mass Statistics:")
        self._print(f"    Heavy mean: {heavy_mean:.4f}")
        self._print(f"    Light mean: {light_mean:.4f}")
        self._print(f"    Separation ratio: {separation:.2f}x")
        
        return mass_data
    
    def _analyze_prediction_confidence(self, model, data_loader, device):
        """Analyze how confident the model is in its predictions"""
        
        confidences = []
        entropies = []
        
        for i in range(5):  # Sample 5 batches
            X, Y = data_loader('train')
            logits, _ = model(X, Y)
            probs = F.softmax(logits, dim=-1)
            
            # Max probability (confidence)
            max_probs, _ = torch.max(probs, dim=-1)
            confidences.extend(max_probs.view(-1).cpu().tolist())
            
            # Entropy
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            entropies.extend(entropy.view(-1).cpu().tolist())
        
        # Statistics
        conf_mean = np.mean(confidences)
        conf_std = np.std(confidences)
        ent_mean = np.mean(entropies)
        
        self._print(f"  Confidence (max prob):")
        self._print(f"    Mean: {conf_mean:.4f} ± {conf_std:.4f}")
        self._print(f"    Min:  {np.min(confidences):.4f}")
        self._print(f"    Max:  {np.max(confidences):.4f}")
        
        self._print(f"\n  Entropy (bits):")
        self._print(f"    Mean: {ent_mean:.4f}")
        self._print(f"    Std:  {np.std(entropies):.4f}")
        
        # Diagnosis
        if conf_mean < 0.1:
            self._print(f"\n  ⚠️  WARNING: Very low confidence - model is guessing randomly")
        elif conf_mean > 0.9:
            self._print(f"\n  ⚠️  WARNING: Overconfident - may be stuck in mode collapse")
        else:
            self._print(f"\n  ✓ Confidence looks healthy")
        
        return {
            'confidence_mean': conf_mean,
            'confidence_std': conf_std,
            'entropy_mean': ent_mean
        }
    
    def _analyze_memory_system(self, model):
        """Analyze memory system health"""
        
        if not hasattr(model, 'reflex') or not hasattr(model.reflex, 'get_memory_stats'):
            self._print("  ⚠️  No memory system found - skipping")
            return {}
        
        stats = model.reflex.get_memory_stats()
        
        self._print(f"  Memory Capacity:")
        self._print(f"    Working:   {stats.get('num_working', 0):4d} / {stats.get('working_capacity', 0):4d}")
        self._print(f"    Buffer:    {stats.get('num_buffer', 0):4d}")
        self._print(f"    Long-term: {stats.get('num_longterm', 0):4d} / {stats.get('longterm_capacity', 0):4d}")
        
        if 'access_counts' in stats:
            access_counts = stats['access_counts']
            if len(access_counts) > 0:
                self._print(f"\n  Memory Access Pattern:")
                self._print(f"    Total accesses: {sum(access_counts)}")
                self._print(f"    Mean access:    {np.mean(access_counts):.2f}")
                self._print(f"    Std access:     {np.std(access_counts):.2f}")
                self._print(f"    Max access:     {np.max(access_counts):.0f}")
                
                # Hot vs cold memories
                hot_threshold = np.percentile(access_counts, 80)
                hot_count = sum(1 for c in access_counts if c > hot_threshold)
                self._print(f"    Hot memories (>80th %ile): {hot_count} ({100*hot_count/len(access_counts):.1f}%)")
        
        # Diagnosis
        num_lt = stats.get('num_longterm', 0)
        if num_lt < 50:
            self._print(f"\n  ⚠️  WARNING: Very few long-term memories - graph too sparse")
        elif num_lt > 1500:
            self._print(f"\n  ⚠️  WARNING: Memory nearing capacity - may need pruning")
        else:
            self._print(f"\n  ✓ Memory count looks reasonable")
        
        return stats
    
    def _analyze_deq_convergence(self, model, data_loader, device):
        """Analyze DEQ convergence quality"""
        
        if not hasattr(model, 'deq'):
            self._print("  ⚠️  No DEQ found - skipping")
            return {}
        
        residuals = []
        num_iters = []
        
        for i in range(3):  # Sample 3 batches
            X, Y = data_loader('train')
            _, _, metrics = model(X, Y, return_metrics=True)
            
            residuals.append(metrics.get('final_residual', 0))
            num_iters.append(metrics.get('num_iters', 0))
        
        res_mean = np.mean(residuals)
        iter_mean = np.mean(num_iters)
        
        self._print(f"  Convergence Metrics:")
        self._print(f"    Mean residual: {res_mean:.2e}")
        self._print(f"    Mean iters:    {iter_mean:.1f}")
        
        # Diagnosis
        if res_mean > 10.0:
            self._print(f"\n  🚨 CRITICAL: Residual very high - DEQ stuck/diverging")
        elif res_mean > 1.0:
            self._print(f"\n  ⚠️  WARNING: Residual high - DEQ not converging well")
        elif res_mean < 0.01:
            self._print(f"\n  ✓ Excellent convergence")
        else:
            self._print(f"\n  ✓ Convergence acceptable")
        
        return {
            'residual_mean': res_mean,
            'iter_mean': iter_mean
        }
    
    def _calculate_entropy(self, counter):
        """Calculate Shannon entropy of distribution"""
        total = sum(counter.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    
    def _create_diagnostic_visualizations(self, iter_num, token_dist, mass_analysis, 
                                         confidence_stats, memory_stats, decode):
        """Create diagnostic visualizations"""
        
        fig = plt.figure(figsize=(18, 10), facecolor='white')
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # 1. Token Distribution Word Cloud
        ax = fig.add_subplot(gs[0, :2])
        ax.set_facecolor('white')
        if len(token_dist['predicted']) > 0:
            try:
                # Create word frequency dict
                word_freq = {}
                for token_id, count in token_dist['predicted'].most_common(100):
                    word = decode([token_id]) if callable(decode) else str(token_id)
                    word_freq[word] = count
                
                wordcloud = WordCloud(width=800, height=400, background_color='white',
                                     colormap='viridis').generate_from_frequencies(word_freq)
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Predicted Token Distribution (Word Cloud)', 
                           fontsize=14, fontweight='bold')
            except Exception as e:
                ax.text(0.5, 0.5, f'Word cloud unavailable\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
        
        # 2. Diversity Over Time
        ax = fig.add_subplot(gs[0, 2])
        ax.set_facecolor('white')
        self.diversity_history.append(token_dist['diversity'])
        if len(self.diversity_history) > 1:
            ax.plot(self.diversity_history, linewidth=2.5, color='#2ecc71')
            ax.set_xlabel('Report #', fontsize=11, fontweight='bold')
            ax.set_ylabel('Unique Tokens', fontsize=11, fontweight='bold')
            ax.set_title('Vocabulary Diversity', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Diversity Tracking\n(need >1 report)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
        
        # 3. Semantic Mass Separation
        ax = fig.add_subplot(gs[1, 0])
        ax.set_facecolor('white')
        if mass_analysis and 'heavy_vals' in mass_analysis:
            heavy = mass_analysis['heavy_vals'][:20]
            light = mass_analysis['light_vals'][:20]
            
            x = np.arange(len(heavy))
            ax.bar(x - 0.2, heavy, width=0.4, label='Heavy (Content)', color='#e74c3c', alpha=0.8)
            ax.bar(x + 0.2, light, width=0.4, label='Light (Function)', color='#3498db', alpha=0.8)
            ax.set_xlabel('Token Rank', fontsize=11, fontweight='bold')
            ax.set_ylabel('Mass Value', fontsize=11, fontweight='bold')
            ax.set_title('Semantic Mass Distribution', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Confidence Distribution
        ax = fig.add_subplot(gs[1, 1])
        ax.set_facecolor('white')
        self.confidence_history.append(confidence_stats['confidence_mean'])
        if len(self.confidence_history) > 0:
            ax.plot(self.confidence_history, linewidth=2.5, color='#9b59b6', marker='o', markersize=6)
            ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.6)
            ax.set_xlabel('Report #', fontsize=11, fontweight='bold')
            ax.set_ylabel('Mean Confidence', fontsize=11, fontweight='bold')
            ax.set_title('Prediction Confidence Over Time', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
        
        # 5. Memory System Status
        ax = fig.add_subplot(gs[1, 2])
        ax.set_facecolor('white')
        if memory_stats:
            categories = ['Working', 'Buffer', 'Long-term']
            values = [
                memory_stats.get('num_working', 0),
                memory_stats.get('num_buffer', 0),
                memory_stats.get('num_longterm', 0)
            ]
            colors = ['#2ecc71', '#f39c12', '#3498db']
            ax.bar(categories, values, color=colors, alpha=0.8)
            ax.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax.set_title('Memory System Status', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add capacity lines
            capacities = [
                memory_stats.get('working_capacity', 0),
                0,  # Buffer has no fixed capacity
                memory_stats.get('longterm_capacity', 0)
            ]
            for i, cap in enumerate(capacities):
                if cap > 0:
                    ax.axhline(y=cap, xmin=i/3, xmax=(i+1)/3, 
                             color='red', linestyle='--', linewidth=2, alpha=0.6)
        
        # 6. Health Dashboard (text summary)
        ax = fig.add_subplot(gs[2, :])
        ax.axis('off')
        
        # Generate health summary
        health_text = f"🏥 SYSTEM HEALTH SUMMARY (Iteration {iter_num})\n\n"
        
        # Token diversity
        diversity = token_dist['diversity']
        if diversity < 50:
            health_text += "🚨 VOCABULARY: Severe collapse (< 50 unique tokens)\n"
        elif diversity < 200:
            health_text += "⚠️  VOCABULARY: Limited diversity (< 200 unique tokens)\n"
        else:
            health_text += f"✓  VOCABULARY: Healthy diversity ({diversity} unique tokens)\n"
        
        # Confidence
        conf = confidence_stats['confidence_mean']
        if conf < 0.15:
            health_text += "🚨 CONFIDENCE: Random guessing (< 15%)\n"
        elif conf > 0.85:
            health_text += "⚠️  CONFIDENCE: Overconfident / mode collapse (> 85%)\n"
        else:
            health_text += f"✓  CONFIDENCE: Healthy uncertainty ({conf:.2%})\n"
        
        # Memory
        if memory_stats:
            num_lt = memory_stats.get('num_longterm', 0)
            if num_lt < 100:
                health_text += f"⚠️  MEMORY: Sparse graph ({num_lt} memories) - need more data\n"
            elif num_lt > 1500:
                health_text += f"⚠️  MEMORY: Near capacity ({num_lt} / {memory_stats.get('longterm_capacity', 2000)})\n"
            else:
                health_text += f"✓  MEMORY: Growing healthily ({num_lt} memories)\n"
        
        ax.text(0.05, 0.5, health_text, transform=ax.transAxes, 
               fontsize=12, fontfamily='monospace', verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Save figure
        output_path = os.path.join(self.reports_dir, f'diagnostics_iter_{iter_num:06d}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self._print(f"  ✅ Diagnostic visualization saved: {output_path}")
