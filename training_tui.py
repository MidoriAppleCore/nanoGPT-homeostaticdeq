"""
Beautiful TUI for training visualization using rich library.

Shows live passage display with highlighted training position,
carrier statistics, edge information, and training metrics.
"""

import torch
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from rich import box
from collections import deque


class TrainingTUI:
    """
    Live training visualization with passage display and stats.
    """
    
    def __init__(self, vocab_meta=None):
        self.console = Console()
        self.vocab_meta = vocab_meta
        self.itos = vocab_meta.get('itos', {}) if vocab_meta else {}
        
        # Training stats
        self.recent_losses = deque(maxlen=100)
        self.edges_trained = set()
        self.current_iteration = 0
        self.total_iterations = 0
        
        # Current training context
        self.current_passage = None
        self.current_passage_id = None
        self.current_edge_idx = None
        self.current_edge = None
        self.current_carrier = None
        self.current_loss = None
        
        # Cache state (for showing dirty markers)
        self.dirty_from = None  # Index where carriers become dirty
        
        # Inference display
        self.inference_text = "Waiting for first inference..."
        self.inference_enabled = False
        
        # Training mode
        self.training_mode = "greedy"  # "greedy" or "global"
        
    def make_layout(self):
        """Create the TUI layout."""
        layout = Layout()
        
        if self.inference_enabled:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="body"),
                Layout(name="inference", size=6),
                Layout(name="footer", size=7)
            )
        else:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="body"),
                Layout(name="footer", size=7)
            )
        
        layout["body"].split_row(
            Layout(name="passage", ratio=2),
            Layout(name="stats", ratio=1)
        )
        
        return layout
    
    def render_header(self):
        """Render header with progress."""
        progress = self.current_iteration / max(self.total_iterations, 1) * 100
        
        header_text = Text()
        header_text.append("🧠 Smart Incremental Training", style="bold cyan")
        
        # Show training mode
        if self.training_mode == "global":
            header_text.append("  🌍 GLOBAL", style="bold magenta")
        elif self.training_mode == "global_training":
            header_text.append("  🌍 GLOBAL (END-TO-END)", style="bold magenta")
        elif self.training_mode == "greedy_burst":
            header_text.append("  ⚡ GREEDY BURST", style="bold green")
        elif self.training_mode == "global_assess":
            header_text.append("  🔍 GLOBAL ASSESS", style="bold blue")
        elif self.training_mode == "greedy_targeted":
            header_text.append("  🎯 TARGETED GREEDY", style="bold yellow")
        elif self.training_mode == "global_polish":
            header_text.append("  ✨ GLOBAL POLISH", style="bold cyan")
        elif self.training_mode == "interleaved":
            header_text.append("  🔄 INTERLEAVED", style="bold magenta")
        else:
            header_text.append("  ⚡ GREEDY", style="bold green")
        
        header_text.append(f"  |  ", style="dim")
        header_text.append(f"{self.current_iteration}/{self.total_iterations}", style="bold yellow")
        header_text.append(f" ({progress:.1f}%)", style="dim")
        
        return Panel(header_text, box=box.HEAVY)
    
    def render_passage(self):
        """Render passage with highlighted current position and carrier states."""
        if self.current_passage is None or self.current_edge_idx is None:
            return Panel("No passage loaded", title="📖 Passage", border_style="blue")
        
        # Convert tokens to text with proper handling of newlines
        # Show passage with carrier state indicators BEFORE each character
        text = Text()
        
        # Determine dirty threshold
        dirty_idx = self.dirty_from if self.dirty_from is not None else len(self.current_passage)
        
        # Convert passage to displayable text
        passage_text = ''.join([self.itos.get(tok.item(), '?') for tok in self.current_passage])
        
        # Build display with inline indicators
        for i, token in enumerate(self.current_passage):
            char = self.itos.get(token.item(), '?')
            
            # Add carrier state indicator before character
            if i < dirty_idx:
                indicator = "●"  # Clean carrier
                indicator_style = "green"
            else:
                indicator = "○"  # Dirty carrier  
                indicator_style = "red"
            
            # Show indicator for positions near current edge
            if abs(i - self.current_edge_idx) <= 3:
                text.append(indicator, style=indicator_style)
            
            # Determine character styling
            if i == self.current_edge_idx:
                # Source token (currently training FROM here)
                text.append(char, style="bold black on bright_green")
            elif i == self.current_edge_idx + 1:
                # Target token (currently training TO here)
                text.append(char, style="bold black on bright_yellow")
            elif char == '\n':
                # Show newline as symbol, then actual newline
                text.append("↵", style="dim cyan")
                text.append("\n")
            else:
                # Normal character
                text.append(char, style="white")
        
        # Add legend
        text.append("\n\n", style="dim")
        text.append("█", style="bold black on bright_green")
        text.append(" Current Source  ", style="dim")
        text.append("█", style="bold black on bright_yellow")
        text.append(" Current Target  ", style="dim")
        text.append("●", style="green")
        text.append(" Clean  ", style="dim")
        text.append("○", style="red")
        text.append(" Dirty", style="dim")
        
        return Panel(
            text,
            title=f"📖 Passage {self.current_passage_id} (pos {self.current_edge_idx}/{len(self.current_passage)-1})",
            border_style="blue",
            padding=(1, 2)
        )
    
    def render_stats(self):
        """Render training statistics."""
        # Edge info table
        edge_table = Table(title="Current Edge", box=box.SIMPLE, show_header=False)
        edge_table.add_column("Metric", style="cyan")
        edge_table.add_column("Value", style="yellow")
        
        if self.current_edge:
            src, tgt = self.current_edge
            src_char = self.itos.get(src, f"#{src}")
            tgt_char = self.itos.get(tgt, f"#{tgt}")
            edge_table.add_row("Edge", f"'{src_char}' → '{tgt_char}'")
            edge_table.add_row("IDs", f"({src}, {tgt})")
        
        # Carrier stats
        carrier_table = Table(title="Carrier State", box=box.SIMPLE, show_header=False)
        carrier_table.add_column("Stat", style="cyan")
        carrier_table.add_column("Value", style="green")
        
        if self.current_carrier is not None:
            carrier_norm = torch.norm(self.current_carrier).item()
            carrier_mean = self.current_carrier.mean().item()
            carrier_std = self.current_carrier.std().item()
            
            carrier_table.add_row("Norm", f"{carrier_norm:.4f}")
            carrier_table.add_row("Mean", f"{carrier_mean:.4f}")
            carrier_table.add_row("Std", f"{carrier_std:.4f}")
        
        # Training stats
        train_table = Table(title="Training", box=box.SIMPLE, show_header=False)
        train_table.add_column("Metric", style="cyan")
        train_table.add_column("Value", style="magenta")
        
        train_table.add_row("Edges trained", f"{len(self.edges_trained)}")
        
        if self.current_loss is not None:
            # Show loss with scientific notation if very small
            if self.current_loss < 0.0001:
                loss_str = f"{self.current_loss:.2e}"
            else:
                loss_str = f"{self.current_loss:.6f}"
            train_table.add_row("Current loss", loss_str)
        
        if self.recent_losses:
            avg_loss = np.mean(list(self.recent_losses))
            min_loss = np.min(list(self.recent_losses))
            max_loss = np.max(list(self.recent_losses))
            
            # Format based on magnitude
            if avg_loss < 0.0001:
                avg_str = f"{avg_loss:.2e}"
                range_str = f"[{min_loss:.2e}, {max_loss:.2e}]"
            else:
                avg_str = f"{avg_loss:.6f}"
                range_str = f"[{min_loss:.6f}, {max_loss:.6f}]"
            
            train_table.add_row("Avg loss (100)", avg_str)
            train_table.add_row("Loss range", range_str)
        
        # Add gradient info if available
        if hasattr(self, 'grad_stats') and self.grad_stats:
            train_table.add_row("", "")  # Spacer
            train_table.add_row("Grad max", f"{self.grad_stats.get('max', 0):.4f}")
            train_table.add_row("Grad median", f"{self.grad_stats.get('median', 0):.4f}")
            train_table.add_row("Grad avg", f"{self.grad_stats.get('avg', 0):.4f}")
            train_table.add_row("Grad std", f"{self.grad_stats.get('std', 0):.4f}")
            train_table.add_row("Grad min", f"{self.grad_stats.get('min', 0):.4f}")
        
        # Add trajectory info if available
        if hasattr(self, 'trajectory_stats') and self.trajectory_stats:
            train_table.add_row("", "")  # Spacer
            train_table.add_row("Traj norm", f"{self.trajectory_stats.get('norm', 0):.4f}")
            train_table.add_row("Traj mean", f"{self.trajectory_stats.get('mean', 0):.6f}")
            train_table.add_row("Traj std", f"{self.trajectory_stats.get('std', 0):.6f}")
        
        # Add 3-Net homeostatic stats if available (Jones 2025 paper)
        if hasattr(self, 'homeostatic_stats') and self.homeostatic_stats:
            train_table.add_row("", "")  # Spacer
            
            # Global Spectral Controller (γ)
            carrier_scale = self.homeostatic_stats.get('carrier_scale', 0)
            trajectory_scale = self.homeostatic_stats.get('trajectory_scale', 0)
            
            # Check health
            scale_learning = abs(carrier_scale - 0.1) > 0.001
            scale_style = "green" if scale_learning else "red"
            
            train_table.add_row("γ carrier", f"{carrier_scale:.6f}", style=scale_style)
            train_table.add_row("γ trajectory", f"{trajectory_scale:.6f}", style=scale_style)
            
            # Local Stabilizer (α)
            alpha_mean = self.homeostatic_stats.get('alpha_mean', 0)
            alpha_std = self.homeostatic_stats.get('alpha_std', 0)
            
            # Handle NaN gracefully
            if alpha_mean != alpha_mean or alpha_std != alpha_std:  # NaN check
                alpha_display = "NaN (error!)"
                alpha_style = "red"
            else:
                # Check health (paper target: ᾱ ∈ [0.28, 0.51])
                alpha_optimal = 0.28 <= alpha_mean <= 0.51
                alpha_adaptive = alpha_std > 0.01
                alpha_style = "green bold" if alpha_optimal and alpha_adaptive else ("yellow" if alpha_adaptive else "red")
                alpha_display = f"{alpha_mean:.4f}±{alpha_std:.4f}"
            
            train_table.add_row("ᾱ (stabilizer)", alpha_display, style=alpha_style)
        
        # Combine tables
        combined = Table.grid(padding=(0, 0))
        combined.add_row(edge_table)
        combined.add_row("")
        combined.add_row(carrier_table)
        combined.add_row("")
        combined.add_row(train_table)
        
        return Panel(combined, title="📊 Statistics", border_style="green")
    
    def render_footer(self):
        """Render footer with recent activity."""
        footer_text = Text()
        
        if self.recent_losses:
            # Show loss sparkline
            losses = list(self.recent_losses)[-50:]  # Last 50
            
            # Filter out NaN values
            valid_losses = [l for l in losses if not (np.isnan(l) or np.isinf(l))]
            
            footer_text.append("Loss trend: ", style="dim")
            
            # Simple ASCII sparkline
            if len(valid_losses) > 1:
                max_loss = max(valid_losses) if max(valid_losses) > 0 else 1
                min_loss = min(valid_losses)
                
                for loss in losses:
                    # Skip NaN/inf values
                    if np.isnan(loss) or np.isinf(loss):
                        footer_text.append("?", style="dim")
                        continue
                    
                    # Normalize to 0-8 range for block characters
                    if max_loss > min_loss:
                        normalized = (loss - min_loss) / (max_loss - min_loss)
                    else:
                        normalized = 0.5
                    
                    # Block characters for sparkline
                    blocks = " ▁▂▃▄▅▆▇█"
                    block_idx = int(normalized * (len(blocks) - 1))
                    color = "red" if loss > np.mean(valid_losses) else "green"
                    footer_text.append(blocks[block_idx], style=color)
            
            # Display recent loss (use valid losses if last one is NaN)
            recent_loss = losses[-1] if not (np.isnan(losses[-1]) or np.isinf(losses[-1])) else (valid_losses[-1] if valid_losses else 0.0)
            footer_text.append(f"\n\nRecent: {recent_loss:.6f}", style="yellow")
            if len(valid_losses) > 10:
                footer_text.append(f"  |  Avg(10): {np.mean(valid_losses[-10:]):.6f}", style="cyan")
        else:
            footer_text.append("Waiting for training data...", style="dim italic")
        
        return Panel(footer_text, title="📈 Loss History", border_style="yellow")
    
    def render_inference(self):
        """Render live inference sample."""
        text = Text()
        
        if self.inference_text:
            # Wrap text nicely
            lines = []
            current_line = ""
            for word in self.inference_text.split():
                if len(current_line) + len(word) + 1 <= 80:
                    current_line += (" " if current_line else "") + word
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            
            for line in lines[:4]:  # Max 4 lines
                text.append(line + "\n", style="bright_white")
        else:
            text.append("Generating...", style="dim italic")
        
        return Panel(text, title="🎭 Live Inference Sample", border_style="magenta")
    
    def render(self):
        """Render the full TUI."""
        layout = self.make_layout()
        
        layout["header"].update(self.render_header())
        layout["passage"].update(self.render_passage())
        layout["stats"].update(self.render_stats())
        if self.inference_enabled:
            layout["inference"].update(self.render_inference())
        layout["footer"].update(self.render_footer())
        
        return layout
    
    def update(self, **kwargs):
        """Update current state."""
        # Store gradient and trajectory stats separately
        if 'grad_stats' in kwargs:
            self.grad_stats = kwargs.pop('grad_stats')
        if 'trajectory_stats' in kwargs:
            self.trajectory_stats = kwargs.pop('trajectory_stats')
        if 'homeostatic_stats' in kwargs:
            self.homeostatic_stats = kwargs.pop('homeostatic_stats')
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Update derived stats
        if 'current_loss' in kwargs and kwargs['current_loss'] is not None:
            self.recent_losses.append(kwargs['current_loss'])
        
        if 'current_edge' in kwargs and kwargs['current_edge'] is not None:
            self.edges_trained.add(kwargs['current_edge'])
    
    def print_summary(self):
        """Print final summary."""
        summary = Table(title="🎉 Training Complete", box=box.DOUBLE_EDGE)
        summary.add_column("Metric", style="cyan bold")
        summary.add_column("Value", style="green bold")
        
        summary.add_row("Total iterations", f"{self.current_iteration:,}")
        summary.add_row("Unique edges trained", f"{len(self.edges_trained):,}")
        
        if self.recent_losses:
            summary.add_row("Final avg loss", f"{np.mean(list(self.recent_losses)):.6f}")
        
        self.console.print(summary)
