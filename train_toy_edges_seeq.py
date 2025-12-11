"""
Train TOY ARITHMETIC edges with SEQUENTIAL TRAINING.

KEY IDEA:
Tests if exponential loss weighting + long sequences force REAL context accumulation!

Toy dataset: Simple arithmetic chains
- Example: "3 + 5 + 2 = 10"
- Cannot be memorized (infinite possible sequences)
- Requires carrier to accumulate running total
- Clear success metric: right or wrong answer

This proves whether the architecture can build context or just learns lookup tables!

Architecture: EXACTLY the same as Shakespeare training
- Same edge networks, same hyperbolic geometry, same Fourier decomposition
- Only difference: arithmetic data instead of text
"""

import os
import sys
import signal
import torch
import torch.nn.functional as F
from pathlib import Path
import pickle
from collections import defaultdict
from tqdm import tqdm
import json
import numpy as np
import random
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box

# Enable CUDA memory optimization for fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Import TUI class
from training_tui import TrainingTUI
from vram_edge_cache import VRAMEdgeCache

# SUBTRACTIVE ENGINEERING: Deterministic network (NO MoE, KEEPS Fourier carrier-signal)
# from hyperbolic_edge_memory import EdgeNeuralNet  # Has MoE routing (non-deterministic)
from edge_neural_net_deterministic import EdgeNeuralNet  # Single-path, keeps Fourier decomposition
from hyperbolic_memory import GeooptLorentzWrapper

# GREYBOX CYBERNETICS: Neuro-Symbolic AI with protected registers
from arithmetic_greybox import wrap_edge_with_greybox


# ============================================================================
# TOY DATASET: ARITHMETIC SEQUENCES
# ============================================================================

def generate_arithmetic_sequences(num_sequences=1000, min_length=3, max_length=8):
    """
    Generate arithmetic sequences like "<START>123 + 456 - 78 = 501\n"
    Uses a set to ensure all sequences are unique.
    Supports 1-4 digit numbers with results in range 0-9999 (no clamping - rejects invalid).
    
    Returns:
        List of strings, each containing one arithmetic problem with START and newline
    """
    sequences = set()
    attempts = 0
    max_attempts = num_sequences * 10  # Try up to 10x to get valid unique sequences
    
    while len(sequences) < num_sequences and attempts < max_attempts:
        attempts += 1
        
        length = random.randint(min_length, max_length)
        
        # Generate numbers with varying digit counts (1-4 digits)
        numbers = []
        for _ in range(length):
            # Weight toward smaller numbers (more 1-2 digit than 3-4 digit)
            num_digits = random.choices([1, 2, 3, 4], weights=[40, 30, 20, 10])[0]
            if num_digits == 1:
                num = random.randint(0, 9)
            elif num_digits == 2:
                num = random.randint(10, 99)
            elif num_digits == 3:
                num = random.randint(100, 999)
            else:  # 4 digits
                num = random.randint(1000, 9999)
            numbers.append(num)
        
        # Build sequence and compute result
        seq = str(numbers[0])
        total = numbers[0]
        
        for i in range(1, length):
            op = random.choice(['+', '-'])
            seq += f" {op} {numbers[i]}"
            if op == '+':
                total += numbers[i]
            else:
                total -= numbers[i]
        
        # ONLY accept if result is in valid range (0-9999)
        # This ensures arithmetic is always correct, no clamping!
        if 0 <= total <= 9999:
            # Add START token at beginning and newline at end
            seq = f"<START>{seq} = {total}\n"
            sequences.add(seq)
    
    return list(sequences)


def create_toy_vocab():
    """
    Create vocabulary for arithmetic:
    - START token: signals beginning of sequence
    - Digits: 0-9
    - Operators: +, -, =
    - Space
    - Newline: signals end of sequence
    """
    chars = ['<START>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '=', ' ', '\n']
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    vocab_meta = {
        'vocab_size': len(chars),
        'stoi': stoi,
        'itos': itos,
        'chars': chars
    }
    
    return vocab_meta


def encode_sequences(sequences, stoi):
    """Convert string sequences to token lists (handle <START> as single token)"""
    encoded = []
    for seq in sequences:
        tokens = []
        i = 0
        while i < len(seq):
            # Check for <START> token
            if seq[i:i+7] == '<START>':
                tokens.append(stoi['<START>'])
                i += 7
            else:
                if seq[i] in stoi:
                    tokens.append(stoi[seq[i]])
                i += 1
        encoded.append(torch.tensor(tokens, dtype=torch.long))
    return encoded


def quick_inference_sample(model_dir, manifold, device, vocab_meta, start_token=23, max_length=30, hidden_dim=257):
    """
    Quick inference for TUI display using convergent EdgeNeuralNet architecture.
    Returns generated text string.
    """
    try:
        import torch.nn as nn
        
        model_dir = Path(model_dir)
        itos = vocab_meta.get('itos', {}) if vocab_meta else {}
        
        # Load decoder head
        decoder_path = model_dir / 'decoder_head.pt'
        if not decoder_path.exists():
            return "Decoder head not trained yet..."
        
        decoder_head = nn.Linear(hidden_dim, 65).to(device)
        decoder_head.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
        decoder_head.eval()
        
        # Initialize carrier and trajectory
        carrier = torch.randn(hidden_dim, device=device) * 0.1
        carrier = carrier / torch.norm(carrier) * 10.0  # Match training norm
        trajectory = torch.randn(hidden_dim, device=device) * 0.1
        
        # Get list of available edges
        edge_files = list(model_dir.glob('*.pt'))
        if len(edge_files) == 0:
            return "No trained edges yet..."
        
        # Generate
        path = [start_token]
        current = start_token
        
        for _ in range(max_length):
            # Find outgoing edges from current token
            outgoing = []
            for f in edge_files:
                if f.stem.startswith(f"{current}_") and f.stem != 'decoder_head':
                    try:
                        tgt = int(f.stem.split('_')[1])
                        outgoing.append((tgt, f.stem))
                    except:
                        continue
            
            if len(outgoing) == 0:
                break
            
            # Sample next token using decoder head (instead of random)
            with torch.no_grad():
                logits = decoder_head(trajectory.to(torch.float32))
                probs = torch.softmax(logits / 0.8, dim=-1)  # temperature=0.8
                
                # Filter to only available outgoing edges
                available_tokens = [tgt for tgt, _ in outgoing]
                if len(available_tokens) == 0:
                    break
                
                # Sample from available edges weighted by model probability
                available_probs = probs[available_tokens]
                available_probs = available_probs / available_probs.sum()
                next_token = available_tokens[torch.multinomial(available_probs, 1).item()]
            
            edge_key = f"{current}_{next_token}"
            
            # Load and apply edge
            try:
                edge_path = model_dir / f'{edge_key}.pt'
                state_dict = torch.load(edge_path, map_location=device, weights_only=True)
                network = EdgeNeuralNet(hidden_dim, manifold, num_heads=4).to(device)
                network.load_state_dict(state_dict, strict=False)  # Allow missing keys
                network.eval()
                
                with torch.no_grad():
                    # Apply convergent edge transformation
                    carrier, trajectory = network(carrier, trajectory, source_pos=current, target_pos=None)
                
                del network, state_dict
                torch.cuda.empty_cache()
            except Exception as e:
                # If edge fails, continue with current state (don't break generation)
                import traceback
                error_msg = traceback.format_exc()[:200]
                # Try to continue without this edge
                continue
            
            path.append(next_token)
            current = next_token
        
        # Convert to text
        return ''.join([itos.get(t, '?') for t in path])
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        if len(error_msg) > 50:
            error_msg = error_msg[:47] + "..."
        return f"Inference error: {error_msg}"


def extract_edges_from_data(data_path='data/shakespeare_char/train.bin'):
    """Extract all unique edges from Shakespeare training data."""
    
    # Load data (numpy binary format)
    data = np.fromfile(data_path, dtype=np.uint16)
    tokens = torch.from_numpy(data.astype(np.int64))
    
    # Extract edges and their positions
    edge_counts = defaultdict(int)
    edge_positions = defaultdict(list)  # edge -> list of positions where it occurs
    
    print(f"📚 Scanning {len(tokens)} tokens for edges...")
    
    for i in range(len(tokens) - 1):
        src = tokens[i].item()
        tgt = tokens[i + 1].item()
        edge = (src, tgt)
        edge_counts[edge] += 1
        edge_positions[edge].append(i)
    
    print(f"✅ Found {len(edge_counts)} unique edges")
    print(f"   Top 10 most frequent:")
    for edge, count in sorted(edge_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"      ({edge[0]}, {edge[1]}): {count} occurrences")
    
    return tokens, edge_counts, edge_positions


def find_paragraph_boundaries(tokens, vocab_meta):
    """
    Find paragraph/section boundaries in text.
    
    A paragraph is defined as text separated by double newlines or major breaks.
    For Shakespeare, this typically corresponds to speaker changes or scene breaks.
    
    Returns list of (start, end) indices for complete paragraphs.
    """
    if vocab_meta:
        itos = vocab_meta.get('itos', {})
        # Find token ID for newline
        newline_token = None
        for token_id, char in itos.items():
            if char == '\n':
                newline_token = token_id
                break
    else:
        # Fallback: assume newline is token 1 (common in char-level)
        newline_token = 1
    
    if newline_token is None:
        # No newline found, treat whole text as one paragraph
        return [(0, len(tokens))]
    
    paragraphs = []
    start = 0
    consecutive_newlines = 0
    
    for i in range(len(tokens)):
        if tokens[i].item() == newline_token:
            consecutive_newlines += 1
            # Double newline (or more) = paragraph boundary
            if consecutive_newlines >= 2:
                if i - consecutive_newlines > start:  # Non-empty paragraph
                    paragraphs.append((start, i - consecutive_newlines + 1))
                start = i + 1
                consecutive_newlines = 0
        else:
            consecutive_newlines = 0
    
    # Add final paragraph if any
    if start < len(tokens):
        paragraphs.append((start, len(tokens)))
    
    return paragraphs


def get_passages_for_edge(edge, edge_positions, tokens, vocab_meta=None, num_passages=10, 
                          max_sentence_length=128, min_sentence_length=5):
    """
    Get COMPLETE PARAGRAPHS containing the target edge.
    
    Returns full paragraphs/sections from start to end, allowing carrier to develop
    meaningful momentum through extended coherent context.
    """
    if edge not in edge_positions:
        return []
    
    positions = edge_positions[edge]
    
    # Find all paragraph boundaries
    paragraphs = find_paragraph_boundaries(tokens, vocab_meta)
    
    # Find which paragraphs contain our target edge
    paragraphs_with_edge = []
    for start, end in paragraphs:
        para_len = end - start
        # Filter by length
        if para_len < min_sentence_length or para_len > max_sentence_length:
            continue
        
        # Check if any edge position falls in this paragraph
        for pos in positions:
            if start <= pos < end - 1:  # -1 because edge is (pos, pos+1)
                paragraphs_with_edge.append({
                    'tokens': tokens[start:end],
                    'edge_pos': pos - start,
                    'start_idx': start,
                    'sentence_len': para_len
                })
                break  # Only add paragraph once even if edge appears multiple times
    
    # Sample random paragraphs
    import random
    num_samples = min(num_passages, len(paragraphs_with_edge))
    if num_samples == 0:
        return []
    
    sampled = random.sample(paragraphs_with_edge, num_samples)
    return sampled


def load_or_create_edge(src, tgt, model_dir, manifold, hidden_dim, device, 
                        use_greybox=False, vocab_size=16, stoi=None):
    """Load edge network if exists, otherwise create new one."""
    edge_path = model_dir / f'{src}_{tgt}.pt'
    
    # hidden_dim should be full Lorentz dimension (257)
    # spatial_dim = 257 - 1 = 256, which is divisible by 4
    # NOTE: use_greybox=False for base network - wrapper adds greybox if needed
    network = EdgeNeuralNet(hidden_dim, manifold, num_heads=4, 
                           use_greybox=False, vocab_size=vocab_size).to(device)
    
    if edge_path.exists():
        try:
            state_dict = torch.load(edge_path, map_location=device, weights_only=False)
            network.load_state_dict(state_dict, strict=False)  # Allow missing keys for greybox
        except Exception as e:
            print(f"   ⚠️  Error loading edge {src}→{tgt}: {e}")
            # Keep fresh network
    
    # WRAP with greybox AFTER loading (or creating fresh)
    if use_greybox:
        network = wrap_edge_with_greybox(network, vocab_size=vocab_size)
        if stoi is not None:
            network.greybox.set_vocab_mapping(stoi)
    
    network.train()
    return network


def save_edge(network, src, tgt, model_dir):
    """Save edge network to disk."""
    edge_path = model_dir / f'{src}_{tgt}.pt'
    torch.save(network.state_dict(), edge_path)


def train_passage_sequence(
    passage_tokens,
    edge_networks,  # Dict of (src, tgt) -> network
    optimizers,     # Dict of (src, tgt) -> optimizer
    manifold,
    device,
    num_steps=10,
    vocab_meta=None,  # For displaying characters
    show_progress=True
):
    """
    Train all edges in a passage together.
    
    Does BOTH forward and backward passes through the sequence.
    This way edges learn to work together!
    """
    
    seq_len = len(passage_tokens)
    if seq_len < 2:
        return {}
    
    # Create fixed positions (simplified - same approach as toy grammar)
    origin = manifold.project(torch.zeros(manifold.dim + 1, device=device))
    
    losses = defaultdict(list)
    
    # Display passage if we have vocab
    if show_progress and vocab_meta:
        itos = vocab_meta.get('itos', {})
        passage_str = ''.join([itos.get(t.item(), '?') for t in passage_tokens[:40]])
        if len(passage_tokens) > 40:
            passage_str += '...'
        print(f"\n    📖 Passage: '{passage_str}'")
    
    # ===================================================================
    # FORWARD PASS: Train sequence left-to-right
    # ===================================================================
    if show_progress:
        print(f"    ➡️  Forward pass ({seq_len-1} edges)...")
    
    forward_pbar = tqdm(range(seq_len - 1), desc="       Forward", leave=False, disable=not show_progress)
    
    for i in forward_pbar:
        src = passage_tokens[i].item()
        tgt = passage_tokens[i + 1].item()
        edge = (src, tgt)
        
        if edge not in edge_networks:
            continue
        
        network = edge_networks[edge]
        optimizer = optimizers[edge]
        
        # Show current edge
        if show_progress and vocab_meta:
            itos = vocab_meta.get('itos', {})
            src_char = itos.get(src, '?')
            tgt_char = itos.get(tgt, '?')
            forward_pbar.set_postfix_str(f"'{src_char}'→'{tgt_char}'")
        
        # Create target position
        target_tangent = torch.randn(manifold.dim, device=device) * 0.1
        target_pos = manifold.exponential_map(origin[1:], target_tangent)
        target_pos = torch.cat([
            torch.sqrt(1 + torch.sum(target_pos**2, dim=-1, keepdim=True)),
            target_pos
        ])
        
        target_direction = manifold.logarithmic_map(origin, target_pos)
        
        # Create input
        z = origin.unsqueeze(0).unsqueeze(0)
        v_in = torch.randn(1, 1, 256, device=device) * 0.1
        
        step_losses = []
        for step in range(num_steps):
            optimizer.zero_grad()
            
            try:
                v_out = network(
                    z=z,
                    source_pos=origin,
                    target_pos=target_pos,
                    v_in=v_in
                )
                
                loss = F.mse_loss(v_out, target_direction.unsqueeze(0).unsqueeze(0))
                
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                    optimizer.step()
                    step_losses.append(loss.item())
                else:
                    break
                    
            except Exception as e:
                break
        
        if step_losses:
            losses[edge].extend(step_losses)
    
    # ===================================================================
    # BACKWARD PASS: Train sequence right-to-left
    # ===================================================================
    if show_progress:
        print(f"    ⬅️  Backward pass ({seq_len-1} edges)...")
    
    backward_pbar = tqdm(range(seq_len - 1, 0, -1), desc="       Backward", leave=False, disable=not show_progress)
    
    for i in backward_pbar:
        src = passage_tokens[i].item()
        tgt = passage_tokens[i - 1].item()
        edge = (src, tgt)  # Reversed direction!
        
        if edge not in edge_networks:
            continue
        
        network = edge_networks[edge]
        optimizer = optimizers[edge]
        
        # Show current edge
        if show_progress and vocab_meta:
            itos = vocab_meta.get('itos', {})
            src_char = itos.get(src, '?')
            tgt_char = itos.get(tgt, '?')
            backward_pbar.set_postfix_str(f"'{src_char}'←'{tgt_char}'")
        
        # Create target position
        target_tangent = torch.randn(manifold.dim, device=device) * 0.1
        target_pos = manifold.exponential_map(origin[1:], target_tangent)
        target_pos = torch.cat([
            torch.sqrt(1 + torch.sum(target_pos**2, dim=-1, keepdim=True)),
            target_pos
        ])
        
        target_direction = manifold.logarithmic_map(origin, target_pos)
        
        # Create input
        z = origin.unsqueeze(0).unsqueeze(0)
        v_in = torch.randn(1, 1, 256, device=device) * 0.1
        
        step_losses = []
        for step in range(num_steps):
            optimizer.zero_grad()
            
            try:
                v_out = network(
                    z=z,
                    source_pos=origin,
                    target_pos=target_pos,
                    v_in=v_in
                )
                
                loss = F.mse_loss(v_out, target_direction.unsqueeze(0).unsqueeze(0))
                
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                    optimizer.step()
                    step_losses.append(loss.item())
                else:
                    break
                    
            except Exception as e:
                break
        
        if step_losses:
            losses[edge].extend(step_losses)
    
    # Return average losses
    avg_losses = {edge: np.mean(loss_list) for edge, loss_list in losses.items()}
    
    if show_progress and avg_losses:
        avg_all = np.mean(list(avg_losses.values()))
        print(f"    📊 Passage avg loss: {avg_all:.6f} ({len(avg_losses)} edges trained)")
    
    return avg_losses


def train_passage_geodesic_refinement(
    passage_tokens,
    model_dir,
    manifold,
    device,
    hidden_dim=257,
    num_refinement_passes=2,
    micro_steps=3,
    vocab_meta=None,
    show_progress=False
):
    """
    Memory-efficient geodesic refinement for a single passage.

    Algorithm:
      1. Forward sweep (no grad): load each edge one-at-a-time, compute v_out -> store carriers on CPU
      2. Refinement: for each edge, load it and train it to map carrier_i -> carrier_{i+1} (micro-steps)
      3. Repeat for num_refinement_passes and also perform a reverse refinement pass

    Returns dict of averaged losses per edge seen in this passage.
    """
    seq_len = len(passage_tokens)
    if seq_len < 2:
        return {}

    origin = manifold.project(torch.zeros(manifold.dim + 1, device=device))

    # Helper to create a random target position on manifold
    def make_target_pos():
        """Create a random point on the manifold (not tangent vector!)."""
        target_tangent = torch.randn(manifold.dim, device=device) * 0.1
        target_pos = manifold.exponential_map(origin, target_tangent, return_lorentz=True)
        return target_pos

    losses = defaultdict(list)

    # Optionally show passage
    if show_progress and vocab_meta:
        itos = vocab_meta.get('itos', {})
        passage_str = ''.join([itos.get(t.item(), '?') for t in passage_tokens])
        # Show full passage with proper formatting
        print(f"\n    📖 Passage ({len(passage_tokens)} tokens):")
        print(f"    ┌{'─' * 60}")
        # Split into lines for readability
        for line in passage_str.split('\n'):
            if line:
                print(f"    │ {line[:58]}")
        print(f"    └{'─' * 60}")

    for pass_i in range(num_refinement_passes):
        # ===== Forward sweep: compute carriers using current edges =====
        if show_progress:
            print(f"    🔁 Refinement pass {pass_i + 1}/{num_refinement_passes} - forward sweep")

        v_carrier = torch.zeros(1, 1, manifold.dim, device=device)
        # ensure carrier matches expected tangent dim (we use hidden spatial dim)
        # store carriers on CPU to avoid VRAM pressure
        carrier_trajectory = [v_carrier.cpu()]

        for i in range(seq_len - 1):
            src = passage_tokens[i].item()
            tgt = passage_tokens[i + 1].item()
            edge = (src, tgt)

            # load network for this edge
            network = load_or_create_edge(src, tgt, model_dir, manifold, hidden_dim, device)
            network.eval()

            with torch.no_grad():
                # Create a target position on manifold for the network call
                target = make_target_pos()
                z = origin.unsqueeze(0).unsqueeze(0)
                
                try:
                    v_out = network(z=z, source_pos=origin, target_pos=target, v_in=v_carrier.to(device))
                except Exception as e:
                    if show_progress:
                        print(f"      ⚠️ Forward refinement error for edge {edge}: {e}")
                    # fallback: random small carrier
                    v_out = torch.randn_like(v_carrier).to(device) * 0.1

            # store and prepare next
            v_carrier = v_out.detach()
            carrier_trajectory.append(v_carrier.cpu())

            # unload
            del network
            torch.cuda.empty_cache()

        # ===== Refinement: train each edge to match next carrier =====
        if show_progress:
            print(f"    🔧 Refinement pass {pass_i + 1}/{num_refinement_passes} - training edges")

        for i in range(seq_len - 1):
            src = passage_tokens[i].item()
            tgt = passage_tokens[i + 1].item()
            edge = (src, tgt)

            network = load_or_create_edge(src, tgt, model_dir, manifold, hidden_dim, device)
            optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
            network.train()

            v_in = carrier_trajectory[i].to(device)
            v_target = carrier_trajectory[i + 1].to(device)

            for m in range(micro_steps):
                optimizer.zero_grad()
                z = origin.unsqueeze(0).unsqueeze(0)
                
                # Create a dummy target position (not used for loss, just for network routing)
                dummy_target = make_target_pos()
                
                try:
                    v_out = network(z=z, source_pos=origin, target_pos=dummy_target, v_in=v_in)
                except Exception as e:
                    # If network fails, skip but log
                    if show_progress:
                        print(f"      ⚠️ Forward refinement error for edge {edge}: {e}")
                    break

                loss = F.mse_loss(v_out, v_target)
                if torch.isnan(loss):
                    if show_progress:
                        print(f"      ⚠️ NaN loss for edge {edge}")
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
                losses[edge].append(loss.item())

            # save and unload
            save_edge(network, src, tgt, model_dir)
            del network, optimizer
            torch.cuda.empty_cache()
        
        # Each refinement pass improves edge cooperation
        # Multiple passes allow iterative convergence

    # return averaged losses
    avg_losses = {edge: float(np.mean(v)) for edge, v in losses.items() if len(v) > 0}
    
    # Debug: print if returning empty
    if not avg_losses and show_progress:
        print(f"    ⚠️ WARNING: No losses collected for this passage!")
    
    return avg_losses


def train_smart_incremental(
    passages_data,
    model_dir,
    manifold,
    device,
    hidden_dim=257,
    num_iterations=10000,
    micro_steps=5,
    propagate_every=50,
    vocab_meta=None,
    show_progress=True,
    debug_mode=False
):
    """
    Smart incremental training with passage cache.
    
    Args:
        debug_mode: If True, force TUI refresh after every iteration (slow but visible)
    
    Algorithm:
    1. Load multiple passages into RAM with carrier caches
    2. Randomly sample (passage, edge) to train
    3. Train edge with current carriers (upstream assumed stable)
    4. Update carrier cache incrementally
    5. Periodically propagate carriers forward when dirty
    
    This is much more efficient than full refinement passes!
    
    Args:
        passages_data: List of token tensors
        model_dir: Directory to save/load edges
        num_iterations: How many random edge training steps
        micro_steps: Training steps per edge per iteration
        propagate_every: How often to propagate dirty carriers
    
    Returns:
        dict of losses per edge
    """
    from passage_cache import PassageCache
    
    # Build passage cache
    if show_progress:
        print(f"\n🗄️  Building passage cache with {len(passages_data)} passages...")
    
    cache = PassageCache(manifold, hidden_dim, device)
    
    for tokens in passages_data:
        cache.add_passage(tokens, vocab_meta)
    
    stats = cache.get_stats()
    if show_progress:
        print(f"   ✅ Cached {stats['num_passages']} passages")
        print(f"      Total edge instances: {stats['total_edges']:,}")
        print(f"      Unique edges: {stats['unique_edges']:,}")
    
    # Initialize carriers for all passages
    if show_progress:
        print(f"\n🌊 Initializing carriers...")
    
    origin = manifold.project(torch.zeros(manifold.dim + 1, device=device))
    
    def make_target_pos():
        """Create a random point on the manifold."""
        target_tangent = torch.randn(manifold.dim, device=device) * 0.1
        target_pos = manifold.exponential_map(origin, target_tangent, return_lorentz=True)
        return target_pos
    
    # Initialize carriers for all passages
    for pid in cache.passages.keys():
        cache.propagate_carriers(pid, model_dir, load_or_create_edge, make_target_pos)
    
    if show_progress:
        print(f"   ✅ Initial carriers computed")
    
    # Initialize VRAM cache for edge networks
    print(f"\n🎮 Initializing VRAM edge cache...")
    vram_cache = VRAMEdgeCache(
        model_dir=model_dir,
        manifold=manifold,
        hidden_dim=hidden_dim,
        device=device,
        max_vram_mb=4000,  # Use up to 4GB
        buffer_mb=500      # Keep 500MB free for training overhead
    )
    print(f"   ✅ VRAM cache ready (limit: 4GB, buffer: 500MB)")
    
    # Wrapper for load_or_create_edge that uses VRAM cache
    def load_or_create_edge_cached(src, tgt, model_dir, manifold, hidden_dim, device):
        network, was_cached = vram_cache.load_edge(
            src, tgt,
            network_class=EdgeNeuralNet,
            network_kwargs={'hidden_dim': hidden_dim, 'manifold': manifold, 'num_heads': 4}
        )
        return network
    
    # Training loop with TUI
    losses = defaultdict(list)
    trained_edges = set()
    
    # Signal handler for graceful shutdown
    shutdown_requested = {'flag': False}
    
    def signal_handler(sig, frame):
        print(f"\n\n⚠️  Received interrupt signal. Finishing current iteration and saving...")
        shutdown_requested['flag'] = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup TUI or fallback
    use_tui = show_progress and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    if use_tui:
        tui = TrainingTUI(vocab_meta=vocab_meta)
        # If num_iterations is None or 0, run indefinitely
        tui.total_iterations = num_iterations if num_iterations > 0 else 999999999
        
        # In debug mode, disable auto-refresh so we can control it manually
        refresh_rate = 1 if debug_mode else 4  # Debug: 1fps for manual control, Normal: 4fps
        
        with Live(tui.render(), refresh_per_second=refresh_rate, console=Console()) as live:
            iter_idx = 0
            while True:
                # Check for shutdown
                if shutdown_requested['flag']:
                    print(f"\n💾 Saving state at iteration {iter_idx}...")
                    break
                
                # Check if we've reached target iterations (if specified)
                if num_iterations > 0 and iter_idx >= num_iterations:
                    break
                
                # Sample random edge instance
                sample = cache.get_random_edge_instance()
                if sample is None:
                    continue
                
                passage_id, edge_idx, src, tgt = sample
                edge = (src, tgt)
                trained_edges.add(edge)
                
                # Get passage tokens for display
                passage_tokens = cache.passages[passage_id]['tokens']
                
                # Preload upcoming edges in this passage (smart prefetching!)
                vram_cache.preload_passage_edges(
                    passage_tokens,
                    current_position=edge_idx,
                    lookahead=5,
                    network_class=EdgeNeuralNet,
                    network_kwargs={'hidden_dim': hidden_dim, 'manifold': manifold, 'num_heads': 4}
                )
                
                # Get carriers
                carrier_in, carrier_target = cache.get_edge_context(passage_id, edge_idx)
                
                # Load edge network from VRAM cache
                network = load_or_create_edge_cached(src, tgt, model_dir, manifold, hidden_dim, device)
                optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
                network.train()
                
                # Train for micro-steps
                for micro_step_idx in range(micro_steps):
                    optimizer.zero_grad()
                    z = origin.unsqueeze(0).unsqueeze(0)
                    dummy_target = make_target_pos()
                    
                    try:
                        v_out = network(
                            z=z,
                            source_pos=origin,
                            target_pos=dummy_target,
                            v_in=carrier_in.to(device)
                        )
                    except Exception:
                        break
                    
                    loss = torch.nn.functional.mse_loss(v_out, carrier_target.to(device))
                    if torch.isnan(loss):
                        break
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                    optimizer.step()
                    
                    losses[edge].append(loss.item())
                    
                    # In debug mode, update TUI after each micro-step
                    if debug_mode and micro_step_idx < micro_steps - 1:
                        # Update loss display
                        tui.update(current_loss=loss.item())
                        live.update(tui.render())
                        live.refresh()
                        import time
                        time.sleep(0.05)  # 50ms pause per micro-step
                
                # Save improved edge (network stays in VRAM cache)
                vram_cache.save_edge(src, tgt, network)
                
                # Compute new carrier with improved edge
                with torch.no_grad():
                    v_new = network(
                        z=z,
                        source_pos=origin,
                        target_pos=make_target_pos(),
                        v_in=carrier_in.to(device)
                    ).detach().cpu()
                
                # Update carrier cache
                needs_propagation = cache.update_carrier(passage_id, edge_idx, v_new)
                
                # Get dirty_from index for visualization
                dirty_from = cache.passages[passage_id]['dirty_from']
                
                # Update TUI
                tui.update(
                    current_iteration=iter_idx + 1,
                    current_passage=passage_tokens,
                    current_passage_id=passage_id,
                    current_edge_idx=edge_idx,
                    current_edge=edge,
                    current_carrier=carrier_in,
                    current_loss=losses[edge][-1] if losses[edge] else None,
                    dirty_from=dirty_from
                )
                live.update(tui.render())
                
                # In debug mode, force refresh and add delay to see updates
                if debug_mode:
                    live.refresh()
                    import time
                    time.sleep(0.2)  # 200ms pause after each edge completes training
                
                # Cleanup optimizer (network stays in VRAM cache)
                del optimizer
                
                # Periodically propagate dirty carriers and show cache stats
                if iter_idx % propagate_every == 0 and iter_idx > 0:
                    if debug_mode:
                        import time
                        time.sleep(0.3)  # Extra pause before propagation in debug mode
                    
                    # Show cache stats every 10 propagations
                    if iter_idx % (propagate_every * 10) == 0:
                        stats = vram_cache.get_stats()
                        print(f"\n🎮 VRAM Cache Stats:")
                        print(f"   Cached edges: {stats['cached_edges']}")
                        print(f"   Hit rate: {stats['hit_rate']*100:.1f}%")
                        print(f"   VRAM usage: {stats['vram_mb']:.0f}MB / {stats['vram_limit_mb']:.0f}MB ({stats['vram_usage_pct']:.1f}%)")
                        print(f"   Preloads: {stats['preloads']}, Evictions: {stats['evictions']}\n")
                    
                    dirty_count = 0
                    for pid in cache.passages.keys():
                        if cache.passages[pid]['dirty_from'] is not None:
                            cache.propagate_carriers(pid, model_dir, load_or_create_edge, make_target_pos)
                            dirty_count += 1
                    
                    # Run quick inference sample periodically
                    if iter_idx % (propagate_every * 5) == 0 and vocab_meta:  # Every 5 propagations
                        tui.inference_enabled = True
                        inference_text = quick_inference_sample(
                            model_dir, manifold, device, vocab_meta,
                            start_token=1,  # Space character - common start
                            max_length=40,
                            hidden_dim=hidden_dim
                        )
                        tui.update(inference_text=inference_text)
                        live.update(tui.render())
                        if debug_mode:
                            live.refresh()
                
                # Increment iteration counter
                iter_idx += 1
        
        # Print final summary
        tui.print_summary()
    
    else:
        # Fallback to tqdm or simple loop
        iter_idx = 0
        
        if show_progress and num_iterations > 0:
            from tqdm import tqdm
            pbar = tqdm(range(num_iterations), desc="Smart training", unit="iter")
        elif show_progress:
            # Infinite mode
            from itertools import count
            pbar = count()
        else:
            # Infinite silent mode
            from itertools import count
            pbar = count()
        
        for _ in pbar:
            # Check for shutdown
            if shutdown_requested['flag']:
                print(f"\n💾 Saving state at iteration {iter_idx}...")
                break
            
            # Check if we've reached target iterations (if specified)
            if num_iterations > 0 and iter_idx >= num_iterations:
                break
            
            # Sample random edge instance
            sample = cache.get_random_edge_instance()
            if sample is None:
                continue
            
            passage_id, edge_idx, src, tgt = sample
            edge = (src, tgt)
            trained_edges.add(edge)
            
            # Get carriers
            carrier_in, carrier_target = cache.get_edge_context(passage_id, edge_idx)
            
            # Load edge network
            network = load_or_create_edge(src, tgt, model_dir, manifold, hidden_dim, device)
            optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
            network.train()
            
            # Train for micro-steps
            for _ in range(micro_steps):
                optimizer.zero_grad()
                z = origin.unsqueeze(0).unsqueeze(0)
                dummy_target = make_target_pos()
                
                try:
                    v_out = network(
                        z=z,
                        source_pos=origin,
                        target_pos=dummy_target,
                        v_in=carrier_in.to(device)
                    )
                except Exception:
                    break
                
                loss = torch.nn.functional.mse_loss(v_out, carrier_target.to(device))
                if torch.isnan(loss):
                    break
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
                
                losses[edge].append(loss.item())
            
            # Save improved edge
            save_edge(network, src, tgt, model_dir)
            
            # Compute new carrier with improved edge
            with torch.no_grad():
                v_new = network(
                    z=z,
                    source_pos=origin,
                    target_pos=make_target_pos(),
                    v_in=carrier_in.to(device)
                ).detach().cpu()
            
            # Update carrier cache
            needs_propagation = cache.update_carrier(passage_id, edge_idx, v_new)
            
            # Cleanup
            del network, optimizer
            torch.cuda.empty_cache()
            
            # Periodically propagate dirty carriers
            if iter_idx % propagate_every == 0 and iter_idx > 0:
                dirty_count = 0
                for pid in cache.passages.keys():
                    if cache.passages[pid]['dirty_from'] is not None:
                        cache.propagate_carriers(pid, model_dir, load_or_create_edge, make_target_pos)
                        dirty_count += 1
                
                if show_progress and dirty_count > 0:
                    print(f"[Iter {iter_idx}] Propagated {dirty_count} passages, {len(trained_edges)} edges trained")
            
            # Increment counter
            iter_idx += 1
    
    # Final propagation
    if show_progress:
        print(f"\n🔄 Final carrier propagation...")
    
    for pid in cache.passages.keys():
        if cache.passages[pid]['dirty_from'] is not None:
            cache.propagate_carriers(pid, model_dir, load_or_create_edge, make_target_pos)
    
    # Return averaged losses
    avg_losses = {edge: float(np.mean(v)) for edge, v in losses.items() if len(v) > 0}
    
    if show_progress:
        print(f"   ✅ Trained {len(trained_edges)} unique edges")
        print(f"   📊 Avg loss: {np.mean(list(avg_losses.values())):.6f}" if avg_losses else "   No losses recorded")
    
    return avg_losses


def train_global_end_to_end(
    passages_data,
    model_dir,
    manifold,
    device,
    hidden_dim=257,
    num_passes=5,
    micro_steps=3,
    vocab_meta=None,
    show_progress=True,
    debug_mode=False
):
    """
    TRUE Global end-to-end training with full passage inference.
    
    This is Phase 2 after greedy training. Each edge is trained with REAL end-to-end loss:
    
    For each position i in passage:
        For each micro-step:
            1. Apply edge i (with gradient)
            2. Infer through ALL downstream edges i+1...end (load each, infer, unload)
            3. Compute loss at the END of passage
            4. Backprop to edge i
            5. Update edge i
    
    This is expensive but ensures edges learn how they affect the ENTIRE passage!
    Early edges learn to "set up" later edges for success.
    """
    from passage_cache import PassageCache
    
    if show_progress:
        print(f"\n🌍 Global End-to-End Training")
        print(f"   Sequential forward passes through passages")
        print(f"   Each edge trained with full passage context")
    
    # Build passage cache
    cache = PassageCache(manifold, hidden_dim, device)
    for tokens in passages_data:
        cache.add_passage(tokens, vocab_meta)
    
    stats = cache.get_stats()
    if show_progress:
        print(f"   ✅ Cached {stats['num_passages']} passages")
        print(f"      Will do {num_passes} forward passes through all")
    
    # Origin and target maker
    origin = manifold.project(torch.zeros(manifold.dim + 1, device=device))
    
    def make_target_pos():
        target_tangent = torch.randn(manifold.dim, device=device) * 0.1
        return manifold.exponential_map(origin, target_tangent, return_lorentz=True)
    
    # Initialize carriers
    for pid in cache.passages.keys():
        cache.propagate_carriers(pid, model_dir, load_or_create_edge, make_target_pos)
    
    # Training stats
    losses = defaultdict(list)
    trained_edges = set()
    total_edge_updates = 0
    
    # Calculate total iterations for progress
    total_iterations = sum(cache.passages[pid]['length'] - 1 for pid in cache.passages.keys()) * num_passes
    
    # Setup TUI or fallback
    use_tui = show_progress and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    if use_tui:
        tui = TrainingTUI(vocab_meta=vocab_meta)
        tui.total_iterations = total_iterations
        tui.training_mode = "global"  # SET GLOBAL MODE
        refresh_rate = 1 if debug_mode else 4
        
        with Live(tui.render(), refresh_per_second=refresh_rate, console=Console()) as live:
            iteration_count = 0
            
            # Do num_passes forward sweeps through all passages
            for pass_num in range(num_passes):
                if show_progress:
                    print(f"\n  🔄 Global Pass {pass_num + 1}/{num_passes}")
                
                # Process each passage sequentially
                for passage_id in cache.passages.keys():
                    passage_tokens = cache.passages[passage_id]['tokens']
                    passage_length = cache.passages[passage_id]['length']
                    
                    # Train edges sequentially from start to end
                    for train_edge_idx in range(passage_length - 1):
                        src = passage_tokens[train_edge_idx].item()
                        tgt = passage_tokens[train_edge_idx + 1].item()
                        edge = (src, tgt)
                        trained_edges.add(edge)
                        total_edge_updates += 1
                        
                        # Get carrier from cache
                        carrier_in = cache.passages[passage_id]['carriers'][train_edge_idx].to(device)
                        
                        # Load edge to train
                        network = load_or_create_edge(src, tgt, model_dir, manifold, hidden_dim, device)
                        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
                        network.train()
                        
                        # Train with micro-steps - EACH with full end-to-end inference!
                        for micro_step_idx in range(micro_steps):
                            optimizer.zero_grad()
                            
                            # 1. Apply edge being trained (with gradient)
                            z = origin.unsqueeze(0).unsqueeze(0)
                            dummy_target = make_target_pos()
                            
                            carrier_out = network(
                                z=z,
                                source_pos=origin,
                                target_pos=dummy_target,
                                v_in=carrier_in.unsqueeze(0).unsqueeze(0)
                            )
                            
                            # 2. INFER THROUGH ALL DOWNSTREAM EDGES (with gradient flow through carrier)
                            #    But freeze downstream edge parameters!
                            carrier = carrier_out.squeeze(0).squeeze(0)
                            
                            for downstream_idx in range(train_edge_idx + 1, passage_length - 1):
                                src_down = passage_tokens[downstream_idx].item()
                                tgt_down = passage_tokens[downstream_idx + 1].item()
                                
                                # Load downstream edge (frozen)
                                edge_down = load_or_create_edge(src_down, tgt_down, model_dir, manifold, hidden_dim, device)
                                edge_down.eval()
                                
                                # Freeze parameters but keep gradient flow through carrier
                                for param in edge_down.parameters():
                                    param.requires_grad = False
                                
                                # Infer forward
                                carrier = edge_down(
                                    z=z,
                                    source_pos=origin,
                                    target_pos=make_target_pos(),
                                    v_in=carrier.unsqueeze(0).unsqueeze(0)
                                ).squeeze(0).squeeze(0)
                                
                                # Unload immediately
                                del edge_down
                                torch.cuda.empty_cache()
                            
                            # 3. Compute END-TO-END loss (where did we end up?)
                            # Target: expected final carrier at end of passage
                            target_final = cache.passages[passage_id]['carriers'][-1].to(device)
                            loss = torch.nn.functional.mse_loss(carrier, target_final)
                            
                            if torch.isnan(loss):
                                break
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                    optimizer.step()
                    
                    losses[edge].append(loss.item())
                    
                    # Debug mode: show micro-step progress
                    if debug_mode and micro_step_idx < micro_steps - 1:
                        tui.update(current_loss=loss.item())
                        live.update(tui.render())
                        live.refresh()
                        import time
                        time.sleep(0.05)
                
                # Save improved edge
                save_edge(network, src, tgt, model_dir)
                
                # IMPORTANT: Update carriers in cache with the improved inference!
                # Re-infer forward with the trained edge to get updated carriers
                with torch.no_grad():
                    network.eval()
                    carrier = carrier_in
                    
                    # Apply trained edge
                    carrier_out = network(
                        z=z,
                        source_pos=origin,
                        target_pos=make_target_pos(),
                        v_in=carrier.unsqueeze(0).unsqueeze(0)
                    ).squeeze(0).squeeze(0).cpu()
                    
                    # Update carrier at position train_edge_idx + 1
                    cache.passages[passage_id]['carriers'][train_edge_idx + 1] = carrier_out
                    
                    # Continue forward through rest of passage to update all downstream carriers
                    carrier = carrier_out.to(device)
                    for i in range(train_edge_idx + 1, passage_length - 1):
                        src_i = passage_tokens[i].item()
                        tgt_i = passage_tokens[i + 1].item()
                        
                        edge_i = load_or_create_edge(src_i, tgt_i, model_dir, manifold, hidden_dim, device)
                        edge_i.eval()
                        
                        carrier = edge_i(
                            z=z,
                            source_pos=origin,
                            target_pos=make_target_pos(),
                            v_in=carrier.unsqueeze(0).unsqueeze(0)
                        ).squeeze(0).squeeze(0)
                        
                        # Update carrier cache
                        cache.passages[passage_id]['carriers'][i + 1] = carrier.cpu()
                        
                        del edge_i
                    
                    # Clear dirty flag since we just updated everything
                    cache.passages[passage_id]['dirty_from'] = None
                
                del network, optimizer
                torch.cuda.empty_cache()
                
                # Update TUI
                iteration_count += 1
                dirty_from = cache.passages[passage_id]['dirty_from']
                tui.update(
                    current_iteration=iteration_count,
                    current_passage=passage_tokens,
                    current_passage_id=passage_id,
                    current_edge_idx=train_edge_idx,
                    current_edge=edge,
                    current_carrier=carrier_in,
                    current_loss=losses[edge][-1] if losses[edge] else None,
                    dirty_from=dirty_from
                )
                live.update(tui.render())
                
                if debug_mode:
                    live.refresh()
                    import time
                    time.sleep(0.2)
                
                if debug_mode:
                    live.refresh()
                    import time
                    time.sleep(0.2)
        
        tui.print_summary()
    
    else:
        # Fallback: simple progress bar or none
        iteration_count = 0
        
        for pass_num in range(num_passes):
            if show_progress:
                print(f"\n  🔄 Global Pass {pass_num + 1}/{num_passes}")
                from tqdm import tqdm
                passage_pbar = tqdm(cache.passages.keys(), desc=f"Pass {pass_num+1}", unit="passage")
            else:
                passage_pbar = cache.passages.keys()
            
            for passage_id in passage_pbar:
                passage_tokens = cache.passages[passage_id]['tokens']
                passage_length = cache.passages[passage_id]['length']
                
                for train_edge_idx in range(passage_length - 1):
                    src = passage_tokens[train_edge_idx].item()
                    tgt = passage_tokens[train_edge_idx + 1].item()
                    edge = (src, tgt)
                    trained_edges.add(edge)
                    total_edge_updates += 1
                    iteration_count += 1
                    
                    carrier_in = cache.passages[passage_id]['carriers'][train_edge_idx].to(device)
                    network = load_or_create_edge(src, tgt, model_dir, manifold, hidden_dim, device)
                    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
                    network.train()
                    
                    for _ in range(micro_steps):
                        optimizer.zero_grad()
                        z = origin.unsqueeze(0).unsqueeze(0)
                        dummy_target = make_target_pos()
                        
                        carrier_out = network(z=z, source_pos=origin, target_pos=dummy_target,
                                            v_in=carrier_in.unsqueeze(0).unsqueeze(0))
                        carrier = carrier_out.squeeze(0).squeeze(0)
                        
                        for i in range(train_edge_idx + 1, passage_length - 1):
                            src_i = passage_tokens[i].item()
                            tgt_i = passage_tokens[i + 1].item()
                            edge_i = load_or_create_edge(src_i, tgt_i, model_dir, manifold, hidden_dim, device)
                            edge_i.eval()
                            with torch.no_grad():
                                for param in edge_i.parameters():
                                    param.requires_grad = False
                            carrier = edge_i(z=z, source_pos=origin, target_pos=make_target_pos(),
                                            v_in=carrier.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                            del edge_i
                        
                        target_carrier = cache.passages[passage_id]['carriers'][-1].to(device)
                        loss = torch.nn.functional.mse_loss(carrier, target_carrier)
                        
                        if torch.isnan(loss):
                            break
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                        optimizer.step()
                        losses[edge].append(loss.item())
                    
                    save_edge(network, src, tgt, model_dir)
                    
                    # Update carriers
                    with torch.no_grad():
                        network.eval()
                        carrier = carrier_in
                        carrier_out = network(z=z, source_pos=origin, target_pos=make_target_pos(),
                                             v_in=carrier.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).cpu()
                        cache.passages[passage_id]['carriers'][train_edge_idx + 1] = carrier_out
                        
                        carrier = carrier_out.to(device)
                        for i in range(train_edge_idx + 1, passage_length - 1):
                            src_i = passage_tokens[i].item()
                            tgt_i = passage_tokens[i + 1].item()
                            edge_i = load_or_create_edge(src_i, tgt_i, model_dir, manifold, hidden_dim, device)
                            edge_i.eval()
                            carrier = edge_i(z=z, source_pos=origin, target_pos=make_target_pos(),
                                            v_in=carrier.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                            cache.passages[passage_id]['carriers'][i + 1] = carrier.cpu()
                            del edge_i
                        
                        cache.passages[passage_id]['dirty_from'] = None
                    
                    del network, optimizer
                    torch.cuda.empty_cache()
    
    avg_losses = {edge: float(np.mean(v)) for edge, v in losses.items() if len(v) > 0}
    
    if show_progress:
        print(f"   ✅ Trained {len(trained_edges)} unique edges globally")
        print(f"   📊 Total edge updates: {total_edge_updates:,}")
        print(f"   📊 Avg end-to-end loss: {np.mean(list(avg_losses.values())):.6f}" if avg_losses else "   No losses recorded")
    
    return avg_losses


def evaluate_global_loss(
    val_tokens,
    edge_positions,
    model_dir,
    manifold,
    hidden_dim,
    device,
    num_passages=10,
    context_size=32
):
    """
    Evaluate global loss on validation set.
    
    Efficient: Load all edges for a passage at once, then test inference.
    """
    
    # Sample random passages from validation
    import random
    total_correct = 0
    total_predictions = 0
    
    passage_starts = random.sample(range(len(val_tokens) - context_size), num_passages)
    
    for start in passage_starts:
        passage = val_tokens[start:start + context_size]
        
        # Load all edges needed for this passage
        edge_networks = {}
        for i in range(len(passage) - 1):
            src = passage[i].item()
            tgt = passage[i + 1].item()
            edge = (src, tgt)
            
            edge_path = model_dir / f'{src}_{tgt}.pt'
            if edge_path.exists():
                try:
                    network = EdgeNeuralNet(hidden_dim, manifold, num_heads=4).to(device)
                    state_dict = torch.load(edge_path, map_location=device, weights_only=False)
                    network.load_state_dict(state_dict)
                    network.eval()
                    edge_networks[edge] = network
                except:
                    pass
        
        # Now test: for each position, can we predict the next token?
        origin = manifold.project(torch.zeros(manifold.dim + 1, device=device))
        
        for i in range(len(passage) - 1):
            current = passage[i].item()
            actual_next = passage[i + 1].item()
            
            # Try all possible next tokens, see which edge gives lowest loss
            best_next = None
            best_score = float('inf')
            
            for candidate_next in range(65):  # vocab_size
                edge = (current, candidate_next)
                
                if edge not in edge_networks:
                    continue
                
                network = edge_networks[edge]
                
                # Create target
                target_tangent = torch.randn(manifold.dim, device=device) * 0.1
                target_pos = manifold.exponential_map(origin[1:], target_tangent)
                target_pos = torch.cat([
                    torch.sqrt(1 + torch.sum(target_pos**2, dim=-1, keepdim=True)),
                    target_pos
                ])
                target_direction = manifold.logarithmic_map(origin, target_pos)
                
                # Forward
                z = origin.unsqueeze(0).unsqueeze(0)
                v_in = torch.randn(1, 1, 256, device=device) * 0.1
                
                with torch.no_grad():
                    try:
                        v_out = network(z=z, source_pos=origin, target_pos=target_pos, v_in=v_in)
                        loss = F.mse_loss(v_out, target_direction.unsqueeze(0).unsqueeze(0))
                        
                        if loss.item() < best_score:
                            best_score = loss.item()
                            best_next = candidate_next
                    except:
                        pass
            
            if best_next == actual_next:
                total_correct += 1
            total_predictions += 1
    
    accuracy = total_correct / max(total_predictions, 1)
    return accuracy


def train_shakespeare_sequential(
    data_path='data/shakespeare_char/train.bin',
    num_passages=1000,
    num_refinement_passes=2,
    micro_steps=3,
    num_epochs=3,
    max_sentence_length=96,  # Without real chunking, 96 is max for 6GB
    min_sentence_length=64,  # Still substantial context
    device='cuda',
    save_every=100
):
    """
    Train Shakespeare edges with RANDOM PARAGRAPH SAMPLING for maximum generalization.
    
    KEY INSIGHT: Sample random complete paragraphs/sections, train all edges in each!
    - Each paragraph is a geodesic path through hyperbolic space
    - LONGER SEQUENCES (64-128 tokens) force carrier to accumulate context over time
    - For dynamical systems, context must propagate through ENTIRE trajectory
    - Edges learn from diverse contexts (not grouped by edge type)
    - Natural curriculum: common edges appear more in random sampling
    - Prevents overfitting to specific passage patterns
    
    CRITICAL: Sequence length balance for 6GB GPU!
    - Literature (RNN/LSTM): 100-500 tokens for language modeling
    - Memory constraint: N edges × 10MB/edge + activations + gradients < 6GB
    - 128 edges × 10MB = 1.3GB networks + ~2GB activations = ~3.5GB total ✓
    - 512 edges × 10MB = 5.1GB networks + activations = OOM! ✗
    - We use 64-128: Forces multi-step context, fits in VRAM
    
    WHY THIS HELPS:
    - 64-128 tokens = 10-20 words = full sentences with context
    - Combined with exponential loss weighting: early edges MUST prepare for token 100+
    - Prevents lookup table memorization (would need 65^100 entries!)
    - Forces carrier to maintain semantic state across sequence
    
    Algorithm:
    1. Find all complete paragraphs in Shakespeare (separated by double newlines)
    2. Each epoch: sample random paragraphs (64-128 chars)
    3. For each paragraph: geodesic refinement (compute carriers, refine edges)
    4. Carrier MUST accumulate context or later predictions fail (exponential weighting!)
    
    Args:
        num_passages: How many random sentences per epoch
        max_sentence_length: Maximum sequence length (128 = ~20 words, fits in 6GB)
        min_sentence_length: Minimum sequence length (64 = ~10 words, forces context) 
        num_refinement_passes: How many forward+backward sweeps per sentence
        micro_steps: Gradient steps per edge per refinement pass
    """
    
    print(f"{'='*60}")
    print(f"🎭 SHAKESPEARE GEODESIC TRAINING (Random Sampling)")
    print(f"{'='*60}")
    print(f"Config:")
    print(f"  Passages per epoch: {num_passages}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Refinement passes: {num_refinement_passes}")
    print(f"  Micro-steps per edge: {micro_steps}")
    print(f"  Sentence length: [{min_sentence_length}, {max_sentence_length}]")
    print(f"  🌊 Random sampling for maximum generalization!")
    print(f"{'='*60}\n")
    
    # Extract edges
    tokens, edge_counts, edge_positions = extract_edges_from_data(data_path)
    vocab_size = tokens.max().item() + 1
    
    print(f"\n📊 Dataset statistics:")
    print(f"   Total tokens: {len(tokens):,}")
    print(f"   Unique edges: {len(edge_counts):,}")
    print(f"   Vocab size: {vocab_size}")
    
    # Load vocabulary metadata for displaying characters
    vocab_meta = None
    try:
        meta_path = Path(data_path).parent / 'meta.pkl'
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                vocab_meta = pickle.load(f)
            print(f"   ✅ Loaded vocabulary: {vocab_meta['vocab_size']} characters")
    except:
        print(f"   ⚠️  Could not load meta.pkl")
    
    # Find all complete paragraphs/sections
    print(f"\n🔍 Finding complete paragraphs/sections...")
    all_paragraphs = find_paragraph_boundaries(tokens, vocab_meta)
    
    # Filter by length
    valid_paragraphs = [
        (start, end) for start, end in all_paragraphs
        if min_sentence_length <= (end - start) <= max_sentence_length
    ]
    
    print(f"   Total paragraphs: {len(all_paragraphs):,}")
    print(f"   Valid length [{min_sentence_length}, {max_sentence_length}]: {len(valid_paragraphs):,}")
    print()
    
    # Create manifold
    manifold = GeooptLorentzWrapper(dim=256)
    hidden_dim = 257
    
    # Create model directory
    model_dir = Path('shakespeare_edges_geodesic')
    model_dir.mkdir(exist_ok=True)
    
    # Track all edges we've trained
    all_trained_edges = set()
    edge_train_counts = defaultdict(int)  # How many times each edge was trained
    
    # Multi-epoch training
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"📖 EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'='*60}\n")
        
        # Sample random paragraphs for this epoch
        import random
        sampled_paragraphs = random.sample(valid_paragraphs, min(num_passages, len(valid_paragraphs)))
        
        all_losses = []
        epoch_start_edges = len(all_trained_edges)  # Track new edges this epoch
        
        # Progress bar for passages
        passage_pbar = tqdm(sampled_paragraphs, desc=f"Epoch {epoch + 1}", unit="passage")
        
        for sent_idx, (start, end) in enumerate(passage_pbar):
            passage_tokens = tokens[start:end]
            
            # Train this passage using geodesic refinement
            losses = train_passage_geodesic_refinement(
                passage_tokens,
                model_dir,
                manifold,
                device,
                hidden_dim=hidden_dim,
                num_refinement_passes=num_refinement_passes,
                micro_steps=micro_steps,
                vocab_meta=vocab_meta,
                show_progress=(sent_idx % 100 == 0)  # Show progress every 100 passages
            )
            
            if losses:
                avg_loss = np.mean(list(losses.values()))
                all_losses.append(avg_loss)
                
                # Track which edges we trained
                for edge in losses.keys():
                    all_trained_edges.add(edge)
                    edge_train_counts[edge] += 1
                
                # Update progress bar with detailed info
                new_edges = len(all_trained_edges) - epoch_start_edges
                passage_pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    trained=len(losses),
                    total=len(all_trained_edges),
                    new=new_edges
                )
            
            # Periodic stats reporting
            if (sent_idx + 1) % 50 == 0:
                print(f"\n{'─'*60}")
                print(f"📊 STATS AFTER {sent_idx + 1} PASSAGES (Epoch {epoch + 1}):")
                print(f"{'─'*60}")
                
                if all_losses:
                    recent_losses = all_losses[-50:]
                    print(f"📉 LOSS: avg={np.mean(recent_losses):.4f}, "
                          f"min={np.min(recent_losses):.4f}, "
                          f"max={np.max(recent_losses):.4f}")
                
                print(f"📈 Unique edges trained: {len(all_trained_edges):,}")
                
                # Show most/least trained edges
                if edge_train_counts:
                    most_trained = sorted(edge_train_counts.items(), key=lambda x: -x[1])[:5]
                    print(f"   Most trained edges:")
                    for edge, count in most_trained:
                        if vocab_meta:
                            itos = vocab_meta.get('itos', {})
                            src_char = itos.get(edge[0], '?')
                            tgt_char = itos.get(edge[1], '?')
                            print(f"      '{src_char}'→'{tgt_char}': {count} times")
                        else:
                            print(f"      {edge}: {count} times")
                print(f"{'─'*60}\n")
            
            # Save checkpoint periodically
            if (sent_idx + 1) % save_every == 0:
                checkpoint = {
                    'epoch': epoch,
                    'passage': sent_idx + 1,
                    'all_trained_edges': list(all_trained_edges),
                    'edge_train_counts': dict(edge_train_counts),
                    'avg_loss': np.mean(all_losses) if all_losses else None
                }
                torch.save(checkpoint, model_dir / f'checkpoint_e{epoch}_p{sent_idx+1}.pt')
        
        print(f"\n✅ Epoch {epoch + 1} complete:")
        print(f"   Passages trained: {len(sampled_paragraphs)}")
        print(f"   Unique edges: {len(all_trained_edges):,}")
        print(f"   Avg loss: {np.mean(all_losses):.4f}" if all_losses else "   No losses recorded")
    
    # Save metadata
    # Convert edge tuples to strings for JSON serialization
    edge_train_counts_str = {f"{src}_{tgt}": count for (src, tgt), count in edge_train_counts.items()}
    
    metadata = {
        'num_edges': len(all_trained_edges),
        'num_epochs': num_epochs,
        'num_passages_per_epoch': num_passages,
        'edge_train_counts': edge_train_counts_str,
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim
    }
    
    with open(model_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"   Total unique edges trained: {len(all_trained_edges):,}")
    print(f"   Saved to: {model_dir}")
    print(f"{'='*60}")


def train_shakespeare_smart(
    data_path='data/shakespeare_char/train.bin',
    num_passages=500,
    num_iterations=10000,
    micro_steps=5,
    num_epochs=1,
    max_sentence_length=96,  # Memory limit without real chunking
    min_sentence_length=64,  # Still forces context
    propagate_every=50,
    device='cuda',
    debug_mode=False,
    global_iterations=0  # NEW: How many global end-to-end iterations to do AFTER greedy
):
    """
    Smart incremental training on Shakespeare.
    
    Uses PassageCache for intelligent carrier management and random edge sampling.
    Much more efficient than naive refinement!
    
    Args:
        global_iterations: If > 0, do global end-to-end training after greedy phase
                          This trains edges with full passage context for better coherence
        debug_mode: If True, slows down TUI updates so you can see every iteration
    """
    print(f"{'='*60}")
    print(f"🧠 SHAKESPEARE SMART INCREMENTAL TRAINING")
    print(f"{'='*60}")
    print(f"Config:")
    print(f"  Passages to cache: {num_passages}")
    print(f"  Training iterations: {num_iterations:,}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Micro-steps per edge: {micro_steps}")
    print(f"  Passage length: [{min_sentence_length}, {max_sentence_length}]")
    if debug_mode:
        print(f"  🐛 DEBUG MODE: Slow TUI updates for visibility")
    print(f"  Propagate every: {propagate_every} iterations")
    print(f"  🎯 Random edge sampling for efficient learning!")
    if global_iterations > 0:
        print(f"  🌍 Global end-to-end phase: {global_iterations:,} iterations")
    print(f"{'='*60}\n")
    
    # Load data
    tokens, edge_counts, edge_positions = extract_edges_from_data(data_path)
    vocab_size = int(tokens.max()) + 1
    
    print(f"\n📚 Scanning {len(tokens):,} tokens for edges...")
    print(f"✅ Found {len(edge_counts):,} unique edges")
    top_edges = sorted(edge_counts.items(), key=lambda x: -x[1])[:10]
    print(f"   Top 10 most frequent:")
    for edge, count in top_edges:
        print(f"      {edge}: {count:,} occurrences")
    
    # Load vocab metadata
    vocab_meta = None
    try:
        meta_path = Path(data_path).parent / 'meta.pkl'
        with open(meta_path, 'rb') as f:
            vocab_meta = pickle.load(f)
            print(f"   ✅ Loaded vocabulary: {vocab_meta['vocab_size']} characters")
    except:
        print(f"   ⚠️  Could not load meta.pkl")
    
    # Find paragraphs
    print(f"\n🔍 Finding complete paragraphs/sections...")
    all_paragraphs = find_paragraph_boundaries(tokens, vocab_meta)
    valid_paragraphs = [
        (start, end) for start, end in all_paragraphs
        if min_sentence_length <= (end - start) <= max_sentence_length
    ]
    
    print(f"   Total paragraphs: {len(all_paragraphs):,}")
    print(f"   Valid length [{min_sentence_length}, {max_sentence_length}]: {len(valid_paragraphs):,}")
    
    # Setup manifold
    hidden_dim = 256  # Spatial dimension
    manifold = GeooptLorentzWrapper(hidden_dim)
    lorentz_dim = hidden_dim + 1  # Full Lorentz dimension = 257
    
    # Create model directory
    model_dir = Path('shakespeare_edges_geodesic')
    model_dir.mkdir(exist_ok=True)
    
    # Track progress
    all_trained_edges = set()
    edge_train_counts = defaultdict(int)
    
    # Multi-epoch training
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"📖 EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'='*60}\n")
        
        # Sample paragraphs
        import random
        sampled_paragraphs = random.sample(valid_paragraphs, min(num_passages, len(valid_paragraphs)))
        
        # Extract tokens for each passage
        passages_data = [tokens[start:end] for start, end in sampled_paragraphs]
        
        # Run smart incremental training
        losses = train_smart_incremental(
            passages_data,
            model_dir,
            manifold,
            device,
            hidden_dim=lorentz_dim,  # Pass full Lorentz dimension (257)
            num_iterations=num_iterations,
            micro_steps=micro_steps,
            propagate_every=propagate_every,
            vocab_meta=vocab_meta,
            show_progress=True,
            debug_mode=debug_mode
        )
        
        # Update tracking
        for edge in losses.keys():
            all_trained_edges.add(edge)
            edge_train_counts[edge] += 1
        
        # Calculate coverage
        total_possible_edges = len(edge_counts)
        coverage = (len(all_trained_edges) / total_possible_edges * 100) if total_possible_edges > 0 else 0
        
        print(f"\n✅ Epoch {epoch + 1} complete:")
        print(f"   Unique edges this epoch: {len(losses):,}")
        print(f"   Total unique edges: {len(all_trained_edges):,} / {total_possible_edges:,}")
        print(f"   📊 Edge coverage: {coverage:.1f}%")
        print(f"   Avg loss: {np.mean(list(losses.values())):.6f}" if losses else "   No losses recorded")
    
    # GLOBAL END-TO-END PHASE (if enabled)
    if global_iterations > 0:
        print(f"\n{'='*60}")
        print(f"🌍 PHASE 2: GLOBAL END-TO-END TRAINING")
        print(f"{'='*60}")
        print(f"Sequential forward passes with end-to-end loss")
        print(f"This ensures edges work together coherently!\n")
        
        # Sample fresh passages for global training
        import random
        sampled_paragraphs = random.sample(valid_paragraphs, min(num_passages, len(valid_paragraphs)))
        passages_data = [tokens[start:end] for start, end in sampled_paragraphs]
        
        # Run global training (num_passes instead of num_iterations)
        global_losses = train_global_end_to_end(
            passages_data,
            model_dir,
            manifold,
            device,
            hidden_dim=lorentz_dim,
            num_passes=global_iterations,  # Changed: treat as number of passes, not iterations
            micro_steps=micro_steps,
            vocab_meta=vocab_meta,
            show_progress=True,
            debug_mode=debug_mode
        )
        
        # Update tracking
        for edge in global_losses.keys():
            all_trained_edges.add(edge)
            edge_train_counts[edge] += 1
        
        # Calculate final coverage after global phase
        total_possible_edges = len(edge_counts)
        coverage = (len(all_trained_edges) / total_possible_edges * 100) if total_possible_edges > 0 else 0
        
        print(f"\n✅ Global phase complete:")
        print(f"   Edges refined globally: {len(global_losses):,}")
        print(f"   Total unique edges: {len(all_trained_edges):,} / {total_possible_edges:,}")
        print(f"   📊 Edge coverage: {coverage:.1f}%")
        print(f"   Avg global loss: {np.mean(list(global_losses.values())):.6f}" if global_losses else "   No losses recorded")
    
    # Calculate edge coverage statistics
    total_possible_edges = len(edge_counts)
    coverage_percentage = (len(all_trained_edges) / total_possible_edges * 100) if total_possible_edges > 0 else 0
    
    # Calculate average training count per edge
    avg_train_count = np.mean(list(edge_train_counts.values())) if edge_train_counts else 0
    edges_trained_once = sum(1 for count in edge_train_counts.values() if count == 1)
    edges_trained_5plus = sum(1 for count in edge_train_counts.values() if count >= 5)
    edges_trained_10plus = sum(1 for count in edge_train_counts.values() if count >= 10)
    
    # Save metadata (for reference)
    edge_train_counts_str = {f"{src}_{tgt}": count for (src, tgt), count in edge_train_counts.items()}
    
    metadata = {
        'num_edges': len(all_trained_edges),
        'num_epochs': num_epochs,
        'num_iterations_per_epoch': num_iterations,
        'num_passages_cached': num_passages,
        'global_iterations': global_iterations,
        'edge_train_counts': edge_train_counts_str,
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim,
        'training_mode': 'smart_incremental' + ('_global' if global_iterations > 0 else ''),
        'total_possible_edges': total_possible_edges,
        'coverage_percentage': coverage_percentage
    }
    
    with open(model_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save metadata_final.json for inference script
    metadata_final = {
        'edge_dir': str(model_dir),
        'hidden_dim': hidden_dim,
        'carrier_cutoff': 20,  # Hardcoded in EdgeNeuralNet
        'num_edges': len(all_trained_edges),
        'vocab_size': vocab_size,
        'training_mode': 'smart_incremental' + ('_global' if global_iterations > 0 else ''),
        'num_epochs': num_epochs,
        'num_iterations_per_epoch': num_iterations,
        'global_iterations': global_iterations,
        'total_possible_edges': total_possible_edges,
        'coverage_percentage': coverage_percentage
    }
    
    with open(model_dir / 'metadata_final.json', 'w') as f:
        json.dump(metadata_final, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ SMART TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"   Total unique edges trained: {len(all_trained_edges):,} / {total_possible_edges:,}")
    print(f"   📊 Edge coverage: {coverage_percentage:.1f}%")
    print(f"   💾 Saved to: {model_dir}")
    print(f"   📄 Metadata: metadata.json, metadata_final.json")
    print(f"{'='*60}")
    
    # Coverage recommendations
    if coverage_percentage < 50:
        print(f"\n⚠️  WARNING: Low edge coverage ({coverage_percentage:.1f}%)")
        print(f"   For coherent text generation, you need at least 70-80% coverage.")
        print(f"   Missing edges: {total_possible_edges - len(all_trained_edges):,}")
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. Increase passages: --num-passages {num_passages * 3} (from {num_passages})")
        print(f"   2. More iterations: --iterations {num_iterations * 2} (from {num_iterations:,})")
        print(f"   3. Multiple epochs: --epochs {num_epochs + 2} (from {num_epochs})")
        print(f"\n   Estimated iterations for 80% coverage: ~{int(num_iterations * (0.8 / max(coverage_percentage/100, 0.01))):,}")
    elif coverage_percentage < 80:
        print(f"\n📊 Moderate edge coverage ({coverage_percentage:.1f}%)")
        print(f"   You'll get some coherent output but may hit dead ends.")
        print(f"   Consider training longer for better results.")
        print(f"   Estimated iterations for 80%: ~{int(num_iterations * (0.8 / max(coverage_percentage/100, 0.01))):,}")
    else:
        print(f"\n✅ Good edge coverage ({coverage_percentage:.1f}%)!")
        print(f"   Model should generate reasonably coherent text.")
        if global_iterations == 0:
            print(f"   💡 Consider adding global training: --global-passes 3")


def train_interleaved(
    data_path=None,  # Not used for toy dataset
    num_passages=500,
    cycle_size=100,
    weak_threshold=0.01,
    micro_steps=3,
    max_sentence_length=20,  # Shorter for arithmetic (e.g., "3 + 5 + 2 = 10")
    min_sentence_length=10,  # Minimum arithmetic sequence
    device='cuda',
    debug_mode=False,
    use_greybox=False  # NEW: Enable greybox cybernetics
):
    """
    INTERLEAVED TRAINING FOR TOY ARITHMETIC DATASET
    
    Exactly the same architecture as Shakespeare, but with arithmetic sequences!
    This tests if the model can REALLY accumulate context (running totals)
    or just memorizes transitions.
    
    This training paradigm prioritizes global coherence while using greedy training
    to rescue struggling edges. The cycle repeats continuously:
    
    PHASE 1: ⚡ GREEDY BURST (cycle_size iterations)
      - Warm up edges with local training across random passages
      - Samples randomly from all passages to avoid overfitting
      - Gets edges roughly functional
    
    PHASE 2: 🌍 GLOBAL TRAINING (PRIMARY - 20 passages)
      - End-to-end backprop through full passage sequences
      - Trains each edge with loss from final passage outcome
      - Rotates through different passages each cycle
      - THIS IS THE PRIMARY LEARNING MECHANISM (teaches coherence)
      - Identifies edges with high global loss as "weak"
    
    PHASE 3: 🎯 TARGETED GREEDY RESCUE (cycle_size/2 iterations)
      - Focused training on weak edges identified in Phase 2
      - Fixes individual edges that struggle in global context
      - Only runs if weak edges were found
    
    PHASE 4: ✨ GLOBAL VERIFICATION (5 passages)
      - Quick verification pass to measure improvement
      - No training, just forward passes to check quality
    
    Key insight: Tests if exponential loss + long sequences force REAL context building!
    
    Args:
        num_passages: Number of arithmetic sequences to generate
        cycle_size: Greedy iterations per cycle  
        weak_threshold: Global loss threshold to identify struggling edges
    """
    from pathlib import Path
    import json
    import time
    from collections import defaultdict
    from rich.live import Live
    from rich.console import Console
    
    print("="*60)
    print("� TOY ARITHMETIC TRAINING")
    print("="*60)
    print(f"  Cycle size: {cycle_size} greedy iterations")
    print(f"  Weak edge threshold: {weak_threshold}")
    print(f"  Sequences: {num_passages}")
    print(f"  Sequence length: [{min_sentence_length}, {max_sentence_length}] chars")
    print("="*60)
    
    # Setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model_dir = Path('toy_arithmetic_edges')
    model_dir.mkdir(exist_ok=True)
    
    # Generate toy dataset (10x more sequences to prevent memorization)
    print("\n📊 Generating arithmetic sequences...")
    sequences = generate_arithmetic_sequences(num_sequences=num_passages*10, 
                                              min_length=3, max_length=8)
    
    # Create vocabulary
    vocab_meta = create_toy_vocab()
    print(f"   ✅ Vocabulary: {vocab_meta['vocab_size']} characters: {vocab_meta['chars']}")
    
    # Encode sequences
    encoded_sequences = encode_sequences(sequences, vocab_meta['stoi'])
    
    # Filter by length
    passages = []
    for seq in encoded_sequences:
        if min_sentence_length <= len(seq) <= max_sentence_length:
            passages.append(seq)
    
    print(f"   ✅ Generated {len(passages)} sequences (length {min_sentence_length}-{max_sentence_length})")
    
    # Save vocab metadata
    with open(model_dir / 'meta.pkl', 'wb') as f:
        pickle.dump(vocab_meta, f)
    
    print(f"   Example sequences:")
    for i in range(min(3, len(passages))):
        seq_str = ''.join([vocab_meta['itos'][t.item()] for t in passages[i]])
        print(f"     {seq_str}")
    
    # Get manifold and hidden dim
    from hyperbolic_memory import GeooptLorentzWrapper
    manifold = GeooptLorentzWrapper(dim=256)
    hidden_dim = 257
    origin = manifold.project(torch.zeros(manifold.dim + 1, device=device))
    
    # Create decoder head: carrier → token logits
    # This is what actually connects carrier state to next token prediction!
    vocab_size = vocab_meta['vocab_size']
    decoder_head = torch.nn.Linear(manifold.dim, vocab_size).to(device)
    decoder_optimizer = torch.optim.AdamW(decoder_head.parameters(), lr=1e-3)
    
    # TOKEN POSITIONS: Fixed positions in hyperbolic space (CRITICAL FOR GEOMETRY!)
    # Each token lives at a fixed point in Lorentz space - same position across ALL passages
    token_positions = torch.nn.Embedding(vocab_size, hidden_dim).to(device)
    # Initialize positions on the manifold
    with torch.no_grad():
        for i in range(vocab_size):
            # Random point on manifold
            pos = torch.randn(hidden_dim, device=device)
            pos = manifold.project(pos)
            token_positions.weight[i] = pos
    token_positions_optimizer = torch.optim.AdamW(token_positions.parameters(), lr=1e-3)
    
    # Try to load existing token positions
    token_positions_path = model_dir / 'token_positions.pt'
    if token_positions_path.exists():
        token_positions.load_state_dict(torch.load(token_positions_path, map_location=device))
        print(f"   ✅ Loaded token positions from {token_positions_path}")
    
    # Try to load existing decoder
    decoder_path = model_dir / 'decoder_head.pt'
    if decoder_path.exists():
        decoder_head.load_state_dict(torch.load(decoder_path, map_location=device))
        print(f"   ✅ Loaded decoder head from {decoder_path}")
    else:
        print(f"   🆕 Created new decoder head (carrier_dim={manifold.dim} → vocab_size={vocab_size})")
    
    # Create VRAM cache
    vram_cache = VRAMEdgeCache(
        model_dir=model_dir,
        manifold=manifold,
        hidden_dim=hidden_dim,
        device=device,
        max_vram_mb=4000,
        buffer_mb=500
    )
    
    def load_or_create_edge_cached(src, tgt, requires_grad=True):
        """Load edge from cache or create new one."""
        network, was_cached = vram_cache.load_edge(
            src, tgt,
            network_class=EdgeNeuralNet,
            network_kwargs={
                'manifold': manifold, 
                'hidden_dim': hidden_dim,
                'use_greybox': use_greybox,
                'vocab_size': vocab_size
            }
        )
        network = network.to(device)
        
        # Configure greybox token mapping if enabled
        if use_greybox:
            network.set_vocab_mapping(vocab_meta['stoi'])
        
        if requires_grad:
            network.train()
            for param in network.parameters():
                param.requires_grad = True
        else:
            network.eval()
            for param in network.parameters():
                param.requires_grad = False
        return network
    
    def make_target_pos():
        """Create dummy target position."""
        return origin.clone()
    
    # Passages already generated above - just verify
    print("\n📝 Using pre-generated arithmetic sequences...")
    passage_lengths = [len(p) for p in passages]
    print(f"   Using {len(passages)} sequences")
    print(f"   Sequence lengths: min={min(passage_lengths)}, max={max(passage_lengths)}, avg={sum(passage_lengths)/len(passage_lengths):.1f}")
    
    # VERIFY: Print first 3 passage lengths explicitly
    print(f"   DEBUG: First 3 passage lengths: {[len(p) for p in passages[:3]]}")
    
    # Initialize TUI
    tui = TrainingTUI(vocab_meta=vocab_meta)
    tui.training_mode = "interleaved"
    refresh_rate = 1 if debug_mode else 4
    
    # Track per-edge statistics
    edge_losses = defaultdict(list)  # (src, tgt) -> [losses]
    edge_weak_counts = defaultdict(int)  # How many times edge was identified as weak
    trained_edges = set()
    
    # Shutdown handler
    shutdown_requested = {'flag': False}
    def signal_handler(sig, frame):
        print(f"\n⚠️  Shutdown requested... saving progress...")
        shutdown_requested['flag'] = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if cycle_size == 0:
        print(f"\n🌍 Starting PURE GLOBAL training (Ctrl+C to stop)...")
        print(f"   No greedy training - edges learn purely through end-to-end backprop!")
    else:
        print(f"\n🔄 Starting interleaved training (Ctrl+C to stop)...")
    
    # Setup CSV logging
    import csv
    csv_path = model_dir / 'training_metrics.csv'
    csv_exists = csv_path.exists()
    csv_file = open(csv_path, 'a', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        'cycle', 'total_global_passes', 'avg_cross_entropy_per_edge', 'avg_total_loss',
        'num_weak_edges', 'avg_carrier_norm', 'num_trained_edges', 'inference_sample'
    ])
    if not csv_exists:
        csv_writer.writeheader()
    print(f"   📊 Logging metrics to {csv_path}")
    
    with Live(tui.render(), refresh_per_second=refresh_rate, console=Console()) as live:
        cycle_num = 0
        total_greedy_iters = 0
        total_global_passes = 0
        
        while not shutdown_requested['flag']:
            cycle_num += 1
            print(f"\n{'='*60}")
            print(f"🔄 CYCLE {cycle_num}")
            print(f"{'='*60}")
            
            # === PHASE 1: GREEDY BURST (skip if cycle_size == 0) ===
            if cycle_size > 0:
                tui.training_mode = "greedy_burst"
                print(f"  ⚡ Phase 1: Greedy burst ({cycle_size} iterations)")
            
            for iter_num in range(cycle_size):
                if shutdown_requested['flag']:
                    break
                
                # Random passage and edge
                passage_tokens = passages[torch.randint(0, len(passages), (1,)).item()]
                edge_idx = torch.randint(0, len(passage_tokens) - 1, (1,)).item()
                src = passage_tokens[edge_idx].item()
                tgt = passage_tokens[edge_idx + 1].item()
                edge = (src, tgt)
                trained_edges.add(edge)
                
                # Smart preloading
                vram_cache.preload_passage_edges(
                    passage_tokens, edge_idx, lookahead=5,
                    network_class=EdgeNeuralNet,
                    network_kwargs={'manifold': manifold, 'hidden_dim': hidden_dim}
                )
                
                # Get carrier (either from cache or compute fresh)
                if edge_idx == 0:
                    carrier_in = torch.zeros(1, 1, manifold.dim, device=device).squeeze(0).squeeze(0)
                else:
                    carrier_in = torch.zeros(1, 1, manifold.dim, device=device).squeeze(0).squeeze(0)
                
                # Load and train edge
                network = load_or_create_edge_cached(src, tgt, requires_grad=True)
                optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
                
                optimizer.zero_grad()
                z = origin.unsqueeze(0).unsqueeze(0)
                target_pos = make_target_pos()
                
                carrier_out = network(
                    z=z,
                    source_pos=origin,
                    target_pos=target_pos,
                    v_in=carrier_in.unsqueeze(0).unsqueeze(0)
                )
                
                # Greedy local loss
                if edge_idx < len(passage_tokens) - 2:
                    next_src = tgt
                    next_tgt = passage_tokens[edge_idx + 2].item()
                    next_edge = load_or_create_edge_cached(next_src, next_tgt, requires_grad=False)
                    
                    # Use detached carrier for target computation (no gradient through next_edge)
                    with torch.no_grad():
                        target_carrier = next_edge(
                            z=z, source_pos=origin, target_pos=target_pos,
                            v_in=carrier_in.detach().unsqueeze(0).unsqueeze(0)
                        )
                    loss = torch.nn.functional.mse_loss(carrier_out, target_carrier.detach())
                else:
                    loss = torch.nn.functional.mse_loss(carrier_out, carrier_in.detach().unsqueeze(0).unsqueeze(0))
                
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                    optimizer.step()
                    edge_losses[edge].append(loss.item())
                
                # Save edge (stays in cache)
                vram_cache.save_edge(src, tgt, network)
                
                # Update TUI
                total_greedy_iters += 1
                tui.update(
                    current_iteration=total_greedy_iters,
                    current_passage=passage_tokens,
                    current_edge_idx=edge_idx,
                    current_edge=edge,
                    current_carrier=carrier_in,
                    current_loss=loss.item() if not torch.isnan(loss) else 0.0,
                    dirty_from=edge_idx + 1
                )
                live.update(tui.render())
            
            # === PHASE 2: GLOBAL TRAINING (PRIMARY) ===
            if shutdown_requested['flag']:
                break
            
            tui.training_mode = "global_training"
            print(f"\n  🌍 Phase 2: Global training (PRIMARY - TRUE end-to-end backprop)")
            print(f"     This is the key insight: train ALL edges in a passage TOGETHER")
            print(f"     Early edges learn from errors that show up later in the text")
            
            weak_edges = set()
            passage_losses = []
            per_edge_global_losses = defaultdict(list)
            actual_mse_losses = []  # Track actual MSE loss (not gradient magnitude)
            base_losses = []  # Track uniform loss (for comparison with exponential weighting)
            carrier_norms = []  # Track carrier norms for CSV logging
            
            # RAPID PASSAGE SWITCHING: Train on MORE passages with FEWER iterations each
            # This gives better exploration and faster edge coverage
            # ULTRA-RAPID MODE: Switch passage EVERY iteration for maximum diversity!
            num_global_passages = min(1000, len(passages))  # 100 → 1000 passages per cycle!
            passage_indices = torch.randperm(len(passages))[:num_global_passages].tolist()
            
            # VRAM budget and checkpointing strategy
            # RAM SAVINGS ANALYSIS (Removing Attention + MoE):
            #   Old EdgeNeuralNet: ~12.6 MB per edge (attention + MoE + stabilizers)
            #   New EdgeNeuralNet: ~1.45 MB per edge (just MLP + Fourier + tiny nets)
            #   Savings: 8.7× smaller!
            #
            # For 200-edge essay:
            #   Old: 200 × 12.6 MB = 2.52 GB (needed checkpointing)
            #   New: 200 × 1.45 MB = 290 MB (fits entirely in VRAM!)
            #
            # DECISION: DISABLE checkpointing, just load all edges!
            max_edges_per_chunk = 300  # Can fit WAY more now
            use_hermetic_checkpointing = True  # Use hermetic for PASS 2 only (no PASS 1 snap)
            hermetic_chunk_size = 50  # (unused)
            #
            # This is the WIN - no checkpointing complexity, just pure end-to-end training!
            
            # Import hermetic checkpointing if needed
            if use_hermetic_checkpointing:
                from hermetic_checkpointing import train_passage_with_hermetic_checkpoint
            
            for passage_idx in passage_indices:
                if shutdown_requested['flag']:
                    break
                
                passage_tokens = passages[passage_idx]
                num_edges = len(passage_tokens) - 1
                
                # === HERMETIC CHECKPOINTING PATH (Staged Convergence) ===
                # Use this for ALL passages to enable snap-to-target training
                if use_hermetic_checkpointing:
                    result = train_passage_with_hermetic_checkpoint(
                        passage_tokens=passage_tokens,
                        model_dir=model_dir,
                        vram_cache=vram_cache,
                        manifold=manifold,
                        decoder_head=decoder_head,
                        decoder_optimizer=decoder_optimizer,
                        device=device,
                        token_positions=token_positions,  # FIXED TOKEN POSITIONS!
                        chunk_size=hermetic_chunk_size,
                        preservation_weight=0.1,   # Lower weight - carrier size less critical
                        target_carrier_norm=10.0,  # Match actual carrier norm
                        weak_threshold=weak_threshold,
                        snap_to_target=False,      # DISABLE PASS 1 - Pure global learning!
                        snap_threshold=2.0,        # (unused)
                        max_snap_steps=0,          # No snap corrections
                        pass2_iterations=1,        # ULTRA-RAPID: 10 → 1 iteration per passage!
                        use_greybox=use_greybox,   # NEW: Pass greybox flag
                        vocab_size=vocab_size,     # NEW: Pass vocab size
                        stoi=vocab_meta['stoi'],   # NEW: Pass token mapping
                        tui=tui,                   # Pass TUI for live updates
                        live=live,                 # Pass Live for rendering
                        verbose=True
                    )
                    
                    # Track results
                    actual_mse_losses.append(result['prediction_loss'])  # Exponentially weighted
                    base_losses.append(result.get('base_loss', result['prediction_loss']))  # Uniform weighting
                    carrier_norms.append(result['carrier_norm'])
                    passage_losses.append(result['loss'])
                    weak_edges.update(result['weak_edges'])
                    trained_edges.update(result['trained_edges'])
                    
                    # Increment global passes counter (was skipped before)
                    total_global_passes += 1
                    
                    # CSV logging every 10 passages for more frequent updates
                    if total_global_passes % 10 == 0:
                        avg_mse_loss = np.mean(actual_mse_losses) if actual_mse_losses else 0.0
                        avg_base_loss = np.mean(base_losses) if base_losses else 0.0
                        avg_carrier_norm = np.mean(carrier_norms) if carrier_norms else 0.0
                        avg_passage_loss = np.mean(passage_losses) if passage_losses else 0.0
                        csv_writer.writerow({
                            'cycle': cycle_num,
                            'total_global_passes': total_global_passes,
                            'avg_cross_entropy_per_edge': avg_mse_loss,
                            'avg_total_loss': avg_passage_loss,
                            'num_weak_edges': len(weak_edges),
                            'avg_carrier_norm': avg_carrier_norm,
                            'num_trained_edges': len(trained_edges),
                            'inference_sample': ""
                        })
                        csv_file.flush()
                    
                    # Save decoder head and token positions periodically
                    if total_global_passes % 5 == 0:
                        torch.save(decoder_head.state_dict(), decoder_path)
                        torch.save(token_positions.state_dict(), token_positions_path)
                        
                        # Also save metadata_final.json for inference compatibility
                        metadata_final = {
                            'vocab_size': len(vocab_meta['chars']),
                            'hidden_dim': hidden_dim,
                            'carrier_cutoff': 20,  # Hardcoded in EdgeNeuralNet
                            'edge_dir': str(model_dir),
                            'manifold': 'lorentz',
                            'total_unique_edges': len(trained_edges),
                            'training_mode': 'interleaved',
                            'use_greybox': use_greybox
                        }
                        with open(model_dir / 'metadata_final.json', 'w') as f:
                            import json
                            json.dump(metadata_final, f, indent=2)
                        
                        print(f"     💾 Saved decoder, positions & metadata (pass {total_global_passes})")
                    
                    # Continue to next passage
                    continue
                
                # === STANDARD PATH (Chunked or Full) ===
                # Chunk long passages to fit in VRAM
                if num_edges > max_edges_per_chunk:
                    num_chunks = (num_edges + max_edges_per_chunk - 1) // max_edges_per_chunk
                    print(f"     Training passage {passage_idx} ({num_edges} edges - chunking into {num_chunks} chunks)")
                else:
                    num_chunks = 1
                    print(f"     Training passage {passage_idx} ({num_edges} edges - full end-to-end gradients)")
                
                # Train each chunk
                carrier = None  # Will be initialized in first chunk
                
                for chunk_idx in range(num_chunks):
                    if shutdown_requested['flag']:
                        break
                    
                    # Calculate chunk boundaries
                    start_edge_idx = chunk_idx * max_edges_per_chunk
                    end_edge_idx = min(start_edge_idx + max_edges_per_chunk, num_edges)
                    
                    # === TRUE GLOBAL TRAINING: All edges in chunk trained simultaneously ===
                    
                    # Step 1: Load ALL edge networks for this chunk into VRAM
                    edge_networks = []
                    edge_optimizers = []
                    edge_tuples = []
                    
                    for edge_idx in range(start_edge_idx, end_edge_idx):
                        src = passage_tokens[edge_idx].item()
                        tgt = passage_tokens[edge_idx + 1].item()
                        edge = (src, tgt)
                        edge_tuples.append(edge)
                        
                        # Load with gradients enabled
                        network = load_or_create_edge_cached(src, tgt, requires_grad=True)
                        optimizer = torch.optim.AdamW(network.parameters(), lr=5e-4)
                        
                        edge_networks.append(network)
                        edge_optimizers.append(optimizer)
                    
                    # Step 2: Initialize or continue carrier
                    if chunk_idx == 0:
                        # First chunk: Initialize carrier with structured signal
                        # IMPORTANT: Explicitly use float32 to prevent dtype promotion issues
                        carrier = torch.zeros(manifold.dim, device=device, dtype=torch.float32)
                        
                        # Encode first token info in carrier (simple hash into manifold)
                        src_token = passage_tokens[start_edge_idx].item()
                        carrier[src_token % manifold.dim] = 1.0  # One-hot-like
                        carrier[(src_token * 7) % manifold.dim] = 0.5  # Secondary encoding
                        carrier = carrier + torch.randn_like(carrier) * 0.01  # Small noise
                        
                        # Normalize to target norm (0.1) to match preservation loss target
                        carrier_norm = torch.norm(carrier)
                        carrier = (carrier / carrier_norm * 0.1).to(torch.float32)  # Explicit float32
                    # else: carrier continues from previous chunk (already set)
                    
                    z = origin.unsqueeze(0).unsqueeze(0)
                    
                    # Zero all gradients
                    for opt in edge_optimizers:
                        opt.zero_grad()
                    
                    # Step 3: SHOOTING-BASED FORWARD PASS with dual carrier-trajectory
                    # TRAINING: We guide the network to hit the correct next token
                    # INFERENCE: Network freely chooses where to shoot
                    
                    # Initialize trajectory vector for this passage
                    trajectory = torch.randn(manifold.dim, device=device, dtype=torch.float32) * 0.1
                    
                    passage_hit = True  # Track if we hit all targets
                    
                    for i, network in enumerate(edge_networks):
                        # Get source and target tokens for this edge
                        src = edge_tuples[i][0]
                        tgt = edge_tuples[i][1]
                        
                        # Get token positions for geometric context
                        src_pos = token_positions(torch.tensor([src], device=device))
                        tgt_pos = token_positions(torch.tensor([tgt], device=device))
                        
                        # Update TUI to show current edge being processed
                        current_global_edge_idx = start_edge_idx + i
                        tui.update(
                            current_iteration=total_global_passes * 100 + current_global_edge_idx,
                            current_passage=passage_tokens,
                            current_edge_idx=current_global_edge_idx,
                            current_edge=edge_tuples[i],
                            current_carrier=carrier,
                            current_loss=0.0,  # Will update after backward
                            dirty_from=current_global_edge_idx + 1
                        )
                        live.update(tui.render())
                        
                        # DUAL INPUT/OUTPUT: Transform both carrier and trajectory
                        carrier, trajectory = network(
                            carrier_in=carrier,  # [256] will be unsqueezed to [1, 1, 256] inside network
                            trajectory_in=trajectory,
                            source_pos=src_pos,
                            target_pos=tgt_pos
                        )
                        
                        # Network returns [B, T, 256], squeeze to [256]
                        carrier = carrier.squeeze(0).squeeze(0).to(torch.float32)
                        trajectory = trajectory.squeeze(0).squeeze(0).to(torch.float32)
                        
                        # NaN protection on carrier
                        if torch.isnan(carrier).any() or torch.isinf(carrier).any():
                            print(f"         ⚠️  NaN/Inf in carrier! Reinitializing.")
                            carrier = torch.randn(manifold.dim, device=device, dtype=torch.float32)
                            carrier = carrier / torch.norm(carrier) * 0.1
                        
                        # NaN protection on trajectory
                        if torch.isnan(trajectory).any() or torch.isinf(trajectory).any():
                            print(f"         ⚠️  NaN/Inf in trajectory! Reinitializing.")
                            trajectory = torch.randn(manifold.dim, device=device, dtype=torch.float32)
                            trajectory = trajectory / torch.norm(trajectory) * 0.1
                        
                        # SHOOTING: Use trajectory to predict next token
                        # Get target token for this edge
                        edge_target_idx = i + 1  # Next token in sequence
                        if edge_target_idx < len(passage_tokens):
                            target_token = passage_tokens[edge_target_idx].to(device)
                            
                            # Decode trajectory → logits (trajectory does prediction!)
                            logits = decoder_head(trajectory)
                            
                            if logits.dim() == 1:
                                logits = logits.unsqueeze(0)
                            if target_token.dim() == 0:
                                target_token = target_token.unsqueeze(0)
                            
                            predicted_token = torch.argmax(logits, dim=-1)
                            
                            # Check if we HIT the target
                            if predicted_token.item() != target_token.item():
                                # MISS! In full shooting mode, we'd restart passage
                                # For now, mark as miss and continue (teacher forcing)
                                passage_hit = False
                                print(f"         ✗ MISS at edge {i}: predicted {predicted_token.item()}, target {target_token.item()}")
                    
                    # Step 4: Compute loss on FINAL prediction
                    # Use the last trajectory to predict the final next token
                    target_token_idx = end_edge_idx
                    if target_token_idx < len(passage_tokens):
                        target_token = passage_tokens[target_token_idx].to(device)
                        
                        # Decode TRAJECTORY (not carrier!) for prediction
                        logits = decoder_head(trajectory.to(torch.float32))
                        
                        if logits.dim() == 1:
                            logits = logits.unsqueeze(0)
                        if target_token.dim() == 0:
                            target_token = target_token.unsqueeze(0)
                        
                        prediction_loss = F.cross_entropy(logits, target_token)
                        
                        # Carrier preservation: carrier should maintain norm, trajectory is for prediction
                        carrier_norm = torch.norm(carrier)
                        target_norm = 0.1
                        preservation_weight = 100.0
                        preservation_loss = preservation_weight * (carrier_norm - target_norm) ** 2
                        
                        # Combined loss
                        global_loss = prediction_loss + preservation_loss
                        
                        # NaN protection
                        if torch.isnan(global_loss) or torch.isinf(global_loss):
                            print(f"         ⚠️  NaN/Inf detected in loss! Skipping.")
                            continue
                        
                        # Track metrics
                        carrier_norms.append(carrier_norm.item())
                        actual_mse_losses.append(prediction_loss.item())
                        
                        # Update TUI with hit/miss status
                        current_global_edge_idx = start_edge_idx + i
                        tui.update(
                            current_iteration=total_global_passes * 100 + current_global_edge_idx,
                            current_passage=passage_tokens,
                            current_edge_idx=current_global_edge_idx,
                            current_edge=edge_tuples[i],
                            current_carrier=carrier,
                            current_loss=global_loss.item(),  # Show actual loss!
                            dirty_from=current_global_edge_idx + 1
                        )
                        live.update(tui.render())
                    else:
                        # End of passage - just use preservation loss
                        carrier_norm = torch.norm(carrier)
                        carrier_norms.append(carrier_norm.item())
                        target_norm = 0.1
                        global_loss = 100.0 * (carrier_norm - target_norm) ** 2  # Match increased weight (was 0.5)
                        actual_mse_losses.append(0.0)  # No prediction loss
                        print(f"         End of passage, using preservation loss only")
                    
                    # Step 5: Backpropagate through ALL edges AND decoder
                    if not torch.isnan(global_loss):
                        # Zero decoder gradient
                        decoder_optimizer.zero_grad()
                        
                        # Backward pass (gradients flow through all edges!)
                        global_loss.backward()
                        
                        # Update decoder head
                        torch.nn.utils.clip_grad_norm_(decoder_head.parameters(), 1.0)
                        decoder_optimizer.step()
                        
                        # Step 6: Update ALL edges (they all learned from the passage's final error)
                        for i, (edge, network, optimizer) in enumerate(zip(edge_tuples, edge_networks, edge_optimizers)):
                            # Clip gradients
                            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                            
                            # Update
                            optimizer.step()
                            
                            # Track per-edge contribution (approximate by gradient magnitude)
                            total_grad = sum(p.grad.abs().sum().item() for p in network.parameters() if p.grad is not None)
                            per_edge_global_losses[edge].append(total_grad)
                            
                            # Identify weak edges (high gradient = struggling to fit)
                            if total_grad > weak_threshold:
                                weak_edges.add(edge)
                                edge_weak_counts[edge] += 1
                            
                            # Save updated network
                            vram_cache.save_edge(edge[0], edge[1], network)
                            
                            # Update TUI with the edge's gradient/loss info
                            current_global_edge_idx = start_edge_idx + i
                            tui.update(
                                current_iteration=total_global_passes * 100 + current_global_edge_idx,
                                current_passage=passage_tokens,
                                current_edge_idx=current_global_edge_idx,
                                current_edge=edge,
                                current_loss=total_grad,
                                dirty_from=len(passage_tokens)  # All edges trained
                            )
                            live.update(tui.render())
                    
                        passage_losses.append(global_loss.item())
                        
                        if num_chunks > 1:
                            print(f"       Chunk {chunk_idx+1}/{num_chunks} loss: {global_loss.item():.6f}")
                        else:
                            print(f"       Global loss: {global_loss.item():.6f}, Weak edges found: {len(weak_edges)}")
                        
                        # Track which edges have been trained
                        trained_edges.update(edge_tuples)
                        
                        # Detach carrier for next chunk (breaks gradient but saves memory)
                        # This is necessary for chunked passages - each chunk gets independent gradients
                        if chunk_idx < num_chunks - 1:
                            carrier = carrier.detach()
                        
                        # Clear VRAM: unload edge networks after updating
                        # This prevents memory fragmentation and ensures we can fit next chunk
                        del edge_networks
                        del edge_optimizers
                        torch.cuda.empty_cache()  # Force garbage collection
                
                # End of passage (all chunks processed)
                if num_chunks > 1:
                    print(f"       Full passage complete: {num_chunks} chunks, avg loss: {np.mean(passage_losses[-num_chunks:]):.6f}")
            
            total_global_passes += 1
            avg_passage_loss = np.mean(passage_losses) if passage_losses else 0.0
            avg_mse_loss = np.mean(actual_mse_losses) if actual_mse_losses else 0.0
            
            # Save decoder head and token positions periodically
            if total_global_passes % 5 == 0:
                torch.save(decoder_head.state_dict(), decoder_path)
                torch.save(token_positions.state_dict(), token_positions_path)
                print(f"     💾 Saved decoder head & token positions")
            
            print(f"     Passages trained: {num_global_passages}")
            print(f"     Weak edges identified: {len(weak_edges)}")
            print(f"     Avg gradient magnitude: {avg_passage_loss:.6f}")
            print(f"     Avg PREDICTION LOSS (cross-entropy): {avg_mse_loss:.6f}")  # TRUE next-token prediction quality!
            
            # === INFERENCE CHECK: See if model can generate coherent text ===
            inference_text = ""
            if cycle_num % 1 == 0:  # Every cycle
                print(f"\n  🔮 Inference check (cycle {cycle_num})...")
                try:
                    inference_text = quick_inference_sample(
                        model_dir=model_dir,
                        manifold=manifold,
                        device=device,
                        vocab_meta=vocab_meta,
                        start_token=23,  # '\n'
                        max_length=50,
                        hidden_dim=hidden_dim
                    )
                    print(f"     Generated: {repr(inference_text[:80])}")
                    
                    # Update TUI with inference
                    tui.update(inference_text=inference_text)
                    live.update(tui.render())
                except Exception as e:
                    print(f"     Inference failed: {e}")
            
            # Write metrics to CSV
            avg_carrier_norm = np.mean(carrier_norms) if carrier_norms else 0.0
            csv_writer.writerow({
                'cycle': cycle_num,
                'total_global_passes': total_global_passes,
                'avg_cross_entropy_per_edge': avg_mse_loss,
                'avg_total_loss': avg_passage_loss,
                'num_weak_edges': len(weak_edges),
                'avg_carrier_norm': avg_carrier_norm,
                'num_trained_edges': len(trained_edges),
                'inference_sample': inference_text[:200] if inference_text else ""  # Allow longer for errors
            })
            csv_file.flush()  # Ensure data is written immediately

            
            # === PHASE 3: TARGETED GREEDY (skip if cycle_size == 0 - pure global mode) ===
            if cycle_size > 0 and weak_edges and not shutdown_requested['flag']:
                tui.training_mode = "greedy_targeted"
                print(f"\n  🎯 Phase 3: Targeted greedy ({cycle_size//2} iterations on {len(weak_edges)} weak edges)")
                
                for iter_num in range(cycle_size // 2):
                    if shutdown_requested['flag']:
                        break
                    
                    # Sample from weak edges
                    edge = list(weak_edges)[torch.randint(0, len(weak_edges), (1,)).item()]
                    src, tgt = edge
                    
                    # Train this edge
                    carrier_in = torch.zeros(1, 1, manifold.dim, device=device).squeeze(0).squeeze(0)
                    network = load_or_create_edge_cached(src, tgt, requires_grad=True)
                    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
                    
                    optimizer.zero_grad()
                    z = origin.unsqueeze(0).unsqueeze(0)
                    carrier_out = network(
                        z=z, source_pos=origin, target_pos=make_target_pos(),
                        v_in=carrier_in.unsqueeze(0).unsqueeze(0)
                    )
                    
                    loss = torch.nn.functional.mse_loss(
                        carrier_out,
                        carrier_in.unsqueeze(0).unsqueeze(0)
                    )
                    
                    if not torch.isnan(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                        optimizer.step()
                        edge_losses[edge].append(loss.item())
                    
                    vram_cache.save_edge(src, tgt, network)
                    total_greedy_iters += 1
            
            # === PHASE 4: GLOBAL POLISH ===
            if not shutdown_requested['flag']:
                tui.training_mode = "global_polish"
                print(f"\n  ✨ Phase 4: Global polish (verify improvements)")
                
                # Quick verification pass on a few passages
                verify_losses = []
                for passage_idx in passage_indices[:5]:  # Check 5 passages
                    if shutdown_requested['flag']:
                        break
                    
                    passage_tokens = passages[passage_idx]
                    carrier = torch.zeros(1, 1, manifold.dim, device=device).squeeze(0).squeeze(0)
                    trajectory = torch.zeros(1, 1, manifold.dim, device=device).squeeze(0).squeeze(0)
                    
                    # Full forward pass (no training, just measurement)
                    with torch.no_grad():
                        for edge_idx in range(len(passage_tokens) - 1):
                            src = passage_tokens[edge_idx].item()
                            tgt = passage_tokens[edge_idx + 1].item()
                            network = load_or_create_edge_cached(src, tgt, requires_grad=False)
                            
                            # Get token positions for geometric context
                            src_pos = token_positions(torch.tensor([src], device=device))
                            tgt_pos = token_positions(torch.tensor([tgt], device=device))
                            
                            # Dual-vector signature with positions
                            carrier, trajectory = network(
                                carrier_in=carrier,
                                trajectory_in=trajectory,
                                source_pos=src_pos,
                                target_pos=tgt_pos
                            )
                            carrier = carrier.squeeze(0).squeeze(0)
                            trajectory = trajectory.squeeze(0).squeeze(0)
                        
                        # Compare to zero carrier (minimal disturbance target)
                        verify_loss = torch.nn.functional.mse_loss(carrier, torch.zeros_like(carrier)).item()
                        verify_losses.append(verify_loss)
                
                avg_verify_loss = np.mean(verify_losses) if verify_losses else 0.0
                print(f"     Verification loss: {avg_verify_loss:.6f}")
                total_global_passes += 1
            
            # Show cycle summary
            print(f"\n  📊 Cycle {cycle_num} complete:")
            print(f"     Total greedy iterations: {total_greedy_iters:,}")
            print(f"     Total global passes: {total_global_passes}")
            print(f"     Unique edges trained: {len(trained_edges):,}")
            print(f"     Cache stats: {vram_cache.hits} hits, {vram_cache.misses} misses")
            
            live.update(tui.render())
    
    # Save final state
    print(f"\n💾 Saving final model...")
    vram_cache.clear()  # Save all cached edges
    
    # Save decoder head and token positions (CRITICAL!)
    torch.save(decoder_head.state_dict(), decoder_path)
    torch.save(token_positions.state_dict(), token_positions_path)
    print(f"   ✅ Saved decoder head to {decoder_path}")
    print(f"   ✅ Saved token positions to {token_positions_path}")
    
    # Save metadata
    total_possible_edges = len(vocab_meta['chars']) ** 2
    coverage_percentage = (len(trained_edges) / total_possible_edges) * 100
    
    metadata = {
        'vocab_size': len(vocab_meta['chars']),
        'hidden_dim': hidden_dim,
        'manifold': 'poincare',
        'total_unique_edges': len(trained_edges),
        'total_possible_edges': total_possible_edges,
        'coverage_percentage': coverage_percentage,
        'training_mode': 'interleaved',
        'total_cycles': cycle_num,
        'total_greedy_iterations': total_greedy_iters,
        'total_global_passes': total_global_passes
    }
    
    with open(model_dir / 'metadata_final.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Close CSV file
    csv_file.close()
    print(f"   📊 Closed training metrics CSV")
    
    print(f"\n{'='*60}")
    print(f"✅ INTERLEAVED TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"   Cycles: {cycle_num}")
    print(f"   Greedy iterations: {total_greedy_iters:,}")
    print(f"   Global passes: {total_global_passes}")
    print(f"   Edges trained: {len(trained_edges):,} / {total_possible_edges:,}")
    print(f"   Coverage: {coverage_percentage:.1f}%")
    print(f"{'='*60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/shakespeare_char/train.bin')
    parser.add_argument('--num-passages', type=int, default=500, help='Base number for sequence generation (actual sequences = num_passages * 10)')
    parser.add_argument('--refinement-passes', type=int, default=2)
    parser.add_argument('--micro-steps', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max-len', type=int, default=20, help='Max sequence length for arithmetic (e.g., "3 + 5 + 2 = 10")')
    parser.add_argument('--min-len', type=int, default=10, help='Min sequence length for arithmetic')
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--smart', action='store_true', default=False, help='Use smart incremental training')
    parser.add_argument('--naive', dest='smart', action='store_false', help='Use naive refinement training')
    parser.add_argument('--interleaved', action='store_true', default=True, help='Use interleaved training (DEFAULT for toy dataset)')
    parser.add_argument('--iterations', type=int, default=0, help='Number of greedy iterations (0 = infinite, stop with Ctrl+C)')
    parser.add_argument('--global-passes', type=int, default=0, help='Number of global end-to-end forward passes (phase 2)')
    parser.add_argument('--greedy-warmup', type=int, default=100, help='Initial greedy iterations before starting interleaved/global training')
    parser.add_argument('--cycle-size', type=int, default=0, help='Greedy iterations per interleaved cycle (0 = HERMETIC ONLY - RECOMMENDED)')
    parser.add_argument('--weak-threshold', type=float, default=0.01, help='Loss threshold to identify weak edges')
    parser.add_argument('--propagate-every', type=int, default=50, help='Propagate carriers every N iterations')
    parser.add_argument('--debug', action='store_true', help='Debug mode: slow TUI updates to see every iteration')
    parser.add_argument('--greybox', action='store_true', help='Enable greybox cybernetics (symbolic arithmetic injection)')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"🔢 TOY ARITHMETIC DATASET TRAINING")
    print(f"   Testing if exponential loss + long sequences force REAL context!")
    print()
    
    if args.interleaved:
        print(f"🔄 Using INTERLEAVED training")
        if args.greybox:
            print(f"🎯 GREYBOX CYBERNETICS ENABLED - Injecting symbolic arithmetic rules!")
        train_interleaved(
            data_path=None,  # Not used for toy dataset
            num_passages=args.num_passages,
            cycle_size=args.cycle_size,
            weak_threshold=args.weak_threshold,
            micro_steps=args.micro_steps,
            max_sentence_length=args.max_len,
            min_sentence_length=args.min_len,
            device=device,
            debug_mode=args.debug,
            use_greybox=args.greybox
        )
        print(f"🧠 Using SMART incremental training")
        train_shakespeare_smart(
            data_path=args.data,
            num_passages=args.num_passages,
            num_iterations=args.iterations,
            micro_steps=args.micro_steps,
            num_epochs=args.epochs,
            max_sentence_length=args.max_len,
            min_sentence_length=args.min_len,
            propagate_every=args.propagate_every,
            device=device,
            debug_mode=args.debug,
            global_iterations=args.global_passes
        )
    else:
        print(f"📚 Using naive refinement training")
        train_shakespeare_sequential(
            data_path=args.data,
            num_passages=args.num_passages,
            num_refinement_passes=args.refinement_passes,
            micro_steps=args.micro_steps,
            num_epochs=args.epochs,
            max_sentence_length=args.max_len,
            min_sentence_length=args.min_len,
            save_every=args.save_every,
            device=device
        )


if __name__ == '__main__':
    main()
