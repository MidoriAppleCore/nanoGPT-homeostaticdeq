"""
Preload Memory System - Seed hyperbolic memory from dataset before training

This module scans the training dataset and extracts semantic chunks to populate
the memory system before training begins. Like giving the model a "textbook" to
read before class!

Key benefits:
1. Faster initial training (no cold start)
2. Better early-stage predictions (knowledge from day 1)
3. Supervised navigation can start immediately (memories already exist)
4. More diverse initial memory substrate

Caching system:
- Preload is SLOW (10K memories = ~10 minutes)
- First run: Creates "preload_cache/" with pristine memories
- Subsequent runs: Copy cached memories to training dir (instant!)
- Training modifies the copied version, cache stays pristine
"""

import os
import pickle
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from contextlib import nullcontext


def get_preload_cache_path(data_dir, num_samples, chunk_size):
    """Get path to cached preloaded memories."""
    # Cache is dataset-specific and config-specific
    cache_name = f"preload_n{num_samples}_c{chunk_size}.pt"
    cache_dir = os.path.join(data_dir, "preload_cache")
    return os.path.join(cache_dir, cache_name)


def load_cached_preload(cache_path, memory_system, verbose=True):
    """Load preloaded memories from cache into memory system."""
    if not os.path.exists(cache_path):
        return False
    
    if verbose:
        print(f"📦 Loading cached preload from: {cache_path}")
    
    try:
        checkpoint = torch.load(cache_path, map_location='cpu')
        
        if hasattr(memory_system, 'longterm'):
            lt_mem = memory_system.longterm
            
            # Check if using DiskBackedTensor
            from disk_backed_tensor import DiskBackedTensor
            is_disk_backed = isinstance(lt_mem.embeddings, DiskBackedTensor)
            
            # Load embeddings
            if is_disk_backed:
                # For DiskBackedTensor, use setitem interface
                for i in range(checkpoint['size'].item()):
                    lt_mem.embeddings[i] = checkpoint['embeddings'][i]
            else:
                # For regular tensor, use direct assignment
                lt_mem.embeddings.data[:checkpoint['size']] = checkpoint['embeddings']
            
            lt_mem.size = checkpoint['size']
            
            # Restore all graph metadata
            if 'adjacency' in checkpoint:
                lt_mem.adjacency = checkpoint['adjacency']
            if 'edge_weights' in checkpoint:
                lt_mem.edge_weights = checkpoint['edge_weights']
            if 'edge_types' in checkpoint:
                lt_mem.edge_types = checkpoint['edge_types']
            if 'type_embeddings' in checkpoint:
                lt_mem.type_embeddings = checkpoint['type_embeddings']
            if 'cluster_ids' in checkpoint:
                lt_mem.cluster_ids = checkpoint['cluster_ids']
            if 'depths' in checkpoint:
                lt_mem.depths = checkpoint['depths']
            if 'rewards' in checkpoint:
                lt_mem.rewards = checkpoint['rewards']
            if 'age' in checkpoint:
                lt_mem.age = checkpoint['age']
            if 'access' in checkpoint:
                lt_mem.access = checkpoint['access']
            if 'edge_traversal_count' in checkpoint:
                lt_mem.edge_traversal_count = checkpoint['edge_traversal_count']
            if 'edge_success_rate' in checkpoint:
                lt_mem.edge_success_rate = checkpoint['edge_success_rate']
            
            if verbose:
                print(f"✅ Loaded {checkpoint['size'].item()} memories from cache")
            return True
    except Exception as e:
        if verbose:
            print(f"⚠️  Failed to load cache: {e}")
        return False


def save_preload_cache(cache_path, memory_system, verbose=True):
    """Save preloaded memories to cache for reuse."""
    if verbose:
        print(f"💾 Saving preload cache to: {cache_path}")
    
    # Create cache directory
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    if hasattr(memory_system, 'longterm'):
        lt_mem = memory_system.longterm
        
        # Check if using DiskBackedTensor
        from disk_backed_tensor import DiskBackedTensor
        is_disk_backed = isinstance(lt_mem.embeddings, DiskBackedTensor)
        
        # Extract embeddings
        if is_disk_backed:
            # For DiskBackedTensor, manually gather the data
            size = lt_mem.size.item()
            embeddings = torch.stack([lt_mem.embeddings[i] for i in range(size)])
        else:
            # For regular tensor, clone directly
            embeddings = lt_mem.embeddings[:lt_mem.size.item()].clone()
        
        checkpoint = {
            'embeddings': embeddings,
            'size': lt_mem.size.clone(),
        }
        
        # Save all graph metadata
        if hasattr(lt_mem, 'adjacency'):
            checkpoint['adjacency'] = lt_mem.adjacency.clone()
        if hasattr(lt_mem, 'edge_weights'):
            checkpoint['edge_weights'] = lt_mem.edge_weights.clone()
        if hasattr(lt_mem, 'edge_types'):
            checkpoint['edge_types'] = lt_mem.edge_types.clone()
        if hasattr(lt_mem, 'type_embeddings'):
            checkpoint['type_embeddings'] = lt_mem.type_embeddings.clone()
        if hasattr(lt_mem, 'cluster_ids'):
            checkpoint['cluster_ids'] = lt_mem.cluster_ids.clone()
        if hasattr(lt_mem, 'depths'):
            checkpoint['depths'] = lt_mem.depths.clone()
        if hasattr(lt_mem, 'rewards'):
            checkpoint['rewards'] = lt_mem.rewards.clone()
        if hasattr(lt_mem, 'age'):
            checkpoint['age'] = lt_mem.age.clone()
        if hasattr(lt_mem, 'access'):
            checkpoint['access'] = lt_mem.access.clone()
        if hasattr(lt_mem, 'edge_traversal_count'):
            checkpoint['edge_traversal_count'] = lt_mem.edge_traversal_count.clone()
        if hasattr(lt_mem, 'edge_success_rate'):
            checkpoint['edge_success_rate'] = lt_mem.edge_success_rate.clone()
        
        torch.save(checkpoint, cache_path)
        
        if verbose:
            cache_size_mb = os.path.getsize(cache_path) / 1024 / 1024
            print(f"✅ Cache saved ({cache_size_mb:.1f} MB)")


def preload_memories_from_dataset(
    model,
    data_dir,
    dataset_name,
    num_samples=1000,
    chunk_size=32,
    stride=16,
    device='cuda',
    dtype=torch.float16,
    verbose=True,
    use_simple_mode=True  # Use simple one-at-a-time mode like the test
):
    """
    Scan dataset and populate memory system with semantic chunks.
    
    Args:
        model: The GrayBoxDEQ model with memory system
        data_dir: Path to data directory
        dataset_name: Name of dataset (e.g., 'oasst')
        num_samples: Number of chunks to extract (default 1000)
        chunk_size: Tokens per chunk (default 32, ~25 words)
        stride: Step between chunks (default 16, 50% overlap)
        device: Device to use
        dtype: Data type for computation
        verbose: Print progress
        use_simple_mode: Use simple one-at-a-time addition (like test, more stable)
        
    Returns:
        num_memories_added: Number of memories successfully added
    """
    
    if verbose:
        print("=" * 70)
        print("🗄️  PRELOADING MEMORIES FROM DATASET")
        print("=" * 70)
    
    # Access memory system first (needed for cache loading)
    if not hasattr(model, 'reflex') or not hasattr(model.reflex, 'memory_retrieval'):
        print("⚠️  Model has no memory system - skipping preload")
        return 0
    
    memory_system = model.reflex.memory_retrieval
    
    # Check for cached preload
    cache_path = get_preload_cache_path(data_dir, num_samples, chunk_size)
    
    if os.path.exists(cache_path):
        if verbose:
            print(f"🎯 Found cached preload! Loading from cache...")
        
        if load_cached_preload(cache_path, memory_system, verbose=verbose):
            if verbose:
                print("=" * 70 + "\n")
            return memory_system.longterm.size.item() if hasattr(memory_system, 'longterm') else 0
        else:
            if verbose:
                print("⚠️  Cache load failed, will rebuild from scratch")
    else:
        if verbose:
            print(f"📊 No cache found at: {cache_path}")
            print(f"   Will build fresh preload (this will take ~10 minutes for 10K memories)")
    
    # Load training data
    # data_dir is already 'data/dataset_name', so just append 'train.bin'
    train_data_path = os.path.join(data_dir, 'train.bin')
    if not os.path.exists(train_data_path):
        print(f"⚠️  Training data not found: {train_data_path}")
        return 0
    
    train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r')
    
    if verbose:
        print(f"📊 Dataset: {len(train_data):,} tokens")
        print(f"🎯 Target: {num_samples} memory chunks")
        print(f"📏 Chunk size: {chunk_size} tokens (~{chunk_size * 0.75:.0f} words)")
        print(f"👣 Stride: {stride} tokens (overlap: {(1 - stride/chunk_size)*100:.0f}%)")
    
    if verbose:
        print(f"🔍 Memory system type: {type(memory_system).__name__}")
        if hasattr(memory_system, 'longterm'):
            print(f"📊 Long-term capacity: {memory_system.longterm.capacity}")
            print(f"📊 Current size: {memory_system.longterm.size.item()}")
        else:
            print(f"⚠️  No longterm attribute found!")
            return 0
    
    # Sample positions using dense sequential scanning from random starting points
    # Strategy: Pick random places in dataset, then extract overlapping chunks
    # This ensures repeated patterns appear multiple times at different contexts
    max_start = len(train_data) - chunk_size
    if max_start <= 0:
        print("⚠️  Dataset too small for chunking")
        return 0
    
    # Dense sampling approach: start from random positions, scan sequentially
    num_sequences = max(1, num_samples // 100)  # ~500 random starting points for 50k samples
    chunks_per_sequence = num_samples // num_sequences
    
    positions = []
    for _ in range(num_sequences):
        # Pick random starting point
        start_pos = np.random.randint(0, max_start - chunks_per_sequence * stride)
        # Extract overlapping chunks from this starting point
        for i in range(chunks_per_sequence):
            pos = start_pos + i * stride
            if pos + chunk_size <= len(train_data):
                positions.append(pos)
    
    # Add some purely random positions for diversity
    random_extras = num_samples - len(positions)
    if random_extras > 0:
        extra_positions = np.random.choice(max_start, size=min(random_extras, max_start), replace=False)
        positions.extend(extra_positions.tolist())
    
    positions = positions[:num_samples]  # Ensure exact count
    np.random.shuffle(positions)  # Shuffle to avoid sequential order
    
    # TWO-PASS SMART SAMPLING (CPU-only, no model needed)
    # Pass 1: Build token co-occurrence graph to find "hub" tokens
    # Pass 2: Prioritize chunks containing highly-connected tokens
    
    if verbose:
        print(f"\n� PASS 1: Analyzing token connectivity (CPU)...")
    
    from collections import defaultdict
    token_connections = defaultdict(int)  # Count of unique co-occurring tokens
    
    # Sample 10% of positions for connectivity analysis
    sample_size = min(len(positions) // 10, 10000)
    sample_positions = np.random.choice(positions, size=sample_size, replace=False)
    
    for pos in tqdm(sample_positions, desc="Building co-occurrence graph", disable=not verbose):
        chunk = train_data[pos:pos+chunk_size]
        if len(chunk) < chunk_size:
            continue
        
        # Count unique tokens in this chunk
        unique_tokens = set(chunk.tolist())
        
        # Each token gets credit for co-occurring with others
        for tok in unique_tokens:
            token_connections[tok] += len(unique_tokens) - 1  # Connected to N-1 other unique tokens
    
    # Find hub tokens (top 20% by connectivity)
    if len(token_connections) > 0:
        connection_values = list(token_connections.values())
        hub_threshold = np.percentile(connection_values, 80)
        hub_tokens = {tok for tok, score in token_connections.items() if score >= hub_threshold}
        
        if verbose:
            print(f"   Found {len(hub_tokens)} hub tokens (threshold: {hub_threshold:.0f} connections)")
            top_hubs = sorted(token_connections.items(), key=lambda x: -x[1])[:5]
            print(f"   Top 5 hubs: {[(tok, cnt) for tok, cnt in top_hubs]}")
    else:
        hub_tokens = set()
        if verbose:
            print("   No hub tokens found, will use random ordering")
    
    # Pass 2: Score all chunks by hub token presence
    if verbose:
        print(f"\n📊 PASS 2: Scoring chunks by hub presence...")
    
    chunk_scores = []
    for pos in tqdm(positions, desc="Scoring chunks", disable=not verbose):
        chunk = train_data[pos:pos+chunk_size]
        if len(chunk) < chunk_size:
            chunk_scores.append((0, pos))
            continue
        
        # Score = sum of connection counts for tokens in chunk
        score = sum(token_connections.get(tok, 0) for tok in chunk.tolist())
        chunk_scores.append((score, pos))
    
    # Sort by score (highest first) - prioritize well-connected chunks
    chunk_scores.sort(reverse=True)
    positions = [pos for score, pos in chunk_scores]
    
    if verbose:
        if len(chunk_scores) > 0:
            print(f"   Scored {len(chunk_scores)} chunks")
            print(f"   Score range: {chunk_scores[-1][0]:.0f} (lowest) to {chunk_scores[0][0]:.0f} (highest)")
            print(f"   Will process high-connectivity chunks first")
    
    if verbose:
        print(f"\n🔄 Encoding & inserting chunks with STREAMING mode (low RAM)...")
    
    model.eval()
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=dtype)
    
    # STREAMING MODE: Encode + insert in small batches to keep RAM low
    streaming_batch_size = 256  # Encode 256 chunks at a time
    batch_embeddings = []
    memories_added = 0
    
    from hyperbolic_memory import PoincareManifold
    if hasattr(memory_system, 'longterm'):
        lt_mem = memory_system.longterm
        poincare = PoincareManifold(dim=lt_mem.memory_dim, c=1.0)
    else:
        if verbose:
            print("⚠️  No longterm memory - skipping")
        return 0
    
    with torch.no_grad():
        iterator = tqdm(positions, desc="Streaming encode+insert") if verbose else positions
        
        for idx, pos in enumerate(iterator):
            # Extract chunk
            chunk = train_data[pos:pos+chunk_size]
            if len(chunk) < chunk_size:
                continue
            
            # Convert to tensor
            x = torch.from_numpy(chunk.astype(np.int64)).to(device)
            
            # Encode chunk through model
            with ctx:
                # Get token embeddings
                tok_emb = model.encoder.wte(x.unsqueeze(0))
                pos_emb = model.encoder.wpe(torch.arange(chunk_size, device=device))
                h = model.encoder.drop(tok_emb + pos_emb)
                
                # Pass through reflex blocks
                if hasattr(model.reflex, 'blocks') and len(model.reflex.blocks) > 0:
                    num_encode_blocks = min(4, len(model.reflex.blocks))
                    for i in range(num_encode_blocks):
                        h = model.reflex.blocks[i](h)
                
                # Pool to chunk embedding
                chunk_embedding = h.mean(dim=1).squeeze(0)
                batch_embeddings.append(chunk_embedding.cpu())  # CPU immediately
            
            # Insert batch when full
            if len(batch_embeddings) >= streaming_batch_size:
                embeddings_tensor = torch.stack(batch_embeddings)  # [256, D]
                
                # Insert with AGGRESSIVE memory limits
                added = lt_mem.add_nodes_batch_gpu(
                    embeddings_tensor,
                    poincare,
                    batch_size=128,  # Small GPU chunks
                    max_existing=500  # VERY AGGRESSIVE: Only use 500 existing for k-NN
                )
                memories_added += added
                
                # Clear batch to free RAM
                batch_embeddings = []
                
                # Force garbage collection every 10 batches
                if memories_added % 2560 == 0:
                    import gc
                    gc.collect()
        
        # Insert remaining batch
        if len(batch_embeddings) > 0:
            embeddings_tensor = torch.stack(batch_embeddings)
            added = lt_mem.add_nodes_batch_gpu(
                embeddings_tensor,
                poincare,
                batch_size=128,
                max_existing=500
            )
            memories_added += added
        
        if verbose:
            print(f"✅ Added {memories_added} memories")
            print(f"📊 Hot tier: {lt_mem.size.item()} / {lt_mem.capacity}")
    
    if verbose:
        print(f"\n✅ Preloaded {memories_added} memories into long-term storage")
        if hasattr(memory_system, 'longterm'):
            lt_mem = memory_system.longterm
            print(f"📊 Memory status: {lt_mem.size.item()} hot / {lt_mem.capacity} capacity")
            if hasattr(lt_mem, 'use_disk') and lt_mem.use_disk:
                disk_count = len(lt_mem.disk_index) if hasattr(lt_mem, 'disk_index') else 0
                print(f"📊 Disk-backed: {disk_count} memories on disk")
    
    # ═══════════════════════════════════════════════════════════════════════
    # FLUSH TO DISK if we have too many memories for edge building
    # ═══════════════════════════════════════════════════════════════════════
    if memories_added > 1 and hasattr(memory_system, 'longterm'):
        lt_mem = memory_system.longterm
        
        # If we have more than 5K memories, flush most to disk before edge building
        # This prevents RAM explosion when computing distance matrices
        if lt_mem.size.item() > 5000 and hasattr(lt_mem, 'flush_to_disk'):
            if verbose:
                print(f"\n💾 FLUSHING TO DISK (too many memories for edge building)")
                print(f"   Keeping 1000 hot, flushing {lt_mem.size.item() - 1000} to disk")
            
            num_to_flush = lt_mem.size.item() - 1000
            flushed = lt_mem.flush_to_disk(num_to_flush=num_to_flush)
            
            if verbose:
                print(f"✅ Flushed {flushed} memories to disk")
                print(f"📊 Hot set: {lt_mem.size.item()} memories")
                # With DiskBackedTensor, all memories are accessible (no separate disk_index)
                if hasattr(lt_mem, 'embeddings') and hasattr(lt_mem.embeddings, '_actual_size'):
                    print(f"📊 Total accessible: {lt_mem.embeddings._actual_size} memories (disk-backed)")

    
    # ═══════════════════════════════════════════════════════════════════════
    # BUILD ADDITIONAL SEMANTIC EDGES between preloaded memories (hot tier only)
    # ═══════════════════════════════════════════════════════════════════════
    if memories_added > 1 and hasattr(memory_system, 'longterm'):
        lt_mem = memory_system.longterm
        # Only build edges for hot tier (disk tier uses previews for approximate search)
        n_hot = lt_mem.size.item()
        
        if verbose:
            print(f"\n🕸️  BUILDING SEMANTIC EDGES (hot tier only: {n_hot} memories)")
            print("=" * 70)
        
        try:
            _build_preload_edges_cpu(memory_system, n_hot, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"⚠️  Could not build edges: {e}")
                import traceback
                traceback.print_exc()
    
    if verbose:
        print("=" * 70 + "\n")
    
    model.train()
    
    # Save to cache for future runs
    if memories_added > 0:
        save_preload_cache(cache_path, memory_system, verbose=verbose)
    
    return memories_added


def _build_preload_edges_cpu(memory_system, n_memories, verbose=True):
    """
    Build graph edges between preloaded memories (all on CPU).
    
    Creates:
    - Proximity edges: k-NN in Euclidean space
    - Semantic edges: High cosine similarity
    """
    import torch.nn.functional as F
    
    lt_mem = memory_system.longterm
    
    if n_memories < 2:
        return
    
    # Get all memory embeddings (on CPU)
    embeddings = lt_mem.embeddings[:n_memories]  # [N, dim]
    
    # Check if we have adjacency structure
    if not hasattr(lt_mem, 'adjacency'):
        if verbose:
            print("⚠️  Long-term memory has no adjacency matrix - creating one")
        # Create adjacency structures sized for n_memories
        k = getattr(memory_system, 'k_neighbors', 20)
        lt_mem.adjacency = torch.full((n_memories, k), -1, dtype=torch.long, device='cpu')
        lt_mem.edge_weights = torch.zeros(n_memories, k, device='cpu')
        if hasattr(lt_mem, 'num_edge_types'):
            lt_mem.edge_types = torch.zeros(n_memories, k, lt_mem.num_edge_types, device='cpu')
    elif lt_mem.adjacency.size(0) < n_memories:
        # Expand adjacency to fit new memories
        if verbose:
            print(f"📈 Expanding adjacency from {lt_mem.adjacency.size(0)} to {n_memories}")
        k = lt_mem.adjacency.size(1)
        new_adjacency = torch.full((n_memories, k), -1, dtype=torch.long, device='cpu')
        new_edge_weights = torch.zeros(n_memories, k, device='cpu')
        
        # Copy existing
        old_size = lt_mem.adjacency.size(0)
        new_adjacency[:old_size] = lt_mem.adjacency
        new_edge_weights[:old_size] = lt_mem.edge_weights
        
        lt_mem.adjacency = new_adjacency
        lt_mem.edge_weights = new_edge_weights
        
        if hasattr(lt_mem, 'edge_types') and lt_mem.edge_types is not None:
            new_edge_types = torch.zeros(n_memories, k, lt_mem.num_edge_types, device='cpu')
            new_edge_types[:old_size] = lt_mem.edge_types
            lt_mem.edge_types = new_edge_types
        
        # Also expand other graph-related tensors
        if hasattr(lt_mem, 'rewards') and lt_mem.rewards.size(0) < n_memories:
            new_rewards = torch.zeros(n_memories, device='cpu')
            new_rewards[:old_size] = lt_mem.rewards
            lt_mem.rewards = new_rewards
        
        if hasattr(lt_mem, 'age') and lt_mem.age.size(0) < n_memories:
            new_age = torch.zeros(n_memories, device='cpu')
            new_age[:old_size] = lt_mem.age
            lt_mem.age = new_age
        
        if hasattr(lt_mem, 'access') and lt_mem.access.size(0) < n_memories:
            new_access = torch.zeros(n_memories, device='cpu')
            new_access[:old_size] = lt_mem.access
            lt_mem.access = new_access
        
        if hasattr(lt_mem, 'edge_traversal_count') and lt_mem.edge_traversal_count.size(0) < n_memories:
            new_etc = torch.zeros(n_memories, k, device='cpu')
            new_etc[:old_size] = lt_mem.edge_traversal_count
            lt_mem.edge_traversal_count = new_etc
        
        if hasattr(lt_mem, 'edge_success_rate') and lt_mem.edge_success_rate.size(0) < n_memories:
            new_esr = torch.zeros(n_memories, k, device='cpu')
            new_esr[:old_size] = lt_mem.edge_success_rate
            lt_mem.edge_success_rate = new_esr
        
        if hasattr(lt_mem, 'cluster_ids') and lt_mem.cluster_ids.size(0) < n_memories:
            new_clusters = torch.full((n_memories,), -1, dtype=torch.long, device='cpu')
            new_clusters[:old_size] = lt_mem.cluster_ids
            lt_mem.cluster_ids = new_clusters
    
    k = lt_mem.adjacency.size(1)  # k neighbors
    k_actual = min(k, n_memories - 1)
    
    if verbose:
        print(f"📊 Building edges for {n_memories} memories (k={k_actual})")
        print(f"⚠️  This will compute ~{(n_memories * n_memories) / 1e6:.1f}M distances - may take several minutes...")
    
    # 1. PROXIMITY EDGES: k-NN in Euclidean space
    # Batched distance computation for speed (process in chunks to avoid OOM)
    with torch.no_grad():
        edges_added = 0
        batch_size = 1000  # Process 1000 nodes at a time
        
        from tqdm import tqdm
        for batch_start in tqdm(range(0, n_memories, batch_size), 
                                desc="Building k-NN edges",
                                disable=not verbose):
            batch_end = min(batch_start + batch_size, n_memories)
            batch_embeddings = embeddings[batch_start:batch_end]  # [batch, dim]
            
            # Compute distances from batch to ALL nodes
            # Using batched norm: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
            batch_norms = (batch_embeddings ** 2).sum(dim=1, keepdim=True)  # [batch, 1]
            all_norms = (embeddings ** 2).sum(dim=1, keepdim=True).T  # [1, N]
            dot_products = torch.mm(batch_embeddings, embeddings.T)  # [batch, N]
            dists = batch_norms + all_norms - 2 * dot_products  # [batch, N]
            dists = torch.sqrt(torch.clamp(dists, min=1e-8))  # Numerical stability
            
            # For each node in batch, get k nearest neighbors
            for local_i, i in enumerate(range(batch_start, batch_end)):
                node_dists = dists[local_i]  # [N]
                
                # Get k+1 nearest (including self), then exclude self
                _, indices = node_dists.topk(k_actual + 1, largest=False)
                neighbors = indices[1:]  # Skip self (index 0)
                neighbor_dists = node_dists[neighbors]
                
                # Store in adjacency matrix
                lt_mem.adjacency[i, :k_actual] = neighbors
                lt_mem.edge_weights[i, :k_actual] = neighbor_dists
                
                # Mark as proximity edges (type 0)
                if hasattr(lt_mem, 'edge_types'):
                    lt_mem.edge_types[i, :k_actual, 0] = 1.0
                
                edges_added += k_actual
    
    if verbose:
        print(f"✅ Added {edges_added} proximity edges (k-NN)")
    
    # 2. SEMANTIC EDGES: High cosine similarity (>0.7)
    # Add semantic edges in batches for speed
    semantic_edges_added = 0
    similarity_threshold = 0.7
    
    if verbose:
        print(f"🔍 Adding semantic edges (cosine similarity > {similarity_threshold})...")
    
    with torch.no_grad():
        # Normalize embeddings for cosine similarity
        emb_norm = F.normalize(embeddings, p=2, dim=1)  # [N, dim]
        
        # Process in batches
        for batch_start in tqdm(range(0, n_memories, batch_size),
                                desc="Adding semantic edges",
                                disable=not verbose):
            batch_end = min(batch_start + batch_size, n_memories)
            batch_norm = emb_norm[batch_start:batch_end]
            
            # Compute similarities for this batch
            similarity = torch.mm(batch_norm, emb_norm.T)  # [batch, N]
            
            # For each node in batch, find high-similarity nodes
            for local_i, i in enumerate(range(batch_start, batch_end)):
                # Find high-similarity nodes (>threshold, excluding self)
                node_similarities = similarity[local_i]
                high_sim_mask = (node_similarities > similarity_threshold) & (torch.arange(n_memories) != i)
                high_sim_indices = high_sim_mask.nonzero(as_tuple=True)[0]
                
                if len(high_sim_indices) > 0:
                    # Check which are NOT already neighbors
                    current_neighbors = lt_mem.adjacency[i, :k_actual]
                    for j in high_sim_indices:
                        if j not in current_neighbors:
                            # Replace furthest neighbor with this high-similarity node
                            furthest_idx = lt_mem.edge_weights[i, :k_actual].argmax()
                            if node_similarities[j] > similarity_threshold:  # Double-check
                                lt_mem.adjacency[i, furthest_idx] = j
                                lt_mem.edge_weights[i, furthest_idx] = 1.0 - node_similarities[j]  # Lower is better
                                
                                # Mark as semantic edge (type 2)
                                if hasattr(lt_mem, 'edge_types'):
                                    lt_mem.edge_types[i, furthest_idx, 0] = 0.0  # Remove proximity
                                    lt_mem.edge_types[i, furthest_idx, 2] = 1.0  # Add semantic
                                
                                semantic_edges_added += 1
                                break  # Only replace one neighbor per high-sim node
    
    if verbose:
        print(f"✅ Added {semantic_edges_added} semantic edges (high similarity)")
        print(f"📊 Total: {edges_added + semantic_edges_added} edges in preloaded graph")


def should_preload_memories(model, min_memories=50):
    """
    Check if we should preload memories (i.e., memory system exists but is empty).
    
    Args:
        model: The model to check
        min_memories: Minimum memories needed to skip preloading
        
    Returns:
        bool: True if we should preload
    """
    if not hasattr(model, 'reflex') or not hasattr(model.reflex, 'memory_retrieval'):
        return False
    
    memory_system = model.reflex.memory_retrieval
    
    if hasattr(memory_system, 'longterm'):
        current_size = memory_system.longterm.size.item()
        return current_size < min_memories
    
    return False
