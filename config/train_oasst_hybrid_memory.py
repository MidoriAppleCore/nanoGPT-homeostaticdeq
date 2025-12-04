# Train on OpenAssistant (OASST1) conversational dataset with hybrid memory
# This is a chatbot/agent training config - dialogue and instruction following
# 🚀 OPTIMIZED FOR SPEED - 2-3x faster than default config
#
# ═══════════════════════════════════════════════════════════════════════════
# 🖕 FIRST PRINCIPLE: GNN IS NON-NEGOTIABLE 🖕
# ═══════════════════════════════════════════════════════════════════════════
# The GNN enables GRAPH NAVIGATION LEARNING. Without it, the system is just
# doing dumb cosine similarity search. The dual memory (preload + online) ONLY
# works if the DEQ can learn intelligent paths through the graph structure.
#
# We WILL make this fit on 6GB consumer GPU. This is the middle finger to
# "AI needs $10K hardware" gatekeeping. Intelligence >> compute.
#
# Optimizations to make Micro-GNN fit:
# - k=12 neighbors (not k=20) → 40% less message passing
# - GNN hidden_dim=256 (not 512) → 50% fewer parameters  
# - Gradient checkpointing → 30-40% less activation memory
# - Batch size=2 → minimal activation footprint
# - DEQ on 6GB is the ENTIRE POINT of this architecture
# ═══════════════════════════════════════════════════════════════════════════
# i mean i think anyone who looks at this repo is going to tell i heavily used LLMs on it but i like this so i'll keep it

# I/O
out_dir = 'out-oasst-hybrid-memory'
log_interval = 10  # Log every 10 iterations for better monitoring
eval_iters = 1  # Just 1 batch - fast validation, still gives signal
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # Start fresh training

# SPEED OPTIMIZATION FLAGS
fast_mode = True  # Reduce expensive monitoring/plotting
plot_interval = 100  # Only plot every 100 iters (was every 20)

# Data
dataset = 'oasst'
gradient_accumulation_steps = 6  # RESTORED: 6 steps (batch=1 × 6 accum = effective 6)
batch_size = 1  # EMERGENCY: 1 for 6GB OOM - AGENT MODE (sequential learning like a human!)
block_size = 256  # REDUCED from 384 - 2/3 context, faster! Still ~200 words

# Model - SCALED INTELLIGENCE ARCHITECTURE
n_layer = 12  # Full 12 layers for encoder
n_head = 8   # 8 heads for 768d (96 dim per head)
n_embd = 768  # Standard dimension for encoder/reflex/memory
dropout = 0.1
bias = False  # No bias for better scaling

# 🚀 SPLIT-DIMENSION DEQ (operates in higher-dimensional space!)
deq_n_embd = 2048  # DEQ thinks in 2048d space (3× smarter!)
deq_n_head = 32    # 32 attention heads (64 dim per head)

# DEQ-specific parameters - ENHANCED INTELLIGENCE
use_deq = True
deq_max_iter = 12  # More iterations for deeper thinking with 200K memory world
deq_tol = 1e-3  # Tighter tolerance (DEQ is more capable)
anderson_accel = True
spectral_norm = True
deq_prenorm = True  # Pre-normalization (already in enhanced DEQ)

# Hamiltonian Dynamics
hamiltonian = True

# Unified Quantum Solver
quantum_solver = False
num_gauge_orbits = 3
symmetry_breaking_iters = 2  # OPTIMIZED: Reduced from 3 - faster gauge search
refinement_iters = 10  # OPTIMIZED: Reduced from 15 - adequate for early training
enable_tunneling = True
tunnel_threshold = 0.95
num_tunnel_rays = 3  # REDUCED: 8 for 6GB GPU (fewer parallel rays = less memory)
temperature_schedule = "exponential"
T_init = 0.1
T_final = 0.01

# Hybrid Memory System - Three-tier graph-structured architecture
use_memory_manifold = True
memory_mode = 'hybrid'  # Three-tier: working (GPU) + buffer (GPU) + long-term (CPU)

# Graph memory parameters - SCALED FOR INTELLIGENCE
memory_k = 12  # k-NN neighbors (MUST match preload cache! Cache has k=12)
gnn_hidden_dim = 128  # Ultra-Micro-GNN (128 for 6GB GPU)
enable_gnn = True  # ENABLED: Ultra-Micro-GNN with gradient checkpointing (~80-100MB VRAM)

# HIERARCHICAL RETRIEVAL & DEQ RE-QUERYING (New features to escape meta-token basin)
use_hierarchical_retrieval = True  # Use cluster→node 2-stage retrieval (better scaling)
deq_requery_memory = True  # Enable dynamic memory navigation during DEQ solving
deq_requery_interval = 2  # Re-query every 2 DEQ iters (more frequent exploration in 200K space)

# Working memory: L1 cache-like, high plasticity (fast learning, fast decay)
working_memory_capacity = 20  # Small active memory on GPU
working_learning_rate_multiplier = 10.0  # Fast plasticity for active dialogue
working_decay_rate = 0.85  # 15% decay per step - volatile short-term

# Consolidation buffer: Hippocampus-like (persists across batches)
consolidation_buffer_size = 100  # Staging area before long-term consolidation

# Long-term memory: Consolidated knowledge with graph structure
longterm_memory_capacity = 50000  # Hot tier: 50K nodes in CPU RAM (~1.5 GB with graph structure)
longterm_disk_path = None  # Will be set to {out_dir}/graph_memory.pt automatically (or use --memory-path)
longterm_max_disk_size = 500000  # Maximum memories (hot + cold disk storage) - 500K for full dataset coverage
longterm_learning_rate_multiplier = 0.1  # Slow consolidation
longterm_decay_rate = 0.999  # 0.1% decay - persistent memory

# Memory dynamics
consolidation_interval = 50  # HOMEOSTATIC: Let semantic/balancer triggers dominate
consolidation_threshold = 0.05  # LOWERED from 0.1 - easier promotion in early training
reconsolidation_threshold = 100  # Bring back if accessed heavily

# Sampling/evaluation - SPEED OPTIMIZATIONS
eval_interval = 200  # OPTIMIZED: Sample every 200 iterations (was 100) - less overhead
num_samples = 30  # OPTIMIZED: Reduced from 50 - faster generation, still good feedback

# Memory manifold - MATCH SCALED ARCHITECTURE
memory_manifold_dim = 768  # Match n_embd (encoder dimension)
memory_dim = 768  # CRITICAL: Must match n_embd to avoid dimension mismatch
hyperbolic_curvature = 1.0

# 🗄️ Memory Preloading (seed memory from dataset before training)
# PMI-BASED SEMANTIC PRELOADING (Church & Hanks 1990)
# Scans entire dataset with intelligent sampling to build semantically-rich initial graph
# Dataset: ~7.9M tokens → ~493K possible chunks (32 tokens, stride 16)
# Strategy: PMI scoring prioritizes chunks with statistically significant associations
# Benefits: Better initial graph structure, highways form faster, improved retrieval
preload_num_samples = 200000  # � FULL SCALE: 200K memories (~40% dataset coverage, ~6GB disk)
                              # Each memory is 32 tokens (~25 words) with 50% overlap
                              # Covers most important semantic patterns in dataset
preload_chunk_size = 32       # Tokens per chunk (~25 words of context)

# 🧠 Memory Navigation Rewards (enable dopamine system)
enable_nav_rewards = True

# 🎓 Supervised Memory Navigation (imitation learning - watch expert paths!)
# NOTE: Automatically activates after ~50 memories exist (around iter 50-100)
# Phase 1 (iters 0-50): Bootstrap memories via exploration
# Phase 2 (iters 50-2000): Teacher-force navigation to oracle memories
# Phase 3 (iters 2000+): Pure exploration and generalization
enable_supervised_nav = True  # Teacher-force memory retrieval to match ground truth
supervised_nav_iters = 2000  # Decay teacher forcing over first 2000 iterations
supervised_lookahead = 4  # Look ahead 4 tokens to find target embedding
supervised_nav_weight = 0.5  # Weight for supervised reward (vs exploration reward)

# Adamw optimizer
learning_rate = 4e-5  # Match your current training (lr=4.00e-05)
max_iters = 600000  # Indefinite training - let it run until stopped00000  # Indefinite training - let it run until stopped
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 50000
min_lr = 3e-5

# System
device = 'cuda'
dtype = 'bfloat16' if hasattr(__import__('torch').cuda, 'is_bf16_supported') and __import__('torch').cuda.is_bf16_supported() else 'float16'
compile = False  # Disabled due to nix ldconfig issue

# Logging
wandb_log = False
wandb_project = 'nanoGPT-oasst'
wandb_run_name = 'oasst-hybrid-memory'
