# Train on OpenAssistant (OASST1) conversational dataset with hybrid memory
# This is a chatbot/agent training config - dialogue and instruction following

# I/O
out_dir = 'out-oasst-hybrid-memory'
log_interval = 1  # Log every iteration to see progress
eval_iters = 1  # Just 1 batch - fast validation, still gives signal
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# Data
dataset = 'oasst'
gradient_accumulation_steps = 4  # Effective batch size = 4 * 8 = 32 (better GPU utilization)
batch_size = 8  # INCREASED from 4 - better throughput
block_size = 256  # REDUCED from 384 - 2/3 context, faster! Still ~200 words

# Model - Balanced for speed + quality (similar to TinyStories size)
n_layer = 3  # REDUCED from 4 - sweet spot for speed
n_head = 8  # Must divide n_embd evenly (256/8=32 dim per head)
n_embd = 256  # REDUCED from 384 - match TinyStories, 2-3x faster!
dropout = 0.1
bias = True  # Use bias in LayerNorm and Linear layers

# DEQ-specific parameters
deq_max_iter = 100
deq_tol = 1e-3
anderson_accel = True
spectral_norm = True

# Hamiltonian Dynamics
hamiltonian = True

# Unified Quantum Solver
quantum_solver = True
num_gauge_orbits = 3
symmetry_breaking_iters = 3
refinement_iters = 15
enable_tunneling = True
tunnel_threshold = 0.95
num_tunnel_rays = 32
temperature_schedule = "exponential"
T_init = 0.1
T_final = 0.01

# Hybrid Memory System - Three-tier architecture
use_memory_manifold = True
memory_mode = 'hybrid'  # Three-tier: working (GPU) + buffer (GPU) + long-term (CPU)

# Working memory: L1 cache-like, high plasticity (fast learning, fast decay)
working_capacity = 20  # Match your current training (was showing W:20)
working_learning_rate_multiplier = 10.0  # Fast plasticity for active dialogue
working_decay_rate = 0.85  # 15% decay per step - volatile short-term

# Consolidation buffer: Hippocampus-like (persists across batches, no decay)
consolidation_buffer_size = 300  # INCREASED - let it accumulate more before sleep
# Buffer gets promoted to long-term during sleep cycle, then resets

# Long-term memory: Consolidated knowledge, slower updates
longterm_capacity = 20000  # UPGRADED: 10x capacity for general LM (300MB RAM, negligible cost)
longterm_learning_rate_multiplier = 0.1  # Slow consolidation
longterm_decay_rate = 0.999  # 0.1% decay - persistent memory
longterm_disk_path = 'out-oasst-hybrid-memory/longterm_disk'  # Disk-backed storage (UNLIMITED!)

# Memory dynamics
consolidation_interval = 1  # Sleep every iteration (matches your current setup)
consolidation_threshold = 0.05  # LOWERED from 0.1 - easier promotion in early training
reconsolidation_threshold = 100  # Bring back if accessed heavily (attention-weighted)

# Sampling/evaluation
eval_interval = 100  # Sample every 100 iterations (was 50 - reduced overhead)
num_samples = 50  # Generate 50 tokens max (fast feedback)

# Memory manifold
memory_manifold_dim = 256  # REDUCED from 384 - match n_embd
hyperbolic_curvature = 1.0

# Adamw optimizer
learning_rate = 4e-5  # Match your current training (lr=4.00e-05)
max_iters = 600000  # Indefinite training - let it run until stopped
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
