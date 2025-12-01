"""
Train TinyStories with Cache-Style Two-Tier Memory

MEMORY ARCHITECTURE (like CPU cache hierarchy):
- Working memory (VRAM) = L1 CACHE: 20 items, 50x LR, 20% decay/step
- Long-term memory (CPU) = RAM: 2000 items, 0.1x LR, 0.1% decay/step
- Promotion every 50 steps (consolidation)

Working memory is TINY and FAST - only what you're thinking about RIGHT NOW
Long-term is LARGE and SLOW - accumulated knowledge from all training
"""

# I/O
out_dir = 'out-tinystories-hybrid-memory'
eval_interval = 250
log_interval = 10
eval_iters = 50
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# wandb logging
wandb_log = False
wandb_project = 'tinystories'
wandb_run_name = 'hybrid-memory'

# data
dataset = 'tinystories'
gradient_accumulation_steps = 4
batch_size = 8  # Reduced from 32 (save memory!)
block_size = 128  # Reduced from 256 (save memory!)

# model
n_layer = 2  # Reflex blocks
n_head = 4  # Reduced from 6
n_embd = 256  # Reduced from 384 (save memory!)
dropout = 0.1
bias = True

# DEQ-specific
deq_max_iter = 30
deq_tol = 1e-3
anderson_accel = True
spectral_norm = False
hamiltonian = True
quantum_solver = False

# Pauli Exclusion (anti-stuttering)
lambda_pauli = 0.5

# ══════════════════════════════════════════════════════════════
# CACHE-STYLE TWO-TIER MEMORY SYSTEM
# ══════════════════════════════════════════════════════════════
use_memory_manifold = True
memory_mode = 'hybrid'  # Dynamic formation (starts empty!)

# Memory dimensions
memory_dim = 256  # Match n_embd (reduced!)
memory_k = 8  # Fewer neighbors (reduced from 16)
memory_alpha = 0.1  # Injection strength
memory_curvature = 1.0  # Hyperbolic curvature

# WORKING MEMORY = L1 CACHE (Immediate context only!)
working_memory_capacity = 15      # Even smaller for GPU memory
working_memory_decay = 0.80       # AGGRESSIVE - 20% decay per step

# LONG-TERM MEMORY = RAM (Background knowledge)
longterm_memory_capacity = 500   # Reduced from 2000
longterm_memory_decay = 0.999     # PERSISTENT

# CONSOLIDATION
memory_promotion_threshold = 0.4  # Lower threshold
memory_promotion_interval = 50    # More frequent

# Learning rates (set automatically in model.configure_optimizers()):
#   - Working memory (cache): 50x base LR = 0.05
#   - Long-term memory (RAM): 0.1x base LR = 1e-4
#   - Regular params: 1x base LR = 1e-3
# ══════════════════════════════════════════════════════════════

# adamw optimizer
learning_rate = 1e-3
max_iters = 10000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 10000
min_lr = 1e-4

# system
device = 'cuda'
dtype = 'bfloat16'
compile = False
