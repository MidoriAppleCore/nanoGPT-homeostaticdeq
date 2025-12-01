# Train Gray Box DEQ - UNIFIED QUANTUM SOLVER
# Combines: Gauge Symmetry + Spontaneous Breaking + Tunneling + Annealing + Path Integral

out_dir = 'out-shakespeare-char-graybox-unified-quantum'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'graybox-deq-unified-quantum'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 256

# Gray Box DEQ architecture
n_layer = 2
n_head = 6
n_embd = 384
dropout = 0.2

# Standard DEQ parameters
deq_max_iter = 15  # Not used directly (quantum solver has phases)
deq_tol = 5e-3
anderson_accel = False  # Not needed with quantum solver
spectral_norm = False

# UNIFIED QUANTUM SOLVER
quantum_solver = True  # Enable the ultimate physics-based solver

# Quantum parameters
num_gauge_orbits = 3          # Sample 3 "phrasings" (gauge symmetry)
symmetry_breaking_iters = 3   # Phase 1: Choose mode/style (spontaneous breaking)
refinement_iters = 5          # Phase 2: Refine & tunnel (total = 8 iters per orbit)

enable_tunneling = True       # Allow quantum jumps between basins
tunnel_threshold = 0.95       # Tunnel if residual > 95% of previous

temperature_schedule = "exponential"  # T decay schedule
T_init = 0.5                  # Hot start (explore)
T_final = 0.01                # Cold end (converge)

# Training
learning_rate = 1e-3
max_iters = 1000000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

compile = False
