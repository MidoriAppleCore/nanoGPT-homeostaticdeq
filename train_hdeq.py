"""
Training script for Gray Box DEQ Language Model.
This is a thin wrapper around train.py that uses model_graybox instead of model.

Usage is identical to train.py:
$ python train_graybox.py config/train_shakespeare_char_graybox.py
$ python train_graybox.py --batch_size=32 --compile=False
"""

import os
import time
import csv
import math
import pickle
from contextlib import nullcontext
from collections import defaultdict
import threading
from queue import Queue
import warnings
import psutil  # For memory monitoring

# Silence C++ stack trace warnings from gradient checkpointing
os.environ['TORCH_DISABLE_ADDR2LINE'] = '1'

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Silence matplotlib/numpy warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

# NOTE: Even with Agg backend, matplotlib is NOT thread-safe when multiple threads
# create figures concurrently. This causes C++ aborts like "terminate called without
# an active exception". All plt.* calls must be protected by BackgroundWorker._matplotlib_lock

# Import from model_graybox instead of model
from model_hdeq import GPTConfig, GPT, compute_pauli_exclusion_loss, HamiltonianOperator, DEQOperator, HomeostaticBalancer

# Memory navigation rewards (dopamine for graph exploration)
from memory_navigation_rewards import MemoryNavigationRewards, MemoryPathReinforcement

# Memory preloading (seed from dataset)
from preload_memories import preload_memories_from_dataset, should_preload_memories

# Homeostatic monitoring system
from homeostatic_monitor import HomeostaticMonitor

# Diagnostic reporting system
from diagnostic_report import DiagnosticReporter

# Gradient flow visualization (topological debugging)
from debug_gradient_flow import visualize_gradient_flow

# For phase space visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️  matplotlib/sklearn not available - phase space visualization disabled")

# Check for --visualizations-off flag (parsed later, but we can peek early)
import sys
if '--visualizations-off' in sys.argv:
    VISUALIZATION_AVAILABLE = False
    print("🚫 Visualizations disabled via --visualizations-off flag")

# -----------------------------------------------------------------------------
# Background Task Worker
# -----------------------------------------------------------------------------

class BackgroundWorker:
    """Non-blocking background worker for visualization/diagnostics"""
    
    # Matplotlib is NOT thread-safe, even with Agg backend
    # Use global lock to prevent simultaneous figure creation
    _matplotlib_lock = threading.Lock()
    
    def __init__(self, max_queue_size=3):
        self.queue = Queue(maxsize=max_queue_size)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self.active_tasks = 0
        self._lock = threading.Lock()
        self.completed_count = 0
        self.failed_count = 0
        
    def _worker(self):
        """Background thread that processes queued tasks"""
        while True:
            task_name, func, args, kwargs = self.queue.get()
            try:
                # Silent execution - graphs speak for themselves
                func(*args, **kwargs)
                with self._lock:
                    self.completed_count += 1
            except Exception as e:
                with self._lock:
                    self.failed_count += 1
                # Only log critical failures
                if "CRITICAL" in str(e) or "CUDA" in str(e):
                    print(f"  ⚠️  [{task_name}] CRITICAL: {e}")
            finally:
                with self._lock:
                    self.active_tasks -= 1
                self.queue.task_done()
    
    def submit(self, task_name, func, *args, **kwargs):
        """Submit a task to run in background (non-blocking)"""
        try:
            with self._lock:
                self.active_tasks += 1
            self.queue.put_nowait((task_name, func, args, kwargs))
            return True
        except:
            # Queue full, skip this task
            with self._lock:
                self.active_tasks -= 1
            return False
    
    def get_status(self):
        """Get current queue status"""
        with self._lock:
            return {
                'active': self.active_tasks,
                'queued': self.queue.qsize(),
                'completed': self.completed_count,
                'failed': self.failed_count
            }

# -----------------------------------------------------------------------------
# Profiling utilities
# -----------------------------------------------------------------------------

class TrainingProfiler:
    """Lightweight profiler for tracking time spent in different training components"""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timers = defaultdict(list)
        self.current_starts = {}
        self.iter_start = None
        
    def start(self, name):
        """Start timing a component"""
        if self.enabled:
            self.current_starts[name] = time.time()
    
    def stop(self, name):
        """Stop timing a component"""
        if self.enabled and name in self.current_starts:
            elapsed = time.time() - self.current_starts[name]
            self.timers[name].append(elapsed)
            del self.current_starts[name]
            return elapsed
        return 0.0
    
    def start_iter(self):
        """Start timing full iteration"""
        if self.enabled:
            self.iter_start = time.time()
    
    def stop_iter(self):
        """Stop timing full iteration"""
        if self.enabled and self.iter_start:
            elapsed = time.time() - self.iter_start
            self.timers['total_iter'].append(elapsed)
            self.iter_start = None
            return elapsed
        return 0.0
    
    def get_stats(self, recent_n=50):
        """Get statistics for recent timings"""
        stats = {}
        for name, times in self.timers.items():
            if len(times) > 0:
                recent = times[-recent_n:]
                stats[name] = {
                    'mean': np.mean(recent),
                    'std': np.std(recent),
                    'min': np.min(recent),
                    'max': np.max(recent),
                    'count': len(recent)
                }
        return stats
    
    def print_stats(self, recent_n=50):
        """Save profiling statistics to CSV (silent mode)"""
        stats = self.get_stats(recent_n)
        if not stats:
            return
        
        # Silent mode - save to CSV instead of printing
        # The data will be available in reports/profiling.csv
        pass
    
    def save_to_csv(self, filepath, recent_n=50):
        """Save profiling stats to CSV file"""
        stats = self.get_stats(recent_n)
        if not stats:
            return
        
        import csv
        import os
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write CSV with component breakdown
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['component', 'mean_ms', 'std_ms', 'percentage'])
            
            total_time = stats.get('total_iter', {}).get('mean', 0)
            
            for name, s in sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True):
                mean_ms = s['mean'] * 1000
                std_ms = s['std'] * 1000
                pct = (s['mean'] / total_time * 100) if total_time > 0 else 0
                writer.writerow([name, f"{mean_ms:.1f}", f"{std_ms:.1f}", f"{pct:.1f}"])
    
    def reset(self):
        """Clear all timers"""
        self.timers.clear()
        self.current_starts.clear()

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out-shakespeare-char-hdeq'
eval_interval = 100
log_interval = 10
eval_iters = 50
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' (gpt2* not supported for DEQ)
# Profiling
enable_profiling = True  # Enable detailed timing profiling
profile_interval = 50  # Log profiling stats every N iterations
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'shakespeare-char'
wandb_run_name = 'hdeq'
# data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 16 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 256
# model
n_layer = 2  # For Gray Box, this controls n_reflex (number of reflex blocks)
n_head = 6
n_embd = 384
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = True # do we use bias inside LayerNorm and Linear layers?
# DEQ-specific parameters
deq_max_iter = 100  # INCREASED: Allow stable system time to reach res < 1e-3 (Profound Fix)
deq_tol = 1e-3
anderson_accel = True
spectral_norm = True  # Disabled for now (device placement issues)

# Hamiltonian Dynamics (energy-conserving symplectic integrator)
hamiltonian = True  # Use Hamiltonian operator instead of dissipative DEQ

# Unified Quantum Solver (combines multiple physics concepts)
quantum_solver = True  # Enable unified quantum-inspired solving

# Quantum solver parameters (when quantum_solver=True)
num_gauge_orbits = 3
symmetry_breaking_iters = 3
refinement_iters = 15  # INCREASED: Need more iters to actually converge (was 5)
enable_tunneling = True
tunnel_threshold = 0.95
num_tunnel_rays = 32  # Sample quantum probability cloud (more rays = better statistics)
temperature_schedule = "exponential"
T_init = 0.1
T_final = 0.01

# Pauli Exclusion (Anti-Stuttering Force)
lambda_pauli = 2.0  # Weight for repetition penalty (0.0 = disabled, 2.0 = strong)

# Algorithmic Efficiency Loss (DISABLED - contradicts giving more iterations!)
lambda_efficiency = 0.0  # Weight for iteration count penalty (DISABLED for stability testing)

# Memory Manifold (Semantic Knowledge Substrate)
use_memory_manifold = False  # Enable memory-augmented reflexes
memory_mode = 'hybrid'  # 'hybrid', 'hyperbolic', 'euclidean'
memory_manifold_path = 'tinystories_memory_manifold.pkl'  # Path for static mode
memory_k = 16  # Number of semantic neighbors to retrieve
memory_alpha = 0.1  # Injection strength (gating parameter)
memory_dim = 384  # Memory embedding dimension
memory_curvature = 1.0  # Hyperbolic curvature
# CACHE-STYLE two-tier memory (working = L1 cache, longterm = RAM)
working_memory_capacity = 20      # TINY - only immediate context (like L1 cache)
working_memory_decay = 0.80       # AGGRESSIVE - 20% decay per step (forget quickly)
longterm_memory_capacity = 20000  # UPGRADED: 10x capacity - better for general LM (300MB RAM)
longterm_memory_decay = 0.999     # PERSISTENT - slow fade
memory_promotion_threshold = 0.4  # Lower threshold for easier promotion
memory_promotion_interval = 50    # More frequent consolidation

# adamw optimizer
learning_rate = 1e-3 # max learning rate
max_iters = 1000 # total number of training iterations
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # Disable for DEQ (can cause issues with implicit differentiation)
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# 🧠 MEMORY FLAGS: Special handling for command-line memory control
# --clean-memory: Start with fresh memory (ignore existing disk storage)
# --memory-path=<path>: Use specific memory file (for resuming/sharing)
# --copy-clean-from=<path>: Copy from clean/reference memory (e.g., preloaded cache)
# --create-clean-to=<path>: After preload, save a clean copy to this path (for reuse)
# --visualizations-off: Disable all matplotlib visualizations (prevents thread-safety crashes)
# --profile: Enable detailed performance profiling (find bottlenecks!)
clean_memory = '--clean-memory' in sys.argv
visualizations_disabled = '--visualizations-off' in sys.argv
enable_profiling = '--profile' in sys.argv
memory_path_override = None
copy_clean_from = None
create_clean_to = None
for arg in sys.argv:
    if arg.startswith('--memory-path='):
        memory_path_override = arg.split('=', 1)[1]
    elif arg.startswith('--copy-clean-from='):
        copy_clean_from = arg.split('=', 1)[1]
    elif arg.startswith('--create-clean-to='):
        create_clean_to = arg.split('=', 1)[1]

# Initialize profiler (after flags parsed)
if enable_profiling:
    from profiler import Profiler
    profiler = Profiler(enabled=True, cuda_sync=True)
    print("🔬 PROFILING ENABLED - Performance analysis active!")
else:
    # Use existing lightweight profiler
    profiler = TrainingProfiler(enabled=True)

# Memory safety check
MAX_RAM_GB = 20.0  # Allow large caches for hyperbolic prefetching
def check_memory_safety():
    """Kill process if RAM usage exceeds limit (prevents runaway leaks)"""
    process = psutil.Process()
    ram_gb = process.memory_info().rss / (1024**3)
    if ram_gb > MAX_RAM_GB:
        print(f"\n{'='*80}")
        print(f"💀 MEMORY SAFETY KILL - RAM usage: {ram_gb:.2f}GB > {MAX_RAM_GB}GB")
        print(f"This protects against memory leaks, not intentional cache usage.")
        print(f"If you need more cache, increase MAX_RAM_GB in train_hdeq.py")
        print(f"{'='*80}")
        import sys
        sys.exit(1)
    return ram_gb

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

# Create unique run folder with timestamp
if master_process:
    if init_from == 'scratch':
        # Create timestamped run directory
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{os.path.basename(out_dir)}_{timestamp}"
        out_dir = os.path.join(os.path.dirname(out_dir) if os.path.dirname(out_dir) else '.', run_name)
        print(f"📁 Creating new run directory: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    
    # 🧠 SET MEMORY PATH: Use run-specific path by default (isolated per run)
    if memory_path_override:
        # User specified explicit path (e.g., to resume from specific memory)
        final_memory_path = memory_path_override
        print(f"🧠 Using custom memory path: {final_memory_path}")
    elif 'longterm_disk_path' in globals() and globals()['longterm_disk_path']:
        # Config specified a path (legacy behavior)
        final_memory_path = globals()['longterm_disk_path']
    else:
        # Default: Run-specific memory DIRECTORY (NEW - each run isolated!)
        final_memory_path = os.path.join(out_dir, 'memory')
        print(f"🧠 Memory directory: {final_memory_path}")
    
    # Apply --clean-memory flag (handle both file and directory)
    import shutil
    if clean_memory:
        if os.path.exists(final_memory_path):
            if os.path.isdir(final_memory_path):
                shutil.rmtree(final_memory_path)
                print(f"🧹 Cleaned old memory directory (--clean-memory flag)")
            else:
                os.remove(final_memory_path)
                print(f"🧹 Cleaned old memory file (--clean-memory flag)")
    
    # Apply --copy-clean-from flag (copy reference memory to this run)
    if copy_clean_from:
        if not os.path.exists(copy_clean_from):
            raise FileNotFoundError(f"❌ --copy-clean-from path doesn't exist: {copy_clean_from}")
        
        # Remove existing memory first (if any)
        if os.path.exists(final_memory_path):
            if os.path.isdir(final_memory_path):
                shutil.rmtree(final_memory_path)
            else:
                os.remove(final_memory_path)
        
        # Copy the clean/reference memory
        if os.path.isdir(copy_clean_from):
            shutil.copytree(copy_clean_from, final_memory_path)
            print(f"📋 Copied clean memory directory: {copy_clean_from} → {final_memory_path}")
        else:
            # Legacy: single file memory
            shutil.copy(copy_clean_from, final_memory_path)
            print(f"📋 Copied clean memory file: {copy_clean_from} → {final_memory_path}")
    
    # Override config with final path
    globals()['longterm_disk_path'] = final_memory_path
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# 🚨 PLATEAU DETECTION & INTERVENTION
# Track loss history for gradient plateau detection
loss_history = []  # Rolling window of recent losses
loss_window_size = 20  # Window for computing loss gradient
plateau_threshold = 1e-4  # If loss gradient < this for N iters, plateau detected
plateau_counter = 0  # How many consecutive iters we've been plateaued
plateau_intervention_trigger = 30  # Soft intervention after 30 plateau iters
plateau_hard_trigger = 100  # Hard intervention after 100 plateau iters
last_intervention_iter = -1000  # Track when we last intervened (cooldown)
intervention_cooldown = 50  # Don't intervene more than once per 50 iters

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  deq_max_iter=deq_max_iter, deq_tol=deq_tol, 
                  anderson_accel=anderson_accel, spectral_norm=spectral_norm,
                  hamiltonian=hamiltonian, quantum_solver=quantum_solver,
                  num_gauge_orbits=num_gauge_orbits, 
                  symmetry_breaking_iters=symmetry_breaking_iters,
                  refinement_iters=refinement_iters,
                  enable_tunneling=enable_tunneling,
                  tunnel_threshold=tunnel_threshold,
                  num_tunnel_rays=num_tunnel_rays,
                  temperature_schedule=temperature_schedule,
                  T_init=T_init, T_final=T_final)

# Memory Manifold Integration (Semantic Knowledge Substrate)
if use_memory_manifold:
    # For HYBRID mode: Memory forms dynamically during training (starts empty!)
    # For STATIC mode: Load pre-built manifold
    if memory_mode == 'hybrid':
        print(f"🧠 Hybrid two-tier memory enabled (dynamic formation)")
        print(f"  Memory starts EMPTY and grows during training")
        print(f"  Working: {working_memory_capacity} capacity on GPU")
        print(f"  Long-term: {longterm_memory_capacity} capacity on CPU")
        
        # ═══════════════════════════════════════════════════════════════════════
        # 🖕 CRITICAL VALIDATION: GNN MUST BE ENABLED 🖕
        # ═══════════════════════════════════════════════════════════════════════
        gnn_enabled = enable_gnn if 'enable_gnn' in locals() else False
        if not gnn_enabled:
            print("\n" + "=" * 80)
            print("❌ ERROR: GNN is DISABLED but is REQUIRED for graph navigation learning!")
            print("=" * 80)
            print("Without GNN, the system can only do dumb cosine similarity search.")
            print("The dual memory (preload + online) architecture REQUIRES graph structure.")
            print("")
            print("To fix: Set enable_gnn=True in your config.")
            print("For 6GB GPU, use: memory_k=12, gnn_hidden_dim=256 (Micro-GNN)")
            print("=" * 80)
            raise ValueError("GNN is non-negotiable for graph-structured memory!")
        
        # SUCCESS BANNER
        gnn_dim = gnn_hidden_dim if 'gnn_hidden_dim' in locals() else 512
        print(f"  ✅ GNN ENABLED (k={memory_k}, hidden={gnn_dim})")
        if gnn_dim <= 256 and memory_k <= 12:
            print(f"  🎯 Micro-GNN optimized for consumer GPU (6GB-friendly)")
        # ═══════════════════════════════════════════════════════════════════════
        
        # Pass memory config to model (no manifold path needed!)
        model_args['use_memory_manifold'] = True
        model_args['memory_mode'] = memory_mode
        model_args['memory_dim'] = memory_dim if 'memory_dim' in locals() else n_embd
        model_args['memory_k'] = memory_k
        model_args['memory_alpha'] = memory_alpha
        model_args['memory_curvature'] = memory_curvature if 'memory_curvature' in locals() else 1.0
        model_args['enable_gnn'] = gnn_enabled
        model_args['gnn_hidden_dim'] = gnn_hidden_dim if 'gnn_hidden_dim' in locals() else 512
        model_args['working_memory_capacity'] = working_memory_capacity
        model_args['working_memory_decay'] = working_memory_decay
        model_args['longterm_memory_capacity'] = longterm_memory_capacity
        model_args['longterm_disk_path'] = globals().get('longterm_disk_path', None)  # FIXED: use globals()
        model_args['longterm_memory_decay'] = longterm_memory_decay
        model_args['memory_promotion_threshold'] = memory_promotion_threshold
        model_args['memory_promotion_interval'] = memory_promotion_interval
        model_args['highway_learning_rate'] = highway_learning_rate if 'highway_learning_rate' in locals() else 0.3  # 🔥 NEW
    else:
        # Legacy static manifold loading
        print(f"🧠 Loading memory manifold from: {memory_manifold_path}")
        if not os.path.exists(memory_manifold_path):
            raise FileNotFoundError(f"Memory manifold not found: {memory_manifold_path}")
        
        with open(memory_manifold_path, 'rb') as f:
            memory_manifold = pickle.load(f)
        
        print(f"  ✓ Loaded {memory_manifold['num_chunks']} semantic chunks")
        print(f"  ✓ Embedding dim: {memory_manifold['embedding_dim']}")
        print(f"  ✓ HNSW index: {memory_manifold['num_chunks']} nodes")
        print(f"  ✓ Retrieval: top-k={memory_k}, α={memory_alpha}")
        
        # Verify dimensions match
        if memory_manifold['embedding_dim'] != n_embd:
            raise ValueError(f"Memory embedding dim ({memory_manifold['embedding_dim']}) must match n_embd ({n_embd})")
        
        # Pass memory config to model
        model_args['use_memory_manifold'] = True
        model_args['memory_manifold_path'] = memory_manifold_path
        model_args['memory_k'] = memory_k
    model_args['memory_alpha'] = memory_alpha
else:
    print("🧠 Memory manifold disabled (use_memory_manifold=False)")

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new Gray Box DEQ model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    # First check if ckpt.pt exists (user-specified resume point)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if os.path.exists(ckpt_path):
        print(f"  Using checkpoint: ckpt.pt")
    else:
        # Fall back to finding latest iteration checkpoint (ckpt_iter*.pt)
        import glob
        iter_ckpts = glob.glob(os.path.join(out_dir, 'ckpt_iter*.pt'))
        if iter_ckpts:
            # Sort by iteration number and take the latest
            # Filter out backup files and only keep standard ckpt_iterXXXX.pt format
            def extract_iter_num(path):
                try:
                    return int(path.split('iter')[-1].split('.')[0])
                except ValueError:
                    return -1  # Put malformed names at the start (will be ignored)
            
            iter_ckpts.sort(key=extract_iter_num)
            # Take the latest valid checkpoint (highest iter number)
            ckpt_path = iter_ckpts[-1]
            print(f"  Found latest checkpoint: {os.path.basename(ckpt_path)}")
        else:
            raise FileNotFoundError(f"No checkpoints found in {out_dir}")
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_model_args = checkpoint['model_args']
    
    # ⚡ FORCE MEMORY UPGRADE OVERRIDE (Zombie Config Fix)
    # When resuming, checkpoint memory settings override current config.
    # Force upgrade to allow the model's "Library" (Long-term memory) to expand.
    # NOTE: Config uses 'longterm_capacity', train_hdeq.py uses 'longterm_memory_capacity'
    old_lt_capacity = checkpoint_model_args.get('longterm_memory_capacity', 2000)
    # Check both variable naming conventions (with/without '_memory_')
    current_lt_capacity = locals().get('longterm_capacity', locals().get('longterm_memory_capacity', old_lt_capacity))
    if current_lt_capacity > old_lt_capacity:
        print(f"⚡ FORCING MEMORY UPGRADE: {old_lt_capacity} → {current_lt_capacity}")
        # Override ALL hybrid memory parameters to ensure consistency
        checkpoint_model_args['longterm_memory_capacity'] = current_lt_capacity
        checkpoint_model_args['working_memory_capacity'] = locals().get('working_capacity', locals().get('working_memory_capacity', 20))
        if 'memory_mode' in locals():
            checkpoint_model_args['memory_mode'] = memory_mode
        if 'working_memory_decay' in locals():
            checkpoint_model_args['working_memory_decay'] = working_memory_decay
        if 'longterm_memory_decay' in locals():
            checkpoint_model_args['longterm_memory_decay'] = longterm_memory_decay
        if 'memory_promotion_threshold' in locals():
            checkpoint_model_args['memory_promotion_threshold'] = memory_promotion_threshold
        if 'memory_promotion_interval' in locals():
            checkpoint_model_args['memory_promotion_interval'] = memory_promotion_interval
        print(f"   Memory will grow from checkpoint's {old_lt_capacity} to new capacity {current_lt_capacity}")
        print(f"   🧠 The 'Library' is now 10× larger - grokking phase can begin!")
    
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # DEQ-specific params
    for k in ['deq_max_iter', 'deq_tol', 'anderson_accel', 'spectral_norm']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    # Memory manifold params (preserve from checkpoint if present)
    for k in ['use_memory_manifold', 'memory_manifold_path', 'memory_k', 'memory_alpha']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    # Filter out memory parameters - they'll be loaded from memory checkpoint
    # This prevents size mismatch errors when memory capacity changes
    memory_keys = [k for k in state_dict.keys() if 'memory_retrieval' in k or 'trajectory_buffer' in k]
    if memory_keys:
        print(f"  Filtering {len(memory_keys)} memory parameters from checkpoint (will load from memory_ckpt)")
        for k in memory_keys:
            del state_dict[k]
    
    # Load model weights (strict=False allows new parameters in current model)
    model.load_state_dict(state_dict, strict=False)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
else:
    raise ValueError(f"init_from='{init_from}' not supported for Gray Box DEQ (only 'scratch' or 'resume')")

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# ═══════════════════════════════════════════════════════════════════════════
# 🔍 VRAM ESTIMATION - Predict if we'll OOM
# ═══════════════════════════════════════════════════════════════════════════
if device == 'cuda':
    import torch.cuda as cuda
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    param_bytes = total_params * 2  # fp16 = 2 bytes
    
    # Estimate activations (very rough)
    B = batch_size if 'batch_size' in locals() else 2
    T = block_size
    D_enc = n_embd
    D_deq = deq_n_embd if 'deq_n_embd' in locals() else n_embd
    
    # Encoder activations: B × T × D × n_layer × 4 (residual+attn+mlp+grad)
    enc_act = B * T * D_enc * n_layer * 4 * 4  # fp32 = 4 bytes
    
    # DEQ activations: B × T × D_deq × deq_max_iter × 2 (forward+backward checkpointing)
    deq_iters = deq_max_iter if 'deq_max_iter' in locals() else 6
    deq_act = B * T * D_deq * deq_iters * 2 * 4
    
    # GNN activations (if enabled)
    gnn_act = 0
    if use_memory_manifold and (enable_gnn if 'enable_gnn' in locals() else False):
        gnn_hidden = gnn_hidden_dim if 'gnn_hidden_dim' in locals() else 512
        k = memory_k
        # Message passing: B × T × k × gnn_hidden (with checkpointing)
        gnn_act = B * T * k * gnn_hidden * 2 * 4  # Reduced by checkpointing
    
    total_act = enc_act + deq_act + gnn_act
    
    # Gradients = params
    grad_bytes = param_bytes
    
    # Optimizer states (AdamW: 2 states per param)
    opt_bytes = param_bytes * 2
    
    # Total estimate
    estimated_mb = (param_bytes + total_act + grad_bytes + opt_bytes) / (1024**2)
    
    # Get actual VRAM
    total_vram = cuda.get_device_properties(0).total_memory / (1024**2)
    
    print("\n" + "=" * 70)
    print("📊 VRAM ESTIMATION")
    print("=" * 70)
    print(f"  Parameters:     {param_bytes/(1024**2):6.1f} MB ({total_params/1e6:.1f}M params)")
    print(f"  Activations:    {total_act/(1024**2):6.1f} MB (B={B}, T={T})")
    print(f"    - Encoder:    {enc_act/(1024**2):6.1f} MB")
    print(f"    - DEQ:        {deq_act/(1024**2):6.1f} MB")
    if gnn_act > 0:
        print(f"    - GNN:        {gnn_act/(1024**2):6.1f} MB (with checkpointing)")
    print(f"  Gradients:      {grad_bytes/(1024**2):6.1f} MB")
    print(f"  Optimizer:      {opt_bytes/(1024**2):6.1f} MB (AdamW states)")
    print(f"  ─────────────────────────")
    print(f"  TOTAL:          {estimated_mb:6.1f} MB / {total_vram:.0f} MB VRAM")
    print(f"  Usage:          {estimated_mb/total_vram*100:5.1f}%")
    print("=" * 70)
    
    if estimated_mb > total_vram * 0.85:
        print("⚠️  WARNING: Estimated VRAM usage > 85%!")
        print("   Risk of OOM. Consider:")
        print("   - Reduce batch_size")
        print("   - Reduce block_size") 
        print("   - Reduce deq_n_embd or deq_max_iter")
        print("   - Reduce memory_k or gnn_hidden_dim")
        raise RuntimeError("Predicted OOM - aborting before training")
    elif estimated_mb > total_vram * 0.95:
        print("❌ ERROR: Estimated VRAM > 95% - will definitely OOM!")
        raise RuntimeError("Configuration won't fit in VRAM")
    else:
        print(f"✅ VRAM headroom: {(total_vram - estimated_mb):.0f} MB ({(1 - estimated_mb/total_vram)*100:.1f}%)")
    print("")
# ═══════════════════════════════════════════════════════════════════════════

# Load memory checkpoint if resuming with memory enabled
if init_from == 'resume' and use_memory_manifold and hasattr(model.reflex, 'load_memory_checkpoint'):
    # Find the corresponding memory checkpoint
    iter_num_str = os.path.basename(ckpt_path).replace('ckpt_iter', '').replace('.pt', '')
    if iter_num_str.isdigit():
        memory_ckpt_path = os.path.join(out_dir, f'memory_ckpt_iter{iter_num_str}.pkl')
    else:
        memory_ckpt_path = os.path.join(out_dir, 'memory_ckpt.pkl')
    
    if os.path.exists(memory_ckpt_path):
        print(f"  Loading memory checkpoint: {os.path.basename(memory_ckpt_path)}")
        
        # Check dimension compatibility before loading
        current_memory_dim = getattr(model_args, 'memory_dim', model_args['n_embd'])
        try:
            model.reflex.load_memory_checkpoint(memory_ckpt_path)
            mem_stats = model.reflex.get_memory_stats()
            print(f"  ✓ Restored memory: W={mem_stats.get('num_working', 0)}, "
                  f"B={mem_stats.get('num_buffer', 0)}, LT={mem_stats.get('num_longterm', 0)}")
        except RuntimeError as e:
            if 'size' in str(e).lower() or 'dimension' in str(e).lower():
                print(f"  ⚠️  Memory dimension mismatch (checkpoint incompatible with current config)")
                print(f"  Expected dim={current_memory_dim}, but checkpoint has different dimension")
                print(f"  Starting with EMPTY memory - will rebuild during training")
            else:
                raise
    else:
        print(f"  ⚠️  Memory checkpoint not found: {memory_ckpt_path}")
        print(f"  Memory will start empty (capacity: {longterm_memory_capacity})")

# ═══════════════════════════════════════════════════════════════════════════
# PRELOAD MEMORIES FROM DATASET (if memory is empty)
# ═══════════════════════════════════════════════════════════════════════════
# Seed the memory system with semantic chunks from the training data.
# This gives the model a "knowledge base" from day 1, enabling:
# 1. Faster convergence (no cold start)
# 2. Immediate supervised navigation (memories already exist)
# 3. Better early predictions (semantic knowledge available)
# ═══════════════════════════════════════════════════════════════════════════

if use_memory_manifold and init_from == 'scratch':
    # Check if we should preload (memory empty or nearly empty)
    needs_preload = should_preload_memories(model, min_memories=50)
    current_mem_size = model.reflex.memory_retrieval.memory.size.item() if hasattr(model.reflex, 'memory_retrieval') else 0
    print(f"\n🔍 Preload check: current_size={current_mem_size}, needs_preload={needs_preload}")
    
    # Note: cluster_ids, depths, type_embeddings are now DiskBackedTensor
    # They auto-load from disk when --copy-clean-from is used. No manual loading needed!
    
    # CRITICAL: Resize RAM-only buffers (rewards, age, access) to match disk-backed size
    # When loading from golden checkpoint, DiskBackedTensors auto-restore their size,
    # but regular buffers (rewards/age/access) stay at size 0 since they don't persist
    if current_mem_size > 0 and hasattr(model.reflex, 'memory_retrieval'):
        memory_tier = model.reflex.memory_retrieval.memory
        if hasattr(memory_tier, 'rewards') and memory_tier.rewards.size(0) == 0:
            # Initialize to zeros matching the loaded memory size
            memory_tier.rewards = torch.zeros(current_mem_size, device=memory_tier.device)
            memory_tier.age = torch.zeros(current_mem_size, device=memory_tier.device)
            memory_tier.access = torch.zeros(current_mem_size, device=memory_tier.device)
            if master_process:
                print(f"✅ Resized RAM buffers (rewards/age/access) to {current_mem_size} to match loaded memories")
    
    if needs_preload:
        print("\n" + "=" * 70)
        print("🗂️  MEMORY PRELOADING - Seeding from dataset")
        print("=" * 70)
        
        # Read preload config from globals (not model_args - it's training config, not model config)
        preload_num_samples = globals().get('preload_num_samples', 1000)
        preload_chunk_size = globals().get('preload_chunk_size', 32)
        
        num_added = preload_memories_from_dataset(
            model=model,
            data_dir=data_dir,
            dataset_name=dataset,
            num_samples=preload_num_samples,
            chunk_size=preload_chunk_size,
            stride=preload_chunk_size // 2,  # 50% overlap
            device=device,
            dtype=torch.float16 if dtype == 'float16' else torch.bfloat16,
            verbose=master_process
        )
        
        if num_added > 0 and master_process:
            print(f"✅ Preloaded {num_added} semantic chunks into memory")
            print(f"🎯 Supervised navigation can start immediately!")
            
            # Apply --create-clean-to flag (save clean copy after preload)
            if create_clean_to:
                import shutil
                from disk_backed_tensor import DiskBackedTensor
                
                # CRITICAL: Flush all disk-backed tensors to ensure ALL data is on disk
                print("💾 Flushing all memory tensors to disk before snapshot...")
                if hasattr(model.reflex, 'memory_retrieval') and hasattr(model.reflex.memory_retrieval, 'memory'):
                    memory_tier = model.reflex.memory_retrieval.memory
                    # Flush ALL disk-backed tensors (graph structure + metadata)
                    tensor_names = [
                        'embeddings', 'adjacency', 'edge_weights', 'edge_types',
                        'edge_traversal_count', 'edge_success_rate',
                        'cluster_ids', 'depths', 'type_embeddings'  # Metadata now disk-backed too!
                    ]
                    for tensor_name in tensor_names:
                        if hasattr(memory_tier, tensor_name):
                            tensor = getattr(memory_tier, tensor_name)
                            if isinstance(tensor, DiskBackedTensor):
                                tensor.flush()
                                print(f"   ✓ Flushed {tensor_name}")
                print("   ✅ All tensors flushed to disk")
                
                # Remove existing clean copy first (if any)
                if os.path.exists(create_clean_to):
                    if os.path.isdir(create_clean_to):
                        shutil.rmtree(create_clean_to)
                    else:
                        os.remove(create_clean_to)
                
                # Copy the freshly preloaded memory
                final_memory_path = globals().get('longterm_disk_path')
                if final_memory_path and os.path.exists(final_memory_path):
                    if os.path.isdir(final_memory_path):
                        shutil.copytree(final_memory_path, create_clean_to)
                        print(f"💾 Saved clean memory snapshot: {final_memory_path} → {create_clean_to}")
                        print(f"   Use --copy-clean-from={create_clean_to} to reuse this preload!")
                    else:
                        shutil.copy(final_memory_path, create_clean_to)
                        print(f"💾 Saved clean memory file: {final_memory_path} → {create_clean_to}")
        
        print("=" * 70 + "\n")
    else:
        if master_process:
            mem_stats = model.reflex.get_memory_stats()
            print(f"ℹ️  Skipping preload - memory already has {mem_stats.get('num_longterm', 0)} entries")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# Initialize Homeostatic Balancer (Bayesian Multi-Objective Learning)
# This automatically balances loss components without manual tuning
loss_names = ["prediction", "jacobian", "novelty", "memory", "nav_reward"]
balancer = HomeostaticBalancer(num_losses=5, loss_names=loss_names).to(device)
# Balancer diagnostics tracked in CSV/graphs - silent initialization

# Initialize Memory Navigation Rewards (Dopamine for graph exploration)
memory_nav_rewards = MemoryNavigationRewards(model_args).to(device)
memory_path_rl = MemoryPathReinforcement(model_args)
print("[Training] Memory navigation rewards enabled")

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# ADD BALANCER PARAMETERS TO OPTIMIZER
# The balancer's σ (uncertainty) parameters must be learned via gradient descent
# Use a slightly higher LR for balancer (it's just 4 parameters)
balancer_lr = learning_rate * 10.0  # 10× higher LR for faster adaptation
optimizer.add_param_group({
    'params': balancer.parameters(),
    'lr': balancer_lr,
    'weight_decay': 0.0  # No weight decay on uncertainty parameters
})
# Balancer config tracked in diagnostics - silent mode

if init_from == 'resume':
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
        # FIX: Restore balancer LR after loading checkpoint (which overwrites it!)
        optimizer.param_groups[-1]['lr'] = balancer_lr
        print("✓ Optimizer state loaded from checkpoint")
    except (ValueError, KeyError) as e:
        print(f"⚠ Warning: Could not load optimizer state: {e}")
        print("  This is normal after architecture changes. Starting optimizer from scratch.")
        print("  Model weights are still loaded correctly!")
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    print("WARNING: torch.compile may not work well with DEQ implicit differentiation")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    """Evaluate the model on train and validation sets"""
    profiler.start('validation')
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y, training_iter=iter_num)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    profiler.stop('validation')
    return out

# generate samples to see what the model is producing
@torch.no_grad()
def generate_samples_with_visualization(iter_num):
    """Generate samples AND visualize reflex network activations as text is generated"""
    model.eval()
    
    # Load meta for decoding and encoding
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # Check if this is a character-level dataset (has 'itos') or BPE (use tiktoken)
        if 'itos' in meta:
            stoi = meta['stoi']
            decode = lambda l: ''.join([meta['itos'][i] for i in l])
            encode = lambda s: [stoi[c] for c in s]
        else:
            # BPE tokenizer (GPT-2, used by WikiText, OASST, etc.)
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            decode = lambda l: enc.decode(l)
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    else:
        decode = lambda l: str(l)  # fallback
        encode = lambda s: [0]  # fallback
    
    print("\n" + "="*70)
    print("SAMPLE GENERATIONS")
    print("="*70)
    
    # Sample 1: Question prompt
    if os.path.exists(meta_path):
        # "Why did Napoleon invade Spain?"
        prompt = "Why did Napoleon invade Spain?"
        start_ids = encode(prompt)
    else:
        start_ids = [0]
    
    x = torch.tensor([start_ids], dtype=torch.long, device=device)
    
    # CAPTURE REFLEX ACTIVATIONS DURING GENERATION
    reflex_activations = []
    generated_tokens = []
    
    # Monkey-patch the reflex forward to capture activations
    if hasattr(raw_model, 'reflex'):
        original_reflex_forward = raw_model.reflex.forward
        
        def capturing_reflex_forward(x, *args, **kwargs):
            output = original_reflex_forward(x, *args, **kwargs)
            # Reflex now returns (reflex_tensor, memory_bundle)
            if isinstance(output, tuple):
                reflex_tensor, memory_bundle = output
            else:
                reflex_tensor = output
            # Capture last token's activation
            if reflex_tensor.ndim == 3:  # (batch, seq, hidden)
                reflex_activations.append(reflex_tensor[0, -1, :].detach().cpu().numpy())
            return output
        
        raw_model.reflex.forward = capturing_reflex_forward
    
    # Generate with activation capture
    y = model.generate(x, max_new_tokens=50, temperature=0.8, top_k=200, effort=1.0)
    generated_tokens = y[0].tolist()
    
    # Restore original reflex forward
    if hasattr(raw_model, 'reflex'):
        raw_model.reflex.forward = original_reflex_forward
    
    print("\n[Question prompt]")
    output = decode(generated_tokens)
    print(output[:200] if len(output) > 200 else output)
    
    # Create animation if we captured activations
    if len(reflex_activations) > 0 and monitor and VISUALIZATION_AVAILABLE:
        print(f"\n  🎬 Creating reflex animation ({len(reflex_activations)} frames)...")
        try:
            import numpy as np
            from matplotlib import pyplot as plt
            from matplotlib.animation import FuncAnimation, PillowWriter
            
            # Limit to first 30 tokens for manageable GIF
            max_frames = min(30, len(reflex_activations))
            reflex_activations = reflex_activations[:max_frames]
            generated_tokens = generated_tokens[:max_frames]
            
            # CRITICAL: Matplotlib is NOT thread-safe, must use lock
            with BackgroundWorker._matplotlib_lock:
                # Create figure with 2 subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), facecolor='white',
                                              gridspec_kw={'height_ratios': [2, 1]})
                
                # Prepare data for trajectory plot
                from sklearn.decomposition import PCA
                activations_array = np.array(reflex_activations)
                pca = PCA(n_components=2)
                trajectory_2d = pca.fit_transform(activations_array)
                
                def update_frame(frame_idx):
                    ax1.clear()
                    ax2.clear()
                    
                    # TOP: 2D trajectory through activation space
                    ax1.set_facecolor('#f0f0f0')
                    
                    # Plot full trajectory (faded)
                    ax1.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 
                            'gray', alpha=0.3, linewidth=1.5, linestyle='--')
                    
                    # Plot trajectory up to current frame (bright)
                    if frame_idx > 0:
                        ax1.plot(trajectory_2d[:frame_idx+1, 0], trajectory_2d[:frame_idx+1, 1],
                                color='#e74c3c', linewidth=3, alpha=0.9, label='Path so far')
                    
                    # Current position (big marker)
                    ax1.scatter(trajectory_2d[frame_idx, 0], trajectory_2d[frame_idx, 1],
                               c='red', s=300, marker='*', zorder=5, edgecolors='black', linewidths=2)
                    
                    # Start position
                    ax1.scatter(trajectory_2d[0, 0], trajectory_2d[0, 1],
                               c='green', s=150, marker='o', zorder=4, edgecolors='black', 
                               linewidths=1.5, label='Start', alpha=0.7)
                    
                    # Labels for a few key points
                    if frame_idx > 0:
                        for i in range(0, min(frame_idx, 10), 3):
                            token_str = decode([generated_tokens[i]])
                            ax1.annotate(token_str, (trajectory_2d[i, 0], trajectory_2d[i, 1]),
                                       fontsize=8, alpha=0.6, ha='center')
                    
                    # Current token label (big)
                    current_token = decode([generated_tokens[frame_idx]])
                    ax1.text(trajectory_2d[frame_idx, 0], trajectory_2d[frame_idx, 1] + 0.5,
                            f'"{current_token}"', fontsize=14, fontweight='bold',
                            ha='center', va='bottom', 
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
                    
                    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', 
                                  fontsize=12, fontweight='bold')
                    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', 
                                  fontsize=12, fontweight='bold')
                    ax1.set_title(f'Reflex Network Trajectory (Token {frame_idx+1}/{len(reflex_activations)})',
                                 fontsize=14, fontweight='bold')
                    ax1.legend(loc='upper right', fontsize=10)
                    ax1.grid(True, alpha=0.3)
                    
                    # BOTTOM: Raw activation heatmap
                    ax2.set_facecolor('white')
                    activation = activations_array[frame_idx]
                    
                    # Show top 50 dimensions for visibility
                    top_dims = min(100, len(activation))
                    im = ax2.imshow(activation[:top_dims].reshape(1, -1), 
                                   cmap='RdBu_r', aspect='auto', 
                                   vmin=-2, vmax=2, interpolation='nearest')
                    
                    ax2.set_title(f'Activation Pattern: "{current_token}"',
                                 fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Neuron', fontsize=10)
                    ax2.set_xlabel(f'Dimension (showing first {top_dims})', fontsize=10)
                    ax2.set_yticks([])
                    
                    # Generated text so far
                    text_so_far = decode(generated_tokens[:frame_idx+1])
                    fig.text(0.5, 0.02, f'Generated: "{text_so_far}"',
                            ha='center', fontsize=11, 
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
                    
                    return ax1, ax2
                
                # Create animation
                anim = FuncAnimation(fig, update_frame, frames=len(reflex_activations),
                                   interval=300, blit=False, repeat=True)
                
                # Save as GIF
                gif_path = os.path.join(monitor.reports_dir, f'reflex_generation_iter_{iter_num:06d}.gif')
                writer = PillowWriter(fps=3)
                anim.save(gif_path, writer=writer)
                plt.close()
            
            print(f"  ✅ Reflex animation saved: {gif_path}")
        except Exception as e:
            print(f"  ⚠️  Could not create reflex animation: {e}")
    
    # CREATE HYPERBOLIC MEMORY NAVIGATION VISUALIZATION
    if hasattr(raw_model, 'reflex') and hasattr(raw_model.reflex, 'memory_retrieval'):
        memory_system = raw_model.reflex.memory_retrieval
        
        # Safety check: memory_system might be None if memory disabled
        if memory_system is not None:
            memory_stats = memory_system.get_memory_stats()
            
            if memory_stats.get('num_longterm', 0) > 10:  # Only if we have enough memories
                print(f"  🗺️  Creating hyperbolic memory map...")
                try:
                    from visualize_hyperbolic_navigation import visualize_memory_state_static
                    
                    map_path = os.path.join(monitor.reports_dir, f'hyperbolic_map_iter_{iter_num:06d}.png')
                    
                    # CRITICAL: visualize_memory_state_static uses matplotlib, must lock
                    with BackgroundWorker._matplotlib_lock:
                        visualize_memory_state_static(memory_system, map_path)
                    
                    # TODO: For animated trajectory showing what model "looks at" per token:
                    # 1. Capture query embeddings during generation (in model.generate())
                    # 2. Capture retrieved memory indices per query
                    # 3. If deq_requery enabled, capture DEQ iteration trajectory
                    # 4. Use visualize_hyperbolic_navigation.create_navigation_gif()
                    # This would show the actual navigation path through hyperbolic space!
                    
                except Exception as e:
                    print(f"  ⚠️  Could not create hyperbolic map: {e}")
    
    print("="*70 + "\n")
    model.train()

@torch.no_grad()
def generate_samples():
    """Generate samples for monitoring (synchronous - model not thread-safe)"""
    model.eval()
    
    # Load meta for decoding and encoding
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # Check if this is a character-level dataset (has 'itos') or BPE (use tiktoken)
        if 'itos' in meta:
            stoi = meta['stoi']
            decode = lambda l: ''.join([meta['itos'][i] for i in l])
            encode = lambda s: [stoi[c] for c in s]
        else:
            # BPE tokenizer (GPT-2, used by WikiText, OASST, etc.)
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            decode = lambda l: enc.decode(l)
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    else:
        decode = lambda l: str(l)  # fallback
        encode = lambda s: [0]  # fallback
    
    # Generate sample
    if os.path.exists(meta_path):
        # "Why did Napoleon invade Spain?"
        prompt = "Why did Napoleon invade Spain?"
        start_ids = encode(prompt)
    else:
        start_ids = [0]
    
    x = torch.tensor([start_ids], dtype=torch.long, device=device)
    
    # Generate with model (minimal tokens for speed - just verify it works)
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=10, temperature=0.8, top_k=200, effort=1.0)
    
    output = decode(y[0].tolist())
    
    # Print sample (compact format)
    print(f"  Sample: {output[:100]}{'...' if len(output) > 100 else ''}")
    
    model.train()
    return output  # Return for potential logging

def visualize_phase_space(iter_num):
    """Generate phase space visualization of DEQ trajectories"""
    if not VISUALIZATION_AVAILABLE:
        return
    
    model.eval()
    
    # Capture DEQ trajectories by monkey-patching
    trajectories = []
    original_solve = model.deq.solve
    
    def captured_solve(u, mask=None, effort=1.0, verbose=False, memory_bundle=None,
                       reflex_module=None, query_embeddings=None):
        B, T, C = u.shape
        z = u.clone()
        z_prev = None
        batch_traj = []
        
        max_iter = model.config.deq_max_iter
        tol = model.config.deq_tol
        gamma = 1.0
        
        for i in range(max_iter):
            batch_traj.append(z.detach().cpu())
            delta_z = model.deq.operator(z, u, mask)
            alpha = model.deq.stabilizer(z, u)
            z_next = z + gamma * alpha * delta_z
            z_next = model.deq.laws.semantic_continuity(z_next, z_prev)
            z_prev = z
            z = z_next
            
            if i > 0:
                diff = (z - batch_traj[-1].to(z.device)).abs().max()
                if i > 3 and diff < tol:
                    break
        
        trajectories.append(torch.stack(batch_traj))
        return model.deq.ln_f(z), i+1, {}
    
    # Apply patch and run inference
    model.deq.solve = captured_solve
    
    # Simple prompt - just use a few common tokens
    # Token 198 = newline, 262 = "the", 257 = "a"
    start_ids = [262, 257]  # "the a" - simple and safe
    x = torch.tensor([start_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        model(x)
    
    # Restore original
    model.deq.solve = original_solve
    
    if len(trajectories) == 0:
        model.train()
        return
    
    # Extract trajectories
    traj = trajectories[0][:, 0, -1, :].numpy()  # Last token
    traj_start = trajectories[0][:, 0, 0, :].numpy()  # First token
    
    # PCA projection
    pca = PCA(n_components=2)
    combined = np.concatenate([traj, traj_start], axis=0)
    pca.fit(combined)
    traj_2d = pca.transform(traj)
    start_2d = pca.transform(traj_start)
    
    # CRITICAL: Matplotlib is NOT thread-safe, even with Agg backend
    # Must use global lock to prevent C++ aborts from concurrent figure creation
    with BackgroundWorker._matplotlib_lock:
        # Plot
        plt.figure(figsize=(10, 8))
        plt.style.use('dark_background')
        
        plt.plot(traj_2d[:, 0], traj_2d[:, 1], 'r-', linewidth=2, label='Last Token Trajectory')
        plt.scatter(traj_2d[0, 0], traj_2d[0, 1], c='white', marker='o', s=100, label='Start', zorder=5)
        plt.scatter(traj_2d[-1, 0], traj_2d[-1, 1], c='red', marker='*', s=300, label='Equilibrium', zorder=5)
        
        plt.plot(start_2d[:, 0], start_2d[:, 1], 'b--', alpha=0.5, linewidth=1.5, label='First Token (Control)')
        plt.scatter(start_2d[-1, 0], start_2d[-1, 1], c='blue', marker='x', s=150, zorder=5)
        
        # Arrows for first few iterations
        for i in range(min(len(traj_2d)-1, 10)):
            dx = traj_2d[i+1,0] - traj_2d[i,0]
            dy = traj_2d[i+1,1] - traj_2d[i,1]
            plt.arrow(traj_2d[i,0], traj_2d[i,1], dx*0.7, dy*0.7,
                     head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.6, zorder=3)
        
        # Diagnostics
        center = traj_2d.mean(axis=0)
        radii = np.linalg.norm(traj_2d - center, axis=1)
        radius_trend = np.polyfit(range(len(radii)), radii, 1)[0]
        
        if abs(radius_trend) < 0.01:
            dynamics_type = "ORBITAL (Hamiltonian)"
        elif radius_trend < -0.01:
            dynamics_type = "SPIRAL (Dissipative)"
        else:
            dynamics_type = "DIVERGING"
        
        plt.title(f'Phase Space @ Iter {iter_num}\nDynamics: {dynamics_type} | DEQ Iters: {len(traj)}', fontsize=14)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
        plt.legend(loc='best', fontsize=9)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        
        # Save to reports
        phase_dir = os.path.join(out_dir, 'reports', 'phase_space')
        os.makedirs(phase_dir, exist_ok=True)
        output_path = os.path.join(phase_dir, f'phase_iter_{iter_num:06d}.png')
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    # Silent mode - phase space saved to graph
    model.train()

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# REFLEX INTEGRATION: Always active for memory-augmented forcing
def get_reflex_gate(it, phase1_iters=500, phase2_iters=1000):
    """
    Reflex Gate β(t): Always 1.0 (full integration).
    
    The reflex module serves two critical functions:
    1. Fast local syntax (shallow attention for bigrams/patterns)
    2. Memory retrieval orchestration (GNN-based graph navigation)
    
    Previously gated to prevent "lizard brain bypass", but with the memory system:
    - Memory provides its own curriculum (few nodes early → many later)
    - Reflex gradients are needed for GNN to learn graph structure
    - Wasting 21% of params (3.7M) for 500 iters is inefficient
    
    New philosophy: Train both subsystems together from iter 0.
    - DEQ learns semantic attractors
    - Reflex learns syntactic patterns + memory navigation
    - Memory graph grows organically, providing natural curriculum
    
    Returns:
        β = 1.0: Full integration always
    """
    # Always use full reflex + memory integration
    return 1.0

# Lyapunov-Guided Optimization: Chaos-Aware Learning Rate
def get_chaos_aware_lr(it, metrics, base_lr):
    """
    Adjusts Learning Rate based on the model's physiological stress.
    
    This is **Homeostatic Optimization**: the learning rate responds to the
    dynamical state of the system, preventing explosions in chaotic regions
    and accelerating in stable basins.
    
    CRITICAL FIX: This function now RESPECTS the normal LR schedule (warmup/decay)
    by calling get_lr() first, then applying chaos as a multiplicative modifier.
    This ensures warmup happens correctly and chaos is a local tweak, not a replacement.
    
    Args:
        it: Current iteration number
        metrics: Dictionary containing DEQ convergence diagnostics:
            - 'num_iters': How many fixed-point iterations were needed
            - 'final_residual': How far from equilibrium (||f(z) - z||)
        base_lr: The base learning rate (passed to get_lr for schedule)
    
    Returns:
        Scheduled LR with chaos-based multiplicative adjustment
    """
    # 1. FIRST: Compute the normal scheduled LR (warmup + cosine decay)
    # This is the SOURCE OF TRUTH for the learning rate trajectory
    lr = get_lr(it)
    
    # 2. THEN: Apply chaos as a multiplicative modifier on that scheduled LR
    # The chaos throttle adjusts the already-scheduled LR, not the base LR
    
    # Stress Signal 1: Thinking too hard (hitting max iters)
    # Normalized to deq_max_iter (typically 30)
    stress_iters = min(1.0, metrics.get('num_iters', 0) / deq_max_iter)
    
    # Stress Signal 2: High Residual (Energy not conserved/minimized)
    # Use LOGARITHMIC scale to distinguish orders of magnitude
    # 🔥 EUSTRESS CALIBRATION: Early exploration needs HIGH residuals to learn!
    # The model MUST be allowed to explore chaotic regions during initialization.
    # Only truly explosive residuals (>10.0) should trigger throttling.
    #
    # RECALIBRATED SCALE (shifted tolerance window):
    # - Residual 100.0: True explosion, dangerous
    # - Residual 10.0: High exploration, ACCEPTABLE early on
    # - Residual 1.0: Healthy orbit search, GOOD (this is where successful runs stabilize)
    # - Residual 0.1: Converging to fixed point
    # - Residual 0.04: Orbital stability (sin(1) ≈ 0.84 throttle observed in good runs)
    # - Residual 1e-2 (0.01): Locked in
    # - Residual 1e-3 (0.001): Perfect equilibrium
    raw_res = metrics.get('final_residual', 0.0)
    if raw_res > 0:
        # NEW: Shift tolerance window - residual of 1.0 is now "safe" (score 0.2, not 0.6)
        # Map log10(residual) from [-3, +2] but with shifted zero point
        # log10(1e-3) = -3 -> -0.33 -> 0.0 (Zen - perfect)
        # log10(0.04) = -1.4 -> -0.07 -> 0.0 (Orbital - successful runs!)
        # log10(0.1) = -1 -> 0.0 -> 0.17 (Good)
        # log10(1.0) = 0 -> +1.0 -> 0.33 (Exploring - ACCEPTABLE)
        # log10(10.0) = +1 -> +2.0 -> 0.67 (High exploration - start watching)
        # log10(100.0) = +2 -> +3.0 -> 0.83 (Panic - explosive)
        log_res = math.log10(max(raw_res, 1e-4))  # Clamp to avoid log(0)
        stress_residual = (log_res + 1.0) / 6.0  # Map [-1, +2] to [0, 0.5], shifted tolerance
        stress_residual = max(0.0, min(1.0, stress_residual))  # Clamp to [0, 1]
    else:
        stress_residual = 0.0
    
    # Combined Chaos Score (0.0 = Zen, 1.0 = Panic)
    chaos_score = max(stress_iters, stress_residual)
    
    # The Valve: If chaos is high, throttle the LR
    # 🔥 CRITICAL: Raise the floor to 0.5 (never drop below 50% LR)
    # This ensures the model retains enough plasticity to escape initialization basin
    # and find the stable 0.84 orbital regime observed in successful runs.
    # If chaos > 0.8, LR drops to 50%. If chaos < 0.2, LR is 100%.
    throttle = 1.0 - max(0, (chaos_score - 0.2) / 0.8)
    throttle = max(0.5, throttle)  # 🚀 NEVER drop below 50% LR (was 0.1)
    
    # REMOVED: Early training override (no longer needed with recalibrated chaos sensor)
    # The new residual scale (log10 mapped from [-3, +1]) handles explosive init correctly:
    # - res = 10.0 → chaos = 1.0 → throttle = 0.1× (prevent explosion)
    # - res = 1.0 → chaos = 0.75 → throttle = 0.4× (gentle descent)
    # - res = 0.1 → chaos = 0.5 → throttle = 0.7× (converging)
    # This allows homeostatic feedback FROM STEP 0, keeping the marble on the fractal edge.
    
    # Apply the chaos throttle to the SCHEDULED lr, not base_lr
    return lr * throttle

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Initialize Homeostatic Monitor
monitor = HomeostaticMonitor(out_dir) if master_process else None

# Initialize Diagnostic Reporter  
diagnostic = DiagnosticReporter(out_dir) if master_process else None

# Prepare reports directory and CSV logging for per-iteration metrics
reports_dir = os.path.join(out_dir, 'reports')
os.makedirs(reports_dir, exist_ok=True)

# Initialize background worker for async viz/diagnostics (non-blocking)
bg_worker = BackgroundWorker(max_queue_size=3) if master_process else None
if master_process:
    print(f"🧵 Background viz worker ready - graphs→ {reports_dir}")
    print(f"   Dashboard updates every 20 iters (silent, check files)\n")

# Adaptive min_lr to prevent training stagnation
lr_plateau = {'min_lr': min_lr, 'plateau_counter': 0, 'last_loss': 1e9}

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

# Note: reports_dir already created above with bg_worker init
metrics_csv = os.path.join(reports_dir, 'metrics.csv')
# write header if new
if master_process and not os.path.exists(metrics_csv):
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iter','loss','time_ms_synced','mfu_percent','deq_iters','timestamp'])

# Track previous metrics for chaos-aware LR
prev_metrics = {'num_iters': 0, 'final_residual': 0.0}

while True:
    # 💀 MEMORY SAFETY: Check RAM usage every iteration
    if master_process:
        ram_gb = check_memory_safety()
        # Print warning at 15GB (intentional large caches are OK, leaks are not)
        if ram_gb > 15.0 and iter_num % 10 == 0:
            print(f"⚠️  RAM usage: {ram_gb:.2f}GB (limit: {MAX_RAM_GB}GB) - large caches active")

    # determine and set the learning rate for this iteration
    # 🚀 FERRARI MODE: Keep LR high, use Thermostat Loss instead of throttling
    # Robot Arm insight: Don't brake the optimizer - teach the weights to stabilize!
    # The stability_loss term (added to balanced_loss) trains the network to self-regulate.
    use_chaos_aware = getattr(config, 'use_chaos_aware_lr', True)  # Can disable for baseline
    
    # CHANGED: Don't throttle LR based on chaos - just use normal schedule
    # The thermostat loss will handle stability training via gradients
    if decay_lr:
        lr = get_lr(iter_num)  # Normal cosine schedule (warmup + decay)
    else:
        lr = learning_rate  # Constant LR (debug mode)
    
    # Optional: Still compute chaos metrics for monitoring (but don't throttle!)
    if use_chaos_aware and iter_num > 0 and prev_metrics:
        # Compute chaos score for diagnostics and stability_loss (BEFORE forward pass)
        # 🔥 RECALIBRATED CHAOS FORMULA - matches eustress calibration
        stress_iters = min(1.0, prev_metrics['num_iters'] / deq_max_iter)
        raw_res = prev_metrics['final_residual']
        if raw_res > 0:
            # NEW SCALE: Residual 1.0 is now "safe" (was "chaos")
            # log10(1e-3) = -3 -> -2.0 -> -0.33 -> 0.0 (Perfect)
            # log10(0.04) = -1.4 -> -0.4 -> -0.07 -> 0.0 (Orbital - sin(1) regime!)
            # log10(0.1) = -1 -> 0.0 -> 0.0 -> 0.0 (Good)
            # log10(1.0) = 0 -> +1.0 -> +0.17 -> 0.17 (Exploring)
            # log10(10.0) = +1 -> +2.0 -> +0.50 -> 0.50 (High exploration)
            # log10(100.0) = +2 -> +3.0 -> +0.83 -> 0.83 (Panic)
            log_res = math.log10(max(raw_res, 1e-4))
            stress_residual = (log_res + 1.0) / 6.0  # NEW: Shifted tolerance window
            stress_residual = max(0.0, min(1.0, stress_residual))
        else:
            stress_residual = 0.0
        chaos_score_current = max(stress_iters, stress_residual)
        
        # Store in a temp variable for stability_loss to use
        # Will be overwritten by actual metrics after forward pass
        current_chaos_estimate = chaos_score_current
    else:
        current_chaos_estimate = 0.0
    
    # Apply LR to param groups, but protect balancer's independent LR
    # The balancer (last param group) should learn faster to regulate the network
    # CRITICAL: Balancer must use CONSTANT LR (immune to chaos throttling)
    # BUT not TOO high - sustained gradients can cause runaway weights
    # DO NOT use 'learning_rate' variable - it may be overridden by config!
    # 1e-3 = 10× network LR (good responsiveness without explosion risk)
    balancer_lr_fixed = 1e-3  # FIXED CONSTANT - not affected by config or chaos
    
    for i, param_group in enumerate(optimizer.param_groups):
        # Last param group is the balancer (4 σ parameters)
        if i == len(optimizer.param_groups) - 1:
            # FORCE balancer LR to stay at 10× base rate (ignore chaos throttle)
            # This allows the balancer to adapt quickly to volatility even when
            # the main network is slowed down by high chaos
            param_group['lr'] = balancer_lr_fixed
        else:
            # Apply chaos-aware LR to main network params only
            param_group['lr'] = lr

    # 🌀 HOMEOSTATIC PERTURBATION: Periodic balancer reset for phase transitions
    # Every 500 iters, reset Bayesian uncertainties to force re-exploration
    # This prevents over-confident priors and enables developmental stage transitions
    if iter_num % 500 == 0 and iter_num > 0 and master_process:
        if hasattr(raw_model, 'balancer') and raw_model.balancer is not None:
            print(f"\n⚡ HOMEOSTATIC PERTURBATION (iter {iter_num})")
            print(f"   Resetting Bayesian uncertainties to escape local optimum")
            print(f"   Old σ values will be forgotten, forcing fresh task re-evaluation")
            # Reset all learned log_sigma parameters back to 0 (σ=1.0)
            for name, param in raw_model.balancer.named_parameters():
                if 'log_sigma' in name:
                    param.data.zero_()
            print(f"   ✓ Balancer reset complete - entering re-exploration phase\n")
    
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        # DISABLED: Validation runs are expensive (100 forward passes every 100 iters)
        # Use diagnostic reports + sample generation instead for faster iteration
        # losses = estimate_loss()
        # print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Log to monitor (using current training loss instead)
        if monitor:
            # monitor.log_loss(iter_num, losses['train'], losses['val'])
            pass  # Skip validation loss logging
        
        # Generate samples to see model progress (skip at iter 0 to speed startup)
        if iter_num > 0:
            # Every 20 iters: create reflex animation GIF
            if iter_num % 20 == 0:
                generate_samples_with_visualization(iter_num)
            else:
                generate_samples()
        
        # Visualize phase space dynamics (every eval, but skip iter 0 if memory was preloaded)
        if iter_num > 0 or not globals().get('preload_num_samples', 0):
            visualize_phase_space(iter_num)
        
        # DISABLED wandb logging (was using validation loss)
        # if wandb_log:
        #     wandb.log({
        #         "iter": iter_num,
        #         "train/loss": losses['train'],
        #         "val/loss": losses['val'],
        #         "lr": lr,
        #         "mfu": running_mfu*100, # convert to percentage
        #     })
        
        # DISABLED validation-based checkpointing (save periodically instead)
        # if losses['val'] < best_val_loss or always_save_checkpoint:
        #     best_val_loss = losses['val']
        #     if iter_num > 0:
        #         checkpoint = {
        #             'model': raw_model.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'model_args': model_args,
        #             'iter_num': iter_num,
        #             'best_val_loss': best_val_loss,
        #             'config': config,
        #         }
        #         ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        #         print(f"saving checkpoint to {out_dir}")
        #         torch.save(checkpoint, ckpt_path)
        #         
        #         # Save MEMORY checkpoint separately (if using hybrid memory)
        #         if use_memory_manifold and hasattr(raw_model.reflex, 'save_memory_checkpoint'):
        #             memory_ckpt_path = os.path.join(out_dir, 'memory_ckpt.pkl')
        #             raw_model.reflex.save_memory_checkpoint(memory_ckpt_path)
        #             print(f"  ✓ Saved memory state to {memory_ckpt_path}")
        #         
        #         # Save checkpoint summary
        #         if monitor:
        #             monitor.save_checkpoint_summary(iter_num, ckpt_path)
    
    # Periodic checkpointing every 100 iters (independent of eval)
    if iter_num > 0 and iter_num % 100 == 0 and master_process:
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': config,
        }
        ckpt_path = os.path.join(out_dir, f'ckpt_iter{iter_num}.pt')
        print(f"💾 Periodic checkpoint → {ckpt_path}")
        torch.save(checkpoint, ckpt_path)
        
        # Save memory too
        if use_memory_manifold and hasattr(raw_model.reflex, 'save_memory_checkpoint'):
            memory_ckpt_path = os.path.join(out_dir, f'memory_ckpt_iter{iter_num}.pkl')
            raw_model.reflex.save_memory_checkpoint(memory_ckpt_path)
    
    # Sample inference every 500 iters (minimal overhead for max speed)
    # This prevents double generation at multiples of eval_interval
    # NOTE: Sample generation is synchronous (model not thread-safe)
    # Reduced frequency + fewer tokens = maximum training throughput
    if iter_num > 0 and iter_num % 500 == 0 and iter_num % eval_interval != 0 and master_process:
        print(f"📝 Generating sample (iter {iter_num})...")
        generate_samples()
    
    if iter_num == 0 and eval_only:
        break

    # Reset profiler after iter 0 to exclude initial validation from ongoing stats
    if iter_num == 1 and enable_profiling:
        profiler.reset()

    # Start iteration timer
    profiler.start_iter()
    
    # Data loading
    profiler.start('data_loading')
    X, Y = get_batch('train')
    profiler.stop('data_loading')

    # Zero gradients before accumulation loop
    # CRITICAL: Must be done BEFORE micro-steps to prevent graph retention errors
    optimizer.zero_grad(set_to_none=True)

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        
        # Forward pass
        profiler.start('forward')
        with ctx:
            # HOMEOSTATIC REFLEX GATING: Compute β(t) gate coefficient
            reflex_gate = get_reflex_gate(iter_num)
            
            # Pass gate to model forward (forces cortical development before spinal automation)
            logits, loss_raw, metrics = model(X, Y, return_metrics=True, training_iter=iter_num, reflex_gate=reflex_gate)
            
            # Track gate value for monitoring
            metrics['reflex_gate'] = reflex_gate
            
            # ═══════════════════════════════════════════════════════════════
            # HOMEOSTATIC BAYESIAN LOSS BALANCING
            # ═══════════════════════════════════════════════════════════════
            # Instead of manually setting λ_pauli, λ_jacobian, etc., we use
            # Bayesian uncertainty balancing to learn optimal task weights.
            # 
            # Collect all raw loss components (DO NOT sum them manually!)
            # CRITICAL: Start with model's loss_components (prediction + jacobian)
            loss_components = metrics.get('loss_components', {}).copy()  # COPY to avoid mutation!
            
            # ═══════════════════════════════════════════════════════════════════
            # BOREDOM/CURIOSITY LOSS (Replaces Pauli Exclusion)
            # ═══════════════════════════════════════════════════════════════════
            # Old: Pauli exclusion (local anti-stutter) - "Don't repeat tokens in sequence"
            # New: True curiosity drive - "Am I stuck in a boring pattern globally?"
            #
            # Two components:
            # 1. EPISTEMIC UNCERTAINTY (Option B): How confident are my predictions?
            #    - Low entropy = peaked distribution = too certain = bored
            #    - High entropy = uncertain = exploring = curious
            #
            # 2. MEMORY EXPLORATION (Option C): Am I using diverse memories?
            #    - Low diversity = stuck in same memory region = bored  
            #    - High diversity = exploring knowledge space = curious
            #
            # Together: Drives the system to explore both internally (uncertainty)
            #           and externally (memory access patterns)
            # ═══════════════════════════════════════════════════════════════════
            
            boredom_loss = torch.tensor(0.0, device=device)
            
            if lambda_pauli > 0:  # Reuse lambda_pauli as boredom weight
                # Component 1: Epistemic Uncertainty (Prediction Entropy)
                # Measure how uncertain the model is about next token
                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)  # [B, T, V]
                    # Entropy per position: -sum(p * log(p))
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, T]
                    avg_entropy = entropy.mean()
                    
                    # Target entropy: We want model to be somewhat uncertain
                    # Too low = overconfident (meta-tokens), too high = random
                    # Natural language entropy ≈ 3-5 bits (8-32 plausible next tokens)
                    entropy_target = 3.5
                    
                    # Boredom from overconfidence
                    # If avg_entropy < target, model is too certain (stuck in rut)
                    epistemic_boredom = F.relu(entropy_target - avg_entropy)
                
                # Component 2: Memory Exploration Diversity
                # Track which memories are being accessed
                memory_diversity_score = torch.tensor(1.0, device=device)  # Default: neutral
                
                if use_memory_manifold and hasattr(raw_model.reflex, 'memory_retrieval'):
                    memory_stats = raw_model.reflex.memory_retrieval.get_memory_stats()
                    
                    # Get access tracking if available
                    if hasattr(raw_model.reflex.memory_retrieval, 'get_diversity_score'):
                        # Memory system provides diversity metric
                        memory_diversity_score = raw_model.reflex.memory_retrieval.get_diversity_score()
                    else:
                        # Fallback: Estimate from memory stats
                        longterm_nodes = memory_stats.get('longterm_nodes', 0)
                        if longterm_nodes > 0:
                            # Simple heuristic: Assume accessing ~20% of memory is good
                            # If stuck, might only access ~5% repeatedly
                            # This is approximate - ideally track actual access patterns
                            diversity_target = 0.2
                            estimated_diversity = 0.15  # Conservative estimate
                            memory_boredom = diversity_target - estimated_diversity
                            memory_diversity_score = torch.tensor(max(0, memory_boredom), device=device)
                
                # Combine both components
                # Weight epistemic higher (0.7) since it's more reliable
                # Memory diversity (0.3) as supporting signal
                w_epistemic = 0.7
                w_memory = 0.3
                
                boredom_loss = (
                    w_epistemic * epistemic_boredom +
                    w_memory * memory_diversity_score
                )
                
                # Store epistemic entropy for monitoring
                metrics['epistemic_entropy'] = avg_entropy.item()
                metrics['entropy_target'] = entropy_target
            
            loss_components['novelty'] = boredom_loss
            
            # Add Memory Reconstruction loss (contrastive quality metric)
            # This measures if retrieved memories are semantically relevant
            # Balancer will learn to downweight when memory is noisy (early training)
            # and upweight when memory becomes organized (late training)
            memory_loss = torch.tensor(0.0, device=device)
            if use_memory_manifold and hasattr(raw_model.reflex, 'get_last_recon_loss'):
                mem_loss = raw_model.reflex.get_last_recon_loss()
                # Safety check: Ensure it's a valid tensor (not None)
                if mem_loss is not None and isinstance(mem_loss, torch.Tensor):
                    memory_loss = mem_loss
            loss_components['memory'] = memory_loss
            
            # ═══════════════════════════════════════════════════════════════════
            # MEMORY NAVIGATION REWARDS (Dopamine for Graph Exploration)
            # ═══════════════════════════════════════════════════════════════════
            # Reward the DEQ for intelligent memory traversal:
            # 1. ACCESS: Using memory vs encoder-only
            # 2. DEPTH: Multi-hop reasoning through graph
            # 3. EFFICIENCY: Short paths in hyperbolic space
            # 4. SUCCESS: Retrieved memories helped prediction
            #
            # These are NEGATIVE losses (rewards), so they reduce total loss
            # when the model explores memory effectively.
            # ═══════════════════════════════════════════════════════════════════
            
            nav_reward = torch.tensor(0.0, device=device)
            if use_memory_manifold and hasattr(raw_model.reflex, 'get_last_memory_bundle'):
                memory_bundle = raw_model.reflex.get_last_memory_bundle()
                
                # Compute navigation rewards
                if memory_bundle is not None:
                    # Get prediction error per token for success reward
                    with torch.no_grad():
                        pred_error = F.cross_entropy(
                            logits.view(-1, logits.size(-1)), 
                            Y.view(-1), 
                            reduction='none'
                        ).view(Y.shape)  # [B, T]
                    
                    # Compute all navigation rewards (exploration)
                    exploration_reward, reward_breakdown = memory_nav_rewards.compute_total_reward(
                        memory_bundle,
                        prediction_error=pred_error,
                        encoder_baseline=None  # Could pass encoder output if available
                    )
                    
                    # ═══════════════════════════════════════════════════════════════
                    # SUPERVISED NAVIGATION: Teacher-force memory paths from ground truth
                    # ═══════════════════════════════════════════════════════════════
                    supervised_reward = torch.tensor(0.0, device=device)
                    if model_args.get('enable_supervised_nav', False):
                        supervised_iters = model_args.get('supervised_nav_iters', 2000)
                        lookahead = model_args.get('supervised_lookahead', 4)
                        
                        # Check if we have enough memories to do supervised navigation
                        has_enough_memories = False
                        if hasattr(raw_model.reflex.memory_retrieval, 'memory'):
                            mem_tier = raw_model.reflex.memory_retrieval.memory
                            has_enough_memories = mem_tier.size.item() >= 50  # Need at least 50 memories
                        
                        # Decay teacher forcing over time (1.0 → 0.0)
                        teacher_weight = max(0.0, 1.0 - iter_num / supervised_iters)
                        
                        if teacher_weight > 0.01 and has_enough_memories:  # Only if memories exist
                            try:
                                # Get ground truth future tokens for each position
                                B, T = Y.shape
                                
                                # For each token, look ahead to get "oracle" context
                                for b in range(min(B, 1)):  # Process first batch item only (speed)
                                    for t in range(min(T - lookahead, 10)):  # Sample 10 positions
                                        # Ground truth continuation
                                        target_tokens = Y[b, t+1:t+1+lookahead]
                                        if len(target_tokens) < lookahead:
                                            continue
                                        
                                        # Embed target tokens into semantic space
                                        with torch.no_grad():
                                            target_emb = raw_model.transformer.wte(target_tokens).mean(dim=0)  # [n_embd]
                                            
                                            # Find "oracle" memories nearest to ground truth
                                            if hasattr(raw_model.reflex.memory_retrieval, 'memory'):
                                                mem_tier = raw_model.reflex.memory_retrieval.memory
                                                if mem_tier.size.item() > 5:
                                                    # Compute distances to all long-term memories
                                                    mem_embs = mem_tier.embeddings[:mem_tier.size.item()]  # [N, n_embd]
                                                    distances = torch.cdist(
                                                        target_emb.unsqueeze(0), 
                                                        mem_embs
                                                    )[0]  # [N]
                                                    
                                                    # Top-5 oracle memories
                                                    oracle_indices = distances.topk(5, largest=False).indices
                                                    
                                                    # Check overlap with actually retrieved memories
                                                    if 'retrieved_indices' in memory_bundle:
                                                        retrieved = memory_bundle['retrieved_indices']
                                                        if retrieved is not None and len(retrieved) > 0:
                                                            overlap = len(set(oracle_indices.cpu().numpy()) & 
                                                                        set(retrieved))
                                                            supervised_reward += overlap / 5.0
                                
                                # Average over positions sampled
                                if supervised_reward.item() > 0:
                                    supervised_reward = supervised_reward / 10.0  # Normalize by positions
                                    
                            except Exception as e:
                                # Fail gracefully if supervised nav has issues
                                if iter_num % 100 == 0:
                                    print(f"  ⚠️  Supervised nav error: {e}")
                                supervised_reward = torch.tensor(0.0, device=device)
                            
                            # Blend exploration + supervised rewards
                            total_reward = (
                                (1 - teacher_weight) * exploration_reward + 
                                teacher_weight * supervised_reward * model_args.get('supervised_nav_weight', 0.5)
                            )
                            
                            reward_breakdown['supervised'] = supervised_reward
                            reward_breakdown['teacher_weight'] = teacher_weight
                        else:
                            # Pure exploration after teacher weight decays
                            total_reward = exploration_reward
                    else:
                        # No supervised navigation
                        total_reward = exploration_reward
                    
                    # Navigation reward is NEGATIVE loss (reward = lower loss)
                    nav_reward = -total_reward  # Flip sign: reward becomes negative loss
                    
                    # Store breakdown for monitoring
                    if iter_num % 50 == 0 and master_process:
                        metrics['nav_rewards'] = {k: v.item() if isinstance(v, torch.Tensor) else v 
                                                   for k, v in reward_breakdown.items()}
            
            loss_components['nav_reward'] = nav_reward
            
            # ═══════════════════════════════════════════════════════════════════
            # COUPLED LOSS: Memory→Prediction Attribution
            # ═══════════════════════════════════════════════════════════════════
            # Problem: Memory and prediction are causally linked, but balancer treats
            #          them as independent. If memo_w is low, memory can't prove value.
            #
            # Solution: Attribute a portion of prediction loss to memory quality.
            #          If memory was retrieved, give it partial credit/blame for
            #          prediction outcomes. This helps balancer see the dependency.
            #
            # Activation: Only when system is stuck (high novelty, low memory weight)
            # ═══════════════════════════════════════════════════════════════════
            
            # Check if coupled loss should activate
            activate_coupling = False
            if use_memory_manifold and hasattr(raw_model.reflex, 'memory_retrieval'):
                memory_stats = raw_model.reflex.memory_retrieval.get_memory_stats()
                longterm_nodes = memory_stats.get('longterm_nodes', 0)
                
                # Get current weights from previous iteration's balance_stats
                # These are stored in metrics from last forward pass
                prev_nove_weight = 1.0
                prev_memo_weight = 1.0
                if 'balance_stats' in metrics:
                    prev_nove_weight = metrics['balance_stats'].get('weight_novelty', 1.0)
                    prev_memo_weight = metrics['balance_stats'].get('weight_memory', 1.0)
                
                # Activation criteria: High exploration pressure + low memory value + sufficient memories
                if (prev_nove_weight > 3.5 and 
                    prev_memo_weight < 0.7 and 
                    longterm_nodes > 500):
                    activate_coupling = True
            
            # Apply coupling if activated
            if activate_coupling and memory_loss.item() > 0:
                # Attribute 30% of prediction loss to memory quality
                # This doesn't boost memory gradient directly - just tells balancer:
                # "When prediction fails, memory shares responsibility"
                memory_contribution_factor = 0.3
                
                # Create coupled memory loss
                prediction_component = loss_components['prediction'].detach()  # Don't double-count gradients
                memory_coupled = memory_loss + memory_contribution_factor * prediction_component
                
                # Replace memory loss with coupled version
                loss_components['memory'] = memory_coupled
                
                # Log coupling activation (only occasionally to avoid spam)
                if iter_num % 100 == 0:
                    print(f"[Coupled Loss] Active: memo_w={prev_memo_weight:.2f}, "
                          f"nove_w={prev_nove_weight:.2f}, LT={longterm_nodes}")
            
            # Apply Homeostatic Balancer
            # This automatically weights each loss by 1/σ² where σ is learned
            # 🔥 ALWAYS ENABLED: Core part of differentiable cybernetics architecture
            USE_BALANCER = True  # Learnable loss weights are critical for auto PDE control
            
            if USE_BALANCER:
                balanced_loss, balance_stats = balancer(loss_components)
            else:
                # Pure prediction loss only - no auxiliary losses!
                balanced_loss = loss_components['prediction']
                balance_stats = {
                    'sigma_prediction': 1.0,
                    'sigma_jacobian': 1.0,
                    'sigma_novelty': 1.0,
                    'sigma_memory': 1.0,
                    'sigma_navigation': 1.0,
                    'weight_prediction': 1.0,
                    'weight_jacobian': 0.0,
                    'weight_novelty': 0.0,
                    'weight_memory': 0.0,
                    'weight_navigation': 0.0
                }
            
            # 🌡️ THERMOSTAT (not Panic Button): Stability as a Loss Term
            # Robot Arm insight: Don't kill the LR when unstable - teach the weights to stabilize!
            # Add a DIFFERENTIABLE penalty that pushes the system toward target entropy/residual
            # This allows the optimizer to LEARN stability, rather than just braking.
            #
            # Target: chaos_score ≈ sin(1) ≈ 0.841 (Natural attractor for limit cycles!)
            # This is the "Golden Ratio" of dynamical systems - stable periodic orbit
            # Penalty scales with distance from target - model learns to self-regulate
            stability_loss_tensor = torch.tensor(0.0, device=device)
            stability_loss_value = 0.0
            if use_chaos_aware and 'current_chaos_estimate' in locals():
                chaos = current_chaos_estimate  # Use PRE-computed chaos from prev step
                chaos_target = math.sin(1.0)  # ≈ 0.841 - Natural limit cycle attractor
                chaos_deviation = abs(chaos - chaos_target)
                
                # Scale penalty: strong when far from target, weak when close
                # γ_thermo (thermostat gain) - how hard to push toward equilibrium
                gamma_thermo = 0.05  # Start gentle (5% of task loss weight)
                stability_loss_tensor = torch.tensor(gamma_thermo * (chaos_deviation ** 2), device=device)
                stability_loss_value = stability_loss_tensor.item()
                
                # Add to total loss - this trains the WEIGHTS to be stable
                balanced_loss = balanced_loss + stability_loss_tensor
                
                # Store for diagnostics
                metrics['stability_loss'] = stability_loss_value
                metrics['chaos_target'] = chaos_target
                metrics['chaos_current'] = chaos
            
            # HOMEOSTATIC FEEDBACK: Update memory system with balancer's uncertainty
            # This creates adaptive consolidation - reduce memory formation when σ_memory is low
            if use_memory_manifold and hasattr(raw_model.reflex, 'memory_retrieval'):
                sigma_memory = balance_stats.get('sigma_memory', 1.0)
                raw_model.reflex.memory_retrieval.update_balancer_feedback(sigma_memory)
                
                # REMOVED: apply_dopamine() - reward tracking no longer needed
                # Quality is now tracked via edge_success_rate and edge_traversal_count
                
                # 🛣️ HIGHWAY STRENGTHENING: Reinforce retrieval paths based on ACTUAL PER-TOKEN loss
                # This creates "highways" through the memory graph - fast paths to useful knowledge!
                # Only strengthen paths that lead to GOOD predictions (token-level precision!)
                if 'loss_per_token' in metrics and metrics['loss_per_token'] is not None:
                    raw_model.reflex.memory_retrieval.strengthen_last_retrieval(metrics['loss_per_token'])
                
                # 🧠 ONLINE MEMORY FORMATION: Store surprising or successful patterns
                # This enables continual learning across datasets!
                # Surprise (high loss) → "I don't know this, remember it!"
                # Success (low loss) → "This works well, consolidate it!"
                if iter_num % 10 == 0 and 'loss_per_token' in metrics and metrics['loss_per_token'] is not None:
                    # Get the reflex state from the model's last forward pass
                    # The reflex output is the final layer norm output before logits
                    if hasattr(raw_model.reflex, 'ln_f'):
                        # During training, we need to do a quick forward to get reflex states
                        # Or we can use the logits and back-project (expensive)
                        # For now, let's just store based on loss without explicit reflex embedding
                        # TODO: Add reflex_state to metrics for efficient access
                        
                        per_token_loss = metrics['loss_per_token']  # [B, T]
                        
                        # Thresholds (configurable)
                        surprise_threshold = 3.0  # Store if loss > 3.0 (very surprised)
                        success_threshold = 0.3   # Store if loss < 0.3 (very successful)
                        
                        B, T = per_token_loss.shape
                        stored_count = 0
                        
                        # Sample a few positions to avoid overhead
                        # Store at most 5 memories per iteration
                        max_stores_per_iter = 5
                        
                        for b in range(B):
                            if stored_count >= max_stores_per_iter:
                                break
                            for t in range(T):
                                if stored_count >= max_stores_per_iter:
                                    break
                                    
                                loss_val = per_token_loss[b, t].item()
                                
                                # Store if surprising OR successful
                                if loss_val > surprise_threshold or loss_val < success_threshold:
                                    # Use the token embedding as a proxy for reflex state
                                    # This is not perfect but allows memory formation without extra forward pass
                                    token_id = X[b, t].item()
                                    token_emb = raw_model.encoder.wte.weight[token_id].detach()
                                    
                                    # Convert loss to reward: lower loss → higher reward
                                    reward = 1.0 / (1.0 + loss_val * 0.5)
                                    reward = max(0.1, min(2.0, reward))  # Clamp to [0.1, 2.0]
                                    
                                    raw_model.reflex.memory_retrieval.store_memory_dynamic(
                                        token_emb,
                                        reward=reward
                                    )
                                    stored_count += 1
                        
                        # Log occasionally
                        if stored_count > 0 and iter_num % 100 == 0:
                            mem_size = raw_model.reflex.memory_retrieval.memory.size.item()
                            print(f"[Memory Formation] Stored {stored_count} new memories (total: {mem_size})")
            
            # Scale for gradient accumulation
            loss = balanced_loss / gradient_accumulation_steps
            
            # Store components for monitoring
            metrics['loss_components_raw'] = {k: v.item() if isinstance(v, torch.Tensor) else v 
                                               for k, v in loss_components.items()}
            metrics['balance_stats'] = balance_stats
            metrics['novelty_drive'] = boredom_loss.item()  # Now boredom/curiosity instead of Pauli
            metrics['loss_base'] = loss_components['prediction'].item()
            metrics['loss_balanced'] = balanced_loss.item()
            
        profiler.stop('forward')
        
        # Data prefetch
        profiler.start('data_prefetch')
        X, Y = get_batch('train')
        profiler.stop('data_prefetch')
        
        # 🔮 MEMORY PREFETCH: Hint which bundles will be needed for next iteration
        # This loads bundles into cache during backward pass (parallel I/O!)
        if iter_num > 0 and use_memory_manifold and hasattr(raw_model.reflex, 'memory_retrieval'):
            try:
                # Quick approximate query using next batch token embeddings (no DEQ needed!)
                with torch.no_grad():
                    approx_query = raw_model.encoder.wte(X).mean(dim=[0, 1])  # [D] - cheap!
                
                # Get likely memory indices for next forward pass
                likely_indices = raw_model.reflex.memory_retrieval.approx_knn_indices(
                    approx_query, k=20
                )
                
                # Hint to prefetch (non-blocking - just queues for background loading)
                if likely_indices:
                    raw_model.reflex.memory_retrieval.memory.hint_prefetch(likely_indices)
            except Exception as e:
                # Prefetch is best-effort - don't break training if it fails
                pass
        
        # Backward pass
        profiler.start('backward')
        scaler.scale(loss).backward()
        
        # 🔮 PROCESS PREFETCH HINTS: Load bundles while gradients are fresh
        # This happens during backward - bundles load in parallel with gradient computation
        if iter_num > 0 and use_memory_manifold and hasattr(raw_model.reflex, 'memory_retrieval'):
            try:
                raw_model.reflex.memory_retrieval.process_prefetch_hints()
            except Exception as e:
                pass
        
        profiler.stop('backward')
    
    # Gradient clipping
    profiler.start('grad_clip')
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        metrics['grad_norm'] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    else:
        # Still compute grad norm even without clipping
        scaler.unscale_(optimizer)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** 0.5
        metrics['grad_norm'] = grad_norm
    profiler.stop('grad_clip')
    
    # 🔬 BALANCER PARAMETER TRACKING (Every 10 iters - lightweight check)
    # Show actual log_vars values to see if they're moving from 0.0
    if iter_num % 10 == 0 and iter_num > 0 and master_process:
        log_vars_values = balancer.log_vars.data.cpu().numpy()
        sigmas = np.exp(0.5 * log_vars_values)
        weights = np.exp(-log_vars_values)
        
        # CRITICAL: Also show ACTUAL balancer LR being used
        actual_balancer_lr = optimizer.param_groups[-1]['lr']
        main_lr = optimizer.param_groups[0]['lr']
        
        # Balancer diagnostics saved to CSV and graphed automatically - no console spam
    
    # 🔬 BALANCER GRADIENT DIAGNOSTIC (saved to CSV and graphed)
    # Verify balancer parameters are being updated - data in diagnostics
    if iter_num % 100 == 0 and iter_num > 0 and master_process:
        balancer_grad_norm = 0.0
        balancer_param_norm = 0.0
        for param in balancer.parameters():
            if param.grad is not None:
                balancer_grad_norm += param.grad.data.norm(2).item() ** 2
                balancer_param_norm += param.data.norm(2).item() ** 2
        balancer_grad_norm = balancer_grad_norm ** 0.5
        balancer_param_norm = balancer_param_norm ** 0.5
        
        # Verify balancer LR is protected (all data in CSV/graphs)
        actual_balancer_lr = optimizer.param_groups[-1]['lr']
        expected_balancer_lr = 1e-3  # FIXED CONSTANT
        main_network_lr = optimizer.param_groups[0]['lr']
    
    # 🔬 GRADIENT FLOW VISUALIZATION: Topological debugging every 20 iters
    # Reveals the local geometry of the loss landscape
    # This shows if we're converging (sink), oscillating (vortex), or bypassing (orthogonal flows)
    # RUN IN BACKGROUND - don't block training!
    if iter_num % 20 == 0 and iter_num > 0 and master_process and bg_worker and VISUALIZATION_AVAILABLE:
        # CRITICAL: visualize_gradient_flow uses matplotlib, must wrap with lock
        def safe_gradient_flow_viz():
            with BackgroundWorker._matplotlib_lock:
                visualize_gradient_flow(raw_model, optimizer, iter_num, out_dir, max_layers=40)
        
        bg_worker.submit(
            "Gradient Flow Viz",
            safe_gradient_flow_viz
        )
    
    # Optimizer step
    profiler.start('optimizer')
    
    # 🚨 CRITICAL PRE-STEP VERIFICATION (silent - data in CSV)
    # Verify balancer LR is ACTUALLY protected before taking the step
    if iter_num % 10 == 0 and master_process:
        actual_bal_lr = optimizer.param_groups[-1]['lr']
        expected_bal_lr = 1e-3  # FIXED CONSTANT (same as line 1075)
        if abs(actual_bal_lr - expected_bal_lr) > 1e-9:
            # Critical error - force correction (this should never happen)
            optimizer.param_groups[-1]['lr'] = expected_bal_lr
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    profiler.stop('optimizer')
    
    # 🔒 BALANCER WEIGHT CLAMPING - Prevent runaway weights
    # Balancer log_vars should stay in reasonable range: [-3, +3]
    # This corresponds to weights in range [0.05, 20.0]
    if balancer is not None:
        with torch.no_grad():
            balancer.log_vars.data.clamp_(-3.0, 3.0)
    
    # ══════════════════════════════════════════════════════════════
    # MEMORY SYSTEM UPDATE (removed - no longer needed)
    # ══════════════════════════════════════════════════════════════
    # REMOVED: apply_dopamine_signal(), memory_step()
    # Quality tracking now happens via edge_success_rate and edge_traversal_count

    # timing and logging
    profiler.stop_iter()
    # make sure GPU work is finished so timings represent wall-clock runtime
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    # Save metrics for next iteration's chaos-aware LR adjustment
    prev_metrics = {
        'num_iters': metrics.get('num_iters', 0),
        'final_residual': metrics.get('final_residual', 0.0),
        'novelty_drive': metrics.get('novelty_drive', 0.0),  # ℂ: Boredom/Curiosity signal
        'loss_base': metrics.get('loss_base', 0.0),  # CE + Jacobian component
        'pauli_component': metrics.get('pauli_component', 0.0)  # Pauli contribution
    }
    
    # Calculate chaos score components for logging (matches NEW recalibrated formula!)
    stress_iters = min(1.0, prev_metrics['num_iters'] / deq_max_iter)
    # Use NEW logarithmic scale for residual (shifted tolerance window)
    raw_res = prev_metrics['final_residual']
    if raw_res > 0:
        log_res = math.log10(max(raw_res, 1e-4))
        stress_residual = (log_res + 1.0) / 6.0  # NEW: Shifted tolerance (was +3.0 / 5.0)
        stress_residual = max(0.0, min(1.0, stress_residual))
    else:
        stress_residual = 0.0
    chaos_score = max(stress_iters, stress_residual)
    throttle = 1.0 - max(0, (chaos_score - 0.2) / 0.8)
    throttle = max(0.5, throttle)  # Floor at 50% (but not used anymore - LR stays high)
    
    # HOMEOSTATIC ADAPTIVE FRICTION: Update Hamiltonian friction based on Chaos Score
    # This makes the damping self-regulate: High chaos → High γ → Aggressive damping

    # HOMEOSTATIC ADAPTIVE FRICTION: Update Hamiltonian friction based on Chaos Score
    # This makes the damping self-regulate: High chaos → High γ → Aggressive damping
    if hasattr(raw_model, 'deq') and hasattr(raw_model.deq, 'operator'):
        if hasattr(raw_model.deq.operator, 'update_friction'):
            gamma_current = raw_model.deq.operator.update_friction(chaos_score)
            # Only log occasionally to avoid spam
            if iter_num % (log_interval * 10) == 0:
                print(f"[Homeostatic] γ(chaos={chaos_score:.3f}) = {gamma_current:.4f}")
    
    # Adaptive minimum LR to prevent training stagnation
    # If loss plateaus for too long, raise min_lr to escape local minima
    if iter_num % 100 == 0 and iter_num > warmup_iters:  # Check every 100 iters after warmup
        current_loss = loss.item() * gradient_accumulation_steps if 'loss' in locals() else lr_plateau['last_loss']
        loss_improvement = lr_plateau['last_loss'] - current_loss
        
        # Plateau detection: loss not improving by at least 0.1%
        if abs(loss_improvement) < 0.001 * lr_plateau['last_loss']:
            lr_plateau['plateau_counter'] += 1
            if lr_plateau['plateau_counter'] >= 5:  # 500 iters of plateau
                # Raise min_lr by 50% (saved to CSV/graphs)
                old_min_lr = lr_plateau['min_lr']
                lr_plateau['min_lr'] = min(lr_plateau['min_lr'] * 1.5, learning_rate * 0.1)  # Cap at 10% of base LR
                lr_plateau['plateau_counter'] = 0  # Reset counter
        else:
            lr_plateau['plateau_counter'] = max(0, lr_plateau['plateau_counter'] - 1)  # Decay counter
        
        lr_plateau['last_loss'] = current_loss
    
    # Apply adaptive min_lr (override the static min_lr)
    lr = max(lr, lr_plateau['min_lr'])
    
    # Log to homeostatic monitor
    if monitor:
        monitor.log_deq(iter_num, prev_metrics['num_iters'], 
                       prev_metrics['final_residual'], dt * 1000.0)
        monitor.log_chaos(iter_num, chaos_score, stress_iters, 
                         stress_residual, throttle)
        monitor.log_lr(iter_num, lr, learning_rate)
        
        # Log reflex gate (developmental phase tracking)
        if 'reflex_gate' in metrics:
            monitor.log_reflex_gate(iter_num, metrics['reflex_gate'])
    
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        
        # 🚨 PLATEAU DETECTION: Track loss history and detect stagnation
        loss_history.append(lossf)
        if len(loss_history) > loss_window_size:
            loss_history.pop(0)
        
        # Compute loss gradient (rate of change) if we have enough history
        if len(loss_history) >= loss_window_size and iter_num > 100:
            # Linear regression slope of recent losses (numerical gradient)
            x = np.arange(len(loss_history))
            y = np.array(loss_history)
            loss_gradient = np.polyfit(x, y, 1)[0]  # Slope of best-fit line
            
            # Plateau detection: gradient near zero means stuck
            if abs(loss_gradient) < plateau_threshold:
                plateau_counter += 1
            else:
                plateau_counter = 0  # Reset if making progress
            
            # Soft intervention: boost exploration when plateaued
            if plateau_counter >= plateau_intervention_trigger and (iter_num - last_intervention_iter) > intervention_cooldown:
                print(f"\n⚠️  PLATEAU DETECTED (iter {iter_num}): Loss gradient {loss_gradient:.6f} < {plateau_threshold}")
                print(f"   Plateau duration: {plateau_counter} iterations")
                print(f"   🔧 SOFT INTERVENTION: Boosting exploration...")
                
                # Temporarily increase novelty weight in balancer
                if hasattr(balancer, 'log_vars'):
                    with torch.no_grad():
                        # Reduce novelty uncertainty → increase weight
                        balancer.log_vars.data[2] -= 0.5  # Index 2 is novelty
                    print(f"      ✓ Increased novelty drive weight")
                
                last_intervention_iter = iter_num
                plateau_counter = 0  # Reset after intervention
        
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        deq_iters = metrics.get('num_iters', 0)
        time_ms = dt * 1000.0
        
        # Log to homeostatic monitor
        if monitor:
            monitor.log_loss(iter_num, lossf)
            monitor.log_deq(iter_num, prev_metrics['num_iters'], 
                           prev_metrics['final_residual'], time_ms)
            monitor.log_chaos(iter_num, chaos_score, stress_iters, 
                             stress_residual, throttle)
            monitor.log_lr(iter_num, lr, learning_rate)
            
            # Log loss components (updated for new boredom-based system)
            if 'loss_components_raw' in metrics:
                raw_components = metrics['loss_components_raw']
                monitor.log_loss_components(
                    iter_num,
                    raw_components.get('prediction', 0.0) + raw_components.get('jacobian', 0.0),  # ce_jacobian combined
                    raw_components.get('novelty', 0.0),  # Now boredom instead of pauli
                    raw_components.get('memory', 0.0),  # Was efficiency, now memory
                    lossf
                )
            
            # 🧠 LOG BAYESIAN BALANCER STATISTICS (Dopamine Dynamics!)
            # This is what the plot_bayesian_brain.py script reads
            if balance_stats and 'loss_components_raw' in metrics:
                monitor.log_balancer(iter_num, balance_stats, metrics['loss_components_raw'])
            
            # Log memory stats
            if use_memory_manifold and hasattr(raw_model.reflex, 'get_memory_stats'):
                mem_stats = raw_model.reflex.get_memory_stats()
                if mem_stats:
                    monitor.log_memory_stats(
                        iter_num,
                        mem_stats.get('num_working', 0),
                        mem_stats.get('num_buffer', 0),
                        mem_stats.get('num_longterm', 0),
                        mem_stats.get('reconsolidations', 0)
                    )
                    
                    # Log memory quality (contrastive loss)
                    if 'loss_components_raw' in metrics:
                        memory_loss = metrics['loss_components_raw'].get('memory', 0.0)
                        monitor.log_memory_quality(
                            iter_num,
                            contrastive_loss=memory_loss,
                            avg_similarity=0.0,  # TODO: Extract from memory_info
                            retrieval_diversity=0.0  # TODO: Compute entropy of retrieval distribution
                        )
        
        # Show both raw residual and log-scaled chaos for transparency
        raw_residual = prev_metrics.get('final_residual', 0.0)
        novelty_drive = prev_metrics.get('novelty_drive', 0.0)
        reflex_gate_val = metrics.get('reflex_gate', 1.0)  # Get β(t)
        
        # Show background worker status periodically (compact)
        if bg_worker and iter_num % 100 == 0 and iter_num > 0:
            s = bg_worker.get_status()
            fail_str = f", {s['failed']} failed" if s['failed'] > 0 else ""
            print(f"  📊 Viz: {s['completed']} done, {s['active']} running, {s['queued']} queued{fail_str}")
        
        # CHAOS BREAKDOWN: Show what's actually driving the chaos score
        chaos_breakdown = f"[σ_iter={stress_iters:.2f}, σ_res={stress_residual:.2f}]"
        
        # BAYESIAN BALANCER: Show learned task weights
        balance_stats = metrics.get('balance_stats', {})
        balancer_str = ""
        if balance_stats:
            # Show which tasks the model "cares about" most (higher weight = more precision)
            weights = [balance_stats.get(f'weight_{name}', 0.0) for name in loss_names]
            sigmas = [balance_stats.get(f'sigma_{name}', 0.0) for name in loss_names]
            # Format: pred(w=1.2,σ=0.9), jac(w=0.8,σ=1.1), ...
            balancer_parts = []
            for i, name in enumerate(loss_names):
                short_name = name[:4]  # pred, jaco, nove, memo
                balancer_parts.append(f"{short_name}(w={weights[i]:.1f},σ={sigmas[i]:.1f})")
            balancer_str = f", BAL=[{', '.join(balancer_parts)}]"
        
        # Reflex integration status (always full now)
        gate_phase = "β=1.0 [REFLEX+MEMORY ACTIVE]"
        
        # MEMORY STATS: Show single-tier memory state
        memory_stats_str = ""
        if use_memory_manifold and hasattr(raw_model.reflex, 'get_memory_stats'):
            mem_stats = raw_model.reflex.get_memory_stats()
            if mem_stats:
                # Single-tier architecture: just show total size
                memory_stats_str = f", mem={mem_stats.get('num_memory', 0)}"
                # Add highway stats if available (compact format)
                if mem_stats.get('highways_formed', 0) > 0:
                    memory_stats_str += f", 🛣️{mem_stats['highways_formed']}"
        
        # LOSS BREAKDOWN: Merge into main line
        loss_breakdown_str = ""
        if 'loss_base' in prev_metrics:
            base = prev_metrics['loss_base']
            loss_breakdown_str = f", base={base:.2f}{balancer_str}"
        
        # Add gradient norm if available
        grad_norm_str = ""
        if 'grad_norm' in metrics:
            grad_norm_str = f", ∇={metrics['grad_norm']:.2e}"
        
        # 🚀 Get edge subsampling stats for clean reporting
        from graph_memory_system import get_edge_subsample_stats
        edge_stats = get_edge_subsample_stats()
        edge_stats_str = ""
        if edge_stats['calls_count'] > 0:
            edge_stats_str = f", edge_sub={edge_stats['calls_count']}calls({edge_stats['reduction_pct']:.0f}%↓)"
        
        # Log line with NOVELTY/EXPLORATION DRIVE (ℂ), chaos breakdown, Bayesian balancer, reflex gate, and memory state
        print(f"iter {iter_num}: loss {lossf:.4f}, time {time_ms:.2f}ms, mfu {running_mfu*100:.2f}%, deq_iters={deq_iters}, lr={lr:.2e}, chaos={chaos_score:.3f}{chaos_breakdown}, res={raw_residual:.2e}, ℂ={novelty_drive:.3e}, {gate_phase}{memory_stats_str}{loss_breakdown_str}{grad_norm_str}{edge_stats_str}")
        
        # � MICRO-PROFILING: Print memory operation stats every iteration
        from graph_memory_system import print_profile_stats, reset_profile_stats
        print_profile_stats()
        reset_profile_stats()
        
        # �🛣️ HIGHWAY REPORT: Every 100 iters, show detailed Hebbian learning stats
        if iter_num % 100 == 0 and iter_num > 0 and master_process and use_memory_manifold:
            if hasattr(raw_model.reflex, 'get_memory_stats'):
                mem_stats = raw_model.reflex.get_memory_stats()
                highways_count = mem_stats.get('highways_formed', 0)
                
                # ALWAYS print highway status (even if 0, so we know it's checking)
                print(f"\n🛣️  HEBBIAN HIGHWAYS (iter {iter_num})")
                print(f"   Total strengthened: {highways_count} edges")
                
                if highways_count > 0:
                    print(f"   Max strengthening: {mem_stats.get('max_highway_strength', 0):.4f}")
                    print(f"   Avg strengthening: {mem_stats.get('avg_highway_strength', 0):.4f}")
                    
                    # Get top-5 highways if available
                    if hasattr(raw_model.reflex.memory_retrieval, 'memory'):
                        highway_details = raw_model.reflex.memory_retrieval.memory.get_highway_stats(top_k=5)
                        if highway_details.get('top_highways'):
                            print(f"   Top 5 highways:")
                            for i, hw in enumerate(highway_details['top_highways'][:5], 1):
                                print(f"      {i}. Edge {hw['source_idx']}→{hw['target_idx']}: "
                                      f"weight {hw['old_weight']:.3f}→{hw['new_weight']:.3f} "
                                      f"(Δ={hw['strengthening']:.4f}, "
                                      f"traversals={hw.get('traversal_count', 0)}, "
                                      f"success={hw.get('success_rate', 0):.2f})")
                else:
                    print(f"   ℹ️  No highways formed yet (highway_log empty)")
                    print(f"   Memory size: {mem_stats.get('num_memory', 0)} memories")
                    # Debug: Check if strengthen_edge is even being called
                    if hasattr(raw_model.reflex.memory_retrieval, 'memory'):
                        mem_tier = raw_model.reflex.memory_retrieval.memory
                        print(f"   Highway log size: {len(mem_tier.highway_log) if hasattr(mem_tier, 'highway_log') else 'N/A'}")
                print()
        
        # Save profiling stats to CSV (silent mode - no console spam)
        if enable_profiling and iter_num % profile_interval == 0 and iter_num > 0 and master_process:
            profiling_csv = os.path.join(out_dir, 'reports', 'profiling.csv')
            profiler.save_to_csv(profiling_csv, recent_n=profile_interval)
        
        # [COMPREHENSIVE DIAGNOSTIC REPORT] Every 10 iters - RUN IN BACKGROUND
        if iter_num % 10 == 0 and iter_num > 0 and diagnostic and master_process and bg_worker and VISUALIZATION_AVAILABLE:
            meta_path_diag = os.path.join(data_dir, 'meta.pkl')
            if os.path.exists(meta_path_diag):
                with open(meta_path_diag, 'rb') as f:
                    meta_diag = pickle.load(f)
                
                # Submit to background - non-blocking, silent mode (data saved to files)
                # CRITICAL: diagnostic.generate_full_report uses matplotlib, must wrap with lock
                def safe_diagnostic_report():
                    with BackgroundWorker._matplotlib_lock:
                        diagnostic.generate_full_report(
                            raw_model, get_batch, meta_diag, iter_num, device=device, silent=True
                        )
                
                bg_worker.submit(
                    "Diagnostic Report",
                    safe_diagnostic_report
                )
                # Note: model.train() will be called at end of diagnostic in background
        
        # [PHYSICS PROBE] Inspect Semantic Mass Matrix (every 1000 iters - saved to CSV)
        if iter_num % 1000 == 0 and iter_num > 0 and hamiltonian:
            
            # We need the tokenizer to decode
            meta_path_inspect = os.path.join(data_dir, 'meta.pkl')
            if os.path.exists(meta_path_inspect):
                with open(meta_path_inspect, 'rb') as f:
                    meta_inspect = pickle.load(f)
                itos = meta_inspect.get('itos', {})
                
                mass_data = raw_model.inspect_concept_mass(top_k=10)
                
                # Log to monitor (data visualized in graphs)
                if monitor:
                    monitor.log_mass_stats(iter_num, mass_data['heavy_vals'], 
                                          mass_data['light_vals'])
        
        # Generate homeostatic dashboard (every 20 iters) - RUN IN BACKGROUND
        # HOMEOSTATIC MONITORING AND DIAGNOSTICS
        # OPTIMIZED: Reduce frequency in fast_mode
        fast_mode = getattr(config, 'fast_mode', False)
        plot_interval = getattr(config, 'plot_interval', 20)
        
        if iter_num % plot_interval == 0 and iter_num > 0 and monitor and bg_worker and VISUALIZATION_AVAILABLE:
            # CRITICAL: monitor.plot_homeostasis uses matplotlib, must wrap with lock
            def safe_homeostasis_plot():
                with BackgroundWorker._matplotlib_lock:
                    monitor.plot_homeostasis(iter_num)
            
            bg_worker.submit(
                "Homeostasis Dashboard",
                safe_homeostasis_plot
            )
            
            # 🧠 BAYESIAN BRAIN DIAGNOSTICS (every 100 iters) - RUN IN BACKGROUND
            # This creates the comprehensive precision-weighted visualization
            if iter_num % 100 == 0:
                def _plot_bayesian():
                    # CRITICAL: plot_bayesian_brain uses matplotlib, must use lock
                    with BackgroundWorker._matplotlib_lock:
                        try:
                            import importlib
                            import plot_bayesian_brain
                            importlib.reload(plot_bayesian_brain)  # Force reload to pick up fixes
                            plot_bayesian_brain.plot_bayesian_brain(monitor.reports_dir, silent=True)
                        except Exception as e:
                            pass  # Errors handled by background worker
                
                bg_worker.submit("Bayesian Brain Plot", _plot_bayesian)
        
        # append to CSV for later plotting/analysis
        if master_process:
            try:
                with open(metrics_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([iter_num, f"{lossf:.6f}", f"{time_ms:.3f}", f"{running_mfu*100:.4f}", deq_iters, time.strftime('%Y-%m-%d %H:%M:%S')])
            except Exception as e:
                print(f"Warning: failed to write metrics CSV: {e}")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
    
    # 🔬 PROFILING: Print stats every 100 iterations to reduce spam
    if enable_profiling and iter_num > 0 and iter_num % 100 == 0 and master_process:
        print(f"\n📊 PROFILING (iter {iter_num}):")
        profiler.print_stats(top_n=5, min_percent=1.0)  # Top 5, >1% only

# 🔬 PROFILING: Final stats at end of training
if enable_profiling and master_process:
    print("\n" + "="*80)
    print("🏁 FINAL PROFILING RESULTS")
    print("="*80)
    profiler.print_stats(top_n=30, min_percent=0.1)
    profiler.save_to_csv(os.path.join(out_dir, 'profiling_stats.csv'))

if ddp:
    destroy_process_group()
