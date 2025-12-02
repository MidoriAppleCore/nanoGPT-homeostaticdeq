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

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Import from model_graybox instead of model
from model_hdeq import GPTConfig, GPT, compute_pauli_exclusion_loss, HamiltonianOperator, DEQOperator, HomeostaticBalancer

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
        """Print profiling statistics"""
        stats = self.get_stats(recent_n)
        if not stats:
            return
        
        print("\n" + "="*80)
        print("⏱️  PROFILING REPORT (last {} iterations)".format(recent_n))
        print("="*80)
        
        # Calculate total time for percentage
        total_time = stats.get('total_iter', {}).get('mean', 0)
        
        # Sort by mean time (descending)
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        for name, s in sorted_stats:
            mean_ms = s['mean'] * 1000
            pct = (s['mean'] / total_time * 100) if total_time > 0 else 0
            
            if name == 'total_iter':
                print(f"\n{'TOTAL ITERATION':<30} {mean_ms:>8.1f}ms ± {s['std']*1000:>5.1f}ms")
                print("-"*80)
            else:
                print(f"  {name:<28} {mean_ms:>8.1f}ms  ({pct:>5.1f}%)  ± {s['std']*1000:>5.1f}ms")
        
        print("="*80 + "\n")
    
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

# Initialize profiler (after config loaded)
profiler = TrainingProfiler(enabled=globals().get('enable_profiling', True))

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
        
        # Pass memory config to model (no manifold path needed!)
        model_args['use_memory_manifold'] = True
        model_args['memory_mode'] = memory_mode
        model_args['memory_dim'] = memory_dim if 'memory_dim' in locals() else n_embd
        model_args['memory_k'] = memory_k
        model_args['memory_alpha'] = memory_alpha
        model_args['memory_curvature'] = memory_curvature if 'memory_curvature' in locals() else 1.0
        model_args['working_memory_capacity'] = working_memory_capacity
        model_args['working_memory_decay'] = working_memory_decay
        model_args['longterm_memory_capacity'] = longterm_memory_capacity
        model_args['longterm_memory_decay'] = longterm_memory_decay
        model_args['memory_promotion_threshold'] = memory_promotion_threshold
        model_args['memory_promotion_interval'] = memory_promotion_interval
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
    # First try to find the latest iteration checkpoint (ckpt_iter*.pt)
    import glob
    iter_ckpts = glob.glob(os.path.join(out_dir, 'ckpt_iter*.pt'))
    if iter_ckpts:
        # Sort by iteration number and take the latest
        iter_ckpts.sort(key=lambda x: int(x.split('iter')[-1].split('.')[0]))
        ckpt_path = iter_ckpts[-1]
        print(f"  Found latest checkpoint: {os.path.basename(ckpt_path)}")
    else:
        # Fall back to ckpt.pt
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
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
        model.reflex.load_memory_checkpoint(memory_ckpt_path)
        mem_stats = model.reflex.get_memory_stats()
        print(f"  ✓ Restored memory: W={mem_stats.get('num_working', 0)}, "
              f"B={mem_stats.get('num_buffer', 0)}, LT={mem_stats.get('num_longterm', 0)}")
    else:
        print(f"  ⚠️  Memory checkpoint not found: {memory_ckpt_path}")
        print(f"  Memory will start empty (capacity: {longterm_memory_capacity})")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Initialize Homeostatic Balancer (Bayesian Multi-Objective Learning)
# This automatically balances loss components without manual tuning
loss_names = ["prediction", "jacobian", "novelty", "memory"]
balancer = HomeostaticBalancer(num_losses=4, loss_names=loss_names).to(device)
print(f"🧠 Initialized Homeostatic Balancer with losses: {loss_names}")
print(f"   Bayesian uncertainty balancing will learn optimal task weights automatically")

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
print(f"   Added {sum(p.numel() for p in balancer.parameters())} balancer parameters to optimizer")
print(f"   Balancer LR: {balancer_lr:.2e} (10× base LR for fast homeostatic adaptation)")

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    # FIX: Restore balancer LR after loading checkpoint (which overwrites it!)
    optimizer.param_groups[-1]['lr'] = balancer_lr
    print(f"   ✓ Restored balancer LR to {balancer_lr:.2e} after checkpoint load")
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
            # Capture last token's activation
            if output.ndim == 3:  # (batch, seq, hidden)
                reflex_activations.append(output[0, -1, :].detach().cpu().numpy())
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
    
    print("="*70 + "\n")
    model.train()

@torch.no_grad()
def generate_samples():
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
    y = model.generate(x, max_new_tokens=50, temperature=0.8, top_k=200, effort=1.0)
    print("\n[Question prompt]")
    output = decode(y[0].tolist())
    # Limit to ~50 characters for quick feedback
    print(output[:200] if len(output) > 200 else output)
    
    print("="*70 + "\n")
    model.train()

def visualize_phase_space(iter_num):
    """Generate phase space visualization of DEQ trajectories"""
    if not VISUALIZATION_AVAILABLE:
        return
    
    model.eval()
    
    # Capture DEQ trajectories by monkey-patching
    trajectories = []
    original_solve = model.deq.solve
    
    def captured_solve(u, mask=None, effort=1.0, verbose=False):
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
    
    print(f"  📊 Phase space saved: {output_path} ({dynamics_type})")
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

# HOMEOSTATIC REFLEX GATING: Force cortical development before spinal automation
def get_reflex_gate(it, phase1_iters=500, phase2_iters=1000):
    """
    Homeostatic Reflex Gate β(t): Controls how much spinal cord signal reaches the brain.
    
    This is the KEY to preventing "lizard brain" optimization.
    
    Phase 1 (0 - 500 iters): β = 0.0
        - PURE DEQ TRAINING
        - Force the cortex to find semantic attractors WITHOUT reflex shortcuts
        - The brain must learn to equilibrate on context alone
        - Like learning to walk before running
    
    Phase 2 (500 - 1000 iters): β ramps 0.0 → 1.0
        - GRADUAL RECONNECTION
        - Slowly reintroduce spinal reflexes
        - Brain has already formed stable attractors
        - Reflexes now enhance, not replace, cortical computation
    
    Phase 3 (1000+ iters): β = 1.0
        - FULL CYBERNETIC INTEGRATION
        - Brain + Spinal Cord working in harmony
        - Both subsystems trained properly
    
    Physical Interpretation:
        - β = 0: "Spinal cord severed" - pure cortical learning
        - β = 0.5: "Partial innervation" - gradual reconnection
        - β = 1.0: "Full nervous system" - complete integration
    
    Formula:
        u_input = context + β(t) * reflex
    
    Biological Analog:
        - Infant brain development: cortex matures before reflexes automate
        - Stroke rehabilitation: brain relearns before reflexes return
    
    Returns:
        β ∈ [0.0, 1.0]: Reflex gating coefficient
    """
    if it < phase1_iters:
        # Phase 1: Pure cortical training
        return 0.0
    elif it < phase2_iters:
        # Phase 2: Linear ramp (smooth reconnection)
        progress = (it - phase1_iters) / (phase2_iters - phase1_iters)
        return progress
    else:
        # Phase 3: Full integration
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
    # RECALIBRATED for quantum solver (which explores before converging)
    # - Residual 100.0: Initialization chaos, explosive
    # - Residual 10.0: Early exploration, very high
    # - Residual 1.0: Mid exploration, acceptable
    # - Residual 0.1: Converging well
    # - Residual 1e-2 (0.01): Locked in
    # - Residual 1e-3 (0.001): Perfect equilibrium
    raw_res = metrics.get('final_residual', 0.0)
    if raw_res > 0:
        # Map log10(residual) from [-3, +2] to [0, 1]
        # log10(1e-3) = -3 -> 0.0 (Zen - perfect)
        # log10(0.01) = -2 -> 0.2 (Excellent)
        # log10(0.1) = -1 -> 0.4 (Good)
        # log10(1.0) = 0 -> 0.6 (Exploring)
        # log10(10.0) = +1 -> 0.8 (High exploration)
        # log10(100.0) = +2 -> 1.0 (Panic - explosive)
        log_res = math.log10(max(raw_res, 1e-4))  # Clamp to avoid log(0)
        stress_residual = (log_res + 3.0) / 5.0  # Map [-3, +2] to [0, 1]
        stress_residual = max(0.0, min(1.0, stress_residual))  # Clamp to [0, 1]
    else:
        stress_residual = 0.0
    
    # Combined Chaos Score (0.0 = Zen, 1.0 = Panic)
    chaos_score = max(stress_iters, stress_residual)
    
    # The Valve: If chaos is high, throttle the LR
    # If chaos > 0.8, LR drops to 10%. If chaos < 0.2, LR is 100%.
    # This prevents "driving into corners at full speed"
    throttle = 1.0 - max(0, (chaos_score - 0.2) / 0.8)
    throttle = max(0.1, throttle)  # Never stop completely, but slow down 10x
    
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

# Adaptive min_lr to prevent training stagnation
lr_plateau = {'min_lr': min_lr, 'plateau_counter': 0, 'last_loss': 1e9}

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

# Prepare reports directory and CSV logging for per-iteration metrics
reports_dir = os.path.join(out_dir, 'reports')
os.makedirs(reports_dir, exist_ok=True)
metrics_csv = os.path.join(reports_dir, 'metrics.csv')
# write header if new
if master_process and not os.path.exists(metrics_csv):
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iter','loss','time_ms_synced','mfu_percent','deq_iters','timestamp'])

# Track previous metrics for chaos-aware LR
prev_metrics = {'num_iters': 0, 'final_residual': 0.0}

while True:

    # determine and set the learning rate for this iteration
    # Use Lyapunov-guided (chaos-aware) LR if we have metrics from previous step
    if decay_lr and iter_num > 0:
        lr = get_chaos_aware_lr(iter_num, prev_metrics, learning_rate)
    else:
        lr = get_lr(iter_num) if decay_lr else learning_rate
    
    # Apply LR to param groups, but protect balancer's independent LR
    # The balancer (last param group) should learn faster to regulate the network
    # CRITICAL: Balancer must use CONSTANT 1e-2 LR (immune to chaos throttling)
    # to quickly adapt task weights in response to loss volatility
    # DO NOT use 'learning_rate' variable - it may be overridden by config!
    balancer_lr_fixed = 1e-2  # FIXED CONSTANT - not affected by config or chaos
    
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
        
        # Visualize phase space dynamics (every eval)
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
    
    # Sample inference every 50 iters (but skip if eval_interval already did it)
    # This prevents double generation at multiples of eval_interval
    if iter_num > 0 and iter_num % 50 == 0 and iter_num % eval_interval != 0 and master_process:
        print(f"\n{'='*80}")
        print(f"📝 SAMPLE INFERENCE (iter {iter_num})")
        print(f"{'='*80}")
        generate_samples()
        print(f"{'='*80}\n")
    
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
            loss_components = metrics.get('loss_components', {})
            
            # Add Pauli Exclusion loss (novelty/anti-stuttering)
            pauli_loss = torch.tensor(0.0, device=device)
            if lambda_pauli > 0:
                pauli_loss = compute_pauli_exclusion_loss(logits, Y)
            loss_components['novelty'] = pauli_loss
            
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
            
            # Apply Homeostatic Balancer
            # This automatically weights each loss by 1/σ² where σ is learned
            balanced_loss, balance_stats = balancer(loss_components)
            
            # Scale for gradient accumulation
            loss = balanced_loss / gradient_accumulation_steps
            
            # Store components for monitoring
            metrics['loss_components_raw'] = {k: v.item() if isinstance(v, torch.Tensor) else v 
                                               for k, v in loss_components.items()}
            metrics['balance_stats'] = balance_stats
            metrics['novelty_drive'] = pauli_loss.item()
            metrics['loss_base'] = loss_components['prediction'].item()
            metrics['loss_balanced'] = balanced_loss.item()
            
        profiler.stop('forward')
        
        # Data prefetch
        profiler.start('data_prefetch')
        X, Y = get_batch('train')
        profiler.stop('data_prefetch')
        
        # Backward pass
        profiler.start('backward')
        scaler.scale(loss).backward()
        profiler.stop('backward')
    
    # Gradient clipping
    profiler.start('grad_clip')
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
        
        print(f"  📊 BALANCER PARAMS: log_vars={log_vars_values}, σ={sigmas}, w={weights}")
        print(f"     ACTUAL LRs: balancer={actual_balancer_lr:.2e}, main={main_lr:.2e}, ratio={actual_balancer_lr/main_lr:.1f}×")
    
    # 🔬 BALANCER GRADIENT DIAGNOSTIC (Check if Diplomat is receiving signals)
    # Log every 100 iters to verify balancer parameters are being updated
    if iter_num % 100 == 0 and iter_num > 0 and master_process:
        balancer_grad_norm = 0.0
        balancer_param_norm = 0.0
        for param in balancer.parameters():
            if param.grad is not None:
                balancer_grad_norm += param.grad.data.norm(2).item() ** 2
                balancer_param_norm += param.data.norm(2).item() ** 2
        balancer_grad_norm = balancer_grad_norm ** 0.5
        balancer_param_norm = balancer_param_norm ** 0.5
        
        # Get ACTUAL balancer LR from optimizer (verify it's protected)
        actual_balancer_lr = optimizer.param_groups[-1]['lr']
        expected_balancer_lr = 1e-2  # FIXED CONSTANT (immune to config/chaos)
        
        # Get main network LR for comparison
        main_network_lr = optimizer.param_groups[0]['lr']
        lr_ratio = actual_balancer_lr / main_network_lr if main_network_lr > 0 else 0
        
        if balancer_grad_norm > 1e-8:
            print(f"  🧠 BALANCER: ||∇||={balancer_grad_norm:.2e}, ||σ||={balancer_param_norm:.2e}")
            print(f"     LR_balancer={actual_balancer_lr:.2e} (expected={expected_balancer_lr:.2e})")
            print(f"     LR_network={main_network_lr:.2e}, Ratio={lr_ratio:.1f}× ✓")
        else:
            print(f"  ⚠️  BALANCER GRADIENT DEAD: ||∇||={balancer_grad_norm:.2e} (expected >1e-8)")
            print(f"     LR_balancer={actual_balancer_lr:.2e}, LR_network={main_network_lr:.2e}")
    
    # 🔬 GRADIENT FLOW VISUALIZATION: Topological debugging every 20 iters
    # Reveals the local geometry of the loss landscape
    # This shows if we're converging (sink), oscillating (vortex), or bypassing (orthogonal flows)
    if iter_num % 20 == 0 and iter_num > 0 and master_process:
        try:
            visualize_gradient_flow(raw_model, optimizer, iter_num, out_dir, max_layers=40)
        except Exception as e:
            print(f"  ⚠️  Gradient flow visualization failed: {e}")
    
    # Optimizer step
    profiler.start('optimizer')
    
    # 🚨 CRITICAL PRE-STEP VERIFICATION (Every iter, minimal overhead)
    # Verify balancer LR is ACTUALLY protected before taking the step
    if iter_num % 10 == 0 and master_process:
        actual_bal_lr = optimizer.param_groups[-1]['lr']
        expected_bal_lr = 1e-2  # FIXED CONSTANT (same as line 1067)
        if abs(actual_bal_lr - expected_bal_lr) > 1e-9:
            print(f"  ⚠️⚠️⚠️  BALANCER LR CORRUPTED! actual={actual_bal_lr:.2e}, expected={expected_bal_lr:.2e}")
            print(f"           Forcing correction...")
            optimizer.param_groups[-1]['lr'] = expected_bal_lr
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    profiler.stop('optimizer')
    
    # ══════════════════════════════════════════════════════════════
    # MEMORY SYSTEM UPDATE (dopamine + aging/decay)
    # ══════════════════════════════════════════════════════════════
    profiler.start('memory_update')
    if use_memory_manifold and hasattr(raw_model.reflex, 'apply_dopamine_signal'):
        raw_model.reflex.apply_dopamine_signal(loss)
    
    if use_memory_manifold and hasattr(raw_model.reflex, 'memory_step'):
        raw_model.reflex.memory_step()
    profiler.stop('memory_update')

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
    
    # Calculate chaos score components for logging (MUST MATCH get_chaos_aware_lr formula!)
    stress_iters = min(1.0, prev_metrics['num_iters'] / deq_max_iter)
    # Use logarithmic scale for residual - UPDATED to match quantum solver range
    raw_res = prev_metrics['final_residual']
    if raw_res > 0:
        log_res = math.log10(max(raw_res, 1e-4))
        stress_residual = (log_res + 3.0) / 5.0  # Map [-3, +2] to [0, 1] (quantum scale)
        stress_residual = max(0.0, min(1.0, stress_residual))
    else:
        stress_residual = 0.0
    chaos_score = max(stress_iters, stress_residual)
    throttle = 1.0 - max(0, (chaos_score - 0.2) / 0.8)
    throttle = max(0.1, throttle)
    
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
                # Raise min_lr by 50%
                old_min_lr = lr_plateau['min_lr']
                lr_plateau['min_lr'] = min(lr_plateau['min_lr'] * 1.5, learning_rate * 0.1)  # Cap at 10% of base LR
                print(f"[LR Plateau] Loss stagnant! Raising min_lr: {old_min_lr:.2e} → {lr_plateau['min_lr']:.2e}")
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
            
            # Log loss components
            if 'loss_base' in prev_metrics and 'pauli_component' in prev_metrics:
                monitor.log_loss_components(
                    iter_num,
                    prev_metrics['loss_base'],
                    prev_metrics['pauli_component'],
                    prev_metrics.get('efficiency_cost', 0.0),
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
        
        # HOMEOSTATIC GATING: Show developmental phase (β=0: cortex only, β=1: full integration)
        if reflex_gate_val < 0.1:
            gate_phase = "β=0.0 [CORTEX ONLY]"
        elif reflex_gate_val < 0.99:
            gate_phase = f"β={reflex_gate_val:.2f} [RECONNECTING]"
        else:
            gate_phase = "β=1.0 [FULL INTEGRATION]"
        
        # MEMORY STATS: Show three-tier memory state
        memory_stats_str = ""
        if use_memory_manifold and hasattr(raw_model.reflex, 'get_memory_stats'):
            mem_stats = raw_model.reflex.get_memory_stats()
            if mem_stats:
                memory_stats_str = f", mem=[W:{mem_stats.get('num_working', 0)}/B:{mem_stats.get('num_buffer', 0)}/LT:{mem_stats.get('num_longterm', 0)}]"
        
        # LOSS BREAKDOWN: Show component contributions
        loss_breakdown_str = ""
        if 'loss_base' in prev_metrics:
            base = prev_metrics['loss_base']
            loss_breakdown_str = f"\n  💡 Loss: base={base:.2f}, {gate_phase}{balancer_str}"
        
        # Log line with NOVELTY/EXPLORATION DRIVE (ℂ), chaos breakdown, Bayesian balancer, reflex gate, and memory state
        print(f"iter {iter_num}: loss {lossf:.4f}, time {time_ms:.2f}ms, mfu {running_mfu*100:.2f}%, deq_iters={deq_iters}, lr={lr:.2e}, chaos={chaos_score:.3f}{chaos_breakdown}, res={raw_residual:.2e}, ℂ={novelty_drive:.3e}, {gate_phase}{memory_stats_str}{loss_breakdown_str}")
        
        # Print profiling stats at profile_interval
        if enable_profiling and iter_num % profile_interval == 0 and iter_num > 0:
            profiler.print_stats(recent_n=profile_interval)
        
        # [COMPREHENSIVE DIAGNOSTIC REPORT] Every 20 iters
        if iter_num % 20 == 0 and iter_num > 0 and diagnostic and master_process:
            try:
                meta_path_diag = os.path.join(data_dir, 'meta.pkl')
                if os.path.exists(meta_path_diag):
                    with open(meta_path_diag, 'rb') as f:
                        meta_diag = pickle.load(f)
                    
                    diagnostic.generate_full_report(
                        raw_model, 
                        get_batch,
                        meta_diag,
                        iter_num,
                        device=device
                    )
                    
                    # CRITICAL: Ensure model is back in training mode
                    # (diagnostic report calls eval() internally)
                    raw_model.train()
                    if ddp:
                        model.train()
            except Exception as e:
                print(f"  ⚠️  Diagnostic report failed: {e}")
                # Ensure training mode even on exception
                raw_model.train()
                if ddp:
                    model.train()
        
        # [PHYSICS PROBE] Inspect Semantic Mass Matrix (every 1000 iters)
        if iter_num % 1000 == 0 and iter_num > 0 and hamiltonian:
            print("\n" + "="*70)
            print("🔬 [PHYSICS PROBE] Inspecting Semantic Mass Matrix...")
            print("="*70)
            
            # We need the tokenizer to decode
            meta_path_inspect = os.path.join(data_dir, 'meta.pkl')
            if os.path.exists(meta_path_inspect):
                with open(meta_path_inspect, 'rb') as f:
                    meta_inspect = pickle.load(f)
                itos = meta_inspect.get('itos', {})
                
                mass_data = raw_model.inspect_concept_mass(top_k=10)
                
                print("\n  ⚛️  HEAVY Concepts (High Inertia - Content Words):")
                for idx, val in zip(mass_data['heavy_ids'], mass_data['heavy_vals']):
                    token = itos.get(idx, f"<{idx}>")
                    # Clean up token for display
                    token_display = repr(token)[1:-1]  # Remove outer quotes
                    print(f"     {token_display:20s}: {val:.4f}")
                
                print("\n  💨 LIGHT Concepts (Agile - Function Words):")
                for idx, val in zip(mass_data['light_ids'], mass_data['light_vals']):
                    token = itos.get(idx, f"<{idx}>")
                    token_display = repr(token)[1:-1]
                    print(f"     {token_display:20s}: {val:.4f}")
                
                # Log to monitor
                if monitor:
                    monitor.log_mass_stats(iter_num, mass_data['heavy_vals'], 
                                          mass_data['light_vals'])
            else:
                print("  ⚠️  No meta.pkl found - cannot decode tokens")
            
            print("="*70 + "\n")
        
        # Generate homeostatic dashboard (every 20 iters)
        if iter_num % 20 == 0 and iter_num > 0 and monitor:
            monitor.plot_homeostasis(iter_num)
            
            # 🧠 BAYESIAN BRAIN DIAGNOSTICS (every 100 iters)
            # This creates the comprehensive precision-weighted visualization
            if iter_num % 100 == 0:
                try:
                    import importlib
                    import plot_bayesian_brain
                    importlib.reload(plot_bayesian_brain)  # Force reload to pick up fixes
                    plot_bayesian_brain.plot_bayesian_brain(monitor.reports_dir)
                except Exception as e:
                    print(f"  ⚠️  Bayesian brain diagnostic failed: {e}")
        
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

if ddp:
    destroy_process_group()
