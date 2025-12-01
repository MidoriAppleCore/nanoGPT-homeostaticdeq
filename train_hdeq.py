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

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Import from model_graybox instead of model
from model_hdeq import GPTConfig, GPT, compute_pauli_exclusion_loss, HamiltonianOperator, DEQOperator

# Homeostatic monitoring system
from homeostatic_monitor import HomeostaticMonitor

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
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out-shakespeare-char-hdeq'
eval_interval = 100
log_interval = 10
eval_iters = 50
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' (gpt2* not supported for DEQ)
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

if master_process:
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
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # DEQ-specific params
    for k in ['deq_max_iter', 'deq_tol', 'anderson_accel', 'spectral_norm']:
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
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
else:
    raise ValueError(f"init_from='{init_from}' not supported for Gray Box DEQ (only 'scratch' or 'resume')")

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
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
    return out

# generate samples to see what the model is producing
@torch.no_grad()
def generate_samples():
    model.eval()
    
    # Load meta for decoding
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        decode = lambda l: ''.join([meta['itos'][i] for i in l])
    else:
        decode = lambda l: str(l)  # fallback
    
    print("\n" + "="*70)
    print("SAMPLE GENERATIONS")
    print("="*70)
    
    # Sample 1: Conditioned on newline token (198 in GPT-2)
    # For TinyStories, use a simple prompt
    if os.path.exists(meta_path):
        # Use newline token ID 198 for GPT-2 tokenizer
        start_ids = [198]  # newline in GPT-2
    else:
        start_ids = [0]
    
    x = torch.tensor([start_ids], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens=200, temperature=0.8, top_k=200, effort=1.0)
    print("\n[Conditioned generation]")
    print(decode(y[0].tolist()))
    
    # Sample 2: Short prompt
    if os.path.exists(meta_path):
        # "Once upon a time" = [7454, 2402, 257, 640]
        start_ids = [7454, 2402, 257, 640]
    else:
        start_ids = [1, 2, 3]
    
    x = torch.tensor([start_ids], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens=200, temperature=0.8, top_k=200, effort=1.0)
    print("\n[Story generation]")
    print(decode(y[0].tolist()))
    
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
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Log to monitor
        if monitor:
            monitor.log_loss(iter_num, losses['train'], losses['val'])
        
        # Generate samples to see model progress
        generate_samples()
        
        # Visualize phase space dynamics (every eval)
        visualize_phase_space(iter_num)
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, ckpt_path)
                
                # Save checkpoint summary
                if monitor:
                    monitor.save_checkpoint_summary(iter_num, ckpt_path)
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss, metrics = model(X, Y, return_metrics=True, training_iter=iter_num)
            
            # Add Pauli Exclusion loss (anti-stuttering force)
            # This is the NOVELTY/EXPLORATION DRIVE (ℂ) - the computational boredom signal
            pauli_loss = 0.0
            if lambda_pauli > 0:
                pauli_loss = compute_pauli_exclusion_loss(logits, Y)
                loss = loss + lambda_pauli * pauli_loss
            
            # ALGORITHMIC EFFICIENCY LOSS: Penalize long thinking
            # Forces the model to learn the FASTEST trajectory to the answer
            # This creates intrinsic motivation for computational efficiency
            efficiency_loss = 0.0
            if lambda_efficiency > 0:
                num_iters = metrics.get('num_iters', 0)
                efficiency_loss = lambda_efficiency * num_iters
                loss = loss + efficiency_loss
            
            # Store for homeostatic monitoring (Free Energy Principle)
            metrics['novelty_drive'] = pauli_loss.item() if isinstance(pauli_loss, torch.Tensor) else pauli_loss
            metrics['efficiency_cost'] = efficiency_loss.item() if isinstance(efficiency_loss, torch.Tensor) else efficiency_loss
            
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
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
        'novelty_drive': metrics.get('novelty_drive', 0.0)  # ℂ: Boredom/Curiosity signal
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
        
        # Show both raw residual and log-scaled chaos for transparency
        raw_residual = prev_metrics.get('final_residual', 0.0)
        novelty_drive = prev_metrics.get('novelty_drive', 0.0)
        
        # CHAOS BREAKDOWN: Show what's actually driving the chaos score
        chaos_breakdown = f"[σ_iter={stress_iters:.2f}, σ_res={stress_residual:.2f}]"
        
        # Log line with NOVELTY/EXPLORATION DRIVE (ℂ) and chaos breakdown
        print(f"iter {iter_num}: loss {lossf:.4f}, time {time_ms:.2f}ms, mfu {running_mfu*100:.2f}%, deq_iters={deq_iters}, lr={lr:.2e}, chaos={chaos_score:.3f}{chaos_breakdown}, res={raw_residual:.2e}, ℂ={novelty_drive:.3e}")
        
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
        
        # Generate homeostatic dashboard (every 500 iters)
        if iter_num % 500 == 0 and iter_num > 0 and monitor:
            monitor.plot_homeostasis(iter_num)
        
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
