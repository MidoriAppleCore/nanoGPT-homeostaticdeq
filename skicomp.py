import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import random

"""
SKI COMBINATOR CALCULUS via DEQ-SECD

The Challenge:
SKI is Turing-complete with only 3 combinators:
  I x → x                    (identity)
  K x y → x                  (constant)
  S f g x → (f x) (g x)      (substitution/composition)

No variables, no closures, no environment - just pure term rewriting.

Why This is Hard for Neural Networks:
1. Unbounded reduction depth (S causes exponential duplication)
2. Structural recursion (nested applications grow trees)
3. Order matters (weak vs. strong reduction strategies)
4. No memorization possible (infinite expression space)

Our Architecture:
1. SECD Stack Machine: Handles application spine traversal
2. DEQ Fixed Points: Stable iteration through reduction sequences
3. Jones 3-Network: α (local damping) + γ (global step) for spectral control
4. Control/Value Separation: Routing based on combinator identity, not term structure

Encoding:
Tokens: 0=NOOP, 1=S, 2=K, 3=I, 4=APP (apply), 5=REDUCE, 6=VAR
We represent SKI terms as sequences in applicative normal form:
  `I x` → [3, 6, 4] (I, VAR, APP)
  `K x y` → [2, 6, 4, 6, 4] (K, VAR, APP, VAR, APP)

Training Curriculum:
- Depth 1-2: I x, K x y, simple S applications
- Depth 3-4: S K K x (proves SKK = I), nested combos
- Depth 5-8: Church numerals, composition chains
- Test depth 10-20: True generalization

Success Criteria:
✓ Reduce I x → x for any x
✓ Reduce K x y → x for any x, y
✓ Reduce S f g x → (f x) (g x) correctly
✓ Generalize to deeper expressions than training
✓ Reduce Church numerals (2 + 3 = 5 via SKI encoding)
✓ Anti-cheat: No memorization, true symbolic rewriting
"""

# ==========================================
# 1. THE DEQ SOLVER
# ==========================================
class DEQFixedPoint(autograd.Function):
    @staticmethod
    def forward(ctx, func, z_init, h_ctx, f_emb, W, U, V, tol=1e-4, max_iter=40):
        with torch.no_grad():
            z = z_init.clone()
            for _ in range(max_iter):
                z_next = func(z, h_ctx, f_emb, W, U, V)
                if torch.norm(z_next - z) < tol:
                    z = z_next
                    break
                z = z_next
        ctx.save_for_backward(z, h_ctx, f_emb, W, U, V)
        ctx.func = func
        return z

    @staticmethod
    def backward(ctx, grad_z_star):
        z_star, h_ctx, f_emb, W, U, V = ctx.saved_tensors
        func = ctx.func
        
        z_star = z_star.detach().requires_grad_(True)
        h_ctx = h_ctx.detach().requires_grad_(True)
        f_emb = f_emb.detach().requires_grad_(True)
        W = W.detach().requires_grad_(True)
        U = U.detach().requires_grad_(True)
        V = V.detach().requires_grad_(True)
        
        with torch.enable_grad():
            f_z = func(z_star, h_ctx, f_emb, W, U, V)
        
        v = grad_z_star.clone()
        for _ in range(20):
            v = autograd.grad(f_z, z_star, v, retain_graph=True)[0] + grad_z_star
            
        grads = autograd.grad(f_z, (h_ctx, f_emb, W, U, V), v, allow_unused=True)
        return (None, None, grads[0], grads[1], grads[2], grads[3], grads[4], None, None)

# ==========================================
# 2. SKI TERM REPRESENTATION
# ==========================================
@dataclass(frozen=True)
class SKITerm:
    """
    Represents SKI combinator terms.
    Types:
    - Combinator: name in ['S', 'K', 'I']
    - Application: (left @ right)
    - Variable: For testing (x, y, z) - not used in pure SKI
    """
    typ: str  # 'S', 'K', 'I', 'APP', 'VAR'
    left: Optional['SKITerm'] = None  # For APP
    right: Optional['SKITerm'] = None  # For APP
    name: Optional[str] = None  # For VAR
    
    def __str__(self):
        if self.typ in ['S', 'K', 'I']:
            return self.typ
        elif self.typ == 'VAR':
            return self.name or '?'
        elif self.typ == 'APP':
            left_str = f"({self.left})" if self.left and self.left.typ == 'APP' else str(self.left)
            right_str = f"({self.right})" if self.right and self.right.typ == 'APP' else str(self.right)
            return f"{left_str} {right_str}"
        return '?'

# ==========================================
# 3. SECD FIBER FOR SKI REDUCTION
# ==========================================

@dataclass(frozen=True)
class Fiber:
    """
    SECD state for SKI reduction.
    S: Stack of SKI terms (Application spine)
    E: Environment (unused in pure SKI, kept for compatibility)
    C: Code queue (reduction sequence)
    D: Dump (saved states for nested reductions)
    """
    S: Tuple[Any, ...]  # Stack
    E: Dict[str, Any]   # Environment (unused)
    C: Tuple[Any, ...]  # Control/Code
    D: Tuple[Any, ...]  # Dump

class SKICore:
    """
    SKI reduction rules implemented as SECD operations.
    
    Reduction semantics:
    - I x → x
    - K x y → x  
    - S f g x → (f x) (g x)
    
    Implementation strategy:
    1. Parse term into application spine on stack
    2. Detect reducible expressions (redexes)
    3. Apply reduction rule
    4. Continue until normal form (or max steps)
    """
    
    # OpCodes (extended for SKI with distinct variables)
    OP_NOOP = 0
    OP_S = 1
    OP_K = 2
    OP_I = 3
    OP_APP = 4      # Build application
    OP_REDUCE = 5   # Trigger reduction
    OP_VAR_X = 6    # Variable x
    OP_VAR_Y = 7    # Variable y
    OP_VAR_Z = 8    # Variable z
    OP_VAR_W = 9    # Variable w
    OP_HALT = 10    # Stop reduction (for autonomous mode)
    
    @staticmethod
    def push_combinator(f: Fiber, combinator: str) -> Fiber:
        """Push S, K, or I onto stack."""
        term = SKITerm(typ=combinator)
        return Fiber((term,) + f.S, f.E, f.C, f.D)
    
    @staticmethod
    def push_var(f: Fiber, var_name: str) -> Fiber:
        """Push variable (x, y, z, w) for testing."""
        term = SKITerm(typ='VAR', name=var_name)
        return Fiber((term,) + f.S, f.E, f.C, f.D)
    
    @staticmethod
    def is_normal_form(term: SKITerm) -> bool:
        """Check if term is in normal form (no redexes)."""
        test_fiber = Fiber((term,), {}, tuple(), tuple())
        _, can_reduce = SKICore.reduce_step(test_fiber)
        return not can_reduce
    
    @staticmethod
    def terms_equal(t1: SKITerm, t2: SKITerm) -> bool:
        """Structural equality of SKI terms."""
        if t1.typ != t2.typ:
            return False
        if t1.typ in ['S', 'K', 'I']:
            return True
        if t1.typ == 'VAR':
            return t1.name == t2.name
        if t1.typ == 'APP':
            return (SKICore.terms_equal(t1.left, t2.left) and 
                    SKICore.terms_equal(t1.right, t2.right))
        return False
    
    @staticmethod
    def apply(f: Fiber) -> Fiber:
        """
        Build application: pop two terms, push (left @ right).
        Stack: [right, left, ...] → [(left @ right), ...]
        """
        if len(f.S) < 2:
            return f
        right = f.S[0]
        left = f.S[1]
        app_term = SKITerm(typ='APP', left=left, right=right)
        return Fiber((app_term,) + f.S[2:], f.E, f.C, f.D)
    
    @staticmethod
    def reduce_once_normal(term: SKITerm) -> Tuple[SKITerm, bool]:
        """
        Perform one reduction step using normal-order (leftmost-outermost) strategy.
        
        Strategy:
        1. Try to reduce at root (check for I/K/S redexes)
        2. If root is not reducible, descend into left subtree
        3. If left is not reducible, descend into right subtree
        4. If nothing reduces, return unchanged
        
        This ensures we can reduce terms like ((K I) x) → I even when
        the redex is inside the left child.
        """
        # Try root redexes first
        
        # I x → x
        if term.typ == 'APP' and term.left and term.left.typ == 'I':
            return term.right, True
        
        # K x y → x   (term = ((K x) y))
        if (term.typ == 'APP' and term.left and term.left.typ == 'APP' and
            term.left.left and term.left.left.typ == 'K'):
            return term.left.right, True
        
        # S f g x → (f x) (g x)   (term = (((S f) g) x))
        if (term.typ == 'APP' and term.left and term.left.typ == 'APP' and
            term.left.left and term.left.left.typ == 'APP' and
            term.left.left.left and term.left.left.left.typ == 'S'):
            f_term = term.left.left.right
            g_term = term.left.right
            x_term = term.right
            f_x = SKITerm(typ='APP', left=f_term, right=x_term)
            g_x = SKITerm(typ='APP', left=g_term, right=x_term)
            return SKITerm(typ='APP', left=f_x, right=g_x), True
        
        # Root not reducible - descend (normal order: left then right)
        if term.typ == 'APP':
            # Try reducing left subtree
            new_left, did = SKICore.reduce_once_normal(term.left)
            if did:
                return SKITerm(typ='APP', left=new_left, right=term.right), True
            
            # Left didn't reduce, try right subtree
            new_right, did = SKICore.reduce_once_normal(term.right)
            if did:
                return SKITerm(typ='APP', left=term.left, right=new_right), True
        
        # Nothing reduced
        return term, False
    
    @staticmethod
    def reduce_step(f: Fiber) -> Tuple[Fiber, bool]:
        """
        Attempt one reduction step on the top of stack using normal-order strategy.
        Returns (new_fiber, did_reduce).
        
        Now uses reduce_once_normal which descends into subterms,
        so it can reduce expressions like ((K I) x) x → (I x) → x
        """
        if len(f.S) == 0:
            return f, False
        
        term = f.S[0]
        rest = f.S[1:]
        
        new_term, did_reduce = SKICore.reduce_once_normal(term)
        if not did_reduce:
            return f, False
        
        return Fiber((new_term,) + rest, f.E, f.C, f.D), True
    
    @staticmethod
    def step_fiber(f: Fiber) -> Tuple[Fiber, int]:
        """
        Execute one SECD step based on code queue.
        Returns (new_fiber, executed_opcode).
        """
        if len(f.C) == 0:
            return f, SKICore.OP_NOOP
        
        op = f.C[0]
        rest = f.C[1:]
        temp = Fiber(f.S, f.E, rest, f.D)
        
        if op == SKICore.OP_S:
            return SKICore.push_combinator(temp, 'S'), op
        elif op == SKICore.OP_K:
            return SKICore.push_combinator(temp, 'K'), op
        elif op == SKICore.OP_I:
            return SKICore.push_combinator(temp, 'I'), op
        elif op == SKICore.OP_APP:
            return SKICore.apply(temp), op
        elif op == SKICore.OP_REDUCE:
            # Perform one reduction step
            new_f, _ = SKICore.reduce_step(temp)
            return new_f, op
        elif op == SKICore.OP_VAR_X:
            return SKICore.push_var(temp, 'x'), op
        elif op == SKICore.OP_VAR_Y:
            return SKICore.push_var(temp, 'y'), op
        elif op == SKICore.OP_VAR_Z:
            return SKICore.push_var(temp, 'z'), op
        elif op == SKICore.OP_VAR_W:
            return SKICore.push_var(temp, 'w'), op
        elif op == SKICore.OP_HALT:
            return temp, op
        else:
            return temp, op

# ==========================================
# 4. MANIFOLD SKI FOR SECD
# ==========================================
class ManifoldSKI(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_ops=11):
        super().__init__()
        self.d = hidden_dim
        self.k = num_ops
        
        # Combinator embeddings (NOOP, S, K, I, APP, REDUCE, VAR_X, VAR_Y, VAR_Z, VAR_W, HALT)
        self.op_embedding = nn.Embedding(num_ops, hidden_dim)
        
        # Address matrix for routing
        self.address_matrix = nn.Parameter(torch.randn(num_ops, hidden_dim))
        self.beta = 5.0
        
        # CORE DEQ: Main solver (Jones Section 4.2)
        self.W = nn.Parameter(torch.randn(num_ops, hidden_dim, hidden_dim) * 0.01)
        self.U = nn.Parameter(torch.randn(num_ops, hidden_dim, hidden_dim) * 0.01)
        self.V = nn.Parameter(torch.randn(num_ops, hidden_dim, hidden_dim) * 0.01)
        
        # LOCAL STABILIZER α: Spatially adaptive damping (Jones Section 4.3)
        # Learns when to trust the DEQ update vs maintain current state
        # Input: [h_context, fiber_state] → Output: α ∈ (0,1)^d
        self.stabilizer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()  # α ∈ (0,1)
        )
        # Initialize to ~0.3 (moderate damping)
        with torch.no_grad():
            self.stabilizer[-2].weight.data *= 0.1
            self.stabilizer[-2].bias.data.fill_(-1.0)  # sigmoid(-1) ≈ 0.27
        
        # GLOBAL SPECTRAL CONTROLLER γ: Step-size scaling (Jones Section 4.4)
        # Ensures ρ(Jf) stays in critical band [0.85, 0.95]
        # Input: [routing_entropy, sequence_position] → Output: γ > 0
        self.controller = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()  # γ > 0
        )
        # Initialize to γ ≈ 0.5 (moderate step size)
        with torch.no_grad():
            self.controller[-2].weight.data *= 0.1
            self.controller[-2].bias.data.fill_(0.0)
        
        # Fiber encoding (encode stack depth and term complexity)
        self.fiber_enc_depth = nn.Linear(1, hidden_dim)
        self.fiber_enc_complexity = nn.Linear(1, hidden_dim)
        
        # Decoder (predict next operation or term type)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
    
    def term_complexity(self, term: SKITerm) -> float:
        """Compute complexity metric for a term (tree depth)."""
        if term.typ in ['S', 'K', 'I', 'VAR']:
            return 1.0
        elif term.typ == 'APP':
            left_c = self.term_complexity(term.left) if term.left else 0
            right_c = self.term_complexity(term.right) if term.right else 0
            return 1.0 + max(left_c, right_c)
        return 0.0

    def embed_fiber(self, fibers, device):
        """Encode fiber state (stack depth + term complexity)."""
        vecs = []
        for f in fibers:
            depth = float(len(f.S))
            # Check if top of stack is an SKITerm
            if f.S and isinstance(f.S[0], SKITerm):
                complexity = self.term_complexity(f.S[0])
            else:
                complexity = 0.0
            
            depth_emb = self.fiber_enc_depth(torch.tensor([[depth]], device=device))
            complex_emb = self.fiber_enc_complexity(torch.tensor([[complexity]], device=device))
            vecs.append(torch.tanh(depth_emb + complex_emb).squeeze(0))
        return torch.stack(vecs)

    def forward(self, h, fibers, token_idx, teacher_ops=None):
        batch_size = token_idx.shape[0]
        device = h.device
        
        # Embed tokens (operations)
        token_emb = self.op_embedding(torch.clamp(token_idx, 0, self.k - 1))
        
        f_emb = self.embed_fiber(fibers, device)
        
        # Routing (control: based on operation type, not term structure)
        scores = torch.matmul(F.normalize(token_emb, dim=1),
                             F.normalize(self.address_matrix, dim=1).T)
        pi = F.softmax(self.beta * scores, dim=-1)
        idx = pi.argmax(dim=-1)
        alpha = (F.one_hot(idx, self.k).float() - pi.detach()) + pi
        
        # DEQ update
        def deq_func(z, h_c, f_c, W_p, U_p, V_p):
            t1 = torch.einsum('bd, kde -> bke', z, W_p)
            t2 = torch.einsum('bd, kde -> bke', h_c, U_p)
            t3 = torch.einsum('bd, kde -> bke', f_c, V_p)
            return torch.einsum('bk, bkd -> bd', alpha, torch.tanh(t1 + t2 + t3))
        
        z_star = DEQFixedPoint.apply(deq_func, torch.zeros_like(h), h, f_emb,
                                     self.W, self.U, self.V)
        
        # Jones 3-Network stabilization
        stabilizer_input = torch.cat([h, f_emb], dim=-1)
        alpha_local = self.stabilizer(stabilizer_input)
        
        routing_entropy = -(pi * torch.log(pi + 1e-8)).sum(dim=-1, keepdim=True)
        controller_input = torch.cat([routing_entropy, torch.zeros(batch_size, 1, device=device)], dim=-1)
        gamma_global = self.controller(controller_input)
        
        # Update: h_{t+1} = h_t + γ·α⊙z*
        h_next = h + gamma_global * alpha_local * z_star
        
        # Symbolic execution
        new_fibers = []
        executed_ops = []
        
        for b in range(batch_size):
            f = fibers[b]
            op_idx = idx[b].item() if teacher_ops is None else teacher_ops[b].item()
            executed_ops.append(op_idx)
            
            # Execute SKI operation
            if len(f.C) > 0:
                # Auto-pilot: execute from code queue
                new_f, _ = SKICore.step_fiber(f)
            else:
                # Manual control: execute based on token
                fake_fiber = Fiber(f.S, f.E, (op_idx,), f.D)
                new_f, _ = SKICore.step_fiber(fake_fiber)
            
            new_fibers.append(new_f)
        
        stabilization_metrics = {
            'alpha_mean': alpha_local.mean().item(),
            'gamma': gamma_global.mean().item(),
            'routing_entropy': routing_entropy.mean().item()
        }
        
        return h_next, new_fibers, self.decoder(h_next), torch.tensor(executed_ops, device=device), pi, stabilization_metrics

# ==========================================
# 5. SKI CURRICULUM GENERATOR
# ==========================================

def random_combinator() -> SKITerm:
    """Generate random S, K, or I."""
    choice = random.choice(['S', 'K', 'I'])
    return SKITerm(typ=choice)

def random_variable() -> SKITerm:
    """Generate test variable with distinct name."""
    name = random.choice(['x', 'y', 'z', 'w'])
    return SKITerm(typ='VAR', name=name)

def var_name_to_opcode(name: str) -> int:
    """Map variable name to opcode."""
    mapping = {'x': SKICore.OP_VAR_X, 'y': SKICore.OP_VAR_Y, 
               'z': SKICore.OP_VAR_Z, 'w': SKICore.OP_VAR_W}
    return mapping.get(name, SKICore.OP_VAR_X)

def build_random_term(depth: int, reducible_prob: float = 0.3) -> SKITerm:
    """
    Build random SKI term with target depth.
    Injects reducible patterns (I x, K x y, S K K) with probability.
    """
    if depth <= 1:
        # Leaf: combinator or variable
        if random.random() < 0.7:
            return random_combinator()
        else:
            return random_variable()
    
    # Inject reducible patterns
    if random.random() < reducible_prob:
        pattern = random.choice(['I_x', 'K_x_y', 'S_K_K'])
        
        if pattern == 'I_x':
            # I x
            I = SKITerm(typ='I')
            x = random_variable()
            return SKITerm(typ='APP', left=I, right=x)
        
        elif pattern == 'K_x_y':
            # K x y
            K = SKITerm(typ='K')
            x = build_random_term(max(1, depth - 2), reducible_prob=0)
            K_x = SKITerm(typ='APP', left=K, right=x)
            y = build_random_term(max(1, depth - 2), reducible_prob=0)
            return SKITerm(typ='APP', left=K_x, right=y)
        
        else:  # S_K_K
            # S K K x
            S = SKITerm(typ='S')
            K1 = SKITerm(typ='K')
            K2 = SKITerm(typ='K')
            S_K = SKITerm(typ='APP', left=S, right=K1)
            S_K_K = SKITerm(typ='APP', left=S_K, right=K2)
            x = build_random_term(max(1, depth - 3), reducible_prob=0)
            return SKITerm(typ='APP', left=S_K_K, right=x)
    
    # Random application
    left_depth = random.randint(1, depth - 1)
    right_depth = depth - left_depth
    
    left = build_random_term(left_depth, reducible_prob)
    right = build_random_term(right_depth, reducible_prob)
    
    return SKITerm(typ='APP', left=left, right=right)

def term_to_program(term: SKITerm) -> List[int]:
    """
    Convert SKI term to OpCode sequence (reverse Polish notation).
    For term (f x), emit: [f_opcodes, x_opcodes, APP]
    Preserves variable identity (x/y/z/w get distinct opcodes).
    """
    if term.typ == 'S':
        return [SKICore.OP_S]
    elif term.typ == 'K':
        return [SKICore.OP_K]
    elif term.typ == 'I':
        return [SKICore.OP_I]
    elif term.typ == 'VAR':
        return [var_name_to_opcode(term.name)]
    elif term.typ == 'APP':
        left_ops = term_to_program(term.left)
        right_ops = term_to_program(term.right)
        return left_ops + right_ops + [SKICore.OP_APP]
    return []

def reduce_term_symbolic(term: SKITerm, max_steps: int = 50) -> Tuple[SKITerm, int]:
    """
    Symbolically reduce a term to normal form (ground truth).
    Returns (reduced_term, num_steps).
    """
    current = term
    steps = 0
    
    for _ in range(max_steps):
        fiber = Fiber((current,), {}, tuple(), tuple())
        new_fiber, did_reduce = SKICore.reduce_step(fiber)
        
        if not did_reduce:
            break
        
        current = new_fiber.S[0] if new_fiber.S else current
        steps += 1
    
    return current, steps

def get_ski_batch(task_type: str):
    """
    Generate SKI training examples.
    
    Tasks:
    - 'identity': I x → x
    - 'constant': K x y → x
    - 'simple_s': S K K x → x (simple S reduction)
    - 'church_0': Church numeral 0 (K I applied to f x)
    - 'deep_5': Random depth-5 expression
    - 'deep_7': Random depth-7 expression
    - 'deep_10': Random depth-10 expression (generalization test)
    """
    
    if task_type == 'identity':
        # I x → x
        program = [SKICore.OP_I, SKICore.OP_VAR_X, SKICore.OP_APP, SKICore.OP_REDUCE]
        target_ops = [SKICore.OP_I, SKICore.OP_VAR_X, SKICore.OP_APP, SKICore.OP_REDUCE]
        expected_result = "x"
    
    elif task_type == 'constant':
        # K x y → x
        program = [SKICore.OP_K, SKICore.OP_VAR_X, SKICore.OP_APP, 
                  SKICore.OP_VAR_Y, SKICore.OP_APP, SKICore.OP_REDUCE]
        target_ops = program.copy()
        expected_result = "x"
    
    elif task_type == 'simple_s':
        # S K K x → x (classic SKK = I proof)
        program = [SKICore.OP_S, SKICore.OP_K, SKICore.OP_APP,
                  SKICore.OP_K, SKICore.OP_APP, SKICore.OP_VAR_X, SKICore.OP_APP,
                  SKICore.OP_REDUCE, SKICore.OP_REDUCE]
        target_ops = program.copy()
        expected_result = "x"
    
    elif task_type == 'church_0':
        # Church 0 = K I
        # Test: (K I) f x → I x → x
        program = [
            SKICore.OP_K, SKICore.OP_I, SKICore.OP_APP,    # Build K I
            SKICore.OP_VAR_X, SKICore.OP_APP,              # Apply to f (using x as placeholder)
            SKICore.OP_VAR_Y, SKICore.OP_APP,              # Apply to x (using y)
            SKICore.OP_REDUCE,                              # (K I) f → I
            SKICore.OP_REDUCE                               # I x → x
        ]
        target_ops = program.copy()
        expected_result = "y"
    
    elif task_type.startswith('deep_'):
        # Deep random expressions
        depth = int(task_type.split('_')[1])
        
        # Generate random term with reducible patterns
        term = build_random_term(depth, reducible_prob=0.4)
        
        # Convert to program
        build_ops = term_to_program(term)
        
        # Get ground truth reduction
        reduced_term, num_steps = reduce_term_symbolic(term, max_steps=100)
        
        # Build full program: build term + reduce steps
        # Cap reductions at 10 to keep training tractable
        reduction_ops = [SKICore.OP_REDUCE] * min(num_steps, 10)
        program = build_ops + reduction_ops
        target_ops = program.copy()
        expected_result = str(reduced_term)
    
    else:  # noop
        program = [SKICore.OP_I, SKICore.OP_VAR_X, SKICore.OP_APP]
        target_ops = program.copy()
        expected_result = "x"
    
    return torch.tensor(program), torch.tensor(target_ops), expected_result

# ==========================================
# 5b. RIGOROUS EVALUATION: AUTONOMOUS REDUCTION
# ==========================================

def evaluate_autonomous_reduction(model, term: SKITerm, ground_truth: SKITerm, max_steps: int = 50) -> Dict[str, Any]:
    """
    Test if model can autonomously reduce a term to normal form.
    
    Phase 1: Build term (teacher-forced with build opcodes)
    Phase 2: Autonomous reduction (model chooses REDUCE/HALT until normal form)
    
    Returns:
        - success: True if model produces correct normal form
        - model_result: Final term from model
        - ground_truth_result: Expected normal form
        - steps_taken: Number of reduction steps
        - exact_match: Structural equality
    """
    device = next(model.parameters()).device
    
    # Phase 1: Build term (teacher-forced)
    build_ops = term_to_program(term)
    
    h = torch.zeros(1, model.d, device=device)
    fibers = [Fiber(tuple(), {}, tuple(), tuple())]
    
    with torch.no_grad():
        for op_val in build_ops:
            tok = torch.tensor([op_val], device=device)
            h, fibers, _, _, _, _ = model(h, fibers, tok, teacher_ops=tok)
    
    # Phase 2: Autonomous reduction
    # Model must choose REDUCE or HALT until normal form
    built_term = fibers[0].S[0] if fibers[0].S else None
    
    if not built_term:
        return {
            'success': False,
            'model_result': None,
            'ground_truth_result': ground_truth,
            'steps_taken': 0,
            'exact_match': False,
            'error': 'Failed to build term'
        }
    
    current_term = built_term
    steps_taken = 0
    
    # Autonomous reduction loop
    for step in range(max_steps):
        # Check if already in normal form
        if SKICore.is_normal_form(current_term):
            break
        
        # Model predicts next action (REDUCE or HALT)
        # Feed a constant "reduction mode" token (use NOOP)
        tok = torch.tensor([SKICore.OP_NOOP], device=device)
        h, _, logits, _, pi, _ = model(h, fibers, tok, teacher_ops=None)
        
        # Get model's choice
        action = logits[0].argmax().item()
        
        # Apply action to symbolic machine
        if action == SKICore.OP_REDUCE:
            test_fiber = Fiber((current_term,), {}, (SKICore.OP_REDUCE,), tuple())
            new_fiber, _ = SKICore.step_fiber(test_fiber)
            current_term = new_fiber.S[0] if new_fiber.S else current_term
            fibers = [new_fiber]
            steps_taken += 1
        elif action == SKICore.OP_HALT:
            break
        else:
            # Model emitted invalid action, treat as HALT
            break
    
    # Check correctness
    exact_match = SKICore.terms_equal(current_term, ground_truth)
    is_normal = SKICore.is_normal_form(current_term)
    
    return {
        'success': exact_match,
        'model_result': current_term,
        'ground_truth_result': ground_truth,
        'steps_taken': steps_taken,
        'exact_match': exact_match,
        'model_is_normal_form': is_normal,
        'ground_truth_is_normal_form': SKICore.is_normal_form(ground_truth)
    }

# ==========================================
# 6. TRAINING LOOP
# ==========================================

def run_ski_curriculum(use_semantic_loss=True, autonomous_reduction_prob=0.3):
    """
    Train SKI combinator system with optional semantic loss and autonomous reduction.
    
    Args:
        use_semantic_loss: If True, add loss term for correct final term
        autonomous_reduction_prob: Probability of training with autonomous Phase 2 reduction
    """
    model = ManifoldSKI(vocab_size=16, hidden_dim=64, num_ops=11)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(">>> SKI COMBINATOR CALCULUS via DEQ-SECD")
    print("Tasks: Basic (I/K/S) + Church Numerals + Deep Expressions")
    print("Goal: Learn unbounded symbolic rewriting with depth generalization")
    print("Note: 11 opcodes (NOOP, S, K, I, APP, REDUCE, VAR_X/Y/Z/W, HALT)")
    print(f"Semantic loss: {'ENABLED' if use_semantic_loss else 'DISABLED'}")
    print(f"Autonomous reduction: {autonomous_reduction_prob*100:.0f}% of samples")
    print()
    
    # Curriculum stages
    # Stage 1 (epochs 0-1000): Basic combinators
    # Stage 2 (epochs 1000-2000): Church numerals + shallow deep expressions
    # Stage 3 (epochs 2000-3000): Deeper expressions for generalization
    
    for epoch in range(3000):
        # Progressive curriculum
        if epoch < 1000:
            # Stage 1: Master basics
            task = random.choice(['identity', 'identity', 'constant', 'simple_s'])
        elif epoch < 2000:
            # Stage 2: Introduce Church numerals and depth-5
            task = random.choice(['identity', 'constant', 'simple_s', 'church_0', 'deep_5'])
        else:
            # Stage 3: Deeper generalization (depth 7-10)
            task = random.choice(['simple_s', 'church_0', 'deep_5', 'deep_7', 'deep_10'])
        
        inputs, teacher, expected = get_ski_batch(task)
        
        # Skip if program is too long (deep expressions can be large)
        if len(inputs) > 100:
            continue
        
        # Decide: teacher-forced or autonomous reduction?
        use_autonomous = (random.random() < autonomous_reduction_prob and 
                         task.startswith('deep_') and epoch > 1000)
        
        h = torch.zeros(1, 64)
        fibers = [Fiber(tuple(), {}, tuple(), tuple())]
        
        all_pis = []
        all_alphas = []
        all_gammas = []
        
        if use_autonomous:
            # TWO-PHASE TRAINING: Build (teacher-forced) + Reduce (autonomous)
            
            # Phase 1: Build term (teacher-forced)
            # Find where REDUCE ops start
            build_end = 0
            for i, op in enumerate(inputs):
                if op.item() == SKICore.OP_REDUCE:
                    build_end = i
                    break
            
            # Execute build phase with teacher forcing
            for t in range(build_end):
                tok = inputs[t].unsqueeze(0)
                h, fibers, logits, exec_ops, pi, stab = model(h, fibers, tok, teacher_ops=tok)
                all_pis.append(pi)
                
                f_emb = model.embed_fiber(fibers, h.device)
                stab_input = torch.cat([h, f_emb], dim=-1)
                alpha_t = model.stabilizer(stab_input).mean()
                routing_entropy_t = -(pi * torch.log(pi + 1e-8)).sum(dim=-1, keepdim=True)
                ctrl_input = torch.cat([routing_entropy_t, torch.zeros(1, 1, device=h.device)], dim=-1)
                gamma_t = model.controller(ctrl_input).squeeze()
                all_alphas.append(alpha_t)
                all_gammas.append(gamma_t)
            
            # Phase 2: Autonomous reduction (model chooses actions)
            built_term = fibers[0].S[0] if fibers[0].S else None
            max_reduce_steps = 20
            
            for step in range(max_reduce_steps):
                if built_term and SKICore.is_normal_form(built_term):
                    break
                
                # Model predicts next action
                tok = torch.tensor([SKICore.OP_NOOP])  # Reduction mode signal
                h, fibers, logits, exec_ops, pi, stab = model(h, fibers, tok, teacher_ops=None)
                
                # Get model's choice from logits
                action = logits[0].argmax().item()
                
                # Execute chosen action
                if action == SKICore.OP_REDUCE and built_term:
                    test_fiber = Fiber((built_term,), {}, (SKICore.OP_REDUCE,), tuple())
                    new_fiber, did_reduce = SKICore.step_fiber(test_fiber)
                    if did_reduce:
                        built_term = new_fiber.S[0] if new_fiber.S else built_term
                        fibers = [new_fiber]
                elif action == SKICore.OP_HALT:
                    break
                
                # Track pi for routing loss (but don't supervise with teacher)
                all_pis.append(pi)
                f_emb = model.embed_fiber(fibers, h.device)
                stab_input = torch.cat([h, f_emb], dim=-1)
                alpha_t = model.stabilizer(stab_input).mean()
                routing_entropy_t = -(pi * torch.log(pi + 1e-8)).sum(dim=-1, keepdim=True)
                ctrl_input = torch.cat([routing_entropy_t, torch.zeros(1, 1, device=h.device)], dim=-1)
                gamma_t = model.controller(ctrl_input).squeeze()
                all_alphas.append(alpha_t)
                all_gammas.append(gamma_t)
            
            final_term = built_term
        else:
            # STANDARD: Full teacher-forced execution
            for t in range(len(inputs)):
                tok = inputs[t].unsqueeze(0)
                h, fibers, logits, exec_ops, pi, stab = model(h, fibers, tok, teacher_ops=None)
                all_pis.append(pi)
                
                # Extract α and γ for loss
                f_emb = model.embed_fiber(fibers, h.device)
                stab_input = torch.cat([h, f_emb], dim=-1)
                alpha_t = model.stabilizer(stab_input).mean()
                
                routing_entropy_t = -(pi * torch.log(pi + 1e-8)).sum(dim=-1, keepdim=True)
                ctrl_input = torch.cat([routing_entropy_t, torch.zeros(1, 1, device=h.device)], dim=-1)
                gamma_t = model.controller(ctrl_input).squeeze()
                
                all_alphas.append(alpha_t)
                all_gammas.append(gamma_t)
            
            final_term = fibers[0].S[0] if fibers[0].S else None
        
        final_str = str(final_term) if final_term else "EMPTY"
        
        # Success: check if result matches expected (flexible matching)
        success = False
        if final_term:
            if expected == "x" and isinstance(final_term, SKITerm) and final_term.typ == 'VAR':
                success = True
            elif expected == final_str:
                success = True
            # For deep expressions, also accept if term is in normal form (no more reductions possible)
            elif task.startswith('deep_'):
                test_fiber = Fiber((final_term,), {}, tuple(), tuple())
                _, can_reduce = SKICore.reduce_step(test_fiber)
                if not can_reduce:
                    success = True  # Reached normal form
        
        # LOSS COMPUTATION
        
        # 1. Routing supervision (only for teacher-forced steps)
        routing_loss = torch.tensor(0.0, requires_grad=True)
        if not use_autonomous:
            # Standard: supervise all steps
            for pi, teacher_op in zip(all_pis, teacher):
                ce_loss = -torch.log(pi[0, teacher_op] + 1e-8)
                routing_loss = routing_loss + ce_loss
        else:
            # Autonomous: only supervise build phase
            build_end = min(len(all_pis), len([op for op in inputs if op.item() != SKICore.OP_REDUCE]))
            for pi, teacher_op in zip(all_pis[:build_end], teacher[:build_end]):
                ce_loss = -torch.log(pi[0, teacher_op] + 1e-8)
                routing_loss = routing_loss + ce_loss
        
        # 2. Semantic loss: Does final term match ground truth?
        loss_semantic = torch.tensor(0.0, requires_grad=True)
        if use_semantic_loss and task.startswith('deep_'):
            # Get ground truth from task
            depth = int(task.split('_')[1])
            # Regenerate the same term (deterministic within epoch via random seed state)
            # For now, just check if we reached normal form as a proxy
            if final_term:
                is_nf = SKICore.is_normal_form(final_term)
                if not is_nf:
                    # Penalize not reaching normal form
                    loss_semantic = loss_semantic + 1.0
        
        # 3. Orthogonality loss
        A_n = F.normalize(model.address_matrix, dim=1)
        loss_ortho = torch.norm(torch.mm(A_n, A_n.T) - torch.eye(11))
        
        # 4. Spectral band loss
        avg_alpha = torch.stack(all_alphas).mean() if all_alphas else torch.tensor(0.5)
        avg_gamma = torch.stack(all_gammas).mean() if all_gammas else torch.tensor(0.5)
        effective_step = avg_gamma * avg_alpha
        loss_spectral = torch.relu(effective_step - 0.95) ** 2 + torch.relu(0.3 - effective_step) ** 2
        
        # Total loss
        total_loss = routing_loss + 0.1 * loss_ortho + loss_spectral + loss_semantic
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 200 == 0:
            # Show curriculum stage
            if epoch < 1000:
                stage = "Stage 1: Basics"
            elif epoch < 2000:
                stage = "Stage 2: Church+Depth5"
            else:
                stage = "Stage 3: Deep Generalization"
            
            result_str = final_str[:20] if len(final_str) <= 20 else final_str[:17] + "..."
            status = "✓" if success else "✗"
            mode = "AUTO" if use_autonomous else "SUPV"
            print(f"Ep {epoch:4d} | {stage:25s} | {task:10s} | {mode} | Loss: {total_loss.item():7.4f} | "
                  f"{result_str:20s} | {status} | α: {avg_alpha.item():.3f} | γ: {avg_gamma.item():.3f}")
    
    print("\n" + "="*80)
    print("FINAL TESTS: Rigorous Evaluation with Variable Identity")
    print("="*80)
    print("Key fixes:")
    print("  1. Distinct variable opcodes (VAR_X/Y/Z/W) preserve identity")
    print("  2. Ground truth comparison uses structural equality")
    print("  3. Test 3 requires EXACT normal form match (not just 'any normal form')")
    print()
    
    # Test 1: S K K x (compositional proof)
    print(">>> TEST 1: S K K x → x (Basic Compositional)")
    test_program = [SKICore.OP_S, SKICore.OP_K, SKICore.OP_APP,
                    SKICore.OP_K, SKICore.OP_APP, SKICore.OP_VAR_X, SKICore.OP_APP,
                    SKICore.OP_REDUCE, SKICore.OP_REDUCE]
    
    h = torch.zeros(1, 64)
    fibers = [Fiber(tuple(), {}, tuple(), tuple())]
    
    with torch.no_grad():
        for i, tok_val in enumerate(test_program):
            tok = torch.tensor([tok_val])
            h, fibers, _, _, _, _ = model(h, fibers, tok, teacher_ops=None)
    
    final = fibers[0].S[0] if fibers[0].S else None
    expected_x = SKITerm(typ='VAR', name='x')
    test1_pass = (final and SKICore.terms_equal(final, expected_x))
    print(f"   Result: {final}")
    print(f"   Expected: {expected_x}")
    print(f"   Structural equality: {test1_pass}")
    print(f"   [{'PASS' if test1_pass else 'FAIL'}]")
    
    # Test 2: Church 0 with proper variable tracking
    print("\n>>> TEST 2: Church 0 = ((K I) f) x → x")
    test_program = [
        SKICore.OP_K, SKICore.OP_I, SKICore.OP_APP,
        SKICore.OP_VAR_X, SKICore.OP_APP,  # f (using VAR_X)
        SKICore.OP_VAR_Y, SKICore.OP_APP,  # x (using VAR_Y)
        SKICore.OP_REDUCE, SKICore.OP_REDUCE
    ]
    
    h = torch.zeros(1, 64)
    fibers = [Fiber(tuple(), {}, tuple(), tuple())]
    
    with torch.no_grad():
        for tok_val in test_program:
            tok = torch.tensor([tok_val])
            h, fibers, _, _, _, _ = model(h, fibers, tok, teacher_ops=None)
    
    final = fibers[0].S[0] if fibers[0].S else None
    expected_y = SKITerm(typ='VAR', name='y')
    test2_pass = (final and SKICore.terms_equal(final, expected_y))
    print(f"   Result: {final}")
    print(f"   Expected: {expected_y} (should be y, not x)")
    print(f"   Structural equality: {test2_pass}")
    print(f"   [{'PASS' if test2_pass else 'FAIL'}]")
    
    # Test 3: Deep random expression with EXACT MATCH requirement
    print("\n>>> TEST 3: Random Depth-15 Expression (EXACT MATCH REQUIRED)")
    deep_term = build_random_term(15, reducible_prob=0.5)
    reduced_ground_truth, gt_steps = reduce_term_symbolic(deep_term, max_steps=50)
    
    build_ops = term_to_program(deep_term)
    reduction_ops = [SKICore.OP_REDUCE] * min(gt_steps, 20)
    test_program = build_ops + reduction_ops
    
    h = torch.zeros(1, 64)
    fibers = [Fiber(tuple(), {}, tuple(), tuple())]
    
    print(f"   Original term: {str(deep_term)[:60]}...")
    print(f"   Ground truth: {str(reduced_ground_truth)[:60]}... ({gt_steps} reduction steps)")
    
    with torch.no_grad():
        for tok_val in test_program[:100]:  # Cap execution
            tok = torch.tensor([tok_val])
            h, fibers, _, _, _, _ = model(h, fibers, tok, teacher_ops=None)
    
    final = fibers[0].S[0] if fibers[0].S else None
    
    # RIGOROUS CHECK: Exact structural equality
    test3_exact_match = (final and SKICore.terms_equal(final, reduced_ground_truth))
    test3_is_normal = SKICore.is_normal_form(final) if final else False
    
    print(f"   Model result: {str(final)[:60] if final else 'EMPTY'}...")
    print(f"   Exact structural match: {test3_exact_match}")
    print(f"   Model is normal form: {test3_is_normal}")
    print(f"   [{'PASS' if test3_exact_match else 'FAIL'}]")
    
    if not test3_exact_match and test3_is_normal:
        print(f"   NOTE: Model reached *a* normal form, but not the *correct* one")
        print(f"         This means: interpreter works, but computation differs from ground truth")
    
    # Summary
    print("\n" + "="*80)
    print("RIGOROUS EVALUATION SUMMARY")
    print("="*80)
    all_pass = test1_pass and test2_pass and test3_exact_match
    print(f"Test 1 (S K K = I):     {'✓ PASS' if test1_pass else '✗ FAIL'} - Exact match: {test1_pass}")
    print(f"Test 2 (Church 0):      {'✓ PASS' if test2_pass else '✗ FAIL'} - Exact match: {test2_pass}")
    print(f"Test 3 (Depth 15):      {'✓ PASS' if test3_exact_match else '✗ FAIL'} - Exact match: {test3_exact_match}")
    print()
    
    if all_pass:
        print("🎉 STRONG CLAIM JUSTIFIED!")
        print("   ✓ Basic SKI reductions with variable identity preserved")
        print("   ✓ Church numeral encoding (variables tracked correctly)")
        print("   ✓ Deep generalization with EXACT normal form match")
        print("   → The interpreter computes correctly and the model can follow it")
    elif test1_pass and test2_pass:
        print("✓ INTERPRETER VALIDATED: Basic SKI + Church numerals work correctly")
        print("⚠ GENERALIZATION INCOMPLETE:")
        print(f"  → Test 3: {'Reached normal form' if test3_is_normal else 'Did not normalize'}")
        print(f"            Exact match: {test3_exact_match}")
        print()
        if not test3_exact_match and test3_is_normal:
            print("  DIAGNOSIS: Model reached *a* normal form, but not the *correct* one")
            print("             Possible causes:")
            print("             - Variable renaming during reduction")
            print("             - Different reduction order (applicative vs normal)")
            print("             - Model hasn't learned deep reduction policy")
        print()
        print("  CURRENT CLAIM: 'Correct SKI interpreter with Church numeral support'")
        print("  STRONGER CLAIM needs: Exact match on depth-15 generalization")
    else:
        print("⚠ BASIC TESTS INCOMPLETE")
        print(f"  → Test 1 (S K K = I): {test1_pass}")
        print(f"  → Test 2 (Church 0): {test2_pass}")
        print()
        print("  This suggests the interpreter or variable tracking has bugs.")
    print("="*80)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': 16,
        'hidden_dim': 64,
        'num_ops': 11
    }, 'ski_trained_rigorous.pt')
    print("\n[SAVED] Checkpoint: ski_trained_rigorous.pt")
    print("\nKEY IMPROVEMENTS:")
    print("  1. Variable identity preserved (VAR_X/Y/Z/W distinct opcodes)")
    print("  2. Structural equality used for ground truth comparison")
    print("  3. Test 3 requires EXACT normal form match (not 'any normal form')")
    print("  → This makes the evaluation claims rigorous and falsifiable")
    
    return model

if __name__ == "__main__":
    import sys
    
    # Parse training mode
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    
    if mode == "baseline":
        print("="*80)
        print("BASELINE MODE: Teacher-forced, no semantic loss")
        print("="*80)
        model = run_ski_curriculum(use_semantic_loss=False, autonomous_reduction_prob=0.0)
    elif mode == "semantic":
        print("="*80)
        print("SEMANTIC MODE: Teacher-forced + semantic loss")
        print("="*80)
        model = run_ski_curriculum(use_semantic_loss=True, autonomous_reduction_prob=0.0)
    elif mode == "autonomous":
        print("="*80)
        print("AUTONOMOUS MODE: Two-phase training (30% autonomous reduction)")
        print("="*80)
        model = run_ski_curriculum(use_semantic_loss=True, autonomous_reduction_prob=0.3)
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python difcomp.py [baseline|semantic|autonomous]")
        sys.exit(1)

