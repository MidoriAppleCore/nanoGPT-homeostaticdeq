"""
GREYBOX CYBERNETIC ARITHMETIC MODULE

Combines TWO powerful techniques:
1. PROTECTED MEMORY REGISTERS (Frequency Domain)
   - Low frequencies = persistent storage (no decay)
   - Allocate specific registers for arithmetic state
   
2. RESIDUAL FORCE BLENDING (Symbolic + Neural)
   - Symbolic handles "easy math" (1+1=2)
   - Neural learns "hard stuff" (parsing, carry bits, operators)
   - Learnable gate decides trust balance

REGISTER ALLOCATION:
- Freq 0: Running accumulator (PROTECTED - never decays)
- Freq 1: Current operation flag (+1 or -1)
- Freq 2-11: Digit buffer (10 registers for multi-digit)
- Freq 12-19: Carry bits and overflow handling (8 registers)
- Freq 20+: Free for neural network to use

This is Neuro-Symbolic AI meets Physics-Informed Networks!
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ArithmeticGreyboxModule(nn.Module):
    """
    Symbolic arithmetic reasoning layer that operates in Fourier frequency space.
    
    PROTECTED REGISTER ALLOCATION (frequency domain):
    - Freq 0: Accumulator (running total, NEVER DECAYS)
    - Freq 1: Operation flag (+1 = add, -1 = subtract)
    - Freq 2-11: Digit buffer (10 registers for multi-digit numbers)
    - Freq 12-19: Carry bits & overflow (8 registers)
    - Freq 20+: Free for neural network
    
    Total: 20 protected registers, 109 free registers
    
    RESIDUAL FORCE: symbolic_force + neural_force
    - Symbolic handles perfect arithmetic
    - Neural learns how to USE the arithmetic module
    """
    
    def __init__(self, vocab_size: int = 16, spatial_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.spatial_dim = spatial_dim
        self.freq_dim = spatial_dim // 2 + 1  # 256 -> 129 complex frequencies
        
        # REGISTER ALLOCATION
        self.REG_ACCUMULATOR = 0      # Running total
        self.REG_OPERATION = 1         # +1 or -1
        self.REG_DIGIT_START = 2       # Digit buffer: 2-11
        self.REG_DIGIT_END = 11
        self.REG_CARRY_START = 12      # Carry bits: 12-19
        self.REG_CARRY_END = 19
        self.NUM_PROTECTED_REGS = 20   # First 20 frequencies are protected
        
        # Learnable blend factor: 0 = pure neural, 1 = pure symbolic
        # Start at 0.5 to give network a strong hint but allow learning
        self.symbolic_blend = nn.Parameter(torch.tensor(0.5))
        
        # PROTECTED MEMORY: Damping multiplier for register frequencies
        # These frequencies NEVER decay (damping = 1.0)
        self.register_protection = nn.Parameter(torch.ones(self.freq_dim), requires_grad=False)
        self.register_protection[self.NUM_PROTECTED_REGS:] = 0.0  # Only protect first 20
        
        # Token type detection (these will be set based on vocab)
        self.digit_tokens = set(range(1, 11))  # Assuming tokens 1-10 are digits 0-9
        self.plus_token = 11   # '+'
        self.minus_token = 12  # '-'
        self.equals_token = 13 # '='
        self.space_token = 14  # ' '
        self.newline_token = 15  # '\n'
        self.start_token = 0   # '<START>'
        
        # Arithmetic state (not learnable, updated during forward pass)
        self.register_buffer('accumulator', torch.zeros(1))
        self.register_buffer('current_digit', torch.zeros(1))
        self.register_buffer('operation', torch.ones(1))  # +1 for add, -1 for subtract
        self.register_buffer('reading_result', torch.zeros(1))  # Flag: are we generating answer?
        
        # CALCULATOR STATE: Accumulate numbers and operations for full computation
        self.register_buffer('current_number', torch.zeros(1))  # Number being built
        self.register_buffer('computed_result', torch.zeros(1))  # Final computed answer
        self.register_buffer('result_ready', torch.zeros(1))  # Flag: result is computed
    
    def set_vocab_mapping(self, stoi: dict):
        """
        Configure token IDs based on actual vocabulary.
        Call this after loading the vocab from meta.pkl
        """
        self.digit_tokens = set(stoi[str(i)] for i in range(10) if str(i) in stoi)
        self.plus_token = stoi.get('+', -1)
        self.minus_token = stoi.get('-', -1)
        self.equals_token = stoi.get('=', -1)
        self.space_token = stoi.get(' ', -1)
        self.newline_token = stoi.get('\n', -1)
        self.start_token = stoi.get('<START>', -1)
    
    def is_digit(self, token_id: int) -> bool:
        """Check if token is a digit 0-9"""
        return token_id in self.digit_tokens
    
    def token_to_digit(self, token_id: int) -> int:
        """Convert digit token to its numeric value"""
        # Assumes tokens 1-10 map to digits 0-9
        # This might need adjustment based on actual vocab ordering
        if token_id in self.digit_tokens:
            return (token_id - 1) % 10
        return 0
    
    def inject_arithmetic_state(self, 
                                carrier_freq: torch.Tensor,
                                src_token: int,
                                tgt_token: int) -> torch.Tensor:
        """
        CALCULATOR GREYBOX: Actually computes the arithmetic!
        
        Network must learn ROUTING, not computation:
        1. Navigate token sequence (hyperbolic edges handle this)
        2. READ computed result from registers 14-16 (DEQ convergence to correct freq)
        3. COPY result to trajectory → decoder (learned frequency-domain routing)
        
        This tests: Can neural networks learn ALGORITHMIC COMPOSITION 
        when given computational tools?
        
        NEW REGISTER ALLOCATION:
        - Freq 0: Accumulator (running total during computation)
        - Freq 1: Operation flag (+1 or -1)
        - Freq 2-11: Digit one-hots (current input digit)
        - Freq 12-13: Carry bits
        - Freq 14: COMPUTED RESULT - ones digit   ← NETWORK READS THIS!
        - Freq 15: COMPUTED RESULT - tens digit   ← NETWORK READS THIS!
        - Freq 16: COMPUTED RESULT - hundreds digit ← NETWORK READS THIS!
        - Freq 17-19: Reserved
        - Freq 20+: Neural network workspace
        
        Args:
            carrier_freq: Carrier in frequency domain (complex tensor)
            src_token: Source token ID
            tgt_token: Target token ID
            
        Returns:
            Modified carrier_freq with computed result in protected registers
        """
        # Clone to avoid in-place modification
        symbolic_carrier = carrier_freq.clone()
        
        # State machine
        if src_token == self.start_token:
            # Reset ALL state at start
            self.current_digit.zero_()
            self.current_number.zero_()
            self.operation.fill_(1.0)
            self.accumulator.zero_()
            self.reading_result.zero_()
            self.result_ready.zero_()
            self.computed_result.zero_()
            # Clear all registers
            symbolic_carrier[..., :self.NUM_PROTECTED_REGS, :] = 0.0
        
        # Build multi-digit numbers as we see digits
        if self.is_digit(src_token):
            digit_val = self.token_to_digit(src_token)
            self.current_digit.fill_(digit_val)
            
            # Accumulate into current number (e.g., 2,7 → 27)
            self.current_number.mul_(10).add_(digit_val)
            
            # Still provide one-hot for network to see input
            symbolic_carrier[..., self.REG_DIGIT_START:self.REG_DIGIT_END+1, 0] = 0.0
            digit_reg = self.REG_DIGIT_START + (digit_val % 10)
            symbolic_carrier[..., digit_reg, 0] = 1.0
        
        # When we see an operation, compute with accumulated number
        if src_token == self.plus_token or src_token == self.minus_token:
            # Apply previous operation to accumulator with current_number
            if self.operation.item() > 0:
                self.accumulator.add_(self.current_number)
            else:
                self.accumulator.sub_(self.current_number)
            
            # Reset for next number
            self.current_number.zero_()
            
            # Set new operation
            if src_token == self.plus_token:
                self.operation.fill_(1.0)
                symbolic_carrier[..., self.REG_OPERATION, 0] = 1.0
            elif src_token == self.minus_token:
                self.operation.fill_(-1.0)
                symbolic_carrier[..., self.REG_OPERATION, 0] = -1.0
        
        # When we see equals, finalize computation and inject result!
        if src_token == self.equals_token:
            # Apply final operation
            if self.operation.item() > 0:
                self.accumulator.add_(self.current_number)
            else:
                self.accumulator.sub_(self.current_number)
            
            # Store computed result
            self.computed_result.copy_(self.accumulator)
            self.result_ready.fill_(1.0)
            self.reading_result.fill_(1.0)
            
            # INJECT COMPUTED RESULT INTO REGISTERS 14-16!
            # Network must learn to READ these and route to decoder
            result_int = int(self.computed_result.item())
            result_int = max(0, min(999, result_int))  # Clamp to 0-999
            
            ones_digit = result_int % 10
            tens_digit = (result_int // 10) % 10
            hundreds_digit = (result_int // 100) % 10
            
            # Put answer in protected frequency registers!
            symbolic_carrier[..., 14, 0] = float(ones_digit)
            symbolic_carrier[..., 15, 0] = float(tens_digit)
            symbolic_carrier[..., 16, 0] = float(hundreds_digit)
            
            # Signal "result mode"
            symbolic_carrier[..., self.REG_OPERATION, 0] = 0.0
            # Clear digit one-hots (not needed in result mode)
            symbolic_carrier[..., self.REG_DIGIT_START:self.REG_DIGIT_END+1, 0] = 0.0
        
        # If we're in result mode and seeing answer digits, keep result in registers
        if self.reading_result.item() > 0 and self.result_ready.item() > 0:
            # Maintain result in registers - network should be copying it to output
            result_int = int(self.computed_result.item())
            result_int = max(0, min(999, result_int))
            
            ones_digit = result_int % 10
            tens_digit = (result_int // 10) % 10
            hundreds_digit = (result_int // 100) % 10
            
            symbolic_carrier[..., 14, 0] = float(ones_digit)
            symbolic_carrier[..., 15, 0] = float(tens_digit)
            symbolic_carrier[..., 16, 0] = float(hundreds_digit)
        
        return symbolic_carrier
    
    def apply_register_protection(self, carrier_freq: torch.Tensor, damping_factor: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency-dependent damping, but PROTECT arithmetic registers.
        
        Protected registers (0-19) never decay (damping = 1.0).
        Other frequencies decay normally according to damping_factor.
        
        This ensures arithmetic state persists across edges!
        
        Args:
            carrier_freq: Carrier in frequency domain (complex)
            damping_factor: Neural network's damping schedule
            
        Returns:
            Modified damping that protects arithmetic registers
        """
        # Protected registers get damping = 1.0 (no decay)
        # Other registers use neural damping
        protected_damping = damping_factor.clone()
        protected_damping[..., :self.NUM_PROTECTED_REGS] = 1.0
        
        return protected_damping
    
    def forward(self,
                carrier_freq_flat: torch.Tensor,
                src_token: Optional[int] = None,
                tgt_token: Optional[int] = None) -> torch.Tensor:
        """
        Blend symbolic arithmetic with neural carrier.
        
        Args:
            carrier_freq_flat: Neural carrier in frequency domain (flattened real/imag)
            src_token: Source token ID (optional)
            tgt_token: Target token ID (optional)
            
        Returns:
            Blended carrier with symbolic arithmetic injected
        """
        if src_token is None or tgt_token is None:
            # No token info, return unmodified
            return carrier_freq_flat
        
        # Reshape flat representation to complex
        freq_dim = carrier_freq_flat.shape[-1] // 2
        carrier_freq_real = carrier_freq_flat.view(*carrier_freq_flat.shape[:-1], freq_dim, 2)
        carrier_freq_complex = torch.view_as_complex(carrier_freq_real)
        
        # Inject symbolic arithmetic
        symbolic_carrier = self.inject_arithmetic_state(carrier_freq_complex, src_token, tgt_token)
        
        # Blend: weighted combination of neural and symbolic
        blend = torch.sigmoid(self.symbolic_blend)  # Clamp to [0, 1]
        blended_carrier = (1 - blend) * carrier_freq_complex + blend * symbolic_carrier
        
        # Convert back to flat representation
        blended_real = torch.view_as_real(blended_carrier)
        blended_flat = blended_real.flatten(-2, -1)
        
        return blended_flat
    
    def get_accumulator_state(self) -> dict:
        """Get current arithmetic state for debugging"""
        return {
            'accumulator': self.accumulator.item(),
            'current_digit': self.current_digit.item(),
            'operation': '+' if self.operation.item() > 0 else '-',
            'reading_result': bool(self.reading_result.item()),
            'symbolic_blend': torch.sigmoid(self.symbolic_blend).item()
        }


class GreyboxEdgeNeuralNet(nn.Module):
    """
    Edge network with greybox arithmetic module integrated.
    
    This is a wrapper around the standard EdgeNeuralNet that adds
    symbolic arithmetic reasoning to guide learning.
    
    KEY INNOVATION: Learnable gate that blends symbolic and neural forces
    - Gate → 0: Trust symbolic arithmetic (for standard math)
    - Gate → 1: Trust neural network (for fuzzy/novel cases)
    """
    
    def __init__(self, base_network, vocab_size: int = 16):
        super().__init__()
        self.base_network = base_network
        self.arithmetic_module = ArithmeticGreyboxModule(
            vocab_size=vocab_size,
            spatial_dim=base_network.hidden_dim - 1
        )
        
        # LEARNABLE GATE: Network decides when to trust symbolic vs neural
        # Input: carrier state + token embeddings
        # Output: blend factor [0, 1]
        freq_dim = (base_network.hidden_dim - 1) // 2 + 1
        self.gate_net = nn.Sequential(
            nn.Linear(freq_dim * 2, freq_dim),
            nn.GELU(),
            nn.Linear(freq_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Initialize gate to 0.5 (balanced trust)
        nn.init.constant_(self.gate_net[-2].bias, 0.0)
        
        # Token tracking for current forward pass
        self.current_src_token = None
        self.current_tgt_token = None
    
    def set_tokens(self, src_token: int, tgt_token: int):
        """Set current token pair before forward pass"""
        self.current_src_token = src_token
        self.current_tgt_token = tgt_token
    
    def forward(self, carrier_in: torch.Tensor, trajectory_in: torch.Tensor,
                source_pos: torch.Tensor, target_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with greybox arithmetic injection using LEARNABLE GATE.
        
        The Cyborg Blend:
        1. Neural network computes its force (learned behavior)
        2. Symbolic module computes correct arithmetic force
        3. Learnable gate decides blend: gate * neural + (1-gate) * symbolic
        
        Result: Network learns to trust symbolic math for standard cases,
                but can override for novel/fuzzy situations!
        """
        # Get neural prediction (standard forward pass)
        carrier_neural, trajectory_out = self.base_network(
            carrier_in, trajectory_in, source_pos, target_pos
        )
        
        # If no token info, return pure neural
        if self.current_src_token is None:
            return carrier_neural, trajectory_out
        
        # Convert carrier to frequency domain for symbolic injection
        carrier_freq = torch.fft.rfft(carrier_neural, dim=-1)
        carrier_freq_real = torch.view_as_real(carrier_freq)
        carrier_freq_flat = carrier_freq_real.flatten(-2, -1)
        
        # Get symbolic arithmetic force
        symbolic_freq_flat = self.arithmetic_module(
            carrier_freq_flat,
            self.current_src_token,
            self.current_tgt_token
        )
        
        # Compute learnable gate based on carrier state
        # Gate = 0: Trust symbolic (standard arithmetic)
        # Gate = 1: Trust neural (novel/fuzzy cases)
        gate = self.gate_net(carrier_freq_flat)  # Shape: [batch, 1]
        
        # THE CYBORG BLEND
        blended_freq = gate * carrier_freq_flat + (1 - gate) * symbolic_freq_flat
        
        # Convert back to spatial domain
        freq_dim = blended_freq.shape[-1] // 2
        blended_real = blended_freq.view(*blended_freq.shape[:-1], freq_dim, 2)
        blended_complex = torch.view_as_complex(blended_real)
        carrier_out = torch.fft.irfft(blended_complex, n=carrier_neural.shape[-1], dim=-1)
        
        return carrier_out, trajectory_out
    
    def get_arithmetic_state(self) -> dict:
        """Get current arithmetic state for debugging - THE ROSETTA STONE"""
        arith_state = self.arithmetic_module.get_accumulator_state()
        
        # Add gate information
        gate_param = self.gate_net[-2].weight
        gate_avg = torch.sigmoid(gate_param.mean()).item()
        
        arith_state['gate_trust_neural'] = gate_avg
        arith_state['gate_trust_symbolic'] = 1.0 - gate_avg
        
        # Add register allocation info
        arith_state['registers'] = {
            'REG_ACCUMULATOR (Freq 0)': f"Protected, stores running total",
            'REG_OPERATION (Freq 1)': f"Protected, stores +1 or -1",
            'REG_DIGIT_BUFFER (Freq 2-11)': f"Protected, 10 registers for digits",
            'REG_CARRY (Freq 12-19)': f"Protected, 8 registers for carry bits",
            'FREE (Freq 20-128)': f"Available for neural network use",
        }
        
        return arith_state


def wrap_edge_with_greybox(edge_network, vocab_size: int = 16) -> GreyboxEdgeNeuralNet:
    """
    Wrap an existing edge network with greybox arithmetic module.
    
    Usage:
        edge = EdgeNeuralNet(...)
        greybox_edge = wrap_edge_with_greybox(edge, vocab_size=16)
        
        # Set tokens before forward pass
        greybox_edge.set_tokens(src_token=3, tgt_token=13)  # '3' -> ' '
        carrier_out, traj_out = greybox_edge(carrier, traj, src_pos, tgt_pos)
        
        # Check arithmetic state
        state = greybox_edge.get_arithmetic_state()
        print(f"Accumulator: {state['accumulator']}")
    """
    return GreyboxEdgeNeuralNet(edge_network, vocab_size)
