{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

let
  # Python with tkinter support
  python = pkgs.python311.withPackages (ps: with ps; [ tkinter ]);
  
  # Detect if NVIDIA GPU is available
  hasNvidiaGpu = builtins.pathExists "/dev/nvidia0";
  
  # For NixOS CUDA support, we need to use the system's NVIDIA drivers
  # This is because Nix can't package proprietary kernel modules
  # We'll set up the environment to find system drivers
  
  cudaPackages = if hasNvidiaGpu then [
    pkgs.cudaPackages.cudatoolkit
    pkgs.cudaPackages.cudnn
  ] else [];
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    python
    python.pkgs.pip
    python.pkgs.virtualenv
    tk  # Tkinter UI support
    nodejs_20
    stdenv.cc.cc.lib  # Provides libstdc++.so.6
    zlib  # Provides libz.so.1
  ] ++ cudaPackages;

  shellHook = ''
    echo "3D Newton Fractal - Development Environment"
    echo "==========================================="
    echo ""
    
    # Set LD_LIBRARY_PATH for core libraries
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:''${LD_LIBRARY_PATH:-}"
    
    # Add CUDA toolkit if available
    ${if hasNvidiaGpu then ''
      export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
      export LD_LIBRARY_PATH="$CUDA_PATH/lib:$CUDA_PATH/lib64:''${LD_LIBRARY_PATH:-}"
      
      # Try to find system NVIDIA libraries (for NixOS)
      # Common locations: /run/opengl-driver/lib, /usr/lib
      for nvidia_path in /run/opengl-driver/lib /run/opengl-driver-32/lib /usr/lib/x86_64-linux-gnu /usr/lib; do
        if [ -d "$nvidia_path" ] && [ -f "$nvidia_path/libnvidia-ml.so.1" ]; then
          export LD_LIBRARY_PATH="$nvidia_path:''${LD_LIBRARY_PATH:-}"
          echo "Found NVIDIA system libraries at: $nvidia_path"
          break
        fi
      done
    '' else ""}
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
      echo "NVIDIA GPU detected:"
      nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
      echo ""
      
      ${if hasNvidiaGpu then ''
        echo "CUDA Toolkit: ${pkgs.cudaPackages.cudatoolkit.version}"
        echo "CUDA_PATH: $CUDA_PATH"
      '' else ""}
    else
      echo "No NVIDIA GPU detected - will use CPU"
    fi
    echo ""
    
    # Setup Python virtual environment
    if [ ! -d "backend/venv" ]; then
      echo "Creating Python virtual environment..."
      cd backend
      python -m venv venv
      source venv/bin/activate
      pip install --upgrade pip
      
      echo "Installing dependencies..."
      ${if hasNvidiaGpu then ''
        echo "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        
        echo "Installing JAX with CUDA 12 support..."
        pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
      '' else ''
        echo "Installing PyTorch (CPU-only)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        
        echo "Installing JAX (CPU-only)..."
        pip install --upgrade "jax[cpu]"
      ''}
      
      pip install -r requirements.txt
      cd ..
    else
      echo "Activating existing virtual environment..."
      source backend/venv/bin/activate
    fi
    
    # Add PyTorch CUDA libraries to LD_LIBRARY_PATH if using CUDA version
    if python -c "import torch; exit(0 if '+cu' in torch.__version__ else 1)" 2>/dev/null; then
      TORCH_CUDA_PATH=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
      if [ -d "$TORCH_CUDA_PATH" ]; then
        export LD_LIBRARY_PATH="$TORCH_CUDA_PATH:''${LD_LIBRARY_PATH:-}"
      fi
    fi
    
    # Verify PyTorch installation
    python -c "import torch; print(f'PyTorch {torch.__version__} installed')" 2>/dev/null || echo "PyTorch not yet installed"
    
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
      python -c "import torch; print(f'✅ PyTorch CUDA available: {torch.cuda.get_device_name(0)}')"
    else
      python -c "import torch; import sys; print(f'⚠️  PyTorch CUDA not available (will use CPU)', file=sys.stderr)" 2>&1 || true
    fi
    
    # Verify JAX installation
    if python -c "import jax" 2>/dev/null; then
      python -c "import jax; print(f'JAX {jax.__version__} installed')"
      
      if python -c "import jax; devs = jax.devices('gpu'); exit(0 if devs else 1)" 2>/dev/null; then
        python -c "import jax; print(f'✅ JAX CUDA available: {jax.devices(\"gpu\")[0]}')"
      else
        echo "⚠️  JAX CUDA not available (will use CPU)"
      fi
    fi
    
    echo ""
    echo "Environment ready!"
    echo ""
    echo "To start the GUI:"
    echo "  cd src && python gui.py"
    echo ""
    echo "To start the web server:"
    echo "  cd backend && python main.py"
    echo ""
  '';
}
