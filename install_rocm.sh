#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              ROCm Installation Script                         ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if running on WSL
if grep -qi microsoft /proc/version; then
    echo "⚠️  WARNING: Running on WSL2"
    echo "   ROCm GPU drivers won't work without AMD GPU hardware"
    echo "   But we can install HIP libraries for building APEX HIP Bridge"
    echo ""
fi

# Install the AMD GPU installer
echo "Step 1: Installing AMD GPU package manager..."
sudo dpkg -i amdgpu-install_6.2.60204-1_all.deb

if [ $? -ne 0 ]; then
    echo "❌ dpkg failed, trying to fix dependencies..."
    sudo apt-get install -f -y
    sudo dpkg -i amdgpu-install_6.2.60204-1_all.deb
fi

echo ""
echo "Step 2: Updating package lists..."
sudo apt-get update

echo ""
echo "Step 3: Installing ROCm (HIP runtime only - no drivers)..."
echo "   This will take 5-10 minutes..."
echo ""

# For WSL/development, install just HIP without kernel drivers
sudo amdgpu-install --usecase=hip,rocm --no-dkms -y

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ ROCm/HIP libraries installed!"
    echo ""
    echo "Adding ROCm to PATH..."
    
    # Add to bashrc if not already there
    if ! grep -q "/opt/rocm/bin" ~/.bashrc; then
        echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    fi
    
    # Add for current session
    export PATH=/opt/rocm/bin:$PATH
    export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
    
    echo ""
    echo "Verifying installation..."
    
    if [ -f /opt/rocm/bin/hipconfig ]; then
        echo "✓ HIP installed:"
        /opt/rocm/bin/hipconfig --version
    else
        echo "⚠️  hipconfig not found"
    fi
    
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                   Installation Complete!                      ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║                                                               ║"
    echo "║  ✓ ROCm/HIP libraries installed                               ║"
    echo "║  ✓ Can now build APEX HIP Bridge                              ║"
    echo "║                                                               ║"
    echo "║  Note: To use AMD GPU, you need AMD hardware                  ║"
    echo "║  (like the MI300X instance on DigitalOcean)                   ║"
    echo "║                                                               ║"
    echo "║  Next steps:                                                  ║"
    echo "║    1. source ~/.bashrc                                        ║"
    echo "║    2. ./build_hip_bridge.sh                                   ║"
    echo "║                                                               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
else
    echo ""
    echo "❌ ROCm installation failed!"
    echo ""
    echo "This might be because:"
    echo "  1. WSL2 compatibility issues"
    echo "  2. Missing dependencies"
    echo "  3. Network/repository issues"
    echo ""
    echo "You can still build the HIP bridge, but testing requires"
    echo "actual AMD GPU hardware (like the MI300X instance)."
    echo ""
fi
