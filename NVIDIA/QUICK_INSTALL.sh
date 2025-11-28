
# Quick Install - Patched CUDA Driver (SM_120 Unlocked)
# Safest method: LD_PRELOAD (no system modification)

set -e

echo "🚀 Patched CUDA Driver - Quick Install"
echo "======================================="
echo ""

# Check if patched file exists
if [ ! -f "libcuda.so.1.1.patched" ]; then
    echo "❌ ERROR: libcuda.so.1.1.patched not found!"
    echo ""
    echo "You need the patched driver file from the APEX project."
    echo "Expected checksum: 131a52575de151f4d67fd50109af7ea94621778a7cb659c831a2eb9c465ee5f9"
    exit 1
fi

echo "✅ Found patched driver file"
echo ""

# Verify checksum
EXPECTED="131a52575de151f4d67fd50109af7ea94621778a7cb659c831a2eb9c465ee5f9"
ACTUAL=$(sha256sum libcuda.so.1.1.patched | awk '{print $1}')

if [ "$ACTUAL" != "$EXPECTED" ]; then
    echo "⚠️  WARNING: Checksum mismatch!"
    echo "Expected: $EXPECTED"
    echo "Got:      $ACTUAL"
    read -p "Continue anyway? (y/N): " CONTINUE
    if [ "$CONTINUE" != "y" ]; then
        exit 1
    fi
fi

# Install using LD_PRELOAD method (safest)
echo "📦 Installing using LD_PRELOAD method (safest)..."
echo ""

sudo mkdir -p /opt/cuda-patched
sudo cp libcuda.so.1.1.patched /opt/cuda-patched/libcuda.so.1.1
sudo chmod 755 /opt/cuda-patched/libcuda.so.1.1

echo "✅ Patched driver installed to /opt/cuda-patched/"
echo ""

# Add to bashrc
if ! grep -q "LD_PRELOAD=/opt/cuda-patched/libcuda.so.1.1" ~/.bashrc 2>/dev/null; then
    echo 'export LD_PRELOAD=/opt/cuda-patched/libcuda.so.1.1' >> ~/.bashrc
    echo "✅ Added to ~/.bashrc"
else
    echo "ℹ️  Already in ~/.bashrc"
fi

echo ""
echo "============================================"
echo "✅ INSTALLATION COMPLETE!"
echo "============================================"
echo ""
echo "To use patched driver:"
echo "  1. Open new terminal (or run: source ~/.bashrc)"
echo "  2. Run any CUDA application"
echo "  3. Test: nvidia-smi"
echo ""
echo "To verify patch is active:"
echo "  echo \$LD_PRELOAD"
echo "  # Should show: /opt/cuda-patched/libcuda.so.1.1"
echo ""
echo "To disable patch temporarily:"
echo "  unset LD_PRELOAD"
echo ""
echo "To remove permanently:"
echo "  sed -i '/LD_PRELOAD.*libcuda/d' ~/.bashrc"
echo ""
echo "SM_120 is now unlocked! 🎉"
echo ""
QUICK

chmod +x QUICK_INSTALL.sh
echo "Created QUICK_INSTALL.sh"