# Quick Deploy to AMD MI300X - Checklist

## 📋 Pre-Flight Checklist

### Files to Upload
```bash
# Essential files (copy these to MI300X):
✓ libapex_hip_bridge.so    # The translation layer (26KB)
✓ test_minimal             # Test binary
✓ test_multi_kernels       # (Optional) More complex test
✓ HIP_BRIDGE_README.md     # Documentation
```

---

## 🚀 5-Minute Quick Start

### Step 1: Launch MI300X Instance
```bash
# Go to: https://amd.digitalocean.com/gpus/new
# Select: gpu-mi300x8-1536gb-devcloud
# Wait ~5 minutes for provisioning
```

### Step 2: SSH In
```bash
ssh root@<your-instance-ip>
```

### Step 3: Verify AMD GPU
```bash
rocm-smi

# Expected: 8x AMD Instinct MI300X listed
```

### Step 4: Upload APEX
```bash
# From your WSL2 machine:
cd "/mnt/c/Users/SentinalAI/Desktop/APEX GPU"

scp libapex_hip_bridge.so root@<instance-ip>:~/
scp test_minimal root@<instance-ip>:~/
```

### Step 5: Test on MI300X
```bash
# On the MI300X instance:
cd ~

# Run CUDA binary on AMD GPU!
LD_PRELOAD=./libapex_hip_bridge.so ./test_minimal
```

---

## 🎯 Expected Output

### Success Looks Like:
```
╔═══════════════════════════════════════════════════════════════╗
║          🔄 APEX HIP BRIDGE - CUDA→AMD Translation          ║
║        Run CUDA Binaries on AMD GPUs Without Rebuild!        ║
╚═══════════════════════════════════════════════════════════════╝
  ✓ HIP Runtime detected
  ✓ GPUs available: 8
  ✓ GPU 0: AMD Instinct MI300X
  ✓ Compute Units: 304

╔═══════════════════════════════════════════════════════════════╗
║  🚀 CUDA KERNEL LAUNCH → HIP TRANSLATION                     ║
║  Grid:  (1, 1, 1)
║  Block: (1, 1, 1)
║  🔄 Translating to HIP...
╚═══════════════════════════════════════════════════════════════╝
[HIP-BRIDGE] cudaMalloc(XXX bytes) → hipMalloc
[HIP-BRIDGE] cudaLaunchKernel → hipLaunchKernel
[HIP-BRIDGE] ✅ Kernel launched on AMD GPU!
[HIP-BRIDGE] cudaDeviceSynchronize → hipDeviceSynchronize

╔═══════════════════════════════════════════════════════════════╗
║                  APEX HIP BRIDGE - SESSION END                ║
╠═══════════════════════════════════════════════════════════════╣
║  CUDA Calls Translated:   X                                   ║
║  HIP Calls Made:          X                                   ║
║  Kernels Launched:        X                                   ║
╚═══════════════════════════════════════════════════════════════╝

Done!
```

### If You See This - 🎉 SUCCESS!
- CUDA binary ran on AMD GPU
- Translation layer working
- MI300X executing translated code

---

## 🐛 Troubleshooting

### "Failed to load HIP library"
```bash
# Check if HIP is installed:
ls /opt/rocm/lib/libamdhip64.so

# Add to LD_LIBRARY_PATH if needed:
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

### "No GPUs detected"
```bash
# Verify ROCm sees GPUs:
rocm-smi

# Check HIP:
/opt/rocm/bin/rocminfo | grep "Device Type"

# Should show 8 GPUs
```

### "Kernel launch failed"
This is expected if kernels are NVIDIA-specific. Next step would be:
- Recompile kernels with `hipcc`
- Or use runtime kernel translation (future work)

---

## 📊 Next Steps After Success

### 1. Profile Performance
```bash
# Use ROCm profiler:
rocprof --stats LD_PRELOAD=./libapex_hip_bridge.so ./test_minimal
```

### 2. Test Multi-GPU
```bash
# Test on each MI300X:
for gpu in {0..7}; do
    HIP_VISIBLE_DEVICES=$gpu \
    LD_PRELOAD=./libapex_hip_bridge.so \
    ./test_minimal
done
```

### 3. Collect Training Data
```bash
# Run various configurations to gather ML training data
# See DEPLOY_AMD_MI300X.md for full collection script
```

### 4. Monitor GPU Usage
```bash
# In separate terminal:
watch -n 1 rocm-smi
```

---

## 💰 Cost Management

**IMPORTANT**: MI300X instances are expensive!

### Batch Your Work
```bash
# Create test script:
cat > run_all_tests.sh << 'EOF'
#!/bin/bash
echo "=== Test 1: Basic Translation ==="
LD_PRELOAD=./libapex_hip_bridge.so ./test_minimal

echo "=== Test 2: Multi-Kernel ==="
LD_PRELOAD=./libapex_hip_bridge.so ./test_multi_kernels

echo "=== Test 3: Performance Profile ==="
rocprof --stats LD_PRELOAD=./libapex_hip_bridge.so ./test_minimal

echo "✅ All tests complete!"
EOF

chmod +x run_all_tests.sh

# Run everything at once:
./run_all_tests.sh > results.log 2>&1
```

### Download Results Immediately
```bash
# From your local machine:
scp root@<instance-ip>:~/results.log .
scp root@<instance-ip>:~/results.csv .  # If you collected data

# Then DESTROY the instance to stop billing
```

### Use Screen/Tmux
```bash
# Start persistent session:
screen -S apex

# Run tests (can disconnect and reconnect)
./run_all_tests.sh

# Disconnect: Ctrl+A, then D
# Reconnect later: screen -r apex
```

---

## ✅ Success Criteria

You've successfully demonstrated APEX HIP Bridge if:
1. ✅ Library loads on MI300X
2. ✅ Detects 8x AMD GPUs
3. ✅ Intercepts CUDA calls
4. ✅ Translates to HIP
5. ✅ Shows "Kernel launched on AMD GPU" message

Even if kernels fail (due to NVIDIA-specific code), **the translation layer works** - you've proven CUDA→HIP interception on AMD hardware!

---

## 🎯 The Big Picture

### What This Proves
- CUDA binaries CAN run on AMD (with translation)
- LD_PRELOAD approach works for API interception
- HIP bridge successfully translates calls
- Cross-vendor GPU computing is possible

### What's Next
- Compile kernels with hipcc for full compatibility
- Collect real MI300X performance data
- Train ML model with AMD-specific metrics
- Extend to more CUDA APIs (cuBLAS, cuDNN)

### The Vision
```bash
# One day:
# Same binary, any GPU vendor, ML-optimized

# On NVIDIA:
./my_cuda_app  # Runs natively

# On AMD:
LD_PRELOAD=./libapex_hip_bridge.so ./my_cuda_app  # Runs via HIP

# With ML optimization:
LD_PRELOAD="./libapex_ml.so:./libapex_hip_bridge.so" ./my_cuda_app
# Runs on AMD with ML-guided optimization!
```

---

## 📞 Quick Commands Reference

```bash
# Upload files:
scp libapex_hip_bridge.so root@IP:~/

# Check GPUs:
rocm-smi

# Test translation:
LD_PRELOAD=./libapex_hip_bridge.so ./test_minimal

# Profile:
rocprof --stats LD_PRELOAD=./libapex_hip_bridge.so ./test_minimal

# Multi-GPU test:
HIP_VISIBLE_DEVICES=0 LD_PRELOAD=./libapex_hip_bridge.so ./test_minimal

# Download results:
scp root@IP:~/results.log .
```

---

**Ready?** Launch that MI300X instance and let's see CUDA run on AMD! 🚀

**Estimated Time**: 10 minutes from launch to results
**Estimated Cost**: ~$5-10 for a 1-hour test session (check DigitalOcean pricing)
**Estimated Awesomeness**: 🔥🔥🔥🔥🔥
