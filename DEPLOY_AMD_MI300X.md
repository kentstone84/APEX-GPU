# Deploy APEX on AMD MI300X (DigitalOcean Cloud)

## 🎯 Instance Details

**GPU**: AMD Instinct MI300X (8x GPUs!)  
**Memory**: 1536GB total (192GB per GPU)  
**Compute Units**: 2432 total (304 per GPU)  
**Architecture**: CDNA3  
**ROCm**: Pre-installed  

**Cost**: Check DigitalOcean pricing for gpu-mi300x8-1536gb-devcloud

## 🚀 Quick Start Guide

### Step 1: Launch Instance

1. Go to: https://amd.digitalocean.com/gpus/new
2. Select: `gpu-mi300x8-1536gb-devcloud`
3. Choose SSH key for access
4. Click "Create Droplet"
5. Wait for instance to provision (~5 minutes)

### Step 2: SSH into Instance

```bash
ssh root@<your-instance-ip>
```

### Step 3: Verify ROCm Installation

```bash
# Check ROCm version
rocm-smi

# Expected output:
# ======================= ROCm System Management Interface =======================
# GPU  Temp   AvgPwr  SCLK     MCLK     Fan     Perf  PwrCap  VRAM%  GPU%
# 0    45.0c  50.0W   1700Mhz  1600Mhz  0.0%    auto  750.0W    0%   0%
# 1    44.0c  50.0W   1700Mhz  1600Mhz  0.0%    auto  750.0W    0%   0%
# ...
# 7    45.0c  50.0W   1700Mhz  1600Mhz  0.0%    auto  750.0W    0%   0%

# Check HIP version
hipconfig --version

# Check number of GPUs
rocminfo | grep "Device Type"
# Should show 8 GPUs
```

### Step 4: Install Build Tools

```bash
# Update system
apt update && apt upgrade -y

# Install build essentials
apt install -y build-essential git wget curl

# Verify GCC
gcc --version

# Install CUDA toolkit (for nvcc to compile test programs)
# Note: We'll compile CUDA programs, then run them via HIP translation!
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt update
apt install -y cuda-toolkit-12-3

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Step 5: Clone/Upload APEX

**Option A: Upload from your machine**
```bash
# From your local machine
scp -r "/mnt/c/Users/SentinalAI/Desktop/APEX GPU"/* root@<instance-ip>:~/apex/
```

**Option B: Create fresh on instance**
```bash
# On the instance
mkdir -p ~/apex
cd ~/apex

# Copy the files we created (you'll need to transfer these)
# For now, let's recreate the key files:
```

Upload these files to the instance:
- `apex_hip_bridge.c`
- `build_hip_bridge.sh`
- `apex_ml_model.h`
- `apex_ml_real.c`
- All test programs (`test_*.cu`)

### Step 6: Build APEX HIP Bridge

```bash
cd ~/apex

# Build the HIP translation layer
chmod +x build_hip_bridge.sh
./build_hip_bridge.sh

# Expected output:
# ✓ ROCm/HIP detected at /opt/rocm
# Building APEX HIP Bridge...
# ✅ libapex_hip_bridge.so built successfully!
```

### Step 7: Build Test Programs (with CUDA)

```bash
# Compile test programs with CUDA toolkit
nvcc -cudart shared test_minimal.cu -o test_minimal
nvcc -cudart shared test_multi_kernels.cu -o test_multi_kernels

# These are CUDA binaries, but we'll run them on AMD!
```

### Step 8: Run CUDA Binary on AMD GPU! 🎉

```bash
# Run CUDA test on AMD MI300X
LD_PRELOAD=./libapex_hip_bridge.so ./test_minimal

# Expected output:
# ╔═══════════════════════════════════════════════════════════════╗
# ║          🔄 APEX HIP BRIDGE - CUDA→AMD Translation          ║
# ║        Run CUDA Binaries on AMD GPUs Without Rebuild!        ║
# ╚═══════════════════════════════════════════════════════════════╝
#   ✓ HIP Runtime detected
#   ✓ AMD GPUs available: 8
#   ✓ GPU 0: AMD Instinct MI300X
#   ✓ Compute Units: 304
# 
# [HIP-BRIDGE] cudaMalloc(4096 bytes) → hipMalloc
# ╔═══════════════════════════════════════════════════════════════╗
# ║  🚀 CUDA KERNEL LAUNCH → HIP TRANSLATION                     ║
# ║  Grid:  (1, 1, 1)
# ║  Block: (1, 1, 1)
# ║  🔄 Translating to HIP...
# ╚═══════════════════════════════════════════════════════════════╝
# [HIP-BRIDGE] ✅ Kernel launched on AMD GPU!
```

## 🧠 Test APEX ML Predictions on MI300X

### Build ML Model for MI300X

Update `apex_ml_model.h` for MI300X specs:

```c
// Update GPU configuration
#define MI300X_CU_COUNT 304  // Compute units per GPU
#define MI300X_MAX_THREADS_PER_CU 2048
#define MI300X_MAX_WAVES_PER_CU 32

// In predict_with_ml(), adjust for MI300X:
unsigned int num_cus = 304;  // Was 84 for RTX 5080
```

### Build and Test

```bash
# Build ML-enhanced version
gcc -shared -fPIC -o libapex_ml_amd.so apex_ml_real.c -ldl -lm

# Test ML predictions on AMD
LD_PRELOAD=./libapex_ml_amd.so ./test_ml_benchmark

# You'll see predictions for MI300X!
```

## 🎯 Collect Training Data from Real AMD GPU

This is **gold** - real MI300X performance data!

```bash
# Create profiling script
cat > collect_mi300x_data.sh << 'SCRIPT'
#!/bin/bash

echo "grid_x,grid_y,grid_z,block_x,block_y,block_z,shared_mem,occupancy,time_ms" > mi300x_training_data.csv

for blocks in 32 64 128 256 512 1024 2048 4096; do
    for threads in 32 64 128 256 512 1024; do
        echo "Profiling: $blocks blocks × $threads threads"
        
        # Run with ROCm profiler
        rocprof --stats ./benchmark_config $blocks $threads 2>&1 | \
            extract_metrics.py >> mi300x_training_data.csv
    done
done

echo "✅ Collected training data from MI300X!"
SCRIPT

chmod +x collect_mi300x_data.sh
./collect_mi300x_data.sh

# This data is PURE GOLD for training your ML model!
```

## 🔥 Advanced: Combine HIP Bridge + ML

```bash
# Build both libraries
gcc -shared -fPIC -o libapex_hip_bridge.so apex_hip_bridge.c -lamdhip64 -ldl
gcc -shared -fPIC -o libapex_ml_amd.so apex_ml_real.c -ldl -lm

# Run CUDA binary with BOTH translation AND ML prediction
LD_PRELOAD="./libapex_ml_amd.so:./libapex_hip_bridge.so" ./test_multi_kernels

# Now you get:
# 1. CUDA→HIP translation (runs on AMD)
# 2. ML performance predictions for MI300X
# 3. Optimization recommendations
```

## 📊 Performance Comparison: RTX 5080 vs MI300X

### Predicted Performance Differences

| Metric | RTX 5080 | MI300X | Ratio |
|--------|----------|---------|-------|
| Compute Units/SMs | 84 | 304 | 3.6x |
| Memory Bandwidth | 960 GB/s | 5.3 TB/s | 5.5x |
| Memory Capacity | 16GB | 192GB | 12x |
| FP32 TFLOPS | ~90 | ~163 | 1.8x |
| AI/ML TFLOPS | ~1400 | ~1300+ | ~1x |

### What This Means for APEX

**Occupancy predictions will be VERY different:**
- MI300X can handle 304 blocks vs RTX's 84
- Optimal block sizes may vary
- Memory patterns completely different

**You'll need to retrain the ML model with MI300X data!**

## 🧪 Test Matrix for MI300X

### Test 1: Basic CUDA Compatibility
```bash
LD_PRELOAD=./libapex_hip_bridge.so ./test_minimal
# Expected: ✅ Works (kernel runs on AMD)
```

### Test 2: Multi-Kernel Workload
```bash
LD_PRELOAD=./libapex_hip_bridge.so ./test_multi_kernels
# Expected: ✅ All kernels run, HIP translations logged
```

### Test 3: ML Predictions
```bash
LD_PRELOAD=./libapex_ml_amd.so ./test_ml_benchmark
# Expected: Predictions for 304 CUs (vs 84 SMs)
```

### Test 4: Combined Translation + ML
```bash
LD_PRELOAD="./libapex_ml_amd.so:./libapex_hip_bridge.so" ./test_multi_kernels
# Expected: CUDA runs on AMD with ML analysis
```

### Test 5: Multi-GPU (8x MI300X!)
```bash
# Set different GPU for each test
for gpu in 0 1 2 3 4 5 6 7; do
    HIP_VISIBLE_DEVICES=$gpu \
    LD_PRELOAD=./libapex_hip_bridge.so ./test_multi_kernels
done
```

## 💰 Cost Optimization

DigitalOcean charges by the hour for GPU instances. Optimize costs:

### Batch Your Testing
```bash
# Create a test suite that runs everything at once
cat > run_all_tests.sh << 'SCRIPT'
#!/bin/bash

echo "Starting APEX test suite on MI300X..."

# Test 1: HIP Bridge
echo "=== Test 1: HIP Translation ==="
LD_PRELOAD=./libapex_hip_bridge.so ./test_minimal

# Test 2: ML Predictions
echo "=== Test 2: ML Predictions ==="
LD_PRELOAD=./libapex_ml_amd.so ./test_ml_benchmark

# Test 3: Combined
echo "=== Test 3: Combined ==="
LD_PRELOAD="./libapex_ml_amd.so:./libapex_hip_bridge.so" ./test_multi_kernels

# Test 4: Data collection
echo "=== Test 4: Training Data Collection ==="
./collect_mi300x_data.sh

echo "✅ All tests complete!"
SCRIPT

chmod +x run_all_tests.sh

# Run everything in one session
./run_all_tests.sh > mi300x_results.log 2>&1

# Download results
# (From your local machine)
scp root@<instance-ip>:~/apex/mi300x_results.log .
scp root@<instance-ip>:~/apex/mi300x_training_data.csv .
```

### Snapshot Your Work
```bash
# Before destroying instance, save your setup
tar -czf apex_mi300x_results.tar.gz ~/apex/
scp root@<instance-ip>:~/apex/apex_mi300x_results.tar.gz .

# Destroy instance (stop paying)
# Next time: restore from snapshot!
```

## 🎯 What You'll Learn

### From This Experiment

1. **CUDA→HIP translation works in practice**
   - Which CUDA APIs translate cleanly
   - Which need special handling
   - Performance overhead

2. **MI300X characteristics**
   - Optimal block/grid sizes for CDNA3
   - Memory bandwidth utilization
   - Wave occupancy patterns

3. **ML model portability**
   - How predictions differ NVIDIA vs AMD
   - Architecture-specific tuning needed
   - Transfer learning possibilities

4. **Real-world training data**
   - Ground truth from actual hardware
   - Better than synthetic data
   - Can train production-quality model

## 📈 Expected Results

### What Should Work
✅ Memory allocation (cudaMalloc → hipMalloc)  
✅ Memory copies (cudaMemcpy → hipMemcpy)  
✅ Kernel launches (<<<>>> → hipLaunchKernel)  
✅ Synchronization (cudaDeviceSynchronize)  
✅ Device queries (cudaGetDeviceCount)  

### What Might Need Work
⚠️ Shared memory (may need tuning)  
⚠️ Texture memory (not implemented yet)  
⚠️ Complex memory patterns  

### What Won't Work (Yet)
❌ CUDA libraries (cuBLAS, cuDNN) - need separate translation  
❌ Pre-compiled CUDA kernels - need HIP compilation  

## 🚀 Next Steps After Testing

### If It Works Well
1. Publish results (blog post, paper)
2. Open source the project
3. Train production ML model with MI300X data
4. Extend to more CUDA APIs

### If You Find Issues
1. Document what doesn't work
2. File issues/TODOs
3. Implement missing translations
4. Improve error handling

## 💡 Pro Tips

### Tip 1: Use Screen/Tmux
```bash
# Start screen session
screen -S apex

# Run tests (can disconnect)
./run_all_tests.sh

# Disconnect: Ctrl+A, then D
# Reconnect: screen -r apex
```

### Tip 2: Monitor GPU Usage
```bash
# In separate terminal
watch -n 1 rocm-smi

# See real-time GPU utilization during tests
```

### Tip 3: Profile with ROCm Tools
```bash
# Detailed profiling
rocprof --stats \
  LD_PRELOAD=./libapex_hip_bridge.so \
  ./test_multi_kernels

# Generates detailed metrics
```

### Tip 4: Multi-GPU Testing
```bash
# Test on all 8 GPUs simultaneously
for gpu in {0..7}; do
    HIP_VISIBLE_DEVICES=$gpu \
    LD_PRELOAD=./libapex_hip_bridge.so \
    ./test_multi_kernels &
done
wait

# 8x parallelism for data collection!
```

## 📊 Deliverables

After running on MI300X, you'll have:

1. **Validation Results**
   - Proof that CUDA→HIP translation works
   - Performance metrics
   - Compatibility matrix

2. **Training Data**
   - Real MI300X occupancy/timing data
   - Can retrain ML model for AMD
   - Much better than synthetic data

3. **Performance Comparison**
   - NVIDIA (RTX 5080) vs AMD (MI300X)
   - Architecture-specific insights
   - Optimization recommendations

4. **Production-Ready Code**
   - Tested HIP bridge
   - AMD-tuned ML model
   - Deployment scripts

## 🎉 The Grand Vision

Imagine:
```bash
# Same CUDA binary
./my_ml_training_app

# On NVIDIA:
./my_ml_training_app  # Uses CUDA natively

# On AMD:
LD_PRELOAD=./libapex_hip_bridge.so ./my_ml_training_app
# Runs on AMD via translation!

# With ML optimization:
LD_PRELOAD="./libapex_ml.so:./libapex_hip_bridge.so" ./my_ml_training_app
# Runs on AMD with ML-guided optimization!
```

**One binary. Multiple vendors. AI-optimized. 🚀**

---

**Budget Note**: MI300X instances are expensive. Plan your tests, run them efficiently, and destroy the instance when done. The knowledge gained is worth it! 💰

**Ready to deploy?** Follow this guide step-by-step and you'll have APEX running on AMD's flagship AI GPU! 🔥
