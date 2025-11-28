# APEX ML - Quick Start Guide

## 🚀 5-Minute Setup

### 1. Build Everything
```bash
./build_apex.sh
```

### 2. Test the Neural Network
```bash
LD_PRELOAD=./libapex_ml_real.so ./test_ml_benchmark
```

You should see predictions like:
```
GPU Occupancy:    65.6% (NN output: 0.6559)
Est. Time:        0.035 ms
SM Utilization:   55.8% (84 / 84 SMs active)
```

### 3. Use with Your Code

#### Option A: Compile your CUDA program correctly
```bash
# ⚠️ IMPORTANT: Use -cudart shared
nvcc -cudart shared myprogram.cu -o myprogram

# Run with APEX
LD_PRELOAD=./libapex_ml_real.so ./myprogram
```

#### Option B: Use with Python/PyTorch
```bash
LD_PRELOAD=./libapex_ml_real.so python your_script.py
```

## 📊 Understanding the Output

### Good Configuration Example
```
╔═══════════════════════════════════════════════════════════════╗
║  🧠 APEX NEURAL NETWORK - Kernel #4                         ║
╠═══════════════════════════════════════════════════════════════╣
║  📊 KERNEL CONFIGURATION                                      ║
║    Grid:             (512, 1, 1) = 512 blocks
║    Block:            (256, 1, 1) = 256 threads/block
║    Total Threads:    131072
║                                                               ║
║  🤖 NEURAL NETWORK PREDICTIONS                                ║
║    GPU Occupancy:    65.6% ✅ GOOD
║    Block Efficiency: 60.2%
║                                                               ║
║  💡 ML OPTIMIZATION RECOMMENDATION                            ║
║    ✓ Configuration looks optimal
╚═══════════════════════════════════════════════════════════════╝
```

### Bad Configuration Example
```
╔═══════════════════════════════════════════════════════════════╗
║  🧠 APEX NEURAL NETWORK - Kernel #1                         ║
╠═══════════════════════════════════════════════════════════════╣
║  📊 KERNEL CONFIGURATION                                      ║
║    Grid:             (64, 1, 1) = 64 blocks
║    Block:            (16, 1, 1) = 16 threads/block
║    Total Threads:    1024
║                                                               ║
║  🤖 NEURAL NETWORK PREDICTIONS                                ║
║    GPU Occupancy:    26.4% ⚠️ LOW
║    Block Efficiency: 20.6%
║                                                               ║
║  💡 ML OPTIMIZATION RECOMMENDATION                            ║
║    ⚠️  CRITICAL: Block size too small (16 threads)
║    → Increase to 128-256 threads for 8x better occupancy
╚═══════════════════════════════════════════════════════════════╝
```

## 🎯 Optimization Rules of Thumb

From the Neural Network's learned patterns:

| Block Size | Expected Occupancy | Action |
|-----------|-------------------|---------|
| < 64 threads | ~26% (LOW) | ⚠️ Increase to 128-256 |
| 64-128 threads | ~40-65% (MEDIUM) | Consider 256 for best results |
| 128-512 threads | ~65-66% (HIGH) | ✅ Optimal range |
| 512-768 threads | ~65% (GOOD) | ✅ Still good |
| > 768 threads | ~47% (REDUCED) | ⚠️ Consider reducing |

### Grid Size Recommendations
- **Minimum**: At least 84 blocks (one per SM)
- **Recommended**: 168+ blocks (2x SMs for better utilization)
- **Optimal**: 1344+ blocks (16x SMs for maximum parallelism)

## 🔧 Troubleshooting

### Problem: "0 kernels detected"
**Solution**: Compile with `-cudart shared`
```bash
# Wrong (static linking)
nvcc myprogram.cu -o myprogram

# Correct (shared linking)
nvcc -cudart shared myprogram.cu -o myprogram
```

Verify with:
```bash
ldd ./myprogram | grep cudart
# Should show: libcudart.so.12 => /path/to/libcudart.so.12
```

### Problem: Predictions seem wrong
The current weights are heuristically initialized. For production:
1. Collect real profiling data from your GPU
2. Train the neural network weights
3. Update `apex_ml_model.h` with trained weights

### Problem: Too much output
Filter the output:
```bash
LD_PRELOAD=./libapex_ml_real.so ./myprogram 2>&1 | grep "Occupancy"
```

## 📁 Key Files

```
libapex_ml_real.so       ← Main library (use this!)
apex_ml_model.h          ← Neural network implementation
apex_ml_real.c           ← APEX integration code
test_ml_benchmark        ← Validation test
build_apex.sh            ← Build everything
```

## 🎓 Advanced Usage

### Save predictions to file
```bash
LD_PRELOAD=./libapex_ml_real.so ./myprogram 2> predictions.log
grep "Occupancy:" predictions.log
```

### Compare before/after optimization
```bash
# Before optimization
LD_PRELOAD=./libapex_ml_real.so ./myprogram_old 2> before.log

# After optimization (increased block size)
LD_PRELOAD=./libapex_ml_real.so ./myprogram_new 2> after.log

# Compare
grep "GPU Occupancy" before.log after.log
```

### Batch testing multiple configurations
```bash
for blocks in 64 128 256 512; do
    for threads in 32 64 128 256 512; do
        echo "Testing: $blocks blocks × $threads threads"
        LD_PRELOAD=./libapex_ml_real.so ./test_config $blocks $threads 2>&1 | \
            grep "Occupancy:"
    done
done
```

## 💡 Example: Optimizing a Real Kernel

### Step 1: Run with APEX
```bash
LD_PRELOAD=./libapex_ml_real.so python train.py 2> apex.log
```

### Step 2: Check predictions
```bash
grep -A 5 "CRITICAL\|WARNING" apex.log
```

Example output:
```
⚠️  CRITICAL: Block size too small (32 threads)
→ Increase to 128-256 threads for 4x better occupancy
```

### Step 3: Modify your code
```python
# Before (32 threads/block)
kernel<<<num_blocks, 32>>>(data)

# After (256 threads/block)
kernel<<<num_blocks, 256>>>(data)
```

### Step 4: Verify improvement
```bash
LD_PRELOAD=./libapex_ml_real.so python train.py 2> apex_improved.log
grep "Occupancy:" apex_improved.log
```

Should see: `GPU Occupancy:    65.6%` (vs ~26% before)

## 🎯 What's Next?

1. **Test with your workloads** - Run APEX on your actual CUDA applications
2. **Collect optimization data** - Use the recommendations to improve performance
3. **Train better weights** - Use real GPU profiling data to improve predictions
4. **Integrate 1.8M model** - Replace the small NN with your full model

---

**Questions?** Check `APEX_ML_SUMMARY.md` for technical details.
