# APEX GPU - Complete File Index

## 🎯 Start Here
- **QUICKSTART.md** - Get running in 5 minutes
- **ACHIEVEMENTS.md** - What we built and why it's awesome
- **APEX_ML_SUMMARY.md** - Technical deep dive

## 🧠 Core ML System (Production)
- **apex_ml_model.h** - Neural network implementation (3-layer FFN, 400 params)
- **apex_ml_real.c** - APEX with real NN integration
- **libapex_ml_real.so** - ⭐ **USE THIS** - Production ML library

## 📚 Alternative Libraries
- **libapex_advanced.so** - Advanced metrics without ML
- **libapex_runtime.so** - Basic Runtime + Driver API interception
- **libapex_kernel.so** - Driver API only (minimal)

## 🔧 Build & Deploy
- **build_apex.sh** - Build all libraries and tests
- **demo_apex_ml.sh** - Interactive demonstration

## 🧪 Tests & Validation
- **test_ml_benchmark.cu** - Tests 8 configurations (validates NN)
- **test_multi_kernels.cu** - Multiple kernel launch patterns
- **test_minimal.cu** - Simple single kernel test
- **test_driver_simple.cu** - Driver API validation

## 📖 Documentation
- **INDEX.md** - This file
- **QUICKSTART.md** - 5-minute setup guide
- **APEX_ML_SUMMARY.md** - Technical architecture
- **ACHIEVEMENTS.md** - What we accomplished
- **APEX_README.md** - (if created) Comprehensive README

## 🗃️ Supporting Files (Reference)
- **apex_kernel.c** - Basic Driver API version
- **apex_runtime.c** - Runtime API version
- **apex_advanced.c** - Advanced metrics version
- **apex_ml_hooks.c** - ML version with placeholders (superseded by apex_ml_real.c)

## 📊 Usage Examples

### Quick Test
```bash
./build_apex.sh
LD_PRELOAD=./libapex_ml_real.so ./test_ml_benchmark
```

### With Your Code
```bash
nvcc -cudart shared myprogram.cu -o myprogram
LD_PRELOAD=./libapex_ml_real.so ./myprogram
```

### With PyTorch
```bash
LD_PRELOAD=./libapex_ml_real.so python train.py
```

## 🎯 Recommended Reading Order

1. **QUICKSTART.md** - Get it running
2. **test_ml_benchmark** - See it in action
3. **ACHIEVEMENTS.md** - Understand what it does
4. **APEX_ML_SUMMARY.md** - Learn how it works
5. **apex_ml_model.h** - Study the neural network
6. **apex_ml_real.c** - Understand the integration

## 🚀 Next Steps

1. Run the benchmark: `LD_PRELOAD=./libapex_ml_real.so ./test_ml_benchmark`
2. Test with your code
3. Train better NN weights with real profiling data
4. Integrate your 1.8M parameter model
5. Deploy in production

---

**Quick Command Reference:**
```bash
# Build everything
./build_apex.sh

# Run validation
LD_PRELOAD=./libapex_ml_real.so ./test_ml_benchmark

# Run demo
./demo_apex_ml.sh

# Check files
ls -lh libapex_ml_real.so
```

---

*All files in this directory are part of the APEX GPU ML Scheduler system.*
