# APEX GPU - Quick Reference Card

## 🚀 **5-Minute AMD MI300X Deployment**

### **1. Upload to AMD**
```bash
scp -r "APEX GPU" user@mi300x:~/
```

### **2. Setup AMD Environment**
```bash
ssh user@mi300x
cd APEX\ GPU
sudo ./setup_amd_mi300x.sh
```

### **3. Verify Setup**
```bash
source apex_env.sh
./test_hello
```

### **4. Run Tests**
```bash
./run_all_tests.sh
```

### **5. Run Your App**
```bash
./your_cuda_app
```

---

## 📝 **Essential Commands**

### **Environment Setup**
```bash
source apex_env.sh  # Load APEX environment
```

### **Quick Tests**
```bash
./verify_apex.sh         # Quick verification
./test_hello             # Minimal CUDA test
./run_all_tests.sh       # Full test suite (5 tests)
```

### **With Profiling**
```bash
APEX_PROFILE=1 APEX_DEBUG=1 ./your_app
```

### **Check GPU**
```bash
rocm-smi                      # GPU status
rocm-smi --showproductname    # GPU model
rocm-smi --showmeminfo        # Memory usage
```

---

## 🔧 **Troubleshooting**

### **Issue: "No GPU detected"**
```bash
rocm-smi  # Check if GPU visible
groups    # Should show: video render
```

### **Issue: "Library not found"**
```bash
ldd libapex_hip_bridge.so  # Check dependencies
source apex_env.sh         # Reload environment
```

### **Issue: "Slow performance"**
```bash
unset APEX_DEBUG APEX_TRACE  # Disable logging
export APEX_STATS=1          # Minimal overhead
```

---

## 📊 **Environment Variables**

| Variable | Values | Purpose |
|----------|--------|---------|
| `APEX_DEBUG` | 0/1 | Debug logging |
| `APEX_PROFILE` | 0/1 | Performance profiling |
| `APEX_TRACE` | 0/1 | Detailed trace |
| `APEX_STATS` | 0/1 | Statistics (default: ON) |
| `APEX_LOG_FILE` | path | Log to file |

---

## 🎯 **Usage Patterns**

### **Basic CUDA App**
```bash
LD_PRELOAD=./libapex_hip_bridge.so ./cuda_app
```

### **cuBLAS App**
```bash
LD_PRELOAD="./libapex_cublas_bridge.so:./libapex_hip_bridge.so" ./app
```

### **PyTorch/TensorFlow**
```bash
export LD_PRELOAD="./libapex_cudnn_bridge.so:./libapex_cublas_bridge.so:./libapex_hip_bridge.so"
python train.py
```

---

## 📁 **Key Files**

### **Bridges**
- `libapex_hip_bridge.so` - CUDA Runtime → HIP
- `libapex_cublas_bridge.so` - cuBLAS → rocBLAS
- `libapex_cudnn_bridge.so` - cuDNN → MIOpen

### **Tests**
- `test_hello` - Minimal validation
- `run_all_tests.sh` - Comprehensive suite
- `verify_apex.sh` - Quick verification

### **Setup**
- `setup_amd_mi300x.sh` - AMD environment setup
- `apex_env.sh` - Environment loader (auto-created)
- `install_rocm.sh` - ROCm installer

### **Documentation**
- `README.md` - Project overview
- `COMPLETE_DEPLOYMENT_GUIDE.md` - Full guide
- `AMD_DEPLOYMENT_CHECKLIST.md` - Deployment steps

---

## ✅ **Success Indicators**

### **APEX is working if:**
- ✅ `./test_hello` shows "Result: 42"
- ✅ `./run_all_tests.sh` shows "100% pass rate"
- ✅ `rocm-smi` shows GPU active during tests
- ✅ No memory leaks in logs
- ✅ Your app produces correct output

---

## 🆘 **Get Help**

1. Check logs: `cat build/*_apex.log`
2. Review checklist: `AMD_DEPLOYMENT_CHECKLIST.md`
3. Read full guide: `COMPLETE_DEPLOYMENT_GUIDE.md`
4. Check status: `cat amd_setup_status.txt`

---

## 🎓 **Remember**

- **Always** `source apex_env.sh` before running
- **Test first** with `./test_hello`
- **Profile** with `APEX_PROFILE=1` to debug
- **Disable** debug logging in production
- **Check** `rocm-smi` to verify GPU usage

---

**APEX GPU - Making CUDA→AMD seamless** 🚀
