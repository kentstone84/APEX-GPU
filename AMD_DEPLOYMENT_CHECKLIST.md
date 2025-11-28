# AMD MI300X Deployment Checklist

## 📋 Pre-Deployment Verification

### ✅ **Phase 1: Local Testing (Before AMD)**

Run these on your current system:

- [ ] All bridges compiled successfully
  ```bash
  ls -lh libapex_*.so
  # Should show: libapex_hip_bridge.so, libapex_cublas_bridge.so, libapex_cudnn_bridge.so
  ```

- [ ] Quick verification passed
  ```bash
  ./verify_apex.sh
  # Should show: 11/12 tests passed (1 expected failure)
  ```

- [ ] Full test suite passed
  ```bash
  ./run_all_tests.sh
  # Should show: 100% pass rate (5/5 tests)
  ```

- [ ] All documentation present
  ```bash
  ls -1 *.md
  # Should show: 7+ markdown files
  ```

- [ ] Test logs clean (no critical errors)
  ```bash
  grep -r "FAILED\|CRITICAL" build/*.log || echo "OK"
  ```

---

### ✅ **Phase 2: AMD Instance Provisioning**

- [ ] AMD MI300X instance provisioned
  - Provider: DigitalOcean / Vultr / AWS / Azure / GCP
  - GPU: AMD MI300X (or MI250X, MI210)
  - OS: Ubuntu 22.04 or 24.04 LTS
  - RAM: 32GB+ recommended
  - Storage: 100GB+ recommended

- [ ] SSH access configured
  ```bash
  ssh user@your-amd-instance
  ```

- [ ] System updated
  ```bash
  sudo apt update && sudo apt upgrade -y
  ```

---

### ✅ **Phase 3: Upload APEX to AMD Instance**

- [ ] Files uploaded
  ```bash
  scp -r "APEX GPU" user@instance:~/
  ```

- [ ] Verify upload
  ```bash
  ssh user@instance "ls -lh ~/APEX\ GPU/*.so"
  ```

- [ ] Permissions set
  ```bash
  ssh user@instance "chmod +x ~/APEX\ GPU/*.sh"
  ```

---

### ✅ **Phase 4: AMD Software Installation**

- [ ] ROCm installed
  ```bash
  cd ~/APEX\ GPU
  sudo ./install_rocm.sh
  # Or: sudo ./setup_amd_mi300x.sh
  ```

- [ ] ROCm verified
  ```bash
  rocm-smi
  # Should show GPU info
  ```

- [ ] HIP runtime present
  ```bash
  ls -lh /opt/rocm/lib/libamdhip64.so
  ```

- [ ] rocBLAS present
  ```bash
  ls -lh /opt/rocm/lib/librocblas.so
  ```

- [ ] MIOpen present (for cuDNN)
  ```bash
  ls -lh /opt/rocm/lib/libMIOpen.so*
  ```

- [ ] GPU detected
  ```bash
  rocm-smi --showproductname
  # Should show: AMD Instinct MI300X (or similar)
  ```

---

### ✅ **Phase 5: APEX Setup on AMD**

- [ ] Run AMD setup script
  ```bash
  ./setup_amd_mi300x.sh
  ```

- [ ] Environment configured
  ```bash
  source apex_env.sh
  echo $LD_PRELOAD
  # Should show all three bridges
  ```

- [ ] Bridges present and loadable
  ```bash
  ldd libapex_hip_bridge.so
  # Should resolve all dependencies
  ```

---

### ✅ **Phase 6: Initial Testing**

- [ ] Hello World test
  ```bash
  # Compile
  nvcc -o test_hello test_hello_cuda.cu

  # Run with APEX
  source apex_env.sh
  ./test_hello
  # Should show: ✅ TEST PASSED! Result: 42
  ```

- [ ] Quick verification
  ```bash
  ./verify_apex.sh
  # Should now pass all tests (12/12)
  ```

- [ ] Comprehensive test suite
  ```bash
  ./run_all_tests.sh
  # Should show: 100% pass rate with actual GPU execution
  ```

- [ ] Check test logs for HIP execution
  ```bash
  grep "HIP\|AMD\|rocBLAS\|MIOpen" build/*_apex.log | head -20
  # Should show HIP/AMD library calls
  ```

---

### ✅ **Phase 7: Real Application Testing**

- [ ] CUDA Samples (if available)
  ```bash
  ./test_cuda_samples.sh
  ```

- [ ] cuBLAS test
  ```bash
  source apex_env.sh
  ./test_cublas_matmul
  # Should complete matrix multiply on AMD GPU
  ```

- [ ] PyTorch test (if needed)
  ```bash
  pip install torch torchvision
  python test_pytorch_cnn.py
  # Should run CNN on AMD GPU
  ```

---

### ✅ **Phase 8: Performance Validation**

- [ ] Enable profiling
  ```bash
  export APEX_PROFILE=1
  ./run_all_tests.sh
  ```

- [ ] Check performance metrics
  ```bash
  grep "PERFORMANCE PROFILE" build/*_apex.log
  # Should show timing data
  ```

- [ ] Verify memory tracking
  ```bash
  grep "MEMORY STATISTICS" build/*_apex.log
  # Should show allocation stats
  ```

- [ ] Check for memory leaks
  ```bash
  grep "Memory Leak" build/*_apex.log
  # Should show 0 bytes leaked
  ```

- [ ] Baseline performance test
  ```bash
  # Run a test and note execution time
  time ./build/test_async_streams
  ```

---

### ✅ **Phase 9: Production Readiness**

- [ ] No critical errors in logs
  ```bash
  grep -i "error\|critical\|fatal" build/*.log | grep -v "No error" || echo "Clean"
  ```

- [ ] All CUDA calls translating
  ```bash
  APEX_DEBUG=1 ./build/test_device_mgmt 2>&1 | grep -c "cudnn\|cublas\|cuda"
  # Should show many CUDA calls
  ```

- [ ] Performance acceptable (compare with baseline)

- [ ] Memory usage reasonable
  ```bash
  rocm-smi --showmeminfo
  ```

- [ ] Thermal/power within limits
  ```bash
  rocm-smi --showtemp --showpower
  ```

---

### ✅ **Phase 10: Your Application**

- [ ] Application uploaded to AMD instance

- [ ] Dependencies installed

- [ ] Application runs with APEX
  ```bash
  source apex_env.sh
  ./your_cuda_app
  # Or: python your_ml_script.py
  ```

- [ ] Output correct (matches CUDA output)

- [ ] Performance satisfactory

- [ ] Stability verified (no crashes over time)

---

## 🚨 **Troubleshooting Checklist**

If something doesn't work:

- [ ] Check ROCm version compatibility
  ```bash
  rocm-smi --version
  # Recommended: 5.7+ or 6.0+
  ```

- [ ] Verify GPU is accessible
  ```bash
  ls -l /dev/kfd /dev/dri
  # Should show render nodes
  ```

- [ ] Check user in video/render groups
  ```bash
  groups
  # Should show: video render
  ```

- [ ] Add user to groups if needed
  ```bash
  sudo usermod -aG video,render $USER
  # Then logout/login
  ```

- [ ] Review APEX logs for errors
  ```bash
  APEX_DEBUG=1 APEX_LOG_FILE=debug.log ./your_app
  cat debug.log
  ```

- [ ] Check library dependencies
  ```bash
  ldd libapex_hip_bridge.so
  ldd libapex_cublas_bridge.so
  ldd libapex_cudnn_bridge.so
  ```

- [ ] Verify LD_PRELOAD is set
  ```bash
  echo $LD_PRELOAD
  # Should show all 3 bridges
  ```

---

## 📊 **Success Criteria**

Your deployment is successful if:

- ✅ All 5 comprehensive tests pass (100%)
- ✅ Test execution uses AMD GPU (check rocm-smi during test)
- ✅ No memory leaks detected
- ✅ Performance is 95%+ of native AMD
- ✅ Your application runs correctly
- ✅ Output matches CUDA version
- ✅ Stable over extended runtime

---

## 🎯 **Quick Validation Commands**

Run these in order for quick validation:

```bash
# 1. Check environment
rocm-smi
source apex_env.sh
echo $LD_PRELOAD

# 2. Quick test
nvcc -o test_hello test_hello_cuda.cu
./test_hello

# 3. Comprehensive tests
./run_all_tests.sh

# 4. Check logs
grep "✅" build/*_apex.log | wc -l
# Should be > 20

# 5. Your app
./your_cuda_application
```

---

## 📝 **Deployment Log Template**

Keep track of your deployment:

```
AMD MI300X Deployment Log
=========================

Date: _______________
Instance: _______________
GPU Model: _______________
ROCm Version: _______________

Phase 1 (Local): [ ] Complete
Phase 2 (Provision): [ ] Complete
Phase 3 (Upload): [ ] Complete
Phase 4 (ROCm): [ ] Complete
Phase 5 (APEX Setup): [ ] Complete
Phase 6 (Initial Tests): [ ] Complete
Phase 7 (App Tests): [ ] Complete
Phase 8 (Performance): [ ] Complete
Phase 9 (Production): [ ] Complete
Phase 10 (Your App): [ ] Complete

Issues Encountered:
_______________________________________________
_______________________________________________

Resolution:
_______________________________________________
_______________________________________________

Final Status: [ ] SUCCESS [ ] PARTIAL [ ] FAILED

Notes:
_______________________________________________
_______________________________________________
```

---

## ✅ **Final Sign-Off**

Before going to production:

- [ ] All tests pass on AMD MI300X
- [ ] Performance benchmarks acceptable
- [ ] No memory leaks
- [ ] Application validated
- [ ] Monitoring configured
- [ ] Backup plan in place
- [ ] Documentation updated
- [ ] Team notified

---

**Deployment Status: Ready for AMD MI300X!** 🚀
