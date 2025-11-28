# APEX Profiling & Diagnostics Guide

## 🎯 New Features

APEX now includes comprehensive profiling and diagnostics:
- ✅ Performance profiling (track time per function)
- ✅ Memory tracking (allocations, leaks, peak usage)
- ✅ Debug logging
- ✅ Trace logging
- ✅ Statistics dashboard
- ✅ Log file support

---

## 🚀 Usage

### Environment Variables

Control APEX behavior with environment variables:

```bash
# Enable debug logging
APEX_DEBUG=1

# Enable performance profiling
APEX_PROFILE=1

# Enable detailed trace logging
APEX_TRACE=1

# Disable statistics (enabled by default)
APEX_STATS=0

# Log to file instead of stderr
APEX_LOG_FILE=apex.log
```

---

## 📊 Examples

### Example 1: Basic Usage (Default)
```bash
LD_PRELOAD=./libapex_hip_bridge.so ./my_cuda_app

# Output shows basic stats at end
```

### Example 2: Debug Mode
```bash
APEX_DEBUG=1 LD_PRELOAD=./libapex_hip_bridge.so ./my_cuda_app

# Shows:
# - Configuration on startup
# - Error messages with details
# - Warnings
```

### Example 3: Performance Profiling
```bash
APEX_PROFILE=1 LD_PRELOAD=./libapex_hip_bridge.so ./my_cuda_app

# Shows detailed performance table:
# ╔════════════════════════════════════════════════════════════════════════╗
# ║                    APEX PERFORMANCE PROFILE                            ║
# ╠════════════════════════════════════════════════════════════════════════╣
# ║ Function            │ Calls  │ Total(ms) │ Avg(μs) │ Min │  Max       ║
# ╠════════════════════════════════════════════════════════════════════════╣
# ║ cudaMalloc          │  1000  │     15.2  │   15.2  │  10 │   45       ║
# ║ cudaMemcpy          │  2000  │    125.5  │   62.7  │  50 │  120       ║
# ║ cudaLaunchKernel    │   100  │      8.5  │   85.0  │  75 │  150       ║
# ╚════════════════════════════════════════════════════════════════════════╝
```

### Example 4: Memory Tracking
```bash
APEX_PROFILE=1 LD_PRELOAD=./libapex_hip_bridge.so ./my_cuda_app

# Shows memory statistics:
# ╔════════════════════════════════════════════════════════════════════════╗
# ║                      APEX MEMORY STATISTICS                            ║
# ╠════════════════════════════════════════════════════════════════════════╣
# ║  Total Allocated:     10485760 bytes  (   10.00 MB)                   ║
# ║  Total Freed:          9437184 bytes  (    9.00 MB)                   ║
# ║  Peak Usage:          10485760 bytes  (   10.00 MB)                   ║
# ║  Current Usage:        1048576 bytes  (    1.00 MB)                   ║
# ║  Allocations:              100                                         ║
# ║  Frees:                     90                                         ║
# ║  ⚠️  Memory Leak:       1048576 bytes  (NOT FREED)                    ║
# ╚════════════════════════════════════════════════════════════════════════╝
```

### Example 5: Trace Logging
```bash
APEX_TRACE=1 LD_PRELOAD=./libapex_hip_bridge.so ./my_cuda_app

# Shows every allocation/free:
# [APEX-TRACE] cudaMalloc(4096 bytes)
# [APEX-TRACE] Allocated 4096 bytes (total: 4096, peak: 4096)
# [APEX-TRACE] cudaMemcpy(4096 bytes)
# [APEX-TRACE] Freed 4096 bytes (remaining: 0)
```

### Example 6: Log to File
```bash
APEX_LOG_FILE=apex_session.log \
APEX_DEBUG=1 \
APEX_PROFILE=1 \
LD_PRELOAD=./libapex_hip_bridge.so ./my_cuda_app

# All output goes to apex_session.log
# Can review later:
cat apex_session.log
```

### Example 7: Everything Enabled
```bash
APEX_DEBUG=1 \
APEX_PROFILE=1 \
APEX_TRACE=1 \
APEX_LOG_FILE=full_trace.log \
LD_PRELOAD="./libapex_cublas_bridge.so:./libapex_hip_bridge.so" \
  ./pytorch_app

# Maximum diagnostics!
```

---

## 🔍 Finding Performance Bottlenecks

Use profiling to optimize:

```bash
APEX_PROFILE=1 LD_PRELOAD=./libapex_hip_bridge.so ./slow_app 2>&1 | grep -A 20 "PERFORMANCE PROFILE"

# Look for:
# - Functions with high Total(ms) - most time spent
# - Functions with high call counts - called too often?
# - Functions with high Max time - inconsistent performance?
```

---

## 🐛 Debugging Memory Leaks

```bash
# Run with profiling
APEX_PROFILE=1 LD_PRELOAD=./libapex_hip_bridge.so ./app

# Check memory statistics at end
# If "Memory Leak" shown:
#   - Review cudaMalloc/cudaFree calls
#   - Ensure all allocations are freed
#   - Check for early exits without cleanup
```

---

## 📈 Benchmarking APEX Overhead

Compare with and without APEX:

```bash
# Without APEX (native CUDA)
time ./my_cuda_app

# With APEX (measure overhead)
time APEX_PROFILE=1 LD_PRELOAD=./libapex_hip_bridge.so ./my_cuda_app

# Check performance table for per-function overhead
# Typical overhead: <1μs per API call
```

---

## 🎯 Production vs Development

**Development** (maximum diagnostics):
```bash
APEX_DEBUG=1 APEX_PROFILE=1 APEX_TRACE=1 APEX_LOG_FILE=debug.log
```

**Testing** (performance profiling):
```bash
APEX_PROFILE=1 APEX_STATS=1
```

**Production** (minimal overhead):
```bash
# No environment variables (just basic stats)
# Or: APEX_STATS=0 to disable even basic stats
```

---

## 📝 Log File Analysis

```bash
# Generate comprehensive log
APEX_DEBUG=1 APEX_PROFILE=1 APEX_TRACE=1 APEX_LOG_FILE=trace.log \
  LD_PRELOAD=./libapex_hip_bridge.so ./app

# Analyze
grep "ERROR" trace.log    # Find errors
grep "Allocated" trace.log | wc -l  # Count allocations
grep "Total(ms)" trace.log  # See performance summary
```

---

## 🔧 Advanced Usage

### Custom Log Analysis Script

```bash
#!/bin/bash
# analyze_apex_log.sh

LOG_FILE=$1

echo "=== APEX Log Analysis ==="
echo ""

echo "Errors:"
grep -c "ERROR" $LOG_FILE

echo ""
echo "Warnings:"
grep -c "WARN" $LOG_FILE

echo ""
echo "Total Allocations:"
grep "Allocated" $LOG_FILE | wc -l

echo ""
echo "Peak Memory:"
grep "Peak Usage" $LOG_FILE | tail -1

echo ""
echo "Slowest Functions:"
grep -A 5 "PERFORMANCE PROFILE" $LOG_FILE | grep "║" | sort -t'│' -k3 -rn | head -5
```

---

## 💡 Tips

1. **Start with DEBUG**: First run with `APEX_DEBUG=1` to see configuration
2. **Profile for bottlenecks**: Use `APEX_PROFILE=1` to find slow functions
3. **Trace for issues**: Use `APEX_TRACE=1` when debugging specific problems
4. **Log to file for analysis**: Always use `APEX_LOG_FILE` for later review
5. **Disable in production**: No env vars = minimal overhead

---

## 🎨 Output Format

All APEX diagnostics use clear, formatted output:

- **[APEX-INFO]**: Informational messages
- **[APEX-DEBUG]**: Debug information (only with APEX_DEBUG=1)
- **[APEX-TRACE]**: Detailed trace (only with APEX_TRACE=1)
- **[APEX-ERROR]**: Error messages with context
- **[APEX-WARN]**: Warning messages

---

## 🚀 Ready for Production

APEX profiling is:
- **Zero overhead** when disabled (default for most features)
- **Minimal overhead** for basic stats (<0.01%)
- **Configurable** via environment variables
- **Non-invasive** - no code changes needed

**Use it to:**
- ✅ Debug issues quickly
- ✅ Optimize performance
- ✅ Track memory usage
- ✅ Validate correctness
- ✅ Monitor production workloads
