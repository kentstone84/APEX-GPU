# APEX GPU - Real-World Application Testing Guide

## 🎯 Overview

This guide shows how to test APEX GPU translation layer with production CUDA applications:
- ✅ NVIDIA CUDA Samples
- ✅ hashcat (Password Recovery)
- ✅ Blender Cycles (3D Rendering)
- ✅ ffmpeg (Video Processing)
- ✅ TensorFlow/PyTorch (Machine Learning)

---

## 📋 Test Categories

### Category 1: CUDA SDK Samples
**Purpose**: Validate core CUDA functionality
**Complexity**: Low to Medium
**Value**: Comprehensive API coverage

### Category 2: Scientific Computing
**Purpose**: Test compute-heavy workloads
**Complexity**: High
**Value**: Real-world performance validation

### Category 3: Media Processing
**Purpose**: Test throughput and latency
**Complexity**: Medium
**Value**: Production use case validation

### Category 4: Machine Learning
**Purpose**: Test ML frameworks
**Complexity**: Very High
**Value**: Primary APEX use case

---

# 1️⃣ NVIDIA CUDA Samples

## Installation

```bash
# Clone CUDA samples
git clone https://github.com/nvidia/cuda-samples.git
cd cuda-samples/Samples

# Or download specific version
wget https://github.com/NVIDIA/cuda-samples/archive/refs/tags/v12.3.tar.gz
tar -xzf v12.3.tar.gz
```

## Key Samples to Test

### 1.1 Device Query
**Path**: `1_Utilities/deviceQuery/`
**Tests**: Device enumeration, properties

```bash
cd 1_Utilities/deviceQuery
make

# Native CUDA
./deviceQuery

# With APEX
APEX_DEBUG=1 LD_PRELOAD=/path/to/libapex_hip_bridge.so ./deviceQuery
```

**Expected**: Should show AMD GPU properties when running on MI300X

---

### 1.2 Vector Add
**Path**: `0_Introduction/vectorAdd/`
**Tests**: Basic kernel launch, memory operations

```bash
cd 0_Introduction/vectorAdd
make

# With APEX profiling
APEX_PROFILE=1 LD_PRELOAD=/path/to/libapex_hip_bridge.so ./vectorAdd
```

**Expected**: Should show performance metrics for cudaMalloc, cudaMemcpy, kernel launch

---

### 1.3 Matrix Multiply
**Path**: `0_Introduction/matrixMul/`
**Tests**: Compute-intensive kernels

```bash
cd 0_Introduction/matrixMul
make

# Baseline (CUDA)
time ./matrixMul

# With APEX
time APEX_PROFILE=1 LD_PRELOAD=/path/to/libapex_hip_bridge.so ./matrixMul
```

**Expected**: Should complete matrix multiplication, compare performance

---

### 1.4 Concurrent Kernels
**Path**: `0_Introduction/concurrentKernels/`
**Tests**: Streams, async operations

```bash
cd 0_Introduction/concurrentKernels
make

# With APEX
APEX_PROFILE=1 APEX_TRACE=1 \
LD_PRELOAD=/path/to/libapex_hip_bridge.so \
./concurrentKernels
```

**Expected**: Should show stream overlap, async profiling

---

### 1.5 Bandwidth Test
**Path**: `1_Utilities/bandwidthTest/`
**Tests**: H2D/D2H transfer speeds

```bash
cd 1_Utilities/bandwidthTest
make

# Compare bandwidth
./bandwidthTest > cuda_bandwidth.txt
APEX_PROFILE=1 LD_PRELOAD=/path/to/libapex_hip_bridge.so \
  ./bandwidthTest > apex_bandwidth.txt

diff cuda_bandwidth.txt apex_bandwidth.txt
```

**Expected**: Bandwidth comparison between native CUDA and APEX

---

## Quick Test Script: test_cuda_samples.sh

```bash
#!/bin/bash
# Test key CUDA samples with APEX

SAMPLES_DIR="cuda-samples/Samples"
APEX_PRELOAD="LD_PRELOAD=/path/to/libapex_hip_bridge.so"

echo "Testing CUDA Samples with APEX"
echo ""

# 1. Device Query
echo "=== Device Query ==="
cd "$SAMPLES_DIR/1_Utilities/deviceQuery"
make -j8
APEX_DEBUG=1 $APEX_PRELOAD ./deviceQuery
echo ""

# 2. Vector Add
echo "=== Vector Add ==="
cd "$SAMPLES_DIR/0_Introduction/vectorAdd"
make -j8
APEX_PROFILE=1 $APEX_PRELOAD ./vectorAdd
echo ""

# 3. Matrix Multiply
echo "=== Matrix Multiply ==="
cd "$SAMPLES_DIR/0_Introduction/matrixMul"
make -j8
time APEX_PROFILE=1 $APEX_PRELOAD ./matrixMul
echo ""

# 4. Concurrent Kernels
echo "=== Concurrent Kernels ==="
cd "$SAMPLES_DIR/0_Introduction/concurrentKernels"
make -j8
APEX_PROFILE=1 $APEX_PRELOAD ./concurrentKernels
echo ""

echo "✅ CUDA Samples testing complete!"
```

---

# 2️⃣ hashcat - Password Recovery Tool

## What is hashcat?
GPU-accelerated password recovery tool. Heavily CUDA-optimized for cryptographic operations.

## Installation

```bash
# Install hashcat
wget https://hashcat.net/files/hashcat-6.2.6.tar.gz
tar -xzf hashcat-6.2.6.tar.gz
cd hashcat-6.2.6

# Or via package manager
apt install hashcat
```

## Testing with APEX

### Test 1: MD5 Hash Cracking

```bash
# Create test hash
echo -n "password123" | md5sum > test.hash

# Benchmark native CUDA
hashcat -b -m 0 -D 2  # Mode 0 = MD5, Device 2 = CUDA

# Run with APEX
APEX_PROFILE=1 \
APEX_LOG_FILE=hashcat_apex.log \
LD_PRELOAD=/path/to/libapex_hip_bridge.so \
hashcat -b -m 0 -D 2
```

**Expected**: Should show GPU kernel launches, memory operations

---

### Test 2: SHA256 Cracking

```bash
# Create SHA256 hash
echo -n "test123" | sha256sum > test_sha256.hash

# Benchmark
hashcat -b -m 1400  # Mode 1400 = SHA256

# With APEX profiling
APEX_PROFILE=1 LD_PRELOAD=/path/to/libapex_hip_bridge.so \
hashcat -b -m 1400
```

**Key metrics to watch**:
- Kernel launch frequency
- Memory transfer overhead
- Hash/s throughput

---

### Test 3: Dictionary Attack

```bash
# Create wordlist
cat > wordlist.txt <<EOF
password
123456
password123
admin
test123
EOF

# Create target hash (MD5 of "password123")
echo "482c811da5d5b4bc6d497ffa98491e38" > target.hash

# Attack with APEX
APEX_PROFILE=1 \
LD_PRELOAD=/path/to/libapex_hip_bridge.so \
hashcat -m 0 -a 0 target.hash wordlist.txt
```

**Expected**: Should crack the hash, show CUDA API usage

---

## hashcat Test Script

```bash
#!/bin/bash
# test_hashcat_apex.sh

echo "╔═══════════════════════════════════════════════════════╗"
echo "║         hashcat + APEX GPU Translation Test          ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Test MD5 benchmark
echo "Test 1: MD5 Benchmark"
echo "---------------------"
APEX_PROFILE=1 \
APEX_LOG_FILE=hashcat_md5.log \
LD_PRELOAD=/path/to/libapex_hip_bridge.so \
hashcat -b -m 0 2>&1 | head -50

echo ""
echo "Test 2: SHA256 Benchmark"
echo "------------------------"
APEX_PROFILE=1 \
APEX_LOG_FILE=hashcat_sha256.log \
LD_PRELOAD=/path/to/libapex_hip_bridge.so \
hashcat -b -m 1400 2>&1 | head -50

echo ""
echo "✅ hashcat tests complete!"
echo "Check logs: hashcat_*.log"
```

---

# 3️⃣ Blender Cycles - 3D Rendering

## What is Blender Cycles?
GPU-accelerated ray tracing renderer. Uses CUDA for rendering acceleration.

## Installation

```bash
# Download Blender
wget https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz
tar -xf blender-4.0.2-linux-x64.tar.xz

# Or via package manager
apt install blender
```

## Testing with APEX

### Test 1: Command-Line Rendering

```bash
# Download test scene (or use included demo)
# Blender comes with demo.blend files

# Render with CPU (baseline)
blender -b demo.blend -f 1 -o //frame_cpu_ -E CYCLES

# Render with CUDA (native)
blender -b demo.blend -f 1 -o //frame_cuda_ -E CYCLES -- --cycles-device CUDA

# Render with APEX
APEX_PROFILE=1 \
APEX_LOG_FILE=blender_apex.log \
LD_PRELOAD=/path/to/libapex_hip_bridge.so \
blender -b demo.blend -f 1 -o //frame_apex_ -E CYCLES -- --cycles-device CUDA
```

---

### Test 2: Benchmark Scene

```bash
# Create simple benchmark scene
cat > benchmark.py <<'EOF'
import bpy

# Clear scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Add sphere
bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0))

# Add camera
bpy.ops.object.camera_add(location=(5, -5, 5))
bpy.context.scene.camera = bpy.context.object

# Add light
bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))

# Set render engine to Cycles
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.samples = 128

# Save
bpy.ops.wm.save_as_mainfile(filepath='benchmark.blend')
EOF

blender --background --python benchmark.py

# Render benchmark
time APEX_PROFILE=1 LD_PRELOAD=/path/to/libapex_hip_bridge.so \
  blender -b benchmark.blend -f 1
```

---

### Test 3: Animation Render

```bash
# Render 30 frames
APEX_PROFILE=1 \
APEX_LOG_FILE=blender_animation.log \
LD_PRELOAD=/path/to/libapex_hip_bridge.so \
blender -b demo.blend -s 1 -e 30 -a -E CYCLES -- --cycles-device CUDA
```

**Expected**: Should render frames using AMD GPU on MI300X

---

# 4️⃣ ffmpeg - Video Processing

## What to test?
ffmpeg's CUDA-accelerated video filters and encoding.

## Installation

```bash
# Install ffmpeg with CUDA support
apt install ffmpeg

# Or compile with CUDA
./configure --enable-cuda-nvcc --enable-cuvid --enable-nvenc
make -j8
```

## Testing with APEX

### Test 1: Video Decode (H.264)

```bash
# Download test video
wget https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_1MB.mp4 \
  -O test.mp4

# Decode with CUDA
ffmpeg -hwaccel cuda -i test.mp4 -f null -

# Decode with APEX
APEX_PROFILE=1 \
LD_PRELOAD=/path/to/libapex_hip_bridge.so \
ffmpeg -hwaccel cuda -i test.mp4 -f null -
```

---

### Test 2: Video Scaling (CUDA filter)

```bash
# Scale with CUDA
ffmpeg -hwaccel cuda -i test.mp4 \
  -vf scale_cuda=1280:720 \
  -c:v h264_nvenc \
  output_cuda.mp4

# Scale with APEX
APEX_PROFILE=1 \
LD_PRELOAD=/path/to/libapex_hip_bridge.so \
ffmpeg -hwaccel cuda -i test.mp4 \
  -vf scale_cuda=1280:720 \
  -c:v h264_nvenc \
  output_apex.mp4
```

---

### Test 3: H.265 Encoding

```bash
# Encode with NVENC
time ffmpeg -i test.mp4 -c:v hevc_nvenc -preset fast output.mp4

# With APEX
time APEX_PROFILE=1 \
LD_PRELOAD=/path/to/libapex_hip_bridge.so \
ffmpeg -i test.mp4 -c:v hevc_nvenc -preset fast output_apex.mp4
```

---

# 5️⃣ Machine Learning Frameworks

## TensorFlow

```bash
# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]

# Test script
cat > test_tf_apex.py <<'EOF'
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Simple computation
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    print("Result:", c)
EOF

# Run with APEX
APEX_PROFILE=1 \
APEX_LOG_FILE=tensorflow_apex.log \
LD_PRELOAD="/path/to/libapex_cublas_bridge.so:/path/to/libapex_hip_bridge.so" \
python test_tf_apex.py
```

---

## PyTorch

```bash
# Install PyTorch
pip install torch torchvision

# Test script
cat > test_pytorch_apex.py <<'EOF'
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))

# Simple tensor operation
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = torch.matmul(x, y)
print("Computation complete:", z.shape)
EOF

# Run with APEX (HIP + cuBLAS bridges)
APEX_PROFILE=1 \
APEX_LOG_FILE=pytorch_apex.log \
LD_PRELOAD="/path/to/libapex_cublas_bridge.so:/path/to/libapex_hip_bridge.so" \
python test_pytorch_apex.py
```

---

# 📊 Performance Comparison Template

```bash
#!/bin/bash
# compare_performance.sh - Compare Native CUDA vs APEX

APP_NAME="$1"
APP_CMD="$2"

echo "Performance Comparison: $APP_NAME"
echo "======================================="

# Native CUDA
echo "Running with native CUDA..."
time $APP_CMD > /dev/null 2>&1
CUDA_TIME=$?

# With APEX
echo "Running with APEX translation..."
time APEX_PROFILE=1 LD_PRELOAD=/path/to/libapex_hip_bridge.so \
  $APP_CMD > /dev/null 2>&1
APEX_TIME=$?

echo ""
echo "Results:"
echo "  Native CUDA: ${CUDA_TIME}s"
echo "  APEX:        ${APEX_TIME}s"
echo "  Overhead:    $((APEX_TIME - CUDA_TIME))s"
```

---

# 🎯 Success Criteria

For each application:

✅ **Interception**: APEX logs show CUDA calls being intercepted
✅ **Translation**: Calls translated to HIP equivalents
✅ **Execution**: Application completes without crashes
✅ **Correctness**: Output matches native CUDA (visual/numerical)
✅ **Performance**: Overhead is acceptable (<10% for compute-heavy)

---

# 📈 Expected Results

## On NVIDIA GPU (Current System)
- ✅ Interception works
- ✅ Translation works
- ⚠️ HIP execution fails (expected - no AMD runtime)

## On AMD MI300X
- ✅ Interception works
- ✅ Translation works
- ✅ HIP execution succeeds
- ✅ Full application functionality

---

# 🚀 Quick Start: All Apps Test

```bash
#!/bin/bash
# test_all_real_world_apps.sh

export APEX_PROFILE=1
export APEX_DEBUG=1
export LD_PRELOAD="/path/to/libapex_cublas_bridge.so:/path/to/libapex_hip_bridge.so"

echo "Testing Real-World CUDA Applications with APEX"
echo ""

# 1. CUDA Samples
./test_cuda_samples.sh

# 2. hashcat
./test_hashcat_apex.sh

# 3. Blender (if available)
if command -v blender &> /dev/null; then
    ./test_blender_apex.sh
fi

# 4. ffmpeg (if available)
if command -v ffmpeg &> /dev/null; then
    ./test_ffmpeg_apex.sh
fi

# 5. ML Frameworks
if python -c "import torch" 2>/dev/null; then
    python test_pytorch_apex.py
fi

echo ""
echo "✅ All real-world application tests complete!"
```

---

*Ready to test APEX with production CUDA applications on AMD MI300X!*
