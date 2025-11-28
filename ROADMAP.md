# APEX GPU Development Roadmap

## ✅ Phase 1: Foundation (COMPLETE)
- [x] CUDA Driver API interception
- [x] CUDA Runtime API interception  
- [x] Real neural network implementation (3-layer, 400 params)
- [x] Feature engineering with log-scaling
- [x] Smart optimization recommendations
- [x] Comprehensive test suite
- [x] Documentation (QUICKSTART, technical guides)

**Status**: Production-ready base system ✅

---

## 🚀 Phase 2: Data Collection & Training (Next 1-2 Weeks)

### Goal: Replace heuristic weights with trained weights from real GPU data

### Step 2.1: Setup Profiling Environment
```bash
# Install NVIDIA profiling tools (if not already installed)
sudo apt-get install nvidia-cuda-toolkit

# Verify ncu (NVIDIA Compute Profiler) works
ncu --version
```

### Step 2.2: Create Benchmark Suite
Create `benchmark_all_configs.cu`:
```cuda
// Systematically test all important configurations
__global__ void benchmark_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple compute workload
        float val = data[idx];
        for (int i = 0; i < 100; i++) {
            val = sqrtf(val * val + 1.0f);
        }
        data[idx] = val;
    }
}

int main() {
    // Test various configurations
    int configs[][2] = {
        {32, 16}, {32, 32}, {32, 64}, {32, 128}, {32, 256}, {32, 512},
        {64, 16}, {64, 32}, {64, 64}, {64, 128}, {64, 256}, {64, 512},
        {128, 16}, {128, 32}, {128, 64}, {128, 128}, {128, 256}, {128, 512},
        // ... more configs
    };
    
    for (auto [blocks, threads] : configs) {
        run_and_profile(blocks, threads);
    }
}
```

### Step 2.3: Collect Ground Truth Data
```bash
#!/bin/bash
# collect_training_data.sh

echo "grid_x,grid_y,grid_z,block_x,block_y,block_z,shared_mem,occupancy,time_ms" > training_data.csv

for blocks in 32 64 128 256 512 1024 2048; do
    for threads in 16 32 64 128 256 512 1024; do
        echo "Testing: $blocks blocks × $threads threads"
        
        # Run with NVIDIA profiler
        ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
            --csv \
            ./benchmark_config $blocks $threads 2>&1 | \
            grep "sm__throughput" | \
            awk -F',' '{print "'$blocks',1,1,'$threads',1,1,0," $NF}' >> training_data.csv
    done
done

echo "✅ Collected $(wc -l < training_data.csv) training samples"
```

### Step 2.4: Train Neural Network
```python
# train_apex_nn.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load training data
data = np.loadtxt('training_data.csv', delimiter=',', skiprows=1)

# Extract features and labels
features = data[:, :7]  # grid, block, shared_mem
labels = data[:, 7:]    # occupancy, time

# Normalize features (same as apex_ml_model.h)
def normalize_features(raw_features):
    normalized = np.zeros((raw_features.shape[0], 8))
    normalized[:, 0] = np.log(raw_features[:, 0] + 1) / 15.0  # grid_x
    normalized[:, 1] = np.log(raw_features[:, 1] + 1) / 15.0  # grid_y
    normalized[:, 2] = np.log(raw_features[:, 2] + 1) / 15.0  # grid_z
    normalized[:, 3] = np.log(raw_features[:, 3] + 1) / 11.0  # block_x
    normalized[:, 4] = np.log(raw_features[:, 4] + 1) / 11.0  # block_y
    normalized[:, 5] = np.log(raw_features[:, 5] + 1) / 11.0  # block_z
    normalized[:, 6] = np.log(raw_features[:, 6] + 1) / 20.0  # shared_mem
    
    # Total threads
    total_threads = (raw_features[:, 0] * raw_features[:, 1] * raw_features[:, 2] *
                     raw_features[:, 3] * raw_features[:, 4] * raw_features[:, 5])
    normalized[:, 7] = np.log(total_threads + 1) / 25.0
    
    return normalized

X = normalize_features(features)
y = labels / 100.0  # Normalize to [0, 1]

# Split train/test
split = int(0.8 * len(X))
X_train, X_test = torch.tensor(X[:split], dtype=torch.float32), torch.tensor(X[split:], dtype=torch.float32)
y_train, y_test = torch.tensor(y[:split], dtype=torch.float32), torch.tensor(y[split:], dtype=torch.float32)

# Define model (matches apex_ml_model.h architecture)
class APEXModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = APEXModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
print("Training APEX neural network...")
for epoch in range(2000):
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        with torch.no_grad():
            test_loss = criterion(model(X_test), y_test)
        print(f"Epoch {epoch:4d} | Train Loss: {loss.item():.6f} | Test Loss: {test_loss.item():.6f}")

# Export weights to C header
def export_weights_to_c(model, filename='apex_trained_weights.h'):
    with open(filename, 'w') as f:
        f.write("// APEX Trained Weights - Auto-generated\n")
        f.write(f"// Training samples: {len(X_train)}\n")
        f.write(f"// Final train loss: {loss.item():.6f}\n\n")
        
        # Layer 1
        w1 = model.fc1.weight.detach().numpy().T  # Transpose for C array layout
        b1 = model.fc1.bias.detach().numpy()
        
        f.write("static const float trained_w1[8][16] = {\n")
        for i in range(8):
            f.write("    {" + ", ".join(f"{w:.8f}f" for w in w1[i]) + "},\n")
        f.write("};\n\n")
        
        f.write("static const float trained_b1[16] = {")
        f.write(", ".join(f"{b:.8f}f" for b in b1))
        f.write("};\n\n")
        
        # Layer 2
        w2 = model.fc2.weight.detach().numpy().T
        b2 = model.fc2.bias.detach().numpy()
        
        f.write("static const float trained_w2[16][8] = {\n")
        for i in range(16):
            f.write("    {" + ", ".join(f"{w:.8f}f" for w in w2[i]) + "},\n")
        f.write("};\n\n")
        
        f.write("static const float trained_b2[8] = {")
        f.write(", ".join(f"{b:.8f}f" for b in b2))
        f.write("};\n\n")
        
        # Layer 3
        w3 = model.fc3.weight.detach().numpy().T
        b3 = model.fc3.bias.detach().numpy()
        
        f.write("static const float trained_w3[8][4] = {\n")
        for i in range(8):
            f.write("    {" + ", ".join(f"{w:.8f}f" for w in w3[i]) + "},\n")
        f.write("};\n\n")
        
        f.write("static const float trained_b3[4] = {")
        f.write(", ".join(f"{b:.8f}f" for b in b3))
        f.write("};\n\n")

export_weights_to_c(model)
print("\n✅ Weights exported to apex_trained_weights.h")
print("   Update apex_ml_model.h to #include this file")
```

### Step 2.5: Update APEX to Use Trained Weights
```c
// In apex_ml_model.h, add at top:
#include "apex_trained_weights.h"

// Replace init_model() weights with:
static void init_model() {
    if (g_model_initialized) return;
    
    // Load trained weights instead of heuristic initialization
    memcpy(g_model.w1, trained_w1, sizeof(trained_w1));
    memcpy(g_model.b1, trained_b1, sizeof(trained_b1));
    memcpy(g_model.w2, trained_w2, sizeof(trained_w2));
    memcpy(g_model.b2, trained_b2, sizeof(trained_b2));
    memcpy(g_model.w3, trained_w3, sizeof(trained_w3));
    memcpy(g_model.b3, trained_b3, sizeof(trained_b3));
    
    g_model_initialized = 1;
}
```

**Deliverables**:
- [ ] `training_data.csv` with 1000+ samples
- [ ] `apex_trained_weights.h` with trained weights
- [ ] Updated `libapex_ml_real.so` with better accuracy

---

## 📈 Phase 3: Scale to Larger Model (Weeks 3-4)

### Goal: Integrate 1.8M parameter model using ONNX Runtime

### Step 3.1: Install ONNX Runtime
```bash
# Download ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz
sudo cp -r onnxruntime-linux-x64-1.16.0/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-1.16.0/lib/* /usr/local/lib/
sudo ldconfig
```

### Step 3.2: Train Larger Model
```python
# train_large_model.py
class LargeAPEXModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 4)
        
        # Total params: ~450K (can scale to 1.8M easily)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

# Train and export to ONNX
model = LargeAPEXModel()
# ... training code ...

# Export to ONNX
dummy_input = torch.randn(1, 8)
torch.onnx.export(model, dummy_input, "apex_large_model.onnx",
                  input_names=['features'],
                  output_names=['predictions'],
                  dynamic_axes={'features': {0: 'batch_size'}})
```

### Step 3.3: Create ONNX-Based APEX
```c
// apex_ml_onnx.c
#include <onnxruntime_c_api.h>

static const OrtApi* g_ort = NULL;
static OrtEnv* g_env = NULL;
static OrtSession* g_session = NULL;

void init_onnx_model() {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "apex", &g_env);
    
    OrtSessionOptions* session_options;
    g_ort->CreateSessionOptions(&session_options);
    
    // Enable CPU optimization (or GPU if available)
    g_ort->SetIntraOpNumThreads(session_options, 1);
    g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL);
    
    g_ort->CreateSession(g_env, "apex_large_model.onnx", session_options, &g_session);
}

MLModelOutput predict_with_onnx(unsigned int gx, unsigned int gy, unsigned int gz,
                                unsigned int bx, unsigned int by, unsigned int bz,
                                size_t shared_mem) {
    float features[8];
    extract_features(gx, gy, gz, bx, by, bz, shared_mem, features);
    
    // Create input tensor
    int64_t input_shape[] = {1, 8};
    size_t input_tensor_size = 8;
    
    OrtMemoryInfo* memory_info;
    g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    
    OrtValue* input_tensor = NULL;
    g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, features, input_tensor_size * sizeof(float),
        input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    
    // Run inference
    const char* input_names[] = {"features"};
    const char* output_names[] = {"predictions"};
    
    OrtValue* output_tensor = NULL;
    g_ort->Run(g_session, NULL, input_names, &input_tensor, 1,
               output_names, 1, &output_tensor);
    
    // Get output
    float* output_data;
    g_ort->GetTensorMutableData(output_tensor, (void**)&output_data);
    
    MLModelOutput result;
    result.occupancy = output_data[0];
    result.execution_time_ms = output_data[1];
    result.sm_utilization = output_data[2];
    result.block_efficiency = output_data[3];
    
    // Cleanup
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    
    return result;
}
```

**Deliverables**:
- [ ] ONNX model with 500K-1.8M parameters
- [ ] `libapex_ml_onnx.so` with ONNX integration
- [ ] Benchmark showing <100μs inference time

---

## 🎯 Phase 4: Production Deployment (Week 5+)

### Features to Add:

#### 4.1: Telemetry & Analytics
```c
// Export predictions to JSON for analysis
void export_prediction_log(const char* filename) {
    FILE* f = fopen(filename, "w");
    fprintf(f, "[\n");
    for (int i = 0; i < kernel_history_size; i++) {
        fprintf(f, "  {\"grid\":[%u,%u,%u], \"block\":[%u,%u,%u], "
                   "\"occupancy\":%.4f, \"time_ms\":%.4f},\n",
                   history[i].grid_x, history[i].grid_y, history[i].grid_z,
                   history[i].block_x, history[i].block_y, history[i].block_z,
                   history[i].predicted_occupancy, history[i].predicted_time);
    }
    fprintf(f, "]\n");
    fclose(f);
}
```

#### 4.2: Auto-Tuning Integration
```c
// Suggest alternative configurations
typedef struct {
    dim3 grid;
    dim3 block;
    float expected_speedup;
} AlternativeConfig;

AlternativeConfig suggest_better_config(dim3 current_grid, dim3 current_block) {
    // Try variations
    AlternativeConfig best = {current_grid, current_block, 1.0f};
    float current_occupancy = predict_occupancy(current_grid, current_block);
    
    // Try different block sizes
    for (int threads = 128; threads <= 512; threads *= 2) {
        dim3 new_block = {threads, 1, 1};
        float new_occupancy = predict_occupancy(current_grid, new_block);
        
        if (new_occupancy > current_occupancy * 1.1f) {
            best.block = new_block;
            best.expected_speedup = new_occupancy / current_occupancy;
        }
    }
    
    return best;
}
```

#### 4.3: Performance Dashboard
Create web dashboard to visualize predictions:
- Real-time kernel launch monitoring
- Occupancy trends over time
- Recommendations histogram
- Performance improvement tracking

**Deliverables**:
- [ ] JSON export functionality
- [ ] Auto-tuning suggestions
- [ ] Web dashboard (optional)
- [ ] Production deployment guide

---

## 🎯 Immediate Action Items (This Week)

### Priority 1: Test Current System
```bash
# Run all tests with real ML model
./build_apex.sh
LD_PRELOAD=./libapex_ml_real.so ./test_ml_benchmark
LD_PRELOAD=./libapex_ml_real.so ./test_multi_kernels
LD_PRELOAD=./libapex_ml_real.so ./test_shared
```

### Priority 2: Document Current State
- [x] QUICKSTART.md
- [x] APEX_ML_SUMMARY.md  
- [x] ACHIEVEMENTS.md
- [x] ROADMAP.md (this file)

### Priority 3: Setup for Next Phase
```bash
# Check if you have profiling tools
which ncu
ncu --version

# Check Python/PyTorch availability (for training)
python3 --version
python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "PyTorch not installed"
```

---

## 📊 Success Metrics

### Current Baseline (Phase 1 Complete)
- ✅ Neural network inference: ~15-80 μs
- ✅ Prediction variance: 26% to 66% occupancy
- ✅ Zero code modification required
- ✅ Works with Runtime and Driver API

### Phase 2 Targets (Trained Model)
- 🎯 Prediction accuracy: >90% correlation with real occupancy
- 🎯 Inference time: <50 μs
- 🎯 Training dataset: 5000+ configurations

### Phase 3 Targets (Large Model)
- 🎯 Model size: 500K-1.8M parameters
- 🎯 Inference time: <100 μs
- 🎯 Prediction accuracy: >95%

### Phase 4 Targets (Production)
- 🎯 Deployed on production ML training servers
- 🎯 Auto-tuning saves 10%+ training time
- 🎯 Monitoring 1000+ jobs/day

---

## 🚧 Known Limitations & Future Work

### Current Limitations
- Heuristic weights (not trained on real data)
- Small model (400 params vs target 1.8M)
- No PyTorch integration yet (requires PyTorch install)
- Single-GPU focused (RTX 5080 specific)

### Future Enhancements
- Multi-GPU support
- Kernel fusion recommendations
- Memory access pattern analysis
- Automatic kernel rewriting
- Cloud deployment (monitor remote GPUs)

---

**Next Step**: Choose which phase to tackle next!
- **Quick win**: Collect profiling data (Phase 2.2-2.3)
- **Deep dive**: Train better weights (Phase 2.4)
- **Scale up**: ONNX integration (Phase 3)
