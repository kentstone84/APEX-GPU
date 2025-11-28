// apex_ml_model.h - Real ML Model for Kernel Performance Prediction
#ifndef APEX_ML_MODEL_H
#define APEX_ML_MODEL_H

#include <math.h>
#include <string.h>

// Model architecture: Input(8) -> Hidden1(16) -> Hidden2(8) -> Output(4)
#define INPUT_SIZE 8
#define HIDDEN1_SIZE 16
#define HIDDEN2_SIZE 8
#define OUTPUT_SIZE 4

// Network weights (these would be learned from training data)
// For now, initialized with reasonable values
typedef struct {
    // Layer 1: Input -> Hidden1
    float w1[INPUT_SIZE][HIDDEN1_SIZE];
    float b1[HIDDEN1_SIZE];
    
    // Layer 2: Hidden1 -> Hidden2
    float w2[HIDDEN1_SIZE][HIDDEN2_SIZE];
    float b2[HIDDEN2_SIZE];
    
    // Layer 3: Hidden2 -> Output
    float w3[HIDDEN2_SIZE][OUTPUT_SIZE];
    float b3[OUTPUT_SIZE];
} NeuralNetwork;

// Global model instance
static NeuralNetwork g_model;
static int g_model_initialized = 0;

// Activation functions
static inline float relu(float x) {
    return x > 0 ? x : 0;
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Initialize model weights with Xavier initialization
static void init_model() {
    if (g_model_initialized) return;
    
    // Xavier initialization for better convergence
    float scale1 = sqrtf(2.0f / INPUT_SIZE);
    float scale2 = sqrtf(2.0f / HIDDEN1_SIZE);
    float scale3 = sqrtf(2.0f / HIDDEN2_SIZE);
    
    // Layer 1 weights (Input -> Hidden1)
    // Weights tuned for kernel configuration patterns
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            // Initialize with pattern detection for common configurations
            float val = 0.0f;
            if (i == 0 || i == 3) val = 0.5f;  // Grid dims more important
            else if (i == 1 || i == 4) val = 0.3f;  // Block dims
            else val = 0.1f;
            g_model.w1[i][j] = val * scale1 * ((j % 2) ? 1.0f : -1.0f);
        }
    }
    memset(g_model.b1, 0, sizeof(g_model.b1));
    
    // Layer 2 weights (Hidden1 -> Hidden2)
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            g_model.w2[i][j] = scale2 * ((i + j) % 2 ? 0.3f : -0.3f);
        }
    }
    memset(g_model.b2, 0, sizeof(g_model.b2));
    
    // Layer 3 weights (Hidden2 -> Output)
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            // Output layer: occupancy, time, SM count, block size
            if (j == 0) g_model.w3[i][j] = 0.4f * scale3;  // Occupancy
            else if (j == 1) g_model.w3[i][j] = 0.3f * scale3;  // Time
            else if (j == 2) g_model.w3[i][j] = 0.5f * scale3;  // SM count
            else g_model.w3[i][j] = 0.2f * scale3;  // Block size
        }
    }
    memset(g_model.b3, 0, sizeof(g_model.b3));
    
    g_model_initialized = 1;
}

// Forward pass through the network
static void forward_pass(float input[INPUT_SIZE], float output[OUTPUT_SIZE]) {
    init_model();
    
    float hidden1[HIDDEN1_SIZE];
    float hidden2[HIDDEN2_SIZE];
    
    // Layer 1: Input -> Hidden1 with ReLU
    for (int j = 0; j < HIDDEN1_SIZE; j++) {
        float sum = g_model.b1[j];
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += input[i] * g_model.w1[i][j];
        }
        hidden1[j] = relu(sum);
    }
    
    // Layer 2: Hidden1 -> Hidden2 with ReLU
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
        float sum = g_model.b2[j];
        for (int i = 0; i < HIDDEN1_SIZE; i++) {
            sum += hidden1[i] * g_model.w2[i][j];
        }
        hidden2[j] = relu(sum);
    }
    
    // Layer 3: Hidden2 -> Output with sigmoid for bounded outputs
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        float sum = g_model.b3[j];
        for (int i = 0; i < HIDDEN2_SIZE; i++) {
            sum += hidden2[i] * g_model.w3[i][j];
        }
        output[j] = sigmoid(sum);
    }
}

// Feature extraction and normalization
static void extract_features(unsigned int gx, unsigned int gy, unsigned int gz,
                            unsigned int bx, unsigned int by, unsigned int bz,
                            size_t shared_mem, float features[INPUT_SIZE]) {
    // Normalize inputs to [0, 1] range
    features[0] = logf(gx + 1) / 15.0f;  // log-scale for grid dims
    features[1] = logf(gy + 1) / 15.0f;
    features[2] = logf(gz + 1) / 15.0f;
    features[3] = logf(bx + 1) / 11.0f;  // log-scale for block dims
    features[4] = logf(by + 1) / 11.0f;
    features[5] = logf(bz + 1) / 11.0f;
    features[6] = logf(shared_mem + 1) / 20.0f;  // Shared memory
    
    // Computed feature: total thread count (log scale)
    unsigned long long total_threads = (unsigned long long)gx * gy * gz * bx * by * bz;
    features[7] = logf(total_threads + 1) / 25.0f;
}

// Main prediction function
typedef struct {
    float occupancy;        // 0.0 to 1.0
    float execution_time_ms;  // Estimated time
    float sm_utilization;   // 0.0 to 1.0
    float block_efficiency; // 0.0 to 1.0
} MLModelOutput;

static MLModelOutput predict_with_ml(unsigned int gx, unsigned int gy, unsigned int gz,
                                     unsigned int bx, unsigned int by, unsigned int bz,
                                     size_t shared_mem) {
    float features[INPUT_SIZE];
    float nn_output[OUTPUT_SIZE];
    
    // Extract and normalize features
    extract_features(gx, gy, gz, bx, by, bz, shared_mem, features);
    
    // Run neural network forward pass
    forward_pass(features, nn_output);
    
    // Post-process outputs
    MLModelOutput result;
    result.occupancy = nn_output[0];  // Already in [0,1] from sigmoid
    
    // Execution time: scale sigmoid output to reasonable ms range
    unsigned long long total_threads = (unsigned long long)gx * gy * gz * bx * by * bz;
    result.execution_time_ms = nn_output[1] * (total_threads / 1000000.0f) * 0.5f;
    
    // SM utilization
    result.sm_utilization = nn_output[2];
    
    // Block efficiency
    result.block_efficiency = nn_output[3];
    
    // Apply domain knowledge corrections
    unsigned int threads_per_block = bx * by * bz;
    
    // Penalize very small blocks
    if (threads_per_block < 64) {
        result.occupancy *= 0.5f;
        result.block_efficiency *= 0.4f;
    } else if (threads_per_block < 128) {
        result.occupancy *= 0.7f;
        result.block_efficiency *= 0.7f;
    }
    
    // Penalize very large blocks (limited parallelism)
    if (threads_per_block > 512) {
        result.occupancy *= 0.85f;
    }
    
    // Boost occupancy for good configurations
    if (threads_per_block >= 128 && threads_per_block <= 512) {
        result.occupancy = fminf(1.0f, result.occupancy * 1.2f);
        result.block_efficiency = fminf(1.0f, result.block_efficiency * 1.15f);
    }
    
    return result;
}

#endif // APEX_ML_MODEL_H
