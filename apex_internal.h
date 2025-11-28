/**
 * APEX GPU Driver - Internal Structures
 * 
 * Internal data structures and constants not exposed in public API.
 * This file should NOT be included by user applications.
 */

#ifndef APEX_INTERNAL_H
#define APEX_INTERNAL_H

#include "apex.h"
#include <stdatomic.h>
#include <pthread.h>
#include <sys/time.h>

/**
 * Ring Buffer Configuration
 */
#define RING_BUFFER_SIZE 4096        // Must be power of 2
#define RING_BUFFER_MASK (RING_BUFFER_SIZE - 1)
#define COMMAND_SIZE 64              // bytes per command

/**
 * Hardware Configuration
 */
#define MAX_DEVICES 16
#define MAX_STREAMS_PER_CONTEXT 256
#define MAX_EVENTS_PER_CONTEXT 1024
#define MAX_GPC_CLUSTERS 18          // For GB100/RTX 5090
#define MAX_SMS_PER_GPC 8
#define TOTAL_SMS (MAX_GPC_CLUSTERS * MAX_SMS_PER_GPC)  // 144

/**
 * Memory Configuration
 */
#define PAGE_SIZE 4096
#define PREFETCH_WINDOW_MS 10        // Prefetch 10ms ahead
#define MEMORY_POOL_SIZE (1ULL << 30) // 1GB initial pool

/**
 * ML Configuration
 */
#define FEATURE_VECTOR_SIZE 128
#define ML_PREDICTION_HORIZON 10     // Predict next 10 kernels
#define ML_UPDATE_FREQUENCY 100      // Update model every 100 launches

/**
 * Command Types (for ring buffer)
 */
typedef enum {
    APEX_CMD_NOP = 0,
    APEX_CMD_LAUNCH_KERNEL = 1,
    APEX_CMD_MEMCPY_ASYNC = 2,
    APEX_CMD_MEMSET_ASYNC = 3,
    APEX_CMD_MALLOC = 4,
    APEX_CMD_FREE = 5,
    APEX_CMD_EVENT_RECORD = 6,
    APEX_CMD_STREAM_WAIT_EVENT = 7,
} ApexCommandType;

/**
 * Command Structure (64 bytes, cache-line aligned)
 */
typedef struct __attribute__((aligned(64))) {
    // Header (8 bytes)
    uint32_t command_type;           // ApexCommandType
    uint32_t sequence;               // Sequence number
    
    // Timing (8 bytes)
    uint64_t timestamp;              // CPU timestamp (for profiling)
    
    // Kernel launch parameters (32 bytes)
    uint64_t kernel_addr;            // Kernel function address
    uint32_t grid_x, grid_y, grid_z;
    uint32_t block_x, block_y, block_z;
    uint64_t args_ptr;               // Pointer to kernel arguments
    uint32_t shared_mem;             // Shared memory per block
    uint32_t stream_id;              // Stream ID
    
    // Memory operation parameters (16 bytes)
    uint64_t src_addr;
    uint64_t dst_addr;
    uint64_t size;
    uint32_t value;                  // For memset
    uint32_t kind;                   // ApexMemcpyKind
    
    // Result pointer (for malloc, etc.)
    uint64_t result_addr;
    
    // Padding to 64 bytes
    uint8_t padding[0];
} ApexCommand;

/**
 * Ring Buffer Structure (shared with GPU)
 */
typedef struct __attribute__((aligned(4096))) {
    // Producer/consumer indices (cache-line aligned)
    atomic_uint_fast32_t producer_index __attribute__((aligned(64)));
    atomic_uint_fast32_t consumer_index __attribute__((aligned(64)));
    
    // Command slots
    ApexCommand commands[RING_BUFFER_SIZE];
    
} ApexRingBuffer;

/**
 * ML Feature Vector (for scheduler)
 */
typedef struct {
    // Kernel signature (32D)
    uint64_t kernel_hash;
    uint32_t grid_size;
    uint32_t block_size;
    uint32_t shared_mem;
    float compute_intensity;         // Estimated FLOPS
    float memory_intensity;          // Estimated bandwidth
    float register_pressure;
    uint32_t kernel_type;            // 0=compute, 1=memory, 2=mixed
    float padding1[24];
    
    // Memory access pattern (32D)
    float read_bandwidth;
    float write_bandwidth;
    float l1_hit_rate;
    float l2_hit_rate;
    float stride_pattern;            // -1=random, 0=sequential, >0=strided
    uint32_t access_count;
    float temporal_locality;
    float spatial_locality;
    float padding2[24];
    
    // SM utilization history (32D)
    float sm_utilization[MAX_GPC_CLUSTERS];
    float sm_variance;
    float load_imbalance;
    float padding3[12];
    
    // Temperature per GPC (18D + padding)
    float temperature[MAX_GPC_CLUSTERS];
    float padding4[14];
    
    // Power metrics (8D)
    float power_watts;
    float voltage;
    float frequency_ghz;
    float energy_efficiency;        // GFLOPS/W
    float padding5[4];
    
    // Previous kernel sequence (6D)
    uint64_t prev_kernels[6];
    
} ApexFeatureVector;

/**
 * ML Schedule Decision
 */
typedef struct {
    // Prefetch decisions
    uint64_t prefetch_addrs[256];
    uint32_t prefetch_sizes[256];
    uint32_t prefetch_count;
    
    // SM assignment (which SMs for each workgroup)
    uint8_t sm_assignment[RING_BUFFER_SIZE];
    
    // Frequency scaling per GPC
    float gpc_frequencies[MAX_GPC_CLUSTERS];
    
    // Speculative execution
    bool speculative_launch;
    float confidence;
    
} ApexScheduleDecision;

/**
 * Memory Page Info (for prefetcher)
 */
typedef struct {
    uint64_t address;
    uint32_t size;
    uint64_t last_access_time;
    uint32_t access_count;
    float access_probability;       // From transformer predictor
    bool resident_in_gpu;
} ApexPageInfo;

/**
 * Device Structure (internal)
 */
struct ApexDevice_st {
    int ordinal;
    ApexDeviceProp properties;
    
    // Hardware info
    int pci_bus_id;
    int pci_device_id;
    void* pci_bar0;                 // PCIe BAR0 mapping
    void* pci_bar1;                 // PCIe BAR1 mapping
    
    // Ring buffer (memory-mapped)
    ApexRingBuffer* ring_buffer;
    volatile uint32_t* doorbell_register;
    
    // Statistics
    atomic_uint_fast64_t kernel_launches;
    atomic_uint_fast64_t memory_transfers;
    atomic_uint_fast64_t prefetch_hits;
    atomic_uint_fast64_t prefetch_misses;
    atomic_uint_fast64_t page_faults;
    
    // ML models (loaded if enabled)
    void* ml_scheduler_model;        // PyTorch JIT model
    void* memory_predictor_model;    // PyTorch JIT model
    
    // Power management
    float current_power_watts;
    float current_temperature_celsius;
    float gpc_frequencies[MAX_GPC_CLUSTERS];
    ApexPowerMode power_mode;
    
    // Configuration flags
    bool ml_scheduling_enabled;
    bool memory_prefetching_enabled;
    bool per_gpc_dvfs_enabled;
    bool arm_offload_enabled;
    
    pthread_mutex_t lock;
};

/**
 * Context Structure (internal)
 */
struct ApexContext_st {
    ApexDevice device;
    uint32_t flags;
    
    // Memory management
    void* device_memory_pool;
    size_t device_memory_size;
    void* managed_memory_pool;
    size_t managed_memory_size;
    
    // Streams
    ApexStream streams[MAX_STREAMS_PER_CONTEXT];
    int stream_count;
    
    // Events
    ApexEvent events[MAX_EVENTS_PER_CONTEXT];
    int event_count;
    
    pthread_mutex_t lock;
};

/**
 * Stream Structure (internal)
 */
struct ApexStream_st {
    uint32_t stream_id;
    ApexContext context;
    int priority;
    
    // Completion tracking
    atomic_uint_fast32_t last_submitted_sequence;
    atomic_uint_fast32_t last_completed_sequence;
    
    pthread_mutex_t lock;
};

/**
 * Event Structure (internal)
 */
struct ApexEvent_st {
    uint32_t event_id;
    ApexContext context;
    ApexStream stream;
    
    // Timing
    struct timeval recorded_time;
    atomic_bool recorded;
    atomic_bool completed;
    
    pthread_mutex_t lock;
};

/**
 * Global State
 */
typedef struct {
    bool initialized;
    uint32_t init_flags;
    
    // Devices
    ApexDevice devices[MAX_DEVICES];
    int device_count;
    
    // Current context
    ApexContext current_context;
    
    // Global statistics
    atomic_uint_fast64_t total_kernel_launches;
    atomic_uint_fast64_t total_memory_transfers;
    
    pthread_mutex_t global_lock;
} ApexGlobalState;

/**
 * Global state instance (defined in apex.c)
 */
extern ApexGlobalState g_apex_state;

/**
 * Internal Functions
 */

// Ring buffer operations
ApexError apex_ring_buffer_init(ApexDevice device);
ApexError apex_ring_buffer_destroy(ApexDevice device);
ApexError apex_ring_buffer_submit(ApexDevice device, ApexCommand* cmd);

// Memory management
ApexError apex_memory_init(ApexContext ctx);
ApexError apex_memory_destroy(ApexContext ctx);
void* apex_malloc_internal(ApexContext ctx, size_t size);
void apex_free_internal(ApexContext ctx, void* ptr);

// ML scheduler
ApexError apex_ml_scheduler_init(ApexDevice device);
ApexError apex_ml_scheduler_destroy(ApexDevice device);
ApexScheduleDecision apex_ml_schedule(ApexDevice device, ApexCommand* cmd);
void apex_ml_update(ApexDevice device, ApexCommand* cmd, ApexScheduleDecision* decision);

// Memory predictor
ApexError apex_memory_predictor_init(ApexDevice device);
ApexError apex_memory_predictor_destroy(ApexDevice device);
void apex_predict_prefetch(ApexDevice device, ApexCommand* cmd, ApexScheduleDecision* decision);

// Power management
ApexError apex_dvfs_init(ApexDevice device);
ApexError apex_dvfs_update(ApexDevice device, ApexScheduleDecision* decision);
void apex_set_gpc_frequency(ApexDevice device, int gpc_id, float freq_ghz);

// Hardware abstraction
ApexError apex_hardware_init(ApexDevice device);
ApexError apex_hardware_destroy(ApexDevice device);
void* apex_map_pci_bar(int bus, int device, int bar);
void apex_unmap_pci_bar(void* addr);
void apex_write_doorbell(ApexDevice device, uint32_t value);

// Timing utilities
static inline uint64_t apex_get_timestamp_ns(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000000ULL + (uint64_t)tv.tv_usec * 1000ULL;
}

// Atomic helpers
static inline uint32_t apex_atomic_inc(atomic_uint_fast32_t* ptr) {
    return atomic_fetch_add(ptr, 1);
}

static inline uint32_t apex_atomic_load(atomic_uint_fast32_t* ptr) {
    return atomic_load(ptr);
}

// Simulation mode (when no real hardware available)
#ifndef APEX_REAL_HARDWARE
#define APEX_SIMULATION_MODE 1
void apex_simulate_kernel_execution(ApexCommand* cmd);
void apex_simulate_memory_transfer(ApexCommand* cmd);
#endif

#endif // APEX_INTERNAL_H