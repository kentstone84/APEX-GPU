/* ========================================================================== */
/*   APEX Profiling & Diagnostics Infrastructure                            */
/*   Performance tracking, debugging, and error diagnostics                  */
/* ========================================================================== */

#ifndef APEX_PROFILER_H
#define APEX_PROFILER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

/* ========================================================================== */
/* Configuration (set via environment variables)                             */
/* ========================================================================== */

typedef struct {
    int debug_enabled;       // APEX_DEBUG=1
    int trace_enabled;       // APEX_TRACE=1
    int profile_enabled;     // APEX_PROFILE=1
    int stats_enabled;       // APEX_STATS=1 (default: on)
    FILE* log_file;          // APEX_LOG_FILE=path
} ApexConfig;

static ApexConfig apex_config = {0};

/* ========================================================================== */
/* Performance Profiling                                                     */
/* ========================================================================== */

#define MAX_FUNCTIONS 100

typedef struct {
    const char* name;
    unsigned long call_count;
    double total_time_us;
    double min_time_us;
    double max_time_us;
} FunctionStats;

static FunctionStats function_stats[MAX_FUNCTIONS];
static int num_tracked_functions = 0;

/* ========================================================================== */
/* Memory Tracking                                                           */
/* ========================================================================== */

typedef struct {
    size_t total_allocated;
    size_t total_freed;
    size_t peak_usage;
    size_t current_usage;
    unsigned long alloc_count;
    unsigned long free_count;
} MemoryStats;

static MemoryStats memory_stats = {0};

/* ========================================================================== */
/* Timing Utilities                                                          */
/* ========================================================================== */

static inline double get_time_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
}

typedef struct {
    const char* func_name;
    double start_time;
    int func_index;
} ProfileScope;

static inline void profile_start(ProfileScope* scope, const char* func_name)
{
    if (!apex_config.profile_enabled) {
        scope->func_index = -1;
        return;
    }

    scope->func_name = func_name;
    scope->start_time = get_time_us();

    // Find or create function stats entry
    scope->func_index = -1;
    for (int i = 0; i < num_tracked_functions; i++) {
        if (strcmp(function_stats[i].name, func_name) == 0) {
            scope->func_index = i;
            break;
        }
    }

    if (scope->func_index == -1 && num_tracked_functions < MAX_FUNCTIONS) {
        scope->func_index = num_tracked_functions++;
        function_stats[scope->func_index].name = func_name;
        function_stats[scope->func_index].call_count = 0;
        function_stats[scope->func_index].total_time_us = 0;
        function_stats[scope->func_index].min_time_us = 1e9;
        function_stats[scope->func_index].max_time_us = 0;
    }
}

static inline void profile_end(ProfileScope* scope)
{
    if (!apex_config.profile_enabled || scope->func_index == -1) return;

    double elapsed = get_time_us() - scope->start_time;

    FunctionStats* stats = &function_stats[scope->func_index];
    stats->call_count++;
    stats->total_time_us += elapsed;
    if (elapsed < stats->min_time_us) stats->min_time_us = elapsed;
    if (elapsed > stats->max_time_us) stats->max_time_us = elapsed;
}

/* ========================================================================== */
/* Memory Tracking Functions                                                 */
/* ========================================================================== */

static inline void track_allocation(size_t size)
{
    memory_stats.total_allocated += size;
    memory_stats.current_usage += size;
    memory_stats.alloc_count++;

    if (memory_stats.current_usage > memory_stats.peak_usage) {
        memory_stats.peak_usage = memory_stats.current_usage;
    }

    if (apex_config.trace_enabled) {
        FILE* out = apex_config.log_file ? apex_config.log_file : stderr;
        fprintf(out, "[APEX-TRACE] Allocated %zu bytes (total: %zu, peak: %zu)\n",
                size, memory_stats.current_usage, memory_stats.peak_usage);
    }
}

static inline void track_free(size_t size)
{
    memory_stats.total_freed += size;
    if (memory_stats.current_usage >= size) {
        memory_stats.current_usage -= size;
    }
    memory_stats.free_count++;

    if (apex_config.trace_enabled) {
        FILE* out = apex_config.log_file ? apex_config.log_file : stderr;
        fprintf(out, "[APEX-TRACE] Freed %zu bytes (remaining: %zu)\n",
                size, memory_stats.current_usage);
    }
}

/* ========================================================================== */
/* Logging Macros                                                            */
/* ========================================================================== */

#define APEX_LOG(level, fmt, ...) do { \
    FILE* out = apex_config.log_file ? apex_config.log_file : stderr; \
    fprintf(out, "[APEX-%s] " fmt "\n", level, ##__VA_ARGS__); \
    if (apex_config.log_file) fflush(apex_config.log_file); \
} while(0)

#define APEX_DEBUG(fmt, ...) do { \
    if (apex_config.debug_enabled) { \
        APEX_LOG("DEBUG", fmt, ##__VA_ARGS__); \
    } \
} while(0)

#define APEX_TRACE(fmt, ...) do { \
    if (apex_config.trace_enabled) { \
        APEX_LOG("TRACE", fmt, ##__VA_ARGS__); \
    } \
} while(0)

#define APEX_ERROR(fmt, ...) APEX_LOG("ERROR", fmt, ##__VA_ARGS__)
#define APEX_WARN(fmt, ...) APEX_LOG("WARN", fmt, ##__VA_ARGS__)
#define APEX_INFO(fmt, ...) APEX_LOG("INFO", fmt, ##__VA_ARGS__)

/* ========================================================================== */
/* Configuration Initialization                                              */
/* ========================================================================== */

static inline void apex_init_config(void)
{
    // Check environment variables
    apex_config.debug_enabled = getenv("APEX_DEBUG") != NULL;
    apex_config.trace_enabled = getenv("APEX_TRACE") != NULL;
    apex_config.profile_enabled = getenv("APEX_PROFILE") != NULL;
    apex_config.stats_enabled = getenv("APEX_STATS") ? atoi(getenv("APEX_STATS")) : 1;

    // Open log file if specified
    const char* log_file = getenv("APEX_LOG_FILE");
    if (log_file) {
        apex_config.log_file = fopen(log_file, "a");
        if (!apex_config.log_file) {
            fprintf(stderr, "[APEX-WARN] Failed to open log file: %s\n", log_file);
        }
    }

    if (apex_config.debug_enabled) {
        APEX_DEBUG("Debug mode enabled");
        APEX_DEBUG("Trace: %s, Profile: %s, Stats: %s",
                   apex_config.trace_enabled ? "ON" : "OFF",
                   apex_config.profile_enabled ? "ON" : "OFF",
                   apex_config.stats_enabled ? "ON" : "OFF");
    }
}

static inline void apex_cleanup_config(void)
{
    if (apex_config.log_file) {
        fclose(apex_config.log_file);
        apex_config.log_file = NULL;
    }
}

/* ========================================================================== */
/* Statistics Reporting                                                      */
/* ========================================================================== */

static inline void apex_print_performance_stats(void)
{
    if (!apex_config.profile_enabled || num_tracked_functions == 0) return;

    FILE* out = apex_config.log_file ? apex_config.log_file : stderr;

    fprintf(out, "\n");
    fprintf(out, "╔════════════════════════════════════════════════════════════════════════════╗\n");
    fprintf(out, "║                      APEX PERFORMANCE PROFILE                              ║\n");
    fprintf(out, "╠════════════════════════════════════════════════════════════════════════════╣\n");
    fprintf(out, "║ Function                    │ Calls    │ Total(ms) │ Avg(μs) │ Min │  Max  ║\n");
    fprintf(out, "╠════════════════════════════════════════════════════════════════════════════╣\n");

    for (int i = 0; i < num_tracked_functions; i++) {
        FunctionStats* s = &function_stats[i];
        double avg_us = s->call_count > 0 ? s->total_time_us / s->call_count : 0;

        fprintf(out, "║ %-27s │ %8lu │ %9.3f │ %7.2f │ %3.0f │ %6.0f ║\n",
                s->name,
                s->call_count,
                s->total_time_us / 1000.0,  // Convert to ms
                avg_us,
                s->min_time_us,
                s->max_time_us);
    }

    fprintf(out, "╚════════════════════════════════════════════════════════════════════════════╝\n");
    fprintf(out, "\n");
}

static inline void apex_print_memory_stats(void)
{
    if (!apex_config.stats_enabled) return;

    FILE* out = apex_config.log_file ? apex_config.log_file : stderr;

    fprintf(out, "╔════════════════════════════════════════════════════════════════════════════╗\n");
    fprintf(out, "║                        APEX MEMORY STATISTICS                              ║\n");
    fprintf(out, "╠════════════════════════════════════════════════════════════════════════════╣\n");
    fprintf(out, "║  Total Allocated:     %12zu bytes  (%8.2f MB)                      ║\n",
            memory_stats.total_allocated,
            memory_stats.total_allocated / (1024.0 * 1024.0));
    fprintf(out, "║  Total Freed:         %12zu bytes  (%8.2f MB)                      ║\n",
            memory_stats.total_freed,
            memory_stats.total_freed / (1024.0 * 1024.0));
    fprintf(out, "║  Peak Usage:          %12zu bytes  (%8.2f MB)                      ║\n",
            memory_stats.peak_usage,
            memory_stats.peak_usage / (1024.0 * 1024.0));
    fprintf(out, "║  Current Usage:       %12zu bytes  (%8.2f MB)                      ║\n",
            memory_stats.current_usage,
            memory_stats.current_usage / (1024.0 * 1024.0));
    fprintf(out, "║  Allocations:         %12lu                                           ║\n",
            memory_stats.alloc_count);
    fprintf(out, "║  Frees:               %12lu                                           ║\n",
            memory_stats.free_count);

    if (memory_stats.current_usage > 0) {
        fprintf(out, "║  ⚠️  Memory Leak:      %12zu bytes  (NOT FREED)                     ║\n",
                memory_stats.current_usage);
    }

    fprintf(out, "╚════════════════════════════════════════════════════════════════════════════╝\n");
}

/* ========================================================================== */
/* Convenience Macro for Profiling Functions                                */
/* ========================================================================== */

#define APEX_PROFILE_FUNCTION() \
    ProfileScope __apex_profile_scope__; \
    profile_start(&__apex_profile_scope__, __FUNCTION__)

#define APEX_PROFILE_END() \
    profile_end(&__apex_profile_scope__)

#endif // APEX_PROFILER_H
