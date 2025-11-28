sentinal@ServerAI:/mnt/c/Users/SentinalAI/Desktop/APEX GPU$ LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ./test_kernel_launch
═══════════════════════════════════════════════════
  APEX ML SCHEDULER - KERNEL LAUNCH TEST
═══════════════════════════════════════════════════

[TEST 1] Vector Addition (1M elements)
─────────────────────────────────────────────────

[APEX-ML] ╔═══════════════════════════════════════════╗
[APEX-ML] ║  APEX GPU DRIVER - ML SCHEDULER MODE     ║
[APEX-ML] ║  1,808,641 Parameters Ready               ║
[APEX-ML] ╚═══════════════════════════════════════════╝


[APEX-ML] ═══════════════════════════════════════════
[APEX-ML] ML SCHEDULER PERFORMANCE STATISTICS
[APEX-ML] ═══════════════════════════════════════════
[APEX-ML] Total ML predictions: 0
[APEX-ML] ═══════════════════════════════════════════

  Grid: (4096, 1, 1)
  Block: (256, 1, 1)
  Launching kernel...

  ✓ Kernel completed

[TEST 2] Matrix Multiplication (1024x1024)
─────────────────────────────────────────────────
  Grid: (64, 64, 1)
  Block: (16, 16, 1)
  Total threads: 1048576
  Launching kernel...

  ✓ Kernel completed

[TEST 3] Multiple Small Kernels (10 iterations)
─────────────────────────────────────────────────
  Grid: (79, 1, 1)
  Block: (128, 1, 1)
  Launching 10 kernels...

  ✓ All kernels completed

═══════════════════════════════════════════════════
  ALL TESTS PASSED
═══════════════════════════════════════════════════