#!/usr/bin/env python3
"""
APEX Interception Test
Tests if APEX is successfully intercepting CUDA calls
"""

import sys

# Test 1: Check if torch is available
try:
    import torch
    print("✓ PyTorch imported successfully")
except ImportError:
    print("✗ PyTorch not installed")
    print("Install with: pip install torch --break-system-packages")
    sys.exit(1)

# Test 2: Check if CUDA is available
if not torch.cuda.is_available():
    print("✗ CUDA not available")
    sys.exit(1)

print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")

# Test 3: Simple CUDA operation
print("\n[TEST] Running simple CUDA operations...")
print("=" * 60)

try:
    # Allocate tensors on GPU
    print("[1] Allocating tensors on GPU...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    
    print("[2] Performing matrix multiplication...")
    z = x @ y
    
    print("[3] Synchronizing...")
    torch.cuda.synchronize()
    
    print("[4] Copying result back to CPU...")
    result = z.cpu()
    
    print(f"\n✓ SUCCESS!")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {result.shape}")
    print(f"   Result sample: {result[0, 0].item():.4f}")
    
except Exception as e:
    print(f"\n✗ FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("If you see APEX interception messages above,")
print("then APEX is successfully intercepting CUDA calls!")
print("=" * 60)