#!/usr/bin/env python3
"""
Test APEX interception with PyTorch
This will launch real CUDA kernels from PyTorch operations
"""

import torch
import torch.nn as nn

print("=" * 60)
print("  APEX + PyTorch Kernel Interception Test")
print("=" * 60)
print()

# Check CUDA availability
if not torch.cuda.is_available():
    print("❌ CUDA not available!")
    exit(1)

device = torch.device('cuda')
print(f"✓ Using device: {torch.cuda.get_device_name(0)}")
print(f"✓ CUDA version: {torch.version.cuda}")
print()

# Simple matrix multiplication (launches multiple kernels)
print("🔥 Test 1: Matrix Multiplication")
print("-" * 60)
A = torch.randn(1024, 1024, device=device)
B = torch.randn(1024, 1024, device=device)
C = torch.matmul(A, B)
torch.cuda.synchronize()
print(f"✓ Matrix multiplication complete: {C.shape}")
print()

# Convolution operation (launches optimized CUDA kernels)
print("🔥 Test 2: Convolution (typical CNN operation)")
print("-" * 60)
conv = nn.Conv2d(3, 64, kernel_size=3, padding=1).cuda()
input_tensor = torch.randn(1, 3, 224, 224, device=device)
output = conv(input_tensor)
torch.cuda.synchronize()
print(f"✓ Convolution complete: {input_tensor.shape} -> {output.shape}")
print()

# Element-wise operations
print("🔥 Test 3: Element-wise operations (ReLU, Add)")
print("-" * 60)
x = torch.randn(10000, 10000, device=device)
y = torch.relu(x)
z = x + y
torch.cuda.synchronize()
print(f"✓ Element-wise operations complete: {z.shape}")
print()

# Reduction operation
print("🔥 Test 4: Reduction (Sum)")
print("-" * 60)
x = torch.randn(1000, 1000, 1000, device=device)
result = torch.sum(x)
torch.cuda.synchronize()
print(f"✓ Reduction complete: sum = {result.item():.6f}")
print()

print("=" * 60)
print("  ✅ All PyTorch tests completed!")
print("=" * 60)
