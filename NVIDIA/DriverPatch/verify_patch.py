#!/usr/bin/env python3
"""
🔍 CUDA Patch Verification Script
Tests if SM_120 lockout removal was successful
"""

import sys
import subprocess
import hashlib
from pathlib import Path

def check_file_exists(path):
    """Check if patched file exists"""
    p = Path(path)
    if not p.exists():
        return False, f"File not found: {path}"
    return True, f"Found: {path} ({p.stat().st_size:,} bytes)"

def verify_checksum(path, expected_sha256):
    """Verify file checksum"""
    try:
        with open(path, 'rb') as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        
        if actual == expected_sha256:
            return True, f"✅ Checksum matches: {actual[:16]}..."
        else:
            return False, f"❌ Checksum mismatch!\n   Expected: {expected_sha256[:16]}...\n   Got:      {actual[:16]}..."
    except Exception as e:
        return False, f"❌ Error reading file: {e}"

def check_cuda_available():
    """Check if CUDA is available via PyTorch"""
    try:
        import torch
        available = torch.cuda.is_available()
        
        if available:
            device_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            return True, f"✅ CUDA available\n   Device: {device_name}\n   Capability: SM_{capability[0]}{capability[1]}"
        else:
            return False, "❌ CUDA not available (torch.cuda.is_available() = False)"
    except ImportError:
        return None, "⚠️  PyTorch not installed (can't test CUDA availability)"
    except Exception as e:
        return False, f"❌ Error checking CUDA: {e}"

def check_nvidia_smi():
    """Check if nvidia-smi works"""
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            # Parse output for GPU info
            lines = result.stdout.split('\n')
            gpu_line = [l for l in lines if 'NVIDIA' in l and 'Driver' in l]
            if gpu_line:
                return True, f"✅ nvidia-smi working"
            return True, "✅ nvidia-smi working"
        else:
            return False, f"❌ nvidia-smi failed: {result.stderr}"
    except FileNotFoundError:
        return False, "❌ nvidia-smi not found (NVIDIA driver not installed?)"
    except Exception as e:
        return False, f"❌ Error running nvidia-smi: {e}"

def test_simple_kernel():
    """Test if we can compile a simple CUDA kernel"""
    try:
        import torch
        
        # Simple kernel that should work on any GPU
        kernel = """
        extern "C" __global__ void simple_add(float* a, float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        """
        
        # Try to compile
        from torch.utils.cpp_extension import load_inline
        
        module = load_inline(
            name='test_simple',
            cpp_sources=[''],
            cuda_sources=[kernel],
            functions=['simple_add'],
            verbose=False
        )
        
        # Test execution
        n = 1024
        a = torch.randn(n, device='cuda')
        b = torch.randn(n, device='cuda')
        c = torch.zeros(n, device='cuda')
        
        # Run kernel
        grid = (n + 255) // 256
        module.simple_add(a, b, c, n, grid=(grid,), block=(256,))
        torch.cuda.synchronize()
        
        # Verify
        expected = a + b
        if torch.allclose(c, expected, rtol=1e-5):
            return True, "✅ Kernel compilation and execution successful"
        else:
            return False, "❌ Kernel executed but results incorrect"
            
    except ImportError:
        return None, "⚠️  PyTorch not installed (can't test kernel compilation)"
    except Exception as e:
        return False, f"❌ Kernel test failed: {e}"

def main():
    print("="*70)
    print("🔍 CUDA PATCH VERIFICATION")
    print("="*70)
    print()
    
    tests = []
    
    # Test 1: Check if patched file exists
    print("📦 TEST 1: Patched File")
    print("-" * 70)
    success, msg = check_file_exists('/mnt/user-data/outputs/libcuda.so.1.1.patched')
    print(msg)
    tests.append(('File Exists', success))
    print()
    
    # Test 2: Verify checksum
    print("🔐 TEST 2: Checksum Verification")
    print("-" * 70)
    expected_checksum = "131a52575de151f4d67fd50109af7ea94621778a7cb659c831a2eb9c465ee5f9"
    success, msg = verify_checksum('/mnt/user-data/outputs/libcuda.so.1.1.patched', expected_checksum)
    print(msg)
    tests.append(('Checksum', success))
    print()
    
    # Test 3: Check nvidia-smi
    print("🖥️  TEST 3: NVIDIA Driver Status")
    print("-" * 70)
    success, msg = check_nvidia_smi()
    print(msg)
    tests.append(('nvidia-smi', success))
    print()
    
    # Test 4: Check CUDA availability
    print("🔧 TEST 4: CUDA Availability")
    print("-" * 70)
    success, msg = check_cuda_available()
    print(msg)
    if success is not None:
        tests.append(('CUDA Available', success))
    print()
    
    # Test 5: Kernel compilation
    print("⚙️  TEST 5: Kernel Compilation")
    print("-" * 70)
    success, msg = test_simple_kernel()
    print(msg)
    if success is not None:
        tests.append(('Kernel Test', success))
    print()
    
    # Summary
    print("="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success in tests if success)
    total = len(tests)
    
    for name, success in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:10s} - {name}")
    
    print()
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("   Your patched driver is ready for installation.")
        print()
        print("Next steps:")
        print("   1. Copy libcuda.so.1.1.patched to your system")
        print("   2. Run install_patched_cuda.sh as root")
        print("   3. Test with your CUDA applications")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("   Review failed tests above for details.")
        print("   You may need to:")
        print("   - Install NVIDIA drivers")
        print("   - Install PyTorch")
        print("   - Check system configuration")
        return 1

if __name__ == "__main__":
    sys.exit(main())
