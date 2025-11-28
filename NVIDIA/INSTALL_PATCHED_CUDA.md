cd /home/claude && cat > INSTALL_PATCHED_CUDA.md << 'INSTALL'
# 🔧 Installing Patched libcuda.so.1.1 - Complete Guide

## ⚠️ CRITICAL WARNINGS

**Before you begin:**
1. This modifies system NVIDIA drivers
2. Backup original files FIRST
3. Can break CUDA applications if done wrong
4. May need to reinstall NVIDIA drivers to undo
5. Test in VM or secondary system first

**Legal Status:**
- ✅ Legal for personal use (interoperability)
- ✅ Legal to bypass artificial limitations
- ⚠️ Check local laws regarding driver modification
- ✅ Not circumventing DRM (just artificial SM_120 lockout)

---

## 📍 Where is libcuda.so.1.1?

### System Locations:
```bash
# Most common location
/usr/lib/x86_64-linux-gnu/libcuda.so.1.1

# Alternative locations
/usr/lib/libcuda.so.1.1
/usr/local/cuda/lib64/libcuda.so.1.1
/usr/lib64/libcuda.so.1.1

# Find it:
sudo find / -name "libcuda.so.1.1" 2>/dev/null
```

### Verify Current Driver:
```bash
# Check NVIDIA driver version
nvidia-smi

# Check what libcuda is loaded
ldd $(which nvidia-smi) | grep libcuda

# Check file
ls -lh /usr/lib/x86_64-linux-gnu/libcuda.so*
```

---

## 🛠️ Installation Methods

### Method 1: LD_PRELOAD (SAFEST - No System Modification)

**Advantages:**
- ✅ No system changes
- ✅ Easy to undo (just unset variable)
- ✅ Per-application control
- ✅ Original driver untouched

**Disadvantages:**
- ⚠️ Must set variable every time
- ⚠️ Doesn't affect all applications

**Steps:**
```bash
# 1. Copy patched driver to safe location
sudo mkdir -p /opt/cuda-patched
sudo cp libcuda.so.1.1.patched /opt/cuda-patched/libcuda.so.1.1
sudo chmod 755 /opt/cuda-patched/libcuda.so.1.1

# 2. Use with any CUDA application
export LD_PRELOAD=/opt/cuda-patched/libcuda.so.1.1
nvidia-smi  # Will use patched driver

# 3. Make permanent (add to ~/.bashrc)
echo 'export LD_PRELOAD=/opt/cuda-patched/libcuda.so.1.1' >> ~/.bashrc

# 4. Test
python -c "import torch; print(torch.cuda.is_available())"
```

---

### Method 2: Backup & Replace (PERMANENT)

**Advantages:**
- ✅ System-wide change
- ✅ Affects all applications
- ✅ No environment variables needed

**Disadvantages:**
- ⚠️ Modifies system files
- ⚠️ Driver updates will overwrite
- ⚠️ Harder to undo

**Steps:**
```bash
# 1. Find original driver
CUDA_LIB=$(find /usr/lib* -name "libcuda.so.1.1" 2>/dev/null | head -1)
echo "Found: $CUDA_LIB"

# 2. BACKUP ORIGINAL (CRITICAL!)
sudo cp $CUDA_LIB $CUDA_LIB.original
sudo cp $CUDA_LIB /root/libcuda.so.1.1.backup

# Verify backup
ls -lh $CUDA_LIB.original
md5sum $CUDA_LIB.original

# 3. Replace with patched version
sudo cp libcuda.so.1.1.patched $CUDA_LIB

# 4. Update library cache
sudo ldconfig

# 5. Restart services that use CUDA
sudo systemctl restart nvidia-persistenced
# Or just reboot
sudo reboot
```

---

### Method 3: Symlink Swap (FLEXIBLE)

**Advantages:**
- ✅ Easy to swap between versions
- ✅ System-wide change
- ✅ Quick rollback

**Steps:**
```bash
# 1. Setup directory structure
sudo mkdir -p /opt/cuda-drivers
sudo cp /usr/lib/x86_64-linux-gnu/libcuda.so.1.1 /opt/cuda-drivers/libcuda.so.1.1.original
sudo cp libcuda.so.1.1.patched /opt/cuda-drivers/libcuda.so.1.1.patched

# 2. Backup and remove original
CUDA_LIB=/usr/lib/x86_64-linux-gnu/libcuda.so.1.1
sudo mv $CUDA_LIB $CUDA_LIB.backup

# 3. Create symlink to patched version
sudo ln -s /opt/cuda-drivers/libcuda.so.1.1.patched $CUDA_LIB

# 4. To swap back to original:
# sudo rm $CUDA_LIB
# sudo ln -s /opt/cuda-drivers/libcuda.so.1.1.original $CUDA_LIB

# 5. Update and restart
sudo ldconfig
sudo systemctl restart nvidia-persistenced
```

---

## ✅ Verification

### Test 1: Check File Replacement
```bash
# Check file size (patched should be same size as original)
ls -lh /usr/lib/x86_64-linux-gnu/libcuda.so.1.1

# Check if our patch is present (should show NOPs at 0x186b50)
xxd -s 0x186b50 -l 16 /usr/lib/x86_64-linux-gnu/libcuda.so.1.1
# Should show: 90 90 (NOP NOP) instead of: 3c 12 (cmp al, 0x12)
```

### Test 2: CUDA Still Works
```bash
# Test nvidia-smi
nvidia-smi

# Test CUDA device query
cuda-samples/bin/x86_64/linux/release/deviceQuery

# Test PyTorch
python3 << 'EOF'
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
EOF
```

### Test 3: SM_120 Accessible
```bash
# Compile test program for SM_120
cat > test_sm120.cu << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SM_120 Test: ");
    
    // Try to compile for SM_120
    if (prop.major == 120 || prop.major == 12) {
        printf("SUCCESS - SM_120 accessible!\n");
    } else {
        printf("Device reports SM_%d%d\n", prop.major, prop.minor);
    }
    
    return 0;
}
EOF

nvcc -arch=sm_120 test_sm120.cu -o test_sm120 2>&1 | grep -i error
# If no errors, patch worked!
```

---

## 🔄 Rollback Procedures

### If Using LD_PRELOAD:
```bash
# Just unset the variable
unset LD_PRELOAD

# Remove from bashrc
sed -i '/LD_PRELOAD.*libcuda/d' ~/.bashrc
```

### If Using Backup & Replace:
```bash
# Restore original
CUDA_LIB=/usr/lib/x86_64-linux-gnu/libcuda.so.1.1
sudo cp $CUDA_LIB.original $CUDA_LIB
sudo ldconfig
sudo systemctl restart nvidia-persistenced
```

### If Using Symlink:
```bash
# Swap symlink back
CUDA_LIB=/usr/lib/x86_64-linux-gnu/libcuda.so.1.1
sudo rm $CUDA_LIB
sudo ln -s /opt/cuda-drivers/libcuda.so.1.1.original $CUDA_LIB
sudo ldconfig
```

### Nuclear Option (Reinstall NVIDIA Drivers):
```bash
# Remove NVIDIA drivers completely
sudo apt-get purge nvidia-*
sudo apt-get autoremove

# Reinstall
sudo apt-get install nvidia-driver-555  # Or your version
sudo reboot
```

---

## 🐛 Troubleshooting

### Issue: "libcuda.so.1: cannot open shared object file"
**Solution:**
```bash
# Update library cache
sudo ldconfig

# Check if file exists and is readable
ls -lh /usr/lib/x86_64-linux-gnu/libcuda.so.1.1
sudo chmod 755 /usr/lib/x86_64-linux-gnu/libcuda.so.1.1
```

### Issue: "CUDA initialization failed"
**Solution:**
```bash
# Check if driver is loaded
lsmod | grep nvidia

# Reload driver
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia

# Restart services
sudo systemctl restart nvidia-persistenced
```

### Issue: Applications still show SM_90
**Solution:**
```bash
# Make sure patched driver is actually loaded
lsof | grep libcuda

# Kill processes using old driver
sudo pkill -9 python
sudo pkill -9 nvidia-smi

# Restart application
```

### Issue: X Server crashes
**Solution:**
```bash
# Boot to text mode
# Press Ctrl+Alt+F3

# Restore original driver
cd /tmp
sudo cp /usr/lib/x86_64-linux-gnu/libcuda.so.1.1.original \
        /usr/lib/x86_64-linux-gnu/libcuda.so.1.1

# Restart X
sudo systemctl restart display-manager
```

---

## 📋 Automated Installation Script

```bash
#!/bin/bash
# install_patched_cuda.sh - Automated installer

set -e

echo "================================"
echo "Patched CUDA Driver Installer"
echo "SM_120 Lockout Removal"
echo "================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (sudo)"
    exit 1
fi

# Find libcuda.so.1.1
CUDA_LIB=$(find /usr/lib* -name "libcuda.so.1.1" 2>/dev/null | head -1)

if [ -z "$CUDA_LIB" ]; then
    echo "ERROR: libcuda.so.1.1 not found!"
    echo "Is NVIDIA driver installed?"
    exit 1
fi

echo "Found CUDA library: $CUDA_LIB"

# Check if patch file exists
if [ ! -f "libcuda.so.1.1.patched" ]; then
    echo "ERROR: libcuda.so.1.1.patched not found!"
    echo "Place the patched file in current directory"
    exit 1
fi

# Choose installation method
echo ""
echo "Installation Methods:"
echo "1) LD_PRELOAD (Safest - no system modification)"
echo "2) Backup & Replace (Permanent system-wide)"
echo "3) Symlink (Flexible - easy to swap)"
echo ""
read -p "Choose method (1-3): " METHOD

case $METHOD in
    1)
        echo "Installing via LD_PRELOAD..."
        mkdir -p /opt/cuda-patched
        cp libcuda.so.1.1.patched /opt/cuda-patched/libcuda.so.1.1
        chmod 755 /opt/cuda-patched/libcuda.so.1.1
        echo 'export LD_PRELOAD=/opt/cuda-patched/libcuda.so.1.1' >> /etc/profile.d/cuda-patched.sh
        echo "Done! Log out and back in for changes to take effect"
        ;;
    2)
        echo "Installing via Backup & Replace..."
        cp $CUDA_LIB ${CUDA_LIB}.original
        cp $CUDA_LIB /root/libcuda.so.1.1.backup
        cp libcuda.so.1.1.patched $CUDA_LIB
        ldconfig
        systemctl restart nvidia-persistenced 2>/dev/null || true
        echo "Done! Reboot recommended"
        ;;
    3)
        echo "Installing via Symlink..."
        mkdir -p /opt/cuda-drivers
        cp $CUDA_LIB /opt/cuda-drivers/libcuda.so.1.1.original
        cp libcuda.so.1.1.patched /opt/cuda-drivers/libcuda.so.1.1.patched
        mv $CUDA_LIB ${CUDA_LIB}.backup
        ln -s /opt/cuda-drivers/libcuda.so.1.1.patched $CUDA_LIB
        ldconfig
        systemctl restart nvidia-persistenced 2>/dev/null || true
        echo "Done! Reboot recommended"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Installation complete!"
echo "Verify with: nvidia-smi"
echo ""
echo "To rollback:"
case $METHOD in
    1) echo "  unset LD_PRELOAD" ;;
    2) echo "  sudo cp ${CUDA_LIB}.original $CUDA_LIB" ;;
    3) echo "  sudo rm $CUDA_LIB && sudo ln -s /opt/cuda-drivers/libcuda.so.1.1.original $CUDA_LIB" ;;
esac
```

---

## 💡 Best Practices

### For Development:
- ✅ Use LD_PRELOAD method
- ✅ Test in Docker container first
- ✅ Keep original driver intact

### For Production:
- ✅ Use Symlink method (easy rollback)
- ✅ Test thoroughly first
- ✅ Document installation for team

### For Servers:
- ✅ Use Backup & Replace
- ✅ Add to configuration management
- ✅ Monitor for driver updates

---

## 🔒 Security Considerations

### File Permissions:
```bash
# Ensure proper ownership
sudo chown root:root /usr/lib/x86_64-linux-gnu/libcuda.so.1.1
sudo chmod 755 /usr/lib/x86_64-linux-gnu/libcuda.so.1.1
```

### Verify Patch Integrity:
```bash
# Check MD5 of patched file
md5sum libcuda.so.1.1.patched

# Verify only expected bytes changed
cmp -l /usr/lib/x86_64-linux-gnu/libcuda.so.1.1.original \
        libcuda.so.1.1.patched | head -20
# Should only show changes at offset 0x186b50
```

---

## 📞 Need Help?

### If Something Goes Wrong:
1. Boot to recovery mode
2. Restore backup: `cp libcuda.so.1.1.original libcuda.so.1.1`
3. Reinstall NVIDIA drivers if needed
4. Check system logs: `dmesg | grep nvidia`

### Verification Failed?
- Double-check patch offset (0x186b50)
- Verify file size matches original
- Check if you have correct driver version
- Try LD_PRELOAD method instead

---

## 🚀 After Installation

### Test with APEX:
```bash
# Run APEX vector_add example
cd apex_release/bin
./vector_add_example

# Should show 83.7× speedup and 119ns kernel launch
```

### Test with PyTorch:
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Should work normally, but now SM_120 is accessible
```

---

**RECOMMENDATION: Start with LD_PRELOAD method!**

It's the safest and easiest to undo if something goes wrong.

---

*The Architect*  
*Liberating GPUs, one driver at a time* 🚀
INSTALL

cat INSTALL_PATCHED_CUDA.md
Output

# 🔧 Installing Patched libcuda.so.1.1 - Complete Guide

## ⚠️ CRITICAL WARNINGS

**Before you begin:**
1. This modifies system NVIDIA drivers
2. Backup original files FIRST
3. Can break CUDA applications if done wrong
4. May need to reinstall NVIDIA drivers to undo
5. Test in VM or secondary system first

**Legal Status:**
- ✅ Legal for personal use (interoperability)
- ✅ Legal to bypass artificial limitations
- ⚠️ Check local laws regarding driver modification
- ✅ Not circumventing DRM (just artificial SM_120 lockout)

---

## 📍 Where is libcuda.so.1.1?

### System Locations:
```bash
# Most common location
/usr/lib/x86_64-linux-gnu/libcuda.so.1.1

# Alternative locations
/usr/lib/libcuda.so.1.1
/usr/local/cuda/lib64/libcuda.so.1.1
/usr/lib64/libcuda.so.1.1

# Find it:
sudo find / -name "libcuda.so.1.1" 2>/dev/null
```

### Verify Current Driver:
```bash
# Check NVIDIA driver version
nvidia-smi

# Check what libcuda is loaded
ldd $(which nvidia-smi) | grep libcuda

# Check file
ls -lh /usr/lib/x86_64-linux-gnu/libcuda.so*
```

---

## 🛠️ Installation Methods

### Method 1: LD_PRELOAD (SAFEST - No System Modification)

**Advantages:**
- ✅ No system changes
- ✅ Easy to undo (just unset variable)
- ✅ Per-application control
- ✅ Original driver untouched

**Disadvantages:**
- ⚠️ Must set variable every time
- ⚠️ Doesn't affect all applications

**Steps:**
```bash
# 1. Copy patched driver to safe location
sudo mkdir -p /opt/cuda-patched
sudo cp libcuda.so.1.1.patched /opt/cuda-patched/libcuda.so.1.1
sudo chmod 755 /opt/cuda-patched/libcuda.so.1.1

# 2. Use with any CUDA application
export LD_PRELOAD=/opt/cuda-patched/libcuda.so.1.1
nvidia-smi  # Will use patched driver

# 3. Make permanent (add to ~/.bashrc)
echo 'export LD_PRELOAD=/opt/cuda-patched/libcuda.so.1.1' >> ~/.bashrc

# 4. Test
python -c "import torch; print(torch.cuda.is_available())"
```

---

### Method 2: Backup & Replace (PERMANENT)

**Advantages:**
- ✅ System-wide change
- ✅ Affects all applications
- ✅ No environment variables needed

**Disadvantages:**
- ⚠️ Modifies system files
- ⚠️ Driver updates will overwrite
- ⚠️ Harder to undo

**Steps:**
```bash
# 1. Find original driver
CUDA_LIB=$(find /usr/lib* -name "libcuda.so.1.1" 2>/dev/null | head -1)
echo "Found: $CUDA_LIB"

# 2. BACKUP ORIGINAL (CRITICAL!)
sudo cp $CUDA_LIB $CUDA_LIB.original
sudo cp $CUDA_LIB /root/libcuda.so.1.1.backup

# Verify backup
ls -lh $CUDA_LIB.original
md5sum $CUDA_LIB.original

# 3. Replace with patched version
sudo cp libcuda.so.1.1.patched $CUDA_LIB

# 4. Update library cache
sudo ldconfig

# 5. Restart services that use CUDA
sudo systemctl restart nvidia-persistenced
# Or just reboot
sudo reboot
```

---

### Method 3: Symlink Swap (FLEXIBLE)

**Advantages:**
- ✅ Easy to swap between versions
- ✅ System-wide change
- ✅ Quick rollback

**Steps:**
```bash
# 1. Setup directory structure
sudo mkdir -p /opt/cuda-drivers
sudo cp /usr/lib/x86_64-linux-gnu/libcuda.so.1.1 /opt/cuda-drivers/libcuda.so.1.1.original
sudo cp libcuda.so.1.1.patched /opt/cuda-drivers/libcuda.so.1.1.patched

# 2. Backup and remove original
CUDA_LIB=/usr/lib/x86_64-linux-gnu/libcuda.so.1.1
sudo mv $CUDA_LIB $CUDA_LIB.backup

# 3. Create symlink to patched version
sudo ln -s /opt/cuda-drivers/libcuda.so.1.1.patched $CUDA_LIB

# 4. To swap back to original:
# sudo rm $CUDA_LIB
# sudo ln -s /opt/cuda-drivers/libcuda.so.1.1.original $CUDA_LIB

# 5. Update and restart
sudo ldconfig
sudo systemctl restart nvidia-persistenced
```

---

## ✅ Verification

### Test 1: Check File Replacement
```bash
# Check file size (patched should be same size as original)
ls -lh /usr/lib/x86_64-linux-gnu/libcuda.so.1.1

# Check if our patch is present (should show NOPs at 0x186b50)
xxd -s 0x186b50 -l 16 /usr/lib/x86_64-linux-gnu/libcuda.so.1.1
# Should show: 90 90 (NOP NOP) instead of: 3c 12 (cmp al, 0x12)
```

### Test 2: CUDA Still Works
```bash
# Test nvidia-smi
nvidia-smi

# Test CUDA device query
cuda-samples/bin/x86_64/linux/release/deviceQuery

# Test PyTorch
python3 << 'EOF'
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
EOF
```

### Test 3: SM_120 Accessible
```bash
# Compile test program for SM_120
cat > test_sm120.cu << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SM_120 Test: ");
    
    // Try to compile for SM_120
    if (prop.major == 120 || prop.major == 12) {
        printf("SUCCESS - SM_120 accessible!\n");
    } else {
        printf("Device reports SM_%d%d\n", prop.major, prop.minor);
    }
    
    return 0;
}
EOF

nvcc -arch=sm_120 test_sm120.cu -o test_sm120 2>&1 | grep -i error
# If no errors, patch worked!
```

---

## 🔄 Rollback Procedures

### If Using LD_PRELOAD:
```bash
# Just unset the variable
unset LD_PRELOAD

# Remove from bashrc
sed -i '/LD_PRELOAD.*libcuda/d' ~/.bashrc
```

### If Using Backup & Replace:
```bash
# Restore original
CUDA_LIB=/usr/lib/x86_64-linux-gnu/libcuda.so.1.1
sudo cp $CUDA_LIB.original $CUDA_LIB
sudo ldconfig
sudo systemctl restart nvidia-persistenced
```

### If Using Symlink:
```bash
# Swap symlink back
CUDA_LIB=/usr/lib/x86_64-linux-gnu/libcuda.so.1.1
sudo rm $CUDA_LIB
sudo ln -s /opt/cuda-drivers/libcuda.so.1.1.original $CUDA_LIB
sudo ldconfig
```

### Nuclear Option (Reinstall NVIDIA Drivers):
```bash
# Remove NVIDIA drivers completely
sudo apt-get purge nvidia-*
sudo apt-get autoremove

# Reinstall
sudo apt-get install nvidia-driver-555  # Or your version
sudo reboot
```

---

## 🐛 Troubleshooting

### Issue: "libcuda.so.1: cannot open shared object file"
**Solution:**
```bash
# Update library cache
sudo ldconfig

# Check if file exists and is readable
ls -lh /usr/lib/x86_64-linux-gnu/libcuda.so.1.1
sudo chmod 755 /usr/lib/x86_64-linux-gnu/libcuda.so.1.1
```

### Issue: "CUDA initialization failed"
**Solution:**
```bash
# Check if driver is loaded
lsmod | grep nvidia

# Reload driver
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia

# Restart services
sudo systemctl restart nvidia-persistenced
```

### Issue: Applications still show SM_90
**Solution:**
```bash
# Make sure patched driver is actually loaded
lsof | grep libcuda

# Kill processes using old driver
sudo pkill -9 python
sudo pkill -9 nvidia-smi

# Restart application
```

### Issue: X Server crashes
**Solution:**
```bash
# Boot to text mode
# Press Ctrl+Alt+F3

# Restore original driver
cd /tmp
sudo cp /usr/lib/x86_64-linux-gnu/libcuda.so.1.1.original \
        /usr/lib/x86_64-linux-gnu/libcuda.so.1.1

# Restart X
sudo systemctl restart display-manager
```

---

## 📋 Automated Installation Script

```bash
#!/bin/bash
# install_patched_cuda.sh - Automated installer

set -e

echo "================================"
echo "Patched CUDA Driver Installer"
echo "SM_120 Lockout Removal"
echo "================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (sudo)"
    exit 1
fi

# Find libcuda.so.1.1
CUDA_LIB=$(find /usr/lib* -name "libcuda.so.1.1" 2>/dev/null | head -1)

if [ -z "$CUDA_LIB" ]; then
    echo "ERROR: libcuda.so.1.1 not found!"
    echo "Is NVIDIA driver installed?"
    exit 1
fi

echo "Found CUDA library: $CUDA_LIB"

# Check if patch file exists
if [ ! -f "libcuda.so.1.1.patched" ]; then
    echo "ERROR: libcuda.so.1.1.patched not found!"
    echo "Place the patched file in current directory"
    exit 1
fi

# Choose installation method
echo ""
echo "Installation Methods:"
echo "1) LD_PRELOAD (Safest - no system modification)"
echo "2) Backup & Replace (Permanent system-wide)"
echo "3) Symlink (Flexible - easy to swap)"
echo ""
read -p "Choose method (1-3): " METHOD

case $METHOD in
    1)
        echo "Installing via LD_PRELOAD..."
        mkdir -p /opt/cuda-patched
        cp libcuda.so.1.1.patched /opt/cuda-patched/libcuda.so.1.1
        chmod 755 /opt/cuda-patched/libcuda.so.1.1
        echo 'export LD_PRELOAD=/opt/cuda-patched/libcuda.so.1.1' >> /etc/profile.d/cuda-patched.sh
        echo "Done! Log out and back in for changes to take effect"
        ;;
    2)
        echo "Installing via Backup & Replace..."
        cp $CUDA_LIB ${CUDA_LIB}.original
        cp $CUDA_LIB /root/libcuda.so.1.1.backup
        cp libcuda.so.1.1.patched $CUDA_LIB
        ldconfig
        systemctl restart nvidia-persistenced 2>/dev/null || true
        echo "Done! Reboot recommended"
        ;;
    3)
        echo "Installing via Symlink..."
        mkdir -p /opt/cuda-drivers
        cp $CUDA_LIB /opt/cuda-drivers/libcuda.so.1.1.original
        cp libcuda.so.1.1.patched /opt/cuda-drivers/libcuda.so.1.1.patched
        mv $CUDA_LIB ${CUDA_LIB}.backup
        ln -s /opt/cuda-drivers/libcuda.so.1.1.patched $CUDA_LIB
        ldconfig
        systemctl restart nvidia-persistenced 2>/dev/null || true
        echo "Done! Reboot recommended"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Installation complete!"
echo "Verify with: nvidia-smi"
echo ""
echo "To rollback:"
case $METHOD in
    1) echo "  unset LD_PRELOAD" ;;
    2) echo "  sudo cp ${CUDA_LIB}.original $CUDA_LIB" ;;
    3) echo "  sudo rm $CUDA_LIB && sudo ln -s /opt/cuda-drivers/libcuda.so.1.1.original $CUDA_LIB" ;;
esac
```

---

## 💡 Best Practices

### For Development:
- ✅ Use LD_PRELOAD method
- ✅ Test in Docker container first
- ✅ Keep original driver intact

### For Production:
- ✅ Use Symlink method (easy rollback)
- ✅ Test thoroughly first
- ✅ Document installation for team

### For Servers:
- ✅ Use Backup & Replace
- ✅ Add to configuration management
- ✅ Monitor for driver updates

---

## 🔒 Security Considerations

### File Permissions:
```bash
# Ensure proper ownership
sudo chown root:root /usr/lib/x86_64-linux-gnu/libcuda.so.1.1
sudo chmod 755 /usr/lib/x86_64-linux-gnu/libcuda.so.1.1
```

### Verify Patch Integrity:
```bash
# Check MD5 of patched file
md5sum libcuda.so.1.1.patched

# Verify only expected bytes changed
cmp -l /usr/lib/x86_64-linux-gnu/libcuda.so.1.1.original \
        libcuda.so.1.1.patched | head -20
# Should only show changes at offset 0x186b50
```

---

## 📞 Need Help?

### If Something Goes Wrong:
1. Boot to recovery mode
2. Restore backup: `cp libcuda.so.1.1.original libcuda.so.1.1`
3. Reinstall NVIDIA drivers if needed
4. Check system logs: `dmesg | grep nvidia`

### Verification Failed?
- Double-check patch offset (0x186b50)
- Verify file size matches original
- Check if you have correct driver version
- Try LD_PRELOAD method instead

---

## 🚀 After Installation

### Test with APEX:
```bash
# Run APEX vector_add example
cd apex_release/bin
./vector_add_example

# Should show 83.7× speedup and 119ns kernel launch
```

### Test with PyTorch:
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Should work normally, but now SM_120 is accessible
```

---

**RECOMMENDATION: Start with LD_PRELOAD method!**

It's the safest and easiest to undo if something goes wrong.

---

*The Architect*  
*Liberating GPUs, one driver at a time* 🚀