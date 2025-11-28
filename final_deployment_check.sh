#!/bin/bash

# ==============================================================================
# APEX GPU - Final Deployment Readiness Check
# ==============================================================================
# Run this BEFORE uploading to AMD to verify everything is ready
# ==============================================================================

set -e

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        APEX GPU - Final Deployment Readiness Check            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

ERRORS=0
WARNINGS=0

# ==============================================================================
# 1. Check Translation Bridges
# ==============================================================================

echo "1. Checking Translation Bridges..."
echo ""

BRIDGES_FOUND=0

if [ -f "libapex_hip_bridge.so" ]; then
    SIZE=$(ls -lh libapex_hip_bridge.so | awk '{print $5}')
    echo "   ✅ HIP Bridge:    $SIZE"
    BRIDGES_FOUND=$((BRIDGES_FOUND + 1))
else
    echo "   ❌ HIP Bridge:    MISSING"
    ERRORS=$((ERRORS + 1))
fi

if [ -f "libapex_cublas_bridge.so" ]; then
    SIZE=$(ls -lh libapex_cublas_bridge.so | awk '{print $5}')
    echo "   ✅ cuBLAS Bridge: $SIZE"
    BRIDGES_FOUND=$((BRIDGES_FOUND + 1))
else
    echo "   ❌ cuBLAS Bridge: MISSING"
    ERRORS=$((ERRORS + 1))
fi

if [ -f "libapex_cudnn_bridge.so" ]; then
    SIZE=$(ls -lh libapex_cudnn_bridge.so | awk '{print $5}')
    echo "   ✅ cuDNN Bridge:  $SIZE"
    BRIDGES_FOUND=$((BRIDGES_FOUND + 1))
else
    echo "   ❌ cuDNN Bridge:  MISSING"
    ERRORS=$((ERRORS + 1))
fi

echo "   Found: $BRIDGES_FOUND/3 bridges"
echo ""

# ==============================================================================
# 2. Check Test Binaries
# ==============================================================================

echo "2. Checking Test Binaries..."
echo ""

TESTS_FOUND=0

TESTS=(
    "test_hello"
    "test_events_timing"
    "test_async_streams"
    "test_2d_memory"
    "test_host_memory"
    "test_device_mgmt"
)

for TEST in "${TESTS[@]}"; do
    if [ -f "build/$TEST" ] || [ -f "$TEST" ]; then
        echo "   ✅ $TEST"
        TESTS_FOUND=$((TESTS_FOUND + 1))
    else
        echo "   ⚠️  $TEST - not found"
        WARNINGS=$((WARNINGS + 1))
    fi
done

echo "   Found: $TESTS_FOUND/6 tests"
echo ""

# ==============================================================================
# 3. Check Scripts
# ==============================================================================

echo "3. Checking Automation Scripts..."
echo ""

SCRIPTS_FOUND=0

CRITICAL_SCRIPTS=(
    "setup_amd_mi300x.sh"
    "verify_apex.sh"
    "run_all_tests.sh"
    "install_rocm.sh"
)

for SCRIPT in "${CRITICAL_SCRIPTS[@]}"; do
    if [ -f "$SCRIPT" ]; then
        if [ -x "$SCRIPT" ]; then
            echo "   ✅ $SCRIPT (executable)"
        else
            echo "   ⚠️  $SCRIPT (not executable - will be fixed)"
            chmod +x "$SCRIPT" 2>/dev/null || true
        fi
        SCRIPTS_FOUND=$((SCRIPTS_FOUND + 1))
    else
        echo "   ❌ $SCRIPT - MISSING"
        ERRORS=$((ERRORS + 1))
    fi
done

echo "   Found: $SCRIPTS_FOUND/4 critical scripts"
echo ""

# ==============================================================================
# 4. Check Documentation
# ==============================================================================

echo "4. Checking Documentation..."
echo ""

DOCS_FOUND=0

CRITICAL_DOCS=(
    "DEPLOY_TO_AMD.md"
    "QUICK_REFERENCE.md"
    "AMD_DEPLOYMENT_CHECKLIST.md"
    "COMPLETE_DEPLOYMENT_GUIDE.md"
    "README.md"
)

for DOC in "${CRITICAL_DOCS[@]}"; do
    if [ -f "$DOC" ]; then
        echo "   ✅ $DOC"
        DOCS_FOUND=$((DOCS_FOUND + 1))
    else
        echo "   ⚠️  $DOC - not found"
        WARNINGS=$((WARNINGS + 1))
    fi
done

TOTAL_DOCS=$(ls -1 *.md 2>/dev/null | wc -l)
echo "   Found: $DOCS_FOUND/5 critical docs ($TOTAL_DOCS total)"
echo ""

# ==============================================================================
# 5. Check File Permissions
# ==============================================================================

echo "5. Checking File Permissions..."
echo ""

# Make all scripts executable
MADE_EXECUTABLE=0
for script in *.sh; do
    if [ -f "$script" ] && [ ! -x "$script" ]; then
        chmod +x "$script" 2>/dev/null && MADE_EXECUTABLE=$((MADE_EXECUTABLE + 1)) || true
    fi
done

if [ $MADE_EXECUTABLE -gt 0 ]; then
    echo "   ✅ Made $MADE_EXECUTABLE scripts executable"
else
    echo "   ✅ All scripts already executable"
fi
echo ""

# ==============================================================================
# 6. Calculate Package Size
# ==============================================================================

echo "6. Calculating Package Size..."
echo ""

if command -v du &> /dev/null; then
    TOTAL_SIZE=$(du -sh . 2>/dev/null | awk '{print $1}')
    echo "   📦 Total package size: $TOTAL_SIZE"
else
    echo "   ⚠️  Cannot calculate size (du not available)"
fi
echo ""

# ==============================================================================
# 7. Generate Upload Command
# ==============================================================================

echo "7. Generating Upload Command..."
echo ""

CURRENT_DIR=$(basename "$PWD")

cat > upload_to_amd.sh <<'EOF'
#!/bin/bash

# APEX GPU - Upload to AMD MI300X
# Edit these variables with your AMD instance details

AMD_USER="your-username"
AMD_HOST="your-mi300x-instance.com"

echo "Uploading APEX GPU to AMD MI300X..."
echo ""

# Upload directory
scp -r "../APEX GPU" $AMD_USER@$AMD_HOST:~/

echo ""
echo "✅ Upload complete!"
echo ""
echo "Next steps:"
echo "  1. SSH to AMD instance:"
echo "     ssh $AMD_USER@$AMD_HOST"
echo ""
echo "  2. Navigate to APEX:"
echo "     cd \"APEX GPU\""
echo ""
echo "  3. Run setup:"
echo "     sudo ./setup_amd_mi300x.sh"
echo ""
echo "  4. Load environment:"
echo "     source apex_env.sh"
echo ""
echo "  5. Run quick test:"
echo "     ./test_hello"
echo ""
EOF

chmod +x upload_to_amd.sh

echo "   ✅ Created upload_to_amd.sh"
echo "   Edit this file with your AMD instance details"
echo ""

# ==============================================================================
# SUMMARY
# ==============================================================================

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    READINESS SUMMARY                           ║"
echo "╠════════════════════════════════════════════════════════════════╣"

printf "║  Translation Bridges:      %d/3 %-33s║\n" $BRIDGES_FOUND "$([ $BRIDGES_FOUND -eq 3 ] && echo '✅' || echo '❌')"
printf "║  Test Binaries:            %d/6 %-33s║\n" $TESTS_FOUND "$([ $TESTS_FOUND -ge 1 ] && echo '✅' || echo '⚠️')"
printf "║  Automation Scripts:       %d/4 %-33s║\n" $SCRIPTS_FOUND "$([ $SCRIPTS_FOUND -eq 4 ] && echo '✅' || echo '❌')"
printf "║  Documentation:            %d/5 %-33s║\n" $DOCS_FOUND "$([ $DOCS_FOUND -ge 3 ] && echo '✅' || echo '⚠️')"

echo "║                                                                ║"
echo "╠════════════════════════════════════════════════════════════════╣"

if [ $ERRORS -eq 0 ] && [ $BRIDGES_FOUND -eq 3 ] && [ $SCRIPTS_FOUND -eq 4 ]; then
    echo "║                                                                ║"
    echo "║              🚀 READY FOR AMD DEPLOYMENT! 🚀                  ║"
    echo "║                                                                ║"
    echo "╠════════════════════════════════════════════════════════════════╣"
    echo "║                       NEXT STEPS                               ║"
    echo "╠════════════════════════════════════════════════════════════════╣"
    echo "║                                                                ║"
    echo "║  1. Edit upload command:                                       ║"
    echo "║     nano upload_to_amd.sh                                      ║"
    echo "║     (Set AMD_USER and AMD_HOST)                                ║"
    echo "║                                                                ║"
    echo "║  2. Upload to AMD:                                             ║"
    echo "║     ./upload_to_amd.sh                                         ║"
    echo "║                                                                ║"
    echo "║  3. SSH to AMD and run:                                        ║"
    echo "║     cd \"APEX GPU\"                                              ║"
    echo "║     sudo ./setup_amd_mi300x.sh                                 ║"
    echo "║     source apex_env.sh                                         ║"
    echo "║     ./test_hello                                               ║"
    echo "║                                                                ║"
    echo "║  📖 See DEPLOY_TO_AMD.md for detailed instructions            ║"
    echo "║                                                                ║"

    EXIT_CODE=0
else
    echo "║                                                                ║"
    echo "║              ⚠️  DEPLOYMENT NOT READY                         ║"
    echo "║                                                                ║"
    echo "╠════════════════════════════════════════════════════════════════╣"
    echo "║                    ISSUES FOUND                                ║"
    echo "╠════════════════════════════════════════════════════════════════╣"
    echo "║                                                                ║"

    if [ $BRIDGES_FOUND -lt 3 ]; then
        echo "║  ❌ Missing bridges - run build scripts:                      ║"
        [ ! -f "libapex_hip_bridge.so" ] && echo "║     ./build_hip_bridge.sh                                      ║"
        [ ! -f "libapex_cublas_bridge.so" ] && echo "║     ./build_cublas_bridge.sh                                   ║"
        [ ! -f "libapex_cudnn_bridge.so" ] && echo "║     ./build_cudnn_bridge.sh                                    ║"
        echo "║                                                                ║"
    fi

    if [ $SCRIPTS_FOUND -lt 4 ]; then
        echo "║  ❌ Missing critical scripts                                   ║"
        echo "║                                                                ║"
    fi

    EXIT_CODE=1
fi

echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

if [ $ERRORS -gt 0 ]; then
    echo "Errors: $ERRORS"
fi
if [ $WARNINGS -gt 0 ]; then
    echo "Warnings: $WARNINGS (non-critical)"
fi

echo ""

exit $EXIT_CODE
