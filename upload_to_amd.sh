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
