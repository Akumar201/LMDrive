#!/bin/bash
# Start the LMDrive ROS2 inference server.
# Fixes the conda environment automatically before launching.
#
# Usage (Terminal 1):
#   conda activate lmdrive
#   bash leaderboard/scripts/run_server.sh
#
# Then in Terminal 2:
#   bash leaderboard/scripts/run_ros_eval.sh ros2_8bit

set -euo pipefail
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$_REPO_ROOT"

# Auto-fix environment (idempotent — skips steps already done)
bash fix_env.sh

# Source ROS2
source ~/ros2_foxy_ws/install/setup.bash

# Force CycloneDDS to use loopback so both processes find each other on the same machine
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><NetworkInterfaceAddress>lo</NetworkInterfaceAddress></General></Domain></CycloneDDS>'

echo ""
echo "=== Starting LMDrive inference server ==="
echo ""
exec python3 leaderboard/team_code/lmdrive_inference_server.py
