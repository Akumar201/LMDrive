#!/bin/bash
# Run LangAuto-Tiny evaluation with the ROS2 split architecture.
#
# This script assumes the inference server is ALREADY RUNNING in a separate terminal:
#   source /opt/ros/humble/setup.bash
#   cd /path/to/LMDrive
#   python3 leaderboard/team_code/lmdrive_inference_server.py
#
# Usage:
#   bash leaderboard/scripts/run_ros_eval.sh [label]
#
# [label] tags the result file (default: ros2_8bit)
#   bash leaderboard/scripts/run_ros_eval.sh ros2_8bit

LABEL=${1:-ros2_8bit}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=results
RESULT_FILE="${RESULTS_DIR}/${LABEL}_${TIMESTAMP}.json"

mkdir -p "$RESULTS_DIR"

# Source ROS2 Foxy built from source (Python 3.8 / CycloneDDS)
source ~/ros2_foxy_ws/install/setup.bash

# Force CycloneDDS to use loopback so both processes discover each other on the same machine
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><NetworkInterfaceAddress>lo</NetworkInterfaceAddress></General></Domain></CycloneDDS>'

export PT=$(($RANDOM % 1000 + 16000))
CARLA_PID=""

kill_carla() {
  if [ -n "$CARLA_PID" ]; then
    echo "" && echo "Shutting down CARLA (PID ${CARLA_PID})..."
    kill "$CARLA_PID" 2>/dev/null
    wait "$CARLA_PID" 2>/dev/null
  fi
}
trap kill_carla EXIT
trap 'kill_carla; exit 130' INT TERM

echo ""
echo "=== LangAuto-Tiny | ROS2 mode | label=${LABEL} | result: ${RESULT_FILE} ==="
echo ""

DISPLAY= bash carla/CarlaUE4.sh --world-port=$PT -opengl -quality-level=Low &
CARLA_PID=$!
sleep 4

export CARLA_ROOT=carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=$PT
export TM_PORT=$(($PT + 500))
export DEBUG_CHALLENGE=0
export REPETITIONS=1
export ROUTES=langauto/benchmark_tiny.xml
export TEAM_AGENT=leaderboard/team_code/lmdrive_ros_agent.py   # <-- ROS2 agent
export TEAM_CONFIG=leaderboard/team_code/lmdriver_config.py
export CHECKPOINT_ENDPOINT=$RESULT_FILE
export SCENARIOS=leaderboard/data/official/all_towns_traffic_scenarios_public.json
export SAVE_PATH=data/eval
export RESUME=False

python3 -u ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
  --scenarios=${SCENARIOS} \
  --routes=${ROUTES} \
  --repetitions=${REPETITIONS} \
  --track=${CHALLENGE_TRACK_CODENAME} \
  --checkpoint=${CHECKPOINT_ENDPOINT} \
  --agent=${TEAM_AGENT} \
  --agent-config=${TEAM_CONFIG} \
  --debug=${DEBUG_CHALLENGE} \
  --record=${RECORD_PATH} \
  --resume=${RESUME} \
  --port=${PORT} \
  --trafficManagerPort=${TM_PORT}

echo ""
echo "Run complete. Results saved to: ${RESULT_FILE}"
echo ""
echo "=== Compare ROS2 vs direct inference ==="
python3 leaderboard/scripts/summarize_results.py \
  --dir "${RESULTS_DIR}" \
  --pattern "*.json"
