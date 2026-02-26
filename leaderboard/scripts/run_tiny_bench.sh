#!/bin/bash
# Run LangAuto-Tiny benchmark and save results to a timestamped file.
#
# Usage:
#   bash leaderboard/scripts/run_tiny_bench.sh [label]
#
# [label] tags the result file so you can group runs by quantization setting.
# Defaults to "8bit". Examples:
#   bash leaderboard/scripts/run_tiny_bench.sh 8bit
#   bash leaderboard/scripts/run_tiny_bench.sh 4bit
#   bash leaderboard/scripts/run_tiny_bench.sh none
#
# Results are saved to: results/tiny_<label>_<YYYYMMDD_HHMMSS>.json
# After the run, a summary table of all matching runs is printed.

LABEL=${1:-8bit}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=results
RESULT_FILE="${RESULTS_DIR}/tiny_${LABEL}_${TIMESTAMP}.json"

mkdir -p "$RESULTS_DIR"

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
echo "=== LangAuto-Tiny | label=${LABEL} | result: ${RESULT_FILE} ==="
echo ""

# Launch CARLA headless (no window, no rendering overhead)
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
export TEAM_AGENT=leaderboard/team_code/lmdriver_agent.py
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
echo "=== All tiny_${LABEL} runs ==="
python3 leaderboard/scripts/summarize_results.py \
  --dir "${RESULTS_DIR}" \
  --pattern "tiny_${LABEL}_*.json"
