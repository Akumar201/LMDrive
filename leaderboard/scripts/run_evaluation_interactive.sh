#!/bin/bash
# Run from LMDrive/ (project root). Prompts to select route and scenario by number, then runs evaluation.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

# --- Route options (path relative to project root) ---
ROUTE_OPTIONS=(
  "langauto/benchmark_long.xml"
  "langauto/benchmark_short.xml"
  "langauto/benchmark_tiny.xml"
  "leaderboard/data/additional_routes/routes_town01_long.xml"
  "leaderboard/data/additional_routes/routes_town02_long.xml"
  "leaderboard/data/additional_routes/routes_town03_long.xml"
  "leaderboard/data/additional_routes/routes_town04_long.xml"
  "leaderboard/data/additional_routes/routes_town06_long.xml"
  "leaderboard/data/training_routes/routes_town01_long.xml"
  "leaderboard/data/training_routes/routes_town01_short.xml"
  "leaderboard/data/training_routes/routes_town02_long.xml"
  "leaderboard/data/training_routes/routes_town05_long.xml"
  "leaderboard/data/official/routes_training.xml"
  "leaderboard/data/official/routes_testing.xml"
  "leaderboard/data/42routes/42routes.xml"
)

# --- Scenario options (path relative to project root) ---
SCENARIO_OPTIONS=(
  "leaderboard/data/official/all_towns_traffic_scenarios_public.json"
  "leaderboard/data/scenarios/no_scenarios.json"
  "leaderboard/data/scenarios/town01_all_scenarios.json"
  "leaderboard/data/scenarios/town02_all_scenarios.json"
  "leaderboard/data/scenarios/town03_all_scenarios.json"
  "leaderboard/data/scenarios/town04_all_scenarios.json"
  "leaderboard/data/scenarios/town05_all_scenarios.json"
  "leaderboard/data/scenarios/town06_all_scenarios.json"
  "leaderboard/data/42routes/42scenarios.json"
)

print_menu() {
  local -n arr=$1
  local i
  for i in "${!arr[@]}"; do
    printf "  %2d) %s\n" "$((i+1))" "${arr[$i]}"
  done
}

echo "=============================================="
echo "  LMDrive evaluation - route & scenario picker"
echo "=============================================="
echo ""
echo "ROUTES (choose one):"
print_menu ROUTE_OPTIONS
echo ""
read -rp "Enter route number [1-${#ROUTE_OPTIONS[@]}]: " route_choice
route_choice="${route_choice:-1}"
if ! [[ "$route_choice" =~ ^[0-9]+$ ]] || (( route_choice < 1 || route_choice > ${#ROUTE_OPTIONS[@]} )); then
  echo "Invalid choice. Using 1."
  route_choice=1
fi
export ROUTES="${ROUTE_OPTIONS[$((route_choice-1))]}"
echo "Selected ROUTES: $ROUTES"
echo ""

echo "SCENARIOS (choose one):"
print_menu SCENARIO_OPTIONS
echo ""
read -rp "Enter scenario number [1-${#SCENARIO_OPTIONS[@]}]: " scenario_choice
scenario_choice="${scenario_choice:-1}"
if ! [[ "$scenario_choice" =~ ^[0-9]+$ ]] || (( scenario_choice < 1 || scenario_choice > ${#SCENARIO_OPTIONS[@]} )); then
  echo "Invalid choice. Using 1."
  scenario_choice=1
fi
export SCENARIOS="${SCENARIO_OPTIONS[$((scenario_choice-1))]}"
echo "Selected SCENARIOS: $SCENARIOS"
echo ""

# --- Same logic as run_evaluation.sh from here ---
export PT=$(($RANDOM % 1000 + 16000))
CARLA_PID=""

kill_carla() {
  if [ -n "$CARLA_PID" ]; then
    echo "" && echo "Shutting down: killing CARLA server (PID ${CARLA_PID})..."
    kill "$CARLA_PID" 2>/dev/null
    wait "$CARLA_PID" 2>/dev/null
  fi
}
on_int_term() {
  echo "" && echo "Interrupted (Ctrl+C). Cleaning up..."
  kill_carla
  exit 130
}
trap kill_carla EXIT
trap on_int_term INT TERM

echo "Starting CARLA server..."
DISPLAY= bash carla/CarlaUE4.sh --world-port=$PT -opengl &
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
export TM_PORT=$(($PT+500))
export DEBUG_CHALLENGE=0
export REPETITIONS=1
export TEAM_AGENT=leaderboard/team_code/lmdriver_agent.py
export TEAM_CONFIG=leaderboard/team_code/lmdriver_config.py
export CHECKPOINT_ENDPOINT=results/sample_result.json
export SAVE_PATH=data/eval
export RESUME=False

echo "Running evaluator with ROUTES=$ROUTES SCENARIOS=$SCENARIOS"
python3 -u "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
  --scenarios="${SCENARIOS}" \
  --routes="${ROUTES}" \
  --repetitions="${REPETITIONS}" \
  --track="${CHALLENGE_TRACK_CODENAME}" \
  --checkpoint="${CHECKPOINT_ENDPOINT}" \
  --agent="${TEAM_AGENT}" \
  --agent-config="${TEAM_CONFIG}" \
  --debug="${DEBUG_CHALLENGE}" \
  --record="${RECORD_PATH}" \
  --resume="${RESUME}" \
  --port="${PORT}" \
  --trafficManagerPort="${TM_PORT}"
