#!/bin/bash
export PT=$(($RANDOM % 1000 + 16000))
CARLA_PID=""

# Graceful exit: kill CARLA server on script exit (normal or Ctrl+C). Ensures GPU/process cleanup.
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

# Start CARLA without rendering (off-screen, no window). -opengl required for headless on Linux.
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
export PORT=$PT # same as the carla server port
export TM_PORT=$(($PT+500)) # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export ROUTES=langauto/benchmark_long.xml
export TEAM_AGENT=leaderboard/team_code/lmdriver_agent.py # agent
export TEAM_CONFIG=leaderboard/team_code/lmdriver_config.py # model checkpoint, not required for expert
export CHECKPOINT_ENDPOINT=results/sample_result.json # results file
#export SCENARIOS=leaderboard/data/scenarios/no_scenarios.json #town05_all_scenarios.json
export SCENARIOS=leaderboard/data/official/all_towns_traffic_scenarios_public.json
export SAVE_PATH=data/eval # path for saving episodes while evaluating
# Set to False to run all routes from start; True to resume from checkpoint
export RESUME=False

echo ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py
python3 -u  ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
  --scenarios=${SCENARIOS}  \
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

