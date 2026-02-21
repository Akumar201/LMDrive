#!/bin/bash
# Print LMDrive model parameter and GPU memory breakdown. Run from LMDrive project root.
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="${PWD}:${PWD}/leaderboard:${PWD}/leaderboard/team_code:${PWD}/LAVIS:${PWD}/vision_encoder:${PYTHONPATH}"
python3 leaderboard/scripts/print_lmdrive_model_size.py
