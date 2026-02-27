# LMDrive — Testing Guide

## Installation

```bash
conda activate lmdrive
cd ~/LMDrive

# Install dependencies
cd vision_encoder && pip install -r requirements.txt && python setup.py develop && cd ..
cd LAVIS && pip install -r requirements.txt && python setup.py develop && cd ..

# Install CARLA
chmod +x setup_carla.sh && ./setup_carla.sh && pip install carla
```

Model weights go in `models/`. Edit `leaderboard/team_code/lmdriver_config.py` to point to them:
```python
preception_model_ckpt = "models/LMDrive-vision-encoder-r50-v1.0/..."
llm_model             = "models/llava-v1.5-7b"
lmdrive_ckpt          = "models/LMDrive-llava-v1.5-7b-v1.0/..."
quantization          = "8bit"   # None, "4bit", or "8bit"
```

---

## Running Without ROS2 (Direct)

Single terminal:

```bash
conda activate lmdrive
cd ~/LMDrive
bash leaderboard/scripts/run_tiny_bench.sh 8bit
```

Results saved to: `results/tiny_8bit_<timestamp>.json`

---

## Running With ROS2 (Split Architecture)

The model runs in a separate process from CARLA. Requires two terminals.

**Build ROS2 once (first time only):**
```bash
conda activate lmdrive
bash ros2_foxy_install.sh   # takes ~15 min
```

**Terminal 1 — start inference server first:**
```bash
conda activate lmdrive
cd ~/LMDrive
source ~/ros2_foxy_ws/install/setup.bash
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><NetworkInterfaceAddress>lo</NetworkInterfaceAddress></General></Domain></CycloneDDS>'
python3 leaderboard/team_code/lmdrive_inference_server.py
```

Wait for: `[INFO] [lmdrive_inference_server]: LMDrive inference server ready — waiting for sensor data.`

**Terminal 2 — start CARLA evaluation after server is ready:**
```bash
conda activate lmdrive
cd ~/LMDrive
bash leaderboard/scripts/run_ros_eval.sh ros2_8bit
```

Results saved to: `results/ros2_8bit_<timestamp>.json`

---

## Comparing Results

```bash
python3 leaderboard/scripts/summarize_results.py --dir results/ --pattern "*.json"
```

Example output:
```
------------------------------------------------------------------------
Run                                   DS    RC (%)        IS    Progress
------------------------------------------------------------------------
tiny_8bit_20260225_192804.json    53.231    65.341     0.769       16/16
ros2_8bit_20260226_205443.json    62.527    83.677     0.758       16/16
------------------------------------------------------------------------
```

- **DS** — Driving Score (overall, higher is better)
- **RC** — Route Completion % (how much of the route was finished)
- **IS** — Infraction Score (safety, closer to 1.0 is better)
