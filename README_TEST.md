# LMDrive — Testing Guide

## Installation

```bash
git clone https://github.com/opendilab/LMDrive.git
cd LMDrive

# Create conda environment
conda deactivate
conda env remove -n lmdrive          # skip if fresh machine
conda create -n lmdrive python=3.8
conda activate lmdrive

# Core dependencies
pip install torch==2.0.1+cu117 torchvision --index-url https://download.pytorch.org/whl/cu117
# torch_scatter must match the exact torch+cuda version — use the PyG index
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

# Vision encoder
cd vision_encoder
pip install -r requirements.txt
python setup.py develop
cd ..

# LLM (LAVIS)
cd LAVIS
pip install -r requirements.txt
python setup.py develop
cd ..

# Pin opencv to a compatible version — newer 4.8+ breaks cv2.dnn.DictValue used deep in timm/LAVIS
pip install "opencv-python-headless==4.5.5.64"

# Required for 8-bit/4-bit quantization — pin to 0.41.3; newer 0.42+ needs triton 2.1+
# which conflicts with the triton 2.0.0 bundled with torch 2.0.1
pip install "bitsandbytes==0.41.3"

# CARLA 0.9.10.1
chmod +x setup_carla.sh
./setup_carla.sh
pip install carla
```

---

## Download Model Weights

Download all three model components into the `models/` directory:

```bash
conda activate lmdrive
pip install huggingface_hub

python3 - <<'EOF'
from huggingface_hub import snapshot_download
# Base LLM
snapshot_download(repo_id="liuhaotian/llava-v1.5-7b",
                  local_dir="models/llava-v1.5-7b")
# Vision encoder
snapshot_download(repo_id="OpenDILabCommunity/LMDrive-vision-encoder-r50-v1.0",
                  local_dir="models/LMDrive-vision-encoder-r50-v1.0")
# LMDrive checkpoint
snapshot_download(repo_id="OpenDILabCommunity/LMDrive-llava-v1.5-7b-v1.0",
                  local_dir="models/LMDrive-llava-v1.5-7b-v1.0")
EOF
```

The config at `leaderboard/team_code/lmdriver_config.py` already points to these paths by default:
```
models/llava-v1.5-7b
models/LMDrive-vision-encoder-r50-v1.0/vision-encoder-r50.pth.tar
models/LMDrive-llava-v1.5-7b-v1.0/llava-v1.5-checkpoint.pth
```

**No editing needed** unless you want to change quantization or display:
```python
quantization  = "8bit"   # None, "4bit", or "8bit"
display_mode  = "none"   # "none" for headless (benchmarking), "pygame" for full HUD
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
pip install 'empy==3.3.4' catkin_pkg   # required by ROS2 Foxy build
bash ros2_foxy_install.sh              # takes ~15 min, builds into ~/ros2_foxy_ws/
```

**Terminal 1 — start inference server first:**
```bash
conda activate lmdrive
cd ~/LMDrive
source ~/ros2_foxy_ws/install/setup.bash
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><NetworkInterfaceAddress>lo</NetworkInterfaceAddress></General></Domain></CycloneDDS>'
python3 leaderboard/team_code/lmdrive_inference_server.py
```

Wait for:
```
[INFO] [lmdrive_inference_server]: LMDrive inference server ready — waiting for sensor data.
```
Model loading takes ~30–60 seconds. **Do not start Terminal 2 until you see this.**

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
