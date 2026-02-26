"""
LMDrive ROS2 Inference Server  (subscriber + model + publisher side)

Runs as a standalone process — start this BEFORE launching the CARLA evaluation.
Loads the full LMDrive model (vision encoder + LLM), subscribes to raw sensor
data published by lmdrive_ros_agent.py, runs inference, and publishes waypoints back.

Usage:
  # Terminal 1 — inference server:
  source /opt/ros/humble/setup.bash
  cd /path/to/LMDrive
  python3 leaderboard/team_code/lmdrive_inference_server.py

  # Terminal 2 — CARLA evaluation:
  bash leaderboard/scripts/run_ros_eval.sh

Topics subscribed  (vehicle → server):
  /lmdrive/rgb_front   sensor_msgs/Image
  /lmdrive/rgb_left    sensor_msgs/Image
  /lmdrive/rgb_right   sensor_msgs/Image
  /lmdrive/rgb_rear    sensor_msgs/Image
  /lmdrive/lidar       std_msgs/Float32MultiArray  N×4 (x,y,z,intensity)
  /lmdrive/instruction std_msgs/String
  /lmdrive/state       std_msgs/Float32MultiArray  [velocity, target_x, target_y, command]
                       *** arrival of this message triggers inference ***

Topics published  (server → vehicle):
  /lmdrive/waypoints   std_msgs/Float32MultiArray  10 floats: 5×(x,y) waypoints
  /lmdrive/timing      std_msgs/Float32MultiArray  [inference_ms]
"""

import imp
import os
import sys
import threading
import time

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import Float32MultiArray, String
# cv_bridge replaced with inline numpy conversion (no Boost.Python dependency needed)

# Add LMDrive root to path so we can import team_code and LAVIS
_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(_REPO_ROOT))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'leaderboard'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'leaderboard', 'team_code'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'scenario_runner'))

from lavis.common.registry import registry
from team_code.lmdriver_agent import (
    lidar_to_raw_features,
    create_carla_rgb_transform,
)

CONFIG_PATH = os.path.join(_REPO_ROOT, 'leaderboard', 'team_code', 'lmdriver_config.py')


class LMDriveInferenceServer(Node):
    """
    ROS2 node that owns the full LMDrive model.
    Triggered by /lmdrive/state (published last by vehicle agent each step).
    """

    def __init__(self):
        super().__init__('lmdrive_inference_server')

        self._lock = threading.Lock()   # protect sensor buffers during inference

        # --- Load config and model ---
        self.config = imp.load_source("MainModel", CONFIG_PATH).GlobalConfig()
        self._load_model()

        # Image transforms (identical to original agent setup)
        self.rgb_front_transform  = create_carla_rgb_transform(224)
        self.rgb_side_transform   = create_carla_rgb_transform(128)
        self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)
        self.softmax = torch.nn.Softmax(dim=1)

        # --- Sensor buffers (latest message for each modality) ---
        self._rgb_front    = None
        self._rgb_left     = None
        self._rgb_right    = None
        self._rgb_rear     = None
        self._lidar_raw    = None   # numpy N×4
        self._instruction  = 'Drive safely.'
        self._state        = None   # [velocity, target_x, target_y, command]

        # Temporal visual feature buffer — same role as in the original agent
        self.visual_feature_buffer = []
        self.sample_rate = self.config.sample_rate * 2  # 20 Hz CARLA tick

        # --- Subscribers ---
        qos = 1
        self.create_subscription(ROSImage,          '/lmdrive/rgb_front',   self._cb_front,   qos)
        self.create_subscription(ROSImage,          '/lmdrive/rgb_left',    self._cb_left,    qos)
        self.create_subscription(ROSImage,          '/lmdrive/rgb_right',   self._cb_right,   qos)
        self.create_subscription(ROSImage,          '/lmdrive/rgb_rear',    self._cb_rear,    qos)
        self.create_subscription(Float32MultiArray, '/lmdrive/lidar',       self._cb_lidar,   qos)
        self.create_subscription(String,            '/lmdrive/instruction', self._cb_instr,   qos)
        # state is subscribed LAST and triggers inference
        self.create_subscription(Float32MultiArray, '/lmdrive/state',       self._cb_state,   qos)

        # --- Publishers ---
        self._pub_waypoints = self.create_publisher(Float32MultiArray, '/lmdrive/waypoints', qos)
        self._pub_timing    = self.create_publisher(Float32MultiArray, '/lmdrive/timing',    qos)

        self.get_logger().info("LMDrive inference server ready — waiting for sensor data.")

    # ------------------------------------------------------------------
    # Model loading (mirrors lmdriver_agent.py setup())
    # ------------------------------------------------------------------
    def _load_model(self):
        quantization = getattr(self.config, 'quantization', None)
        model_cls = registry.get_model_class('vicuna_drive')

        self.get_logger().info("Building model...")
        model = model_cls(
            preception_model=self.config.preception_model,
            preception_model_ckpt=self.config.preception_model_ckpt,
            llm_model=self.config.llm_model,
            max_txt_len=64,
            use_notice_prompt=self.config.agent_use_notice,
            quantization=quantization,
        )

        self.get_logger().info("Loading checkpoint...")
        ckpt = torch.load(self.config.lmdrive_ckpt)["model"]
        if quantization:
            ckpt = {k: v for k, v in ckpt.items() if not k.startswith('llm_model.')}
        model.load_state_dict(ckpt, strict=False)

        if quantization:
            for param in model.parameters():
                if param.data.device.type == 'cpu':
                    param.data = param.data.cuda()
            for buf in model.buffers():
                if buf.device.type == 'cpu':
                    buf.data = buf.data.cuda()
        else:
            model.cuda()

        model.eval()
        self.net = model
        self.get_logger().info("Model loaded and ready.")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Image conversion (replaces cv_bridge — no Boost.Python needed)
    # ------------------------------------------------------------------
    @staticmethod
    def _imgmsg_to_np(msg):
        """Convert sensor_msgs/Image (rgb8) to uint8 HxWx3 numpy array."""
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

    # Sensor callbacks — just store the latest value
    # ------------------------------------------------------------------
    def _cb_front(self, msg):
        self._rgb_front = self._imgmsg_to_np(msg)

    def _cb_left(self, msg):
        self._rgb_left = self._imgmsg_to_np(msg)

    def _cb_right(self, msg):
        self._rgb_right = self._imgmsg_to_np(msg)

    def _cb_rear(self, msg):
        self._rgb_rear = self._imgmsg_to_np(msg)

    def _cb_lidar(self, msg):
        pts = np.array(msg.data, dtype=np.float32)
        self._lidar_raw = pts.reshape(-1, 4)

    def _cb_instr(self, msg):
        if msg.data != self._instruction:
            self.get_logger().info(f"Instruction changed: '{msg.data}' — resetting visual buffer")
            self._instruction = msg.data
            self.visual_feature_buffer = []  # reset temporal history on new route segment

    def _cb_state(self, msg):
        """
        State is published last by the vehicle agent each step.
        Its arrival signals that all sensor data for this step is ready.
        """
        self._state = list(msg.data)  # [velocity, target_x, target_y, command]
        # Guard: don't run inference if any sensor hasn't arrived yet
        if any(x is None for x in [
            self._rgb_front, self._rgb_left, self._rgb_right,
            self._rgb_rear, self._lidar_raw
        ]):
            self.get_logger().warn("State arrived but some sensors are still None — skipping step")
            return
        self._run_inference()

    # ------------------------------------------------------------------
    # Inference — runs synchronously in the ROS2 spin thread
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_inference(self):
        t0 = time.time()

        velocity = self._state[0]
        target_x = self._state[1]
        target_y = self._state[2]

        # --- Prepare image tensors (same transforms as original agent) ---
        rgb_front = (self.rgb_front_transform(PILImage.fromarray(self._rgb_front))
                     .unsqueeze(0).cuda().float())
        rgb_left  = (self.rgb_side_transform(PILImage.fromarray(self._rgb_left))
                     .unsqueeze(0).cuda().float())
        rgb_right = (self.rgb_side_transform(PILImage.fromarray(self._rgb_right))
                     .unsqueeze(0).cuda().float())
        rgb_rear  = (self.rgb_side_transform(PILImage.fromarray(self._rgb_rear))
                     .unsqueeze(0).cuda().float())
        # Centre crop from front camera (same as original agent)
        rgb_center = (self.rgb_center_transform(
                          PILImage.fromarray(cv2.resize(self._rgb_front, (800, 600))))
                      .unsqueeze(0).cuda().float())

        # --- Prepare LiDAR ---
        lidar_processed, num_points = lidar_to_raw_features(self._lidar_raw)

        # --- Build input dict ---
        input_data = {
            'rgb_front':    rgb_front,
            'rgb_left':     rgb_left,
            'rgb_right':    rgb_right,
            'rgb_center':   rgb_center,
            'rgb_rear':     rgb_rear,
            'target_point': torch.tensor([target_x, target_y]).cuda().view(1, 2).float(),
            'lidar':        torch.from_numpy(lidar_processed).float().cuda().unsqueeze(0),
            'num_points':   torch.tensor([num_points]).cuda().unsqueeze(0),
            'velocity':     torch.tensor([velocity]).cuda().view(1, 1).float(),
            'text_input':   [self._instruction],
        }

        # --- Visual encoder + temporal buffer (same logic as update_and_collect) ---
        image_embeds = self.net.visual_encoder(input_data)
        self.visual_feature_buffer.append(image_embeds)

        # Subsample buffer at sample_rate (same as original agent)
        sampled = self.visual_feature_buffer[::self.sample_rate]
        if (len(self.visual_feature_buffer) - 1) % self.sample_rate != 0:
            sampled.append(self.visual_feature_buffer[-1])
        image_embeds_seq = torch.stack(sampled, 1)
        input_data['valid_frames'] = [image_embeds_seq.size(1)]

        # --- LLM inference ---
        with torch.cuda.amp.autocast(enabled=True):
            waypoints, is_end = self.net(
                input_data, inference_mode=True, image_embeds=image_embeds_seq)

        waypoints = waypoints[-1].view(5, 2)  # shape [5, 2], same as original

        # Reset buffer when route segment ends (end_prob > 0.75)
        end_prob = self.softmax(is_end)[-1][1].item()
        if end_prob > 0.75:
            self.visual_feature_buffer = []

        # Cap buffer length (matches original agent: 400 frames max)
        if len(self.visual_feature_buffer) > 400:
            self.visual_feature_buffer = []

        inference_ms = (time.time() - t0) * 1000

        # --- Publish waypoints ---
        wp_msg = Float32MultiArray()
        wp_msg.data = waypoints.cpu().numpy().flatten().tolist()
        self._pub_waypoints.publish(wp_msg)

        # --- Publish timing ---
        timing_msg = Float32MultiArray()
        timing_msg.data = [float(inference_ms)]
        self._pub_timing.publish(timing_msg)

        self.get_logger().info(
            f"Inference: {inference_ms:.1f}ms | "
            f"buf_len={len(self.visual_feature_buffer)} | "
            f"end_prob={end_prob:.2f}")


def main():
    rclpy.init()
    server = LMDriveInferenceServer()
    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        pass
    finally:
        server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
