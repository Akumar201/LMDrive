"""
LMDrive ROS2 Vehicle Agent  (publisher side)

This agent replaces lmdriver_agent.py in the CARLA evaluation pipeline.
It does NOT load the LLM/vision model. Instead, it:
  1. Collects raw sensor data from CARLA each step
  2. Publishes it to ROS2 topics
  3. Blocks until the inference server returns waypoints
  4. Converts waypoints → steering/throttle/brake via PID

Run the inference server first (in a separate terminal):
  python3 leaderboard/team_code/lmdrive_inference_server.py

Then launch evaluation using the ROS2 run script:
  bash leaderboard/scripts/run_ros_eval.sh

Topics published  (vehicle → inference server):
  /lmdrive/rgb_front   sensor_msgs/Image        1200×900 front camera
  /lmdrive/rgb_left    sensor_msgs/Image         400×300 left camera
  /lmdrive/rgb_right   sensor_msgs/Image         400×300 right camera
  /lmdrive/rgb_rear    sensor_msgs/Image         400×300 rear camera
  /lmdrive/lidar       std_msgs/Float32MultiArray N×4 point cloud (x,y,z,intensity)
  /lmdrive/instruction std_msgs/String            navigation instruction text
  /lmdrive/state       std_msgs/Float32MultiArray [velocity, target_x, target_y, command]
                       *** published LAST — triggers inference on server side ***

Topics subscribed  (inference server → vehicle):
  /lmdrive/waypoints   std_msgs/Float32MultiArray 10 floats: 5×(x,y) waypoints
  /lmdrive/timing      std_msgs/Float32MultiArray [inference_ms] for latency logging
"""

import imp
import math
import os
import threading
import time

import carla
import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import Float32MultiArray, String
# cv_bridge replaced with inline numpy conversion (no Boost.Python dependency needed)

from leaderboard.autoagents import autonomous_agent
from team_code.planner import RoutePlanner, InstructionPlanner
from team_code.pid_controller import PIDController
from team_code.lmdriver_agent import lidar_to_raw_features

SAVE_PATH = os.environ.get("SAVE_PATH", None)
WAYPOINT_TIMEOUT = 2.0  # seconds to wait for waypoints before falling back to prev_control


def get_entry_point():
    return "LMDriveROSAgent"


class LMDriveROSAgent(autonomous_agent.AutonomousAgent):
    """
    Vehicle-side agent: publishes raw sensor data to ROS2 and applies
    waypoints received from the remote inference server via PID control.
    No model is loaded here — all inference happens in lmdrive_inference_server.py.
    """

    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.initialized = False
        self.prev_lidar = None
        self.prev_control = carla.VehicleControl()
        self.prev_control.brake = 1.0
        self.curr_instruction = 'Drive safely.'

        self.config = imp.load_source("MainModel", path_to_conf_file).GlobalConfig()

        # PID controllers (same as original agent)
        self.turn_controller = PIDController(
            K_P=self.config.turn_KP, K_I=self.config.turn_KI,
            K_D=self.config.turn_KD, n=self.config.turn_n)
        self.speed_controller = PIDController(
            K_P=self.config.speed_KP, K_I=self.config.speed_KI,
            K_D=self.config.speed_KD, n=self.config.speed_n)

        # ROS2 setup
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node('lmdrive_vehicle_agent')

        # Publishers (sensor data → inference server)
        qos = 1  # queue depth
        self._pub_front = self._node.create_publisher(ROSImage,          '/lmdrive/rgb_front',   qos)
        self._pub_left  = self._node.create_publisher(ROSImage,          '/lmdrive/rgb_left',    qos)
        self._pub_right = self._node.create_publisher(ROSImage,          '/lmdrive/rgb_right',   qos)
        self._pub_rear  = self._node.create_publisher(ROSImage,          '/lmdrive/rgb_rear',    qos)
        self._pub_lidar = self._node.create_publisher(Float32MultiArray, '/lmdrive/lidar',       qos)
        self._pub_instr = self._node.create_publisher(String,            '/lmdrive/instruction', qos)
        self._pub_state = self._node.create_publisher(Float32MultiArray, '/lmdrive/state',       qos)

        # Waypoint subscriber (inference server → vehicle)
        self._waypoints = None
        self._waypoint_event = threading.Event()
        self._node.create_subscription(
            Float32MultiArray, '/lmdrive/waypoints', self._on_waypoints, qos)

        # Timing subscriber — optional, for latency logging
        self._last_inference_ms = 0.0
        self._node.create_subscription(
            Float32MultiArray, '/lmdrive/timing',
            lambda msg: setattr(self, '_last_inference_ms', msg.data[0]), qos)

        # Spin ROS2 in a background daemon thread so it doesn't block CARLA
        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self._node,), daemon=True)
        self._spin_thread.start()

        print("[ROS2 agent] Vehicle node started. Waiting for inference server...")

    # ------------------------------------------------------------------
    # Waypoint callback (runs in the ROS2 spin thread)
    # ------------------------------------------------------------------
    @staticmethod
    def _np_to_imgmsg(arr):
        """Convert uint8 HxWx3 numpy array to sensor_msgs/Image (rgb8, no cv_bridge)."""
        msg = ROSImage()
        msg.height, msg.width = arr.shape[:2]
        msg.encoding = 'rgb8'
        msg.step = arr.strides[0]
        msg.data = arr.tobytes()
        return msg

    def _on_waypoints(self, msg):
        self._waypoints = np.array(msg.data, dtype=np.float32).reshape(5, 2)
        self._waypoint_event.set()

    # ------------------------------------------------------------------
    # CARLA sensor definitions (identical to original agent)
    # ------------------------------------------------------------------
    def sensors(self):
        return [
            {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "width": 1200, "height": 900, "fov": 100, "id": "rgb_front"},
            {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3,
             "roll": 0.0, "pitch": 0.0, "yaw": -60.0,
             "width": 400, "height": 300, "fov": 100, "id": "rgb_left"},
            {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3,
             "roll": 0.0, "pitch": 0.0, "yaw": 60.0,
             "width": 400, "height": 300, "fov": 100, "id": "rgb_right"},
            {"type": "sensor.camera.rgb", "x": -1.3, "y": 0.0, "z": 2.3,
             "roll": 0.0, "pitch": 0.0, "yaw": 180.0,
             "width": 400, "height": 300, "fov": 100, "id": "rgb_rear"},
            {"type": "sensor.lidar.ray_cast", "x": 1.3, "y": 0.0, "z": 2.5,
             "roll": 0.0, "pitch": 0.0, "yaw": -90.0, "id": "lidar"},
            {"type": "sensor.other.imu", "x": 0.0, "y": 0.0, "z": 0.0,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "sensor_tick": 0.05, "id": "imu"},
            {"type": "sensor.other.gnss", "x": 0.0, "y": 0.0, "z": 0.0,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "sensor_tick": 0.01, "id": "gps"},
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        ]

    # ------------------------------------------------------------------
    # Init (called on first real step)
    # ------------------------------------------------------------------
    def _init(self):
        self._route_planner = RoutePlanner(5, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self._instruction_planner = InstructionPlanner(self.scenario_cofing_name, True)
        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data["gps"]
        return (gps - self._route_planner.mean) * self._route_planner.scale

    # ------------------------------------------------------------------
    # Sensor preprocessing (same as original agent tick())
    # ------------------------------------------------------------------
    def tick(self, input_data):
        rgb_front = cv2.cvtColor(input_data["rgb_front"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left  = cv2.cvtColor(input_data["rgb_left"][1][:, :, :3],  cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data["rgb_right"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_rear  = cv2.cvtColor(input_data["rgb_rear"][1][:, :, :3],  cv2.COLOR_BGR2RGB)

        speed   = input_data["speed"][1]["speed"]
        compass = input_data["imu"][1][-1]
        if math.isnan(compass):
            compass = 0.0
        gps = input_data["gps"][1][:2]

        result = {
            "rgb_front": rgb_front, "rgb_left": rgb_left,
            "rgb_right": rgb_right, "rgb_rear": rgb_rear,
            "gps": gps, "speed": speed, "compass": compass,
        }
        pos = self._get_position(result)
        result["gps"] = pos

        # LiDAR: merge current + previous frame (same as original agent)
        lidar_raw = input_data["lidar"][1][..., :4]
        if self.prev_lidar is not None:
            lidar_full = np.concatenate([lidar_raw, self.prev_lidar])
        else:
            lidar_full = lidar_raw
        self.prev_lidar = lidar_raw
        result["lidar_raw"] = lidar_full

        next_wp, next_cmd = self._route_planner.run_step(pos)
        result["next_waypoint"] = next_wp
        result["next_command"] = next_cmd.value
        result["speed"] = speed

        theta = compass + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        local_cmd_pt = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        result["target_point"] = R.T.dot(local_cmd_pt)

        return result

    # ------------------------------------------------------------------
    # Main step — called by CARLA each tick
    # ------------------------------------------------------------------
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        self.step += 1

        # Hold brake for the first 20 frames (model warm-up in original agent)
        if self.step < 20:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            return control

        tick_data = self.tick(input_data)
        t_publish_start = time.time()

        # --- 1. Publish camera images ---
        self._pub_front.publish(self._np_to_imgmsg(tick_data["rgb_front"]))
        self._pub_left.publish( self._np_to_imgmsg(tick_data["rgb_left"]))
        self._pub_right.publish(self._np_to_imgmsg(tick_data["rgb_right"]))
        self._pub_rear.publish( self._np_to_imgmsg(tick_data["rgb_rear"]))

        # --- 2. Publish LiDAR point cloud ---
        lidar_msg = Float32MultiArray()
        lidar_msg.data = tick_data["lidar_raw"].flatten().tolist()
        self._pub_lidar.publish(lidar_msg)

        # --- 3. Publish navigation instruction (only when it changes) ---
        instruction = self._instruction_planner.command2instruct(
            self.town_id, tick_data, self._route_planner.route)
        if instruction != self.curr_instruction:
            self.curr_instruction = instruction
        instr_msg = String()
        instr_msg.data = self.curr_instruction
        self._pub_instr.publish(instr_msg)

        # --- 4. Publish ego state LAST — triggers inference on server side ---
        state_msg = Float32MultiArray()
        state_msg.data = [
            float(tick_data["speed"]),
            float(tick_data["target_point"][0]),
            float(tick_data["target_point"][1]),
            float(tick_data["next_command"]),
        ]
        self._waypoint_event.clear()
        self._pub_state.publish(state_msg)

        publish_ms = (time.time() - t_publish_start) * 1000

        # --- 5. Wait for waypoints from inference server ---
        got_wp = self._waypoint_event.wait(timeout=WAYPOINT_TIMEOUT)
        if not got_wp:
            print(f"[ROS2 agent] step={self.step}: waypoint timeout ({WAYPOINT_TIMEOUT}s), "
                  f"holding previous control")
            return self.prev_control

        roundtrip_ms = (time.time() - t_publish_start) * 1000
        print(f"[ROS2 agent] step={self.step} | "
              f"publish={publish_ms:.1f}ms | "
              f"inference={self._last_inference_ms:.1f}ms | "
              f"roundtrip={roundtrip_ms:.1f}ms")

        # --- 6. Convert waypoints → VehicleControl via PID ---
        waypoints_tensor = torch.from_numpy(self._waypoints)
        steer, throttle, brake, _ = self.control_pid(waypoints_tensor, tick_data["speed"])

        if brake < 0.05:
            brake = 0.0
        if brake > 0.1:
            throttle = 0.0

        control = carla.VehicleControl()
        control.steer    = float(steer) * 0.8
        control.throttle = float(throttle)
        control.brake    = float(brake)
        self.prev_control = control
        return control

    # ------------------------------------------------------------------
    # PID control — copied from original agent so we don't need the model
    # ------------------------------------------------------------------
    def control_pid(self, waypoints, velocity):
        assert waypoints.size(0) == 5
        waypoints = waypoints.data.cpu().numpy()
        waypoints[:, 1] *= -1  # flip y (forward is negative in our waypoints)

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        brake = (desired_speed < self.config.brake_speed or
                 (velocity / (desired_speed + 1e-6)) > self.config.brake_ratio)

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        if velocity < 0.01:
            angle = np.array(0.0)
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        delta = np.clip(desired_speed - velocity, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': float(velocity),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
        }
        return steer, throttle, brake, metadata

    def destroy(self):
        self._node.destroy_node()
        rclpy.shutdown()
