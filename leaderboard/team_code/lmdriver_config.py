import os


class GlobalConfig:
    """base architecture configurations"""

    # Controller
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40  # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40  # buffer size

    max_throttle = 0.75  # upper limit on throttle signal value in dataset
    brake_speed = 0.1  # desired speed below which brake is triggered
    brake_ratio = 1.1  # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.35  # maximum change in speed input to logitudinal controller

    # Paths relative to project root (run_evaluation.sh is executed from LMDrive/)
    llm_model = 'models/llava-v1.5-7b'
    preception_model = 'memfuser_baseline_e1d3_return_feature'
    preception_model_ckpt = 'models/LMDrive-vision-encoder-r50-v1.0/vision-encoder-r50.pth.tar'
    lmdrive_ckpt = 'models/LMDrive-llava-v1.5-7b-v1.0/llava-v1.5-checkpoint.pth'

    agent_use_notice = False
    sample_rate = 2
    quantization = "8bit"  # None, "4bit", or "8bit"
    display_mode = "camera"  # "pygame" (full HUD) or "camera" (lightweight cv2)
    display_update_interval = 1  # update display every N steps (1=every step, 2=every 2nd, etc.)


    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
