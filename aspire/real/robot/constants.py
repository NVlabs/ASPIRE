import os
import sys

# ---------------------------------------------------------------------------
# CAN bus type: gs_usb on macOS (USB-CAN adapters), socketcan on Linux
# ---------------------------------------------------------------------------
CAN_BUSTYPE = "gs_usb" if sys.platform == "darwin" else "socketcan"

# ---------------------------------------------------------------------------
# Arm CAN interfaces
# Linux (socketcan): kernel interface names like "can_leader_l"
# macOS (gs_usb): USB serial number strings — discover with:
#   python -c "from gs_usb.gs_usb import GsUsb
#   for i,d in enumerate(GsUsb.scan()): print(i, d.serial_number)"
# ---------------------------------------------------------------------------
if sys.platform == "darwin":
    RIGHT_LEADER_CAN_INTERFACE = os.environ.get("YAM_RIGHT_LEADER_CAN_INTERFACE", "")
    LEFT_LEADER_CAN_INTERFACE = os.environ.get("YAM_LEFT_LEADER_CAN_INTERFACE", "")
    LEFT_FOLLOWER_CAN_INTERFACE = os.environ.get("YAM_LEFT_FOLLOWER_CAN_INTERFACE", "")
    RIGHT_FOLLOWER_CAN_INTERFACE = os.environ.get("YAM_RIGHT_FOLLOWER_CAN_INTERFACE", "")
else:
    LEFT_FOLLOWER_CAN_INTERFACE = "can_follow_l"
    RIGHT_FOLLOWER_CAN_INTERFACE = "can_follow_r"
    LEFT_LEADER_CAN_INTERFACE = "can_leader_l"
    RIGHT_LEADER_CAN_INTERFACE = "can_leader_r"

# Arm server ports
LEFT_FOLLOWER_PORT = 11333
RIGHT_FOLLOWER_PORT = 11334
LEFT_LEADER_PORT = 11335
RIGHT_LEADER_PORT = 11336
PICO_PORT = 8963

# YAM arm motor configuration
YAM_ARM_MOTOR_IDS = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06]
YAM_ARM_MOTOR_TYPES = ["4340", "4340", "4340", "4310", "4310", "4310"]
YAM_GRIPPER_MOTOR_ID = 0x07
YAM_GRIPPER_MOTOR_TYPE = "4310"  # linear_4310 gripper

# Default PD gains (from i2rt/get_yam_robot)
YAM_ARM_KP = [80.0, 80.0, 80.0, 40.0, 10.0, 10.0]
YAM_ARM_KD = [5.0, 5.0, 5.0, 1.5, 1.5, 1.5]
YAM_GRIPPER_KP = 20.0
YAM_GRIPPER_KD = 0.5

# Gripper FORCE_POS control parameters
YAM_GRIPPER_VEL_LIMIT = 30.0  # rad/s (0-100)
YAM_GRIPPER_TORQUE_LIMIT_NM = 0.75  # Nm (4310 T_max=10 Nm)

# Gripper motor direction: motor_pos → env_pos = SIGN * motor_pos
YAM_GRIPPER_SIGN = -1

# ---------------------------------------------------------------------------
# Safety: max joint velocity (rad/s)
#
# Single limit enforced at both the policy layer and the env layer.
# Per-step delta is computed at runtime:
#   max_delta_per_step = MAX_JOINT_VELOCITY_RAD_S / control_hz
#
# At 30 Hz:  6 / 30 = 0.2 rad/step
# At 60 Hz:  6 / 60 = 0.1 rad/step
# ---------------------------------------------------------------------------
MAX_JOINT_VELOCITY_RAD_S = 6  # rad/s — max safe joint speed for all layers. 6 rad/s  (~344 deg/s)
