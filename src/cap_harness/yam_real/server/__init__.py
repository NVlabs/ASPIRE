"""Level 2: the follower arm servers that own the CAN bus.

One process per arm. Each buffers the most recent validated target and re-sends
it to the motors at the profile's hold rate, so a late or dropped RPC leaves the
arm holding its last valid pose rather than going slack.

These run outside a harness run and outlive it. Nothing in the client half
starts or stops them, because stopping one de-energizes the motors.
"""

from cap_harness.yam_real.server.arm_server import FollowerArmServer
from cap_harness.yam_real.server.motors import DaMiaoBus, MotorBus, MotorState
from cap_harness.yam_real.server.yam_robot import YamRobot

__all__ = [
    "DaMiaoBus",
    "FollowerArmServer",
    "MotorBus",
    "MotorState",
    "YamRobot",
]
