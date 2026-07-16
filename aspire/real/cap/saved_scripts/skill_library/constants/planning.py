"""Motion-planning defaults inherited from saved-script baselines."""

PLANNING_SPEED = 1.5
IK_ERROR_THRESHOLD_M = 0.01
IK_XYZ_WEIGHT = 1.0
IK_RPY_WEIGHT = 0.3
BATCH_TOP_K = 16
BATCH_SOLVER_SPEED = "fast"
BATCH_VALIDATE_TRAJECTORY = False
MOTION_PLANNER_BACKEND = "curobo"
