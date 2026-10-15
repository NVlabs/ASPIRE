# Bimanual control smoke (RoboSuite, two_arm_lift). Public-API replacement for
# the former private validate_robosuite_control gate. It verifies the arm-control
# invariants that the bimanual task programs do not isolate -- independent arm
# actuation, gripper-target persistence under motion, and homing -- using only
# allowlisted CaP tools, and asserts on any violation so a failure surfaces as
# program_ok == False. It is provider-free by construction.
#
# Simultaneous bimanual execution via the public move_synchronized (integrated
# cuRobo) path is exercised by the bimanual acceptance plan
# (configs/validation/robosuite-bimanual.yaml, two_arm_lift), so it is not
# re-checked here.
report = {"checks": {}, "details": {}}

home = get_robot_state()
arms = list(home.arms)

# 1. Independent actuation: moving one arm must not disturb the other.
for arm in arms:
    other = "secondary" if arm == "primary" else "primary"
    before = get_robot_state()
    target = before.joint_positions[arm].copy()
    target[0] = target[0] + 0.03
    result = move_to_joints(target, arm=arm, tolerance=0.01, max_steps=120)
    after = get_robot_state()
    other_delta = float(max(abs(after.joint_positions[other] - before.joint_positions[other])))
    ok = bool(result.ok) and other_delta < 0.01
    report["checks"][arm + "_independent"] = ok
    report["details"][arm + "_other_arm_delta_rad"] = other_delta
    assert ok, arm + " actuation disturbed the other arm (delta=" + str(other_delta) + ")"

# 2. Gripper-target persistence: a closed gripper stays closed through arm motion.
close_gripper(arm="primary")
gripper_before = get_robot_state().gripper_positions["primary"]
persist_target = get_robot_state().joint_positions["primary"].copy()
persist_target[0] = persist_target[0] + 0.02
motion = move_to_joints(persist_target, arm="primary", tolerance=0.01, max_steps=120)
gripper_after = get_robot_state().gripper_positions["primary"]
persisted = bool(motion.ok) and gripper_after <= gripper_before + 1e-6
report["checks"]["gripper_target_persisted"] = persisted
report["details"]["primary_gripper_before_after"] = [float(gripper_before), float(gripper_after)]
assert persisted, "gripper target did not persist through motion"

# 3. Homing: both arms return to their initial configuration.
for arm in arms:
    go_home(arm=arm)
final = get_robot_state()
home_error = {
    arm: float(max(abs(final.joint_positions[arm] - home.joint_positions[arm]))) for arm in arms
}
homed = max(home_error.values()) < 0.01
report["checks"]["home_error_below_0_01_rad"] = homed
report["details"]["home_error_rad"] = home_error
assert homed, "arms did not return home: " + str(home_error)

report["success"] = all(report["checks"].values())
