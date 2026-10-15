# Home releases both grippers before the single whole-robot home motion.
for arm in ("left", "right"):
    result = set_gripper(1.0, arm=arm)
    if not result.ok:
        break
else:
    result = go_home(arm="both")
