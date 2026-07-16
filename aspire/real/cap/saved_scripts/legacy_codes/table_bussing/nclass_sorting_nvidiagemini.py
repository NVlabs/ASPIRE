from skill_library.constants.manipulation import MAX_NO_PROGRESS_ROUNDS
from skill_library.constants.planning import BATCH_SOLVER_SPEED, BATCH_TOP_K
from skill_library.constants.sorting import SORT_VLM_CONFIG, TABLE_SORT_RUN_CONFIG
from skill_library.namespace import go_home
from skill_library.pick_place import pick_and_place
from skill_library.vlm_classification import build_assignments, survey


TARGETS = ["red plate"]  # ["green basket"]
CLASSES = ["red grapes"]  # ["red grapes"]
ASSIGNMENTS = build_assignments(CLASSES, TARGETS)

RUN_CONFIG = dict(TABLE_SORT_RUN_CONFIG)
VLM_CONFIG = dict(
    SORT_VLM_CONFIG,
    backend="nvidia",
    model="gcp/google/gemini-3-flash-preview",
    birdseye_kwargs=RUN_CONFIG,
)

print(f"[DEBUG] class_to_target mapping: {dict(ASSIGNMENTS)}")
print(
    "[DEBUG] Table sorting stack: NVIDIA Gemini survey + SAM3 + AnyGrasp(top16) + "
    f"batch cuRobo(top_k={BATCH_TOP_K}, solver={BATCH_SOLVER_SPEED})"
)

objects_by_target, remaining = survey(ASSIGNMENTS, None, **VLM_CONFIG)
no_progress_rounds = 0

while remaining > 0:
    moved_any = False
    for obj_class, target_name in ASSIGNMENTS:
        items = objects_by_target.get(target_name, [])
        print(f"[DEBUG] class={obj_class}, target={target_name}, items={items}")
        for obj_name in items:
            print(f"--- Sorting obj={obj_name} (class={obj_class}) -> {target_name} ---")
            moved_any |= pick_and_place(obj_name, target_name, grasp_mode="anygrasp", **RUN_CONFIG)

    if not moved_any:
        no_progress_rounds += 1
        print(f"[DEBUG] No objects moved ({no_progress_rounds}/{MAX_NO_PROGRESS_ROUNDS})")
        if no_progress_rounds >= MAX_NO_PROGRESS_ROUNDS:
            break
    else:
        no_progress_rounds = 0

    objects_by_target, remaining = survey(ASSIGNMENTS, None, **VLM_CONFIG)
go_home()
print("\nAll done! Table sorting complete.")
