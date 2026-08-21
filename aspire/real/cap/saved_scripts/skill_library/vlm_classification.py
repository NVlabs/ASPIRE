# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    from skill_library.namespace import *  # noqa: F401,F403
except AttributeError:
    pass

import collections
import json

try:
    from skill_library.constants.vision import (  # type: ignore
        DEFAULT_VLM_CAMERAS,
        MAJORITY,
        NUM_VOTES,
        VLM_BACKEND,
        VLM_MODEL,
    )
except Exception:
    VLM_BACKEND = "gemini"
    VLM_MODEL = "gemini-2.5-flash"
    DEFAULT_VLM_CAMERAS = ("top", "left", "right")
    NUM_VOTES = 1
    MAJORITY = 1


def _camera_media(cameras=None):
    if cameras is None:
        cameras = DEFAULT_VLM_CAMERAS
    if isinstance(cameras, str):
        cameras = [c.strip() for c in cameras.split(",")]
    media = [f"camera:{str(camera).strip()}" for camera in cameras if str(camera).strip()]
    return media or ["camera:top"]


def query(prompt, cameras=None, backend=None, model=None):
    return vlm_query(
        text=prompt,
        backend=backend or VLM_BACKEND,
        model=model or VLM_MODEL,
        media=_camera_media(cameras),
    )


def _normalize_items(value):
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "none":
            return []
        raw_items = text.split(",")
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = [str(value)]

    items = []
    seen = set()
    for raw_item in raw_items:
        item = str(raw_item).strip().lower()
        if len(item) >= 2 and item[0] == item[-1] and item[0] in {"'", '"'}:
            item = item[1:-1].strip()
        item = item.strip("`*_ ")
        item = item.lstrip("-•* ")
        item = item.rstrip(" .;:")
        item = " ".join(item.split())
        if not item or item == "none" or item in seen:
            continue
        seen.add(item)
        items.append(item)
    return items


def _assignment_fields(assignment):
    if len(assignment) == 2:
        class_name, target_name = assignment
    elif len(assignment) >= 3:
        class_name, _, target_name = assignment[:3]
    else:
        raise ValueError(f"Invalid assignment: {assignment!r}")
    return str(class_name), str(target_name)


def _target_names(assignments):
    return [_assignment_fields(assignment)[1] for assignment in assignments]


def build_assignments(classes, targets):
    if len(classes) != len(targets):
        raise ValueError(
            f"classes ({len(classes)}) and targets ({len(targets)}) must have the same length"
        )
    assignments = [(str(class_name), str(target_name)) for class_name, target_name in zip(classes, targets)]
    if not assignments:
        raise ValueError("At least one class-target assignment is required")
    class_names = [class_name for class_name, _ in assignments]
    target_names = [target_name for _, target_name in assignments]
    if len(set(class_names)) != len(class_names):
        raise ValueError(f"classes must be unique: {class_names}")
    if len(set(target_names)) != len(target_names):
        raise ValueError(f"targets must be unique: {target_names}")
    return assignments


def _build_json_example(assignments):
    return json.dumps({target_name: [] for target_name in _target_names(assignments)})


def build_prompt(assignments):
    lines = [
        "Identify objects on the table that still need sorting.",
        "For each class-target pair below, list only the objects of that class that still need to be moved to that target.",
        "Return JSON only, using the exact target names as keys and arrays of object names as values.",
        "Do not include markdown fences or any extra commentary.",
        f"Example: {_build_json_example(assignments)}",
    ]
    for index, assignment in enumerate(assignments, start=1):
        class_name, target_name = _assignment_fields(assignment)
        lines.append(f"{index}. {class_name} -> {target_name}")
    return "\n".join(lines)


def _extract_bracket_lists(text):
    lists = []
    start = None
    depth = 0
    for index, char in enumerate(text):
        if char == "[":
            if depth == 0:
                start = index
            depth += 1
        elif char == "]" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                lists.append(text[start : index + 1])
                start = None
    return lists


def _parse_bracket_list(text):
    text = text.strip()
    if not text.startswith("[") or not text.endswith("]"):
        return _normalize_items(text)
    inner = text[1:-1].strip()
    if not inner:
        return []
    return _normalize_items(inner.split(","))


def _find_assignment_sections(text, assignments):
    lowered = text.lower()
    matches = []
    for index, assignment in enumerate(assignments):
        class_name, target_name = _assignment_fields(assignment)
        positions = []
        for marker in (class_name, target_name):
            marker = marker.strip().lower()
            if marker:
                pos = lowered.find(marker)
                if pos != -1:
                    positions.append(pos)
        if positions:
            matches.append((min(positions), index))

    matches.sort()
    sections = []
    for match_index, (start, assignment_index) in enumerate(matches):
        end = matches[match_index + 1][0] if match_index + 1 < len(matches) else len(text)
        sections.append((assignment_index, text[start:end]))
    return sections


def parse_response(response, assignments):
    text = str(response).strip()
    target_names = _target_names(assignments)
    result = {target_name: [] for target_name in target_names}

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            for target_name in target_names:
                result[target_name] = _normalize_items(parsed.get(target_name))
            return result

    found_targets = set()
    for index, section in _find_assignment_sections(text, assignments):
        _, target_name = _assignment_fields(assignments[index])
        bracket_lists = _extract_bracket_lists(section)
        if bracket_lists:
            result[target_name] = _parse_bracket_list(bracket_lists[0])
            found_targets.add(target_name)
            continue
        if ":" in section:
            _, value = section.split(":", 1)
            result[target_name] = _normalize_items(value)
            found_targets.add(target_name)

    ordered_lists = [_parse_bracket_list(item) for item in _extract_bracket_lists(text)]
    if len(ordered_lists) >= len(assignments):
        fallback_lists = ordered_lists[::2][: len(assignments)] if len(ordered_lists) >= 2 * len(assignments) else ordered_lists[: len(assignments)]
        for assignment, items in zip(assignments, fallback_lists):
            _, target_name = _assignment_fields(assignment)
            if target_name not in found_targets:
                result[target_name] = items
    return result


def survey(
    classes,
    targets,
    cameras=None,
    backend=None,
    model=None,
    num_votes=None,
    majority=None,
    go_birdeye=True,
    birdseye_kwargs=None,
):
    if go_birdeye:
        try:
            from skill_library.pick_place import go_birdeye as _go_birdeye

            _go_birdeye(**(birdseye_kwargs or {}))
        except Exception as exc:
            print(f"  Birdseye move before VLM survey failed/skipped: {exc}")

    assignments = classes if targets is None else build_assignments(classes, targets)
    target_names = _target_names(assignments)
    prompt = build_prompt(assignments)
    counts = {target_name: collections.Counter() for target_name in target_names}
    votes = int(NUM_VOTES if num_votes is None else num_votes)
    threshold = int(MAJORITY if majority is None else majority)

    for vote_index in range(votes):
        response = query(prompt, cameras=cameras, backend=backend, model=model)
        parsed = parse_response(response, assignments)
        print(f"    Vote {vote_index + 1}: {str(response).strip()}")
        print(f"    Parsed {vote_index + 1}: {parsed}")
        for target_name in target_names:
            for item in parsed.get(target_name, []):
                counts[target_name][item] += 1

    survey_result = {
        target_name: [item for item, count in counts[target_name].items() if count >= threshold]
        for target_name in target_names
    }
    remaining_objects = []
    for target_name in target_names:
        for item in survey_result[target_name]:
            if item not in remaining_objects:
                remaining_objects.append(item)

    print("\n=== Detect objects to sort with VLM ===")
    for assignment in assignments:
        class_name, target_name = _assignment_fields(assignment)
        print(
            f"    Majority {class_name} -> {target_name}: "
            f"{survey_result[target_name]} (from counts: {dict(counts[target_name])})"
        )
    print(f"Objects remaining to sort: {remaining_objects} (count={len(remaining_objects)})")
    return survey_result, len(remaining_objects)
