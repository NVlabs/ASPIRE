"""Validate pose JSON locally; these tests start no MCP server or robot session."""

import jsonschema
import numpy as np
import pytest

from cap_harness.agent.mcp import _pose_input_schema
from cap_harness.geometry import pose_from_mapping


@pytest.mark.parametrize(
    "orientation",
    [{"rpy_deg": [0, 0, 90]}, {"quaternion_wxyz": [2**-0.5, 0, 0, 2**-0.5]}],
)
def test_pose_schema_accepts_both_orientation_formats(orientation):
    target = {"position": [0.5, 0.2, 0.9], "frame": "world", **orientation}
    jsonschema.validate(target, _pose_input_schema())

    pose = pose_from_mapping(target)

    np.testing.assert_allclose(pose.as_matrix()[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    assert pose.frame == "world"
    np.testing.assert_allclose(pose.position, target["position"])


@pytest.mark.parametrize(
    "orientation",
    [
        {},
        {"rpy_deg": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
        {"rpy_deg": [0, 0]},
        {"rpy_deg": [0, 0, 0, 0]},
        {"rpy_deg": [[0, 0, 0]]},
        {"rpy_deg": [True, 0, 0]},
        {"rpy_deg": ["0", 0, 0]},
        {"rpy_deg": None},
        {"quaternion_wxyz": [1, 0, 0]},
        {"rpy": [0, 0, 0]},
        {"rpy_deg": [0, 0, 0], "unknown": 1},
    ],
)
def test_pose_schema_rejects_ambiguous_or_malformed_orientations(orientation):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {"position": [0.5, 0.2, 0.9], "frame": "world", **orientation},
            _pose_input_schema(),
        )


@pytest.mark.parametrize("field", ["position", "frame"])
def test_pose_target_requires_position_and_frame(field):
    target = {"position": [0.5, 0.2, 0.9], "frame": "world", "rpy_deg": [0, 0, 0]}
    del target[field]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(target, _pose_input_schema())
    with pytest.raises(ValueError, match="position and frame"):
        pose_from_mapping(target)
