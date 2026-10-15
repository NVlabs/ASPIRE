"""Typed Contact-GraspNet client using the service's numpy/base64 wire format."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
import io

import numpy as np
import requests

from cap_harness.contracts import GraspCandidate, GraspSet, ObjectGeometry, PointCloud
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import matrix_to_pose

from ..http import HttpProviderClient, ProviderHttpError
from .wire import PLAN_POINT_CLOUDS_PATH

DEFAULT_GRASPNET_URL = "http://127.0.0.1:8115"
_LOCAL_Z_90_DEG = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _encode_numpy(array: np.ndarray) -> str:
    with io.BytesIO() as buffer:
        np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_numpy(value: object) -> np.ndarray:
    if not isinstance(value, str):
        raise ValueError("numpy payload must be a base64 string")
    try:
        raw = base64.b64decode(value, validate=True)
        with io.BytesIO(raw) as buffer:
            result = np.load(buffer, allow_pickle=False)
    except (ValueError, OSError, binascii.Error) as exc:
        raise ValueError("invalid numpy/base64 payload") from exc
    return np.asarray(result)


class ContactGraspNetProvider:
    """Generate grasp transforms in exactly the input point-cloud frame."""

    def __init__(
        self,
        base_url: str = DEFAULT_GRASPNET_URL,
        *,
        timeout_s: float | tuple[float, float] = 120.0,
        max_retries: int = 2,
        retry_backoff_s: float = 0.1,
        service_max_retries: int = 7,
        forward_passes: int = 1,
        default_width_m: float = 0.08,
        session: requests.Session | None = None,
    ) -> None:
        if service_max_retries < 0:
            raise ValueError("service_max_retries must be non-negative")
        if forward_passes <= 0:
            raise ValueError("forward_passes must be positive")
        if not np.isfinite(default_width_m) or default_width_m < 0:
            raise ValueError("default_width_m must be finite and non-negative")
        self._http = HttpProviderClient(
            base_url,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            session=session,
        )
        self.service_max_retries = int(service_max_retries)
        self.forward_passes = int(forward_passes)
        self.default_width_m = float(default_width_m)

    @property
    def base_url(self) -> str:
        return self._http.base_url

    def health(self) -> bool:
        return self._http.health()

    def generate_grasps(
        self,
        point_cloud: PointCloud,
        *,
        scene_point_cloud: PointCloud | None = None,
        geometry: ObjectGeometry | None = None,
        max_candidates: int | None = None,
    ) -> GraspSet:
        if not isinstance(point_cloud, PointCloud):
            return self._failure(ErrorCode.INVALID_REQUEST, "point_cloud must be PointCloud")
        if geometry is not None:
            if not isinstance(geometry, ObjectGeometry):
                return self._failure(ErrorCode.INVALID_REQUEST, "geometry must be ObjectGeometry")
            if geometry.frame != point_cloud.frame:
                return self._failure(
                    ErrorCode.INVALID_REQUEST,
                    "geometry and point_cloud frames do not match",
                    details={
                        "geometry_frame": geometry.frame,
                        "point_cloud_frame": point_cloud.frame,
                    },
                )
        if scene_point_cloud is not None:
            if not isinstance(scene_point_cloud, PointCloud):
                return self._failure(
                    ErrorCode.INVALID_REQUEST,
                    "scene_point_cloud must be PointCloud or None",
                )
            if scene_point_cloud.frame != point_cloud.frame:
                return self._failure(
                    ErrorCode.INVALID_REQUEST,
                    "scene and segment point-cloud frames do not match",
                    details={
                        "scene_frame": scene_point_cloud.frame,
                        "segment_frame": point_cloud.frame,
                    },
                )
        if max_candidates is not None and (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates <= 0
        ):
            return self._failure(
                ErrorCode.INVALID_REQUEST, "max_candidates must be a positive integer or None"
            )

        payload = {
            "pc_full_base64": _encode_numpy((scene_point_cloud or point_cloud).points),
            "pc_segment_base64": _encode_numpy(point_cloud.points),
            "segmap_id": 1,
            "local_regions": True,
            "filter_grasps": True,
            "forward_passes": self.forward_passes,
            "max_retries": self.service_max_retries,
        }
        try:
            response = self._http.post_json(PLAN_POINT_CLOUDS_PATH, payload)
            transforms = _decode_numpy(response.get("grasps_base64"))
            scores = _decode_numpy(response.get("scores_base64"))
            contacts = _decode_numpy(response.get("contact_pts_base64"))
            candidates = self._decode_candidates(
                transforms,
                scores,
                contacts,
                frame=point_cloud.frame,
                max_candidates=max_candidates,
            )
        except ProviderHttpError as exc:
            return GraspSet(ok=False, error=exc.error)
        except (TypeError, ValueError, KeyError) as exc:
            return self._failure(
                ErrorCode.GRASP_FAILED,
                "Contact-GraspNet returned an invalid response",
                details={"provider": "contact_graspnet", "error": str(exc)},
            )

        if not candidates:
            return self._failure(
                ErrorCode.NO_GRASP,
                "Contact-GraspNet returned no grasp candidates",
                details={"provider": "contact_graspnet", "frame": point_cloud.frame},
            )
        return GraspSet(
            ok=True,
            grasps=candidates,
            diagnostics={"provider": "contact_graspnet", "frame": point_cloud.frame},
        )

    def _decode_candidates(
        self,
        transforms: np.ndarray,
        scores: np.ndarray,
        contacts: np.ndarray,
        *,
        frame: str,
        max_candidates: int | None,
    ) -> list[GraspCandidate]:
        if transforms.size == 0:
            return []
        if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
            raise ValueError(f"grasps must have shape (N, 4, 4), got {transforms.shape}")
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(scores) != len(transforms):
            raise ValueError("grasp and score counts do not match")
        if not bool(np.all(np.isfinite(transforms))) or not bool(np.all(np.isfinite(scores))):
            raise ValueError("grasp transforms and scores must be finite")
        if bool(np.any((scores < 0.0) | (scores > 1.0))):
            raise ValueError("grasp scores must be in [0, 1]")
        if contacts.size == 0:
            contacts = np.empty((len(transforms), 0), dtype=np.float64)
        elif contacts.shape != (len(transforms), 3) or not bool(np.all(np.isfinite(contacts))):
            raise ValueError("contact points must have finite shape (N, 3)")

        order = np.argsort(-scores, kind="stable")
        if max_candidates is not None:
            order = order[:max_candidates]
        candidates: list[GraspCandidate] = []
        for index in order:
            raw_transform = np.asarray(transforms[index], dtype=np.float64)
            transform = raw_transform @ _LOCAL_Z_90_DEG
            metadata: dict[str, object] = {
                "orientation_correction": "right_multiply_local_z_90_deg"
            }
            if contacts.shape[1:] == (3,):
                metadata["contact_point"] = tuple(float(value) for value in contacts[index])
            candidates.append(
                GraspCandidate(
                    pose=matrix_to_pose(transform, frame=frame),
                    score=float(scores[index]),
                    width_m=self.default_width_m,
                    metadata=metadata,
                )
            )
        return candidates

    @staticmethod
    def _failure(
        code: ErrorCode,
        message: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> GraspSet:
        return GraspSet(
            ok=False,
            error=ApiError(code=code, message=message, details=details or {}),
        )


GraspNetProvider = ContactGraspNetProvider
ContactGraspNetClient = ContactGraspNetProvider

__all__ = [
    "DEFAULT_GRASPNET_URL",
    "ContactGraspNetClient",
    "ContactGraspNetProvider",
    "GraspNetProvider",
]
