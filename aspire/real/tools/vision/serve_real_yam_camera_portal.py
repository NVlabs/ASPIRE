#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only Real-YAM camera Portal server for BundleSDF.

This is a small replacement for the legacy CAP server camera RPC surface in the
trimmed OpenForge real-YAM runtime. It intentionally exposes only the methods
that tools/vision/serve_bundlesdf.py calls:

  get_camera_image(camera)
  get_camera_depth(camera)
  get_camera_intrinsics(camera)
  get_camera_extrinsics(camera)

It does not expose any robot motion commands.
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from typing import Any

import numpy as np
import portal

from cap.env.real_bimanual_yam.env import RealYamEnv


log = logging.getLogger("real_yam_camera_portal")


class RealYamCameraPortal:
    def __init__(self, port: int = 8300) -> None:
        self._port = int(port)
        self._env = RealYamEnv(enable_cameras=True)
        self._server = portal.Server(self._port, errors=False)
        self._server.bind("health", self.health)
        self._server.bind("list_cameras", self.list_cameras)
        self._server.bind("get_camera_image", self.get_camera_image)
        self._server.bind("get_camera_depth", self.get_camera_depth)
        self._server.bind("get_camera_intrinsics", self.get_camera_intrinsics)
        self._server.bind("get_camera_extrinsics", self.get_camera_extrinsics)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "server": "real_yam_camera_portal",
            "configured_cameras": list(getattr(self._env, "camera_names", ())),
            "available_cameras": sorted(getattr(self._env, "_cameras", {}).keys()),
        }

    def list_cameras(self) -> dict[str, Any]:
        return {
            "configured": list(getattr(self._env, "camera_names", ())),
            "available": sorted(getattr(self._env, "_cameras", {}).keys()),
        }

    def _require_camera(self, camera: str) -> str:
        name = str(camera).strip().lower()
        configured = set(getattr(self._env, "camera_names", ()))
        if name not in configured:
            expected = ", ".join(getattr(self._env, "camera_names", ())) or "<none>"
            raise ValueError(f"Unknown camera {camera!r}; expected one of {expected}")
        return name

    def get_camera_image(self, camera: str) -> np.ndarray:
        name = self._require_camera(camera)
        rgb = self._env.render_rgb(name)
        if rgb is None:
            raise RuntimeError(f"No RGB frame available for camera {name!r}")
        return np.ascontiguousarray(rgb)

    def get_camera_depth(self, camera: str) -> np.ndarray:
        name = self._require_camera(camera)
        depth = self._env.render_depth(name)
        if depth is None:
            raise RuntimeError(f"No depth frame available for camera {name!r}")
        return np.ascontiguousarray(depth.astype(np.float32, copy=False))

    def get_camera_intrinsics(self, camera: str) -> list[float]:
        name = self._require_camera(camera)
        intrinsics = self._env.get_camera_intrinsics(name)
        if not intrinsics or len(intrinsics) != 4 or all(float(v) == 0.0 for v in intrinsics):
            raise RuntimeError(f"No intrinsics available for camera {name!r}")
        return [float(v) for v in intrinsics]

    def get_camera_extrinsics(self, camera: str) -> dict[str, Any]:
        name = self._require_camera(camera)
        return self._env.get_camera_extrinsics(name)

    def serve(self) -> None:
        self._server.start(block=False)
        log.info("Real-YAM camera portal serving on port %s", self._port)
        try:
            while True:
                if not self._server.loop.running:
                    raise RuntimeError(
                        "Portal loop exited unexpectedly "
                        f"with exitcode={self._server.loop.exitcode}"
                    )
                if not self._server.socket.thread.running:
                    error = self._server.socket.error
                    if error is not None:
                        raise RuntimeError("Portal socket thread crashed") from error
                    raise RuntimeError("Portal socket thread exited unexpectedly")
                time.sleep(0.2)
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            log.exception("Failed to close RealYamEnv")
        if self._server.running:
            self._server.close(timeout=1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve read-only Real-YAM camera RPCs for BundleSDF"
    )
    parser.add_argument("--port", type=int, default=8300)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

    server = RealYamCameraPortal(port=args.port)

    def _stop(signum, _frame) -> None:
        log.info("Received signal %s; shutting down", signum)
        server.close()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    server.serve()


if __name__ == "__main__":
    main()
