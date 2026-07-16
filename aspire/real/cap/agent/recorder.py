"""Video recorder for cap scripts and agent runs.

ScriptRecorder writes camera frames to MP4 files, then re-encodes to H.264.

Two modes:

  CapServer mode — timer-based pull from server cameras (hardware/legacy):
      recorder = ScriptRecorder(server, Path("video"))

  Env push mode — frames pushed after each env.step(), no background timer:
      recorder = ScriptRecorder.from_env(camera_names, Path("video"))
      env.set_recorder(recorder)   # env calls recorder.push_frame() on step
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from cap.server.cap_server import CapServer

_RECORD_FPS = 5


class PreviewRecordingError(RuntimeError):
    """Raised when BundleSDF preview recording cannot produce video evidence."""


class ScriptRecorder:
    def __init__(self, server: CapServer, output_dir: Path) -> None:
        """CapServer mode — timer pulls frames from server cameras."""
        self._server = server
        self._cam_names: list[str] = list(server._cameras.keys())
        self._get_image = server.get_camera_image
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._frame_queue: queue.Queue[tuple[str, np.ndarray] | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._capture_thread: threading.Thread | None = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="script-recorder-capture",
        )
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="script-recorder-writer",
        )

    @classmethod
    def from_env(
        cls,
        camera_names: list[str],
        output_dir: Path,
    ) -> ScriptRecorder:
        """Env push mode — no timer thread; frames pushed via push_frame()."""
        instance = object.__new__(cls)
        instance._server = None
        instance._cam_names = list(camera_names)
        instance._get_image = None
        instance._output_dir = output_dir
        instance._output_dir.mkdir(parents=True, exist_ok=True)
        instance._writers = {}
        instance._frame_queue = queue.Queue()
        instance._stop_event = threading.Event()
        instance._capture_thread = None  # no timer in push mode
        instance._writer_thread = threading.Thread(
            target=instance._writer_loop,
            daemon=True,
            name="script-recorder-writer",
        )
        return instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        print(f"[recorder] Saving to {self._output_dir}")
        if self._capture_thread is not None:
            self._capture_thread.start()
        self._writer_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join()
        self._frame_queue.put(None)  # sentinel
        self._writer_thread.join()
        for w in self._writers.values():
            w.release()
        self._reencode_videos()
        print(f"[recorder] Videos saved to {self._output_dir}")

    def push_frame(self, cam_name: str, frame: np.ndarray) -> None:
        """Push a frame directly (env push mode). Safe to call from any thread."""
        if self._stop_event.is_set():
            return
        if frame is not None and frame.ndim == 3 and frame.shape[0] > 1:
            self._frame_queue.put((cam_name, frame.copy()))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        import time
        interval = 1.0 / _RECORD_FPS
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            for cam in self._cam_names:
                try:
                    frame = self._get_image(cam)
                    if frame is not None and frame.shape[0] > 1:
                        self._frame_queue.put((cam, frame.copy()))
                except Exception:
                    pass
            elapsed = time.monotonic() - t0
            remaining = interval - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)

    def _writer_loop(self) -> None:
        while True:
            item = self._frame_queue.get()
            if item is None:
                break
            cam, frame = item
            if cam not in self._writers:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                path = str(self._output_dir / f"{cam}.mp4")
                self._writers[cam] = cv2.VideoWriter(path, fourcc, _RECORD_FPS, (w, h))
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self._writers[cam].write(bgr)

    def _reencode_videos(self) -> None:
        for mp4 in sorted(self._output_dir.glob("*.mp4")):
            tmp = mp4.with_suffix(".tmp.mp4")
            mp4.rename(tmp)
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(tmp),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "30",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-pix_fmt", "yuv420p", str(mp4),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                tmp.unlink()
            except (subprocess.CalledProcessError, FileNotFoundError):
                tmp.rename(mp4)


class PreviewStreamRecorder:
    """Record HTTP MJPEG preview streams to MP4 with ffmpeg.

    Real YAM usually runs scripts with ``runtime.no_cameras=true`` so the
    BundleSDF camera Portal keeps exclusive ownership of the RealSense devices.
    In that mode ``ScriptRecorder`` cannot pull frames from ``env.render_rgb``.
    This recorder captures the already-running BundleSDF preview streams
    instead, preserving real camera video without opening camera devices again.
    """

    def __init__(
        self,
        base_url: str,
        camera_names: list[str],
        output_dir: Path,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._cam_names = list(camera_names)
        self._output_dir = output_dir
        self._processes: dict[str, subprocess.Popen] = {}
        self._logs: dict[str, object] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._thread_errors: dict[str, str] = {}
        self._stop_event = threading.Event()
        self._backend = os.environ.get("OPENFORGE_PREVIEW_RECORDER_BACKEND", "ffmpeg").strip().lower() or "ffmpeg"

    def start(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        if not self._cam_names:
            raise PreviewRecordingError(
                "BundleSDF preview recording requested but no cameras were configured"
            )
        print(f"[preview-recorder] Saving BundleSDF previews to {self._output_dir}")
        self._preflight_preview_streams()
        if self._backend in {"python", "cv2", "opencv"}:
            self._start_python_recorders()
            return
        for cam in self._cam_names:
            url = f"{self._base_url}/preview/{cam}"
            video_path = self._output_dir / f"{cam}.mp4"
            log_path = self._output_dir / f"{cam}.ffmpeg.log"
            log_file = None
            try:
                log_file = log_path.open("w", encoding="utf-8")
                proc = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "warning",
                        "-y",
                        "-i",
                        url,
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "28",
                        "-pix_fmt",
                        "yuv420p",
                        str(video_path),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=log_file,
                )
            except FileNotFoundError:
                if log_file is not None:
                    log_file.close()
                self._close_logs()
                raise PreviewRecordingError(
                    "ffmpeg not found; BundleSDF preview recording cannot produce MP4 evidence"
                )
            except OSError as exc:
                if log_file is not None:
                    log_file.close()
                self._close_logs()
                self._terminate_processes()
                raise PreviewRecordingError(
                    f"failed to start {cam} preview recorder for {url}: {exc}"
                ) from exc
            self._processes[cam] = proc
            self._logs[cam] = log_file
            print(f"[preview-recorder] {cam}: {url} -> {video_path}")

    def stop(self) -> None:
        if self._backend in {"python", "cv2", "opencv"}:
            self._stop_python_recorders()
            self._reencode_python_outputs_to_h264()
        else:
            self._terminate_processes()
            self._close_logs()
        result = self._validate_outputs()
        result_path = self._output_dir / "preview_recording_result.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        failures = [item for item in result["videos"] if not item["ok"]]
        if failures:
            summary = ", ".join(
                f"{item['camera']}:{item['problem']}" for item in failures
            )
            raise PreviewRecordingError(
                "BundleSDF preview recording failed to produce expected MP4 evidence "
                f"({summary}); details: {result_path}"
            )
        print(f"[preview-recorder] Videos saved to {self._output_dir}")

    def _probe_timeout_s(self) -> float:
        raw = os.environ.get("OPENFORGE_PREVIEW_RECORDER_PROBE_TIMEOUT_S", "").strip()
        try:
            return max(0.25, float(raw)) if raw else 3.0
        except ValueError:
            return 3.0

    def _probe_min_bytes(self) -> int:
        raw = os.environ.get("OPENFORGE_PREVIEW_RECORDER_PROBE_MIN_BYTES", "").strip()
        try:
            return max(1, int(raw)) if raw else 64
        except ValueError:
            return 64

    def _require_h264_outputs(self) -> bool:
        raw = os.environ.get(
            "OPENFORGE_PREVIEW_RECORDER_REQUIRE_H264",
            "1",
        ).strip().lower()
        return raw not in {"0", "false", "no", "off"}

    def _preflight_preview_streams(self) -> None:
        results = []
        failures = []
        for cam in self._cam_names:
            url = f"{self._base_url}/preview/{cam}"
            result = self._probe_preview_stream(cam, url)
            results.append(result)
            if not result["ok"]:
                failures.append(result)
        path = self._output_dir / "preview_recording_preflight.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "openforge.preview_recording_preflight.v1",
                    "base_url": self._base_url,
                    "cameras": self._cam_names,
                    "results": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if failures:
            summary = ", ".join(
                f"{item['camera']}:{item['problem']}" for item in failures
            )
            raise PreviewRecordingError(
                "BundleSDF preview recording preflight failed before script execution "
                f"({summary}); details: {path}"
            )

    def _probe_preview_stream(self, cam: str, url: str) -> dict[str, object]:
        timeout_s = self._probe_timeout_s()
        min_bytes = self._probe_min_bytes()
        started = time.monotonic()
        result: dict[str, object] = {
            "camera": cam,
            "url": url,
            "ok": False,
            "timeout_s": timeout_s,
            "min_bytes": min_bytes,
            "bytes_read": 0,
            "status": None,
            "content_type": None,
            "problem": None,
        }
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "openforge-preview-recorder/1"},
            )
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                status = getattr(response, "status", None) or response.getcode()
                content_type = response.headers.get("Content-Type")
                result["status"] = status
                result["content_type"] = content_type
                chunks: list[bytes] = []
                while (
                    time.monotonic() - started < timeout_s
                    and sum(len(chunk) for chunk in chunks) < min_bytes
                ):
                    chunk = response.read(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                data = b"".join(chunks)
                result["bytes_read"] = len(data)
                result["elapsed_s"] = round(time.monotonic() - started, 3)
                if status and int(status) >= 400:
                    result["problem"] = f"HTTP {status}"
                elif len(data) < min_bytes:
                    result["problem"] = (
                        f"preview stream yielded {len(data)} bytes, below required {min_bytes}"
                    )
                else:
                    result["ok"] = True
                    result["has_jpeg_marker"] = b"\xff\xd8" in data
        except urllib.error.HTTPError as exc:
            result.update(
                {
                    "status": exc.code,
                    "content_type": exc.headers.get("Content-Type") if exc.headers else None,
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "problem": f"HTTP {exc.code}: {exc.reason}",
                }
            )
        except urllib.error.URLError as exc:
            result.update(
                {
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "problem": f"URL error: {exc.reason}",
                }
            )
        except TimeoutError:
            result.update(
                {
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "problem": f"timed out waiting for preview bytes after {timeout_s:.2f}s",
                }
            )
        except OSError as exc:
            result.update(
                {
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "problem": f"I/O error while probing preview stream: {exc}",
                }
            )
        log_path = self._output_dir / f"{cam}.preview_probe.log"
        log_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    def _stop_timeout_s(self) -> float:
        raw = os.environ.get("OPENFORGE_PREVIEW_RECORDER_STOP_TIMEOUT_S", "").strip()
        try:
            return max(1.0, float(raw)) if raw else 8.0
        except ValueError:
            return 8.0

    def _start_python_recorders(self) -> None:
        self._stop_event.clear()
        self._thread_errors.clear()
        for cam in self._cam_names:
            url = f"{self._base_url}/preview/{cam}"
            video_path = self._output_dir / f"{cam}.mp4"
            log_path = self._output_dir / f"{cam}.preview_recorder.log"
            thread = threading.Thread(
                target=self._record_preview_stream_python,
                args=(cam, url, video_path, log_path),
                daemon=True,
                name=f"preview-recorder-{cam}",
            )
            self._threads[cam] = thread
            thread.start()
            print(f"[preview-recorder] {cam}: {url} -> {video_path} (python)")

    def _record_preview_stream_python(
        self,
        cam: str,
        url: str,
        video_path: Path,
        log_path: Path,
    ) -> None:
        writer = None
        frames = 0
        started = time.monotonic()
        frame_period = 1.0 / float(_RECORD_FPS)
        transient_errors: list[str] = []
        log: dict[str, object] = {
            "camera": cam,
            "url": url,
            "camera_portal": os.environ.get(
                "OPENFORGE_PREVIEW_RECORDER_CAMERA_PORTAL_ADDR",
                "127.0.0.1:8300",
            ),
            "path": str(video_path),
            "backend": "python",
            "frames": 0,
            "transient_errors": [],
            "problem": None,
        }
        try:
            import portal

            client = portal.Client(str(log["camera_portal"]))
            while not self._stop_event.is_set():
                loop_start = time.monotonic()
                try:
                    rgb = client.get_camera_image(cam).result(timeout=3)
                except Exception as exc:
                    if self._stop_event.is_set():
                        break
                    msg = f"{type(exc).__name__}: {exc}"
                    transient_errors.append(msg)
                    if frames == 0:
                        raise RuntimeError(f"no camera frame before error: {msg}") from exc
                    time.sleep(frame_period)
                    continue

                arr = np.asarray(rgb)
                if arr.ndim != 3 or arr.shape[2] < 3:
                    raise RuntimeError(f"camera {cam!r} returned invalid frame shape {arr.shape}")
                frame = np.ascontiguousarray(arr[..., :3][..., ::-1])
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(video_path),
                        fourcc,
                        _RECORD_FPS,
                        (int(w), int(h)),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"failed to open VideoWriter for {video_path}")
                writer.write(frame)
                frames += 1
                remaining = frame_period - (time.monotonic() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)
        except Exception as exc:
            problem = f"{type(exc).__name__}: {exc}"
            self._thread_errors[cam] = problem
            log["problem"] = problem
        finally:
            if writer is not None:
                writer.release()
            log.update(
                {
                    "frames": frames,
                    "transient_errors": transient_errors[-8:],
                    "elapsed_s": round(time.monotonic() - started, 3),
                }
            )
            try:
                log_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
            except Exception:
                pass

    def _stop_python_recorders(self) -> None:
        self._stop_event.set()
        timeout_s = self._stop_timeout_s()
        deadline = time.monotonic() + timeout_s
        for cam, thread in self._threads.items():
            remaining = max(0.1, deadline - time.monotonic())
            thread.join(timeout=remaining)
            if thread.is_alive():
                self._thread_errors[cam] = (
                    f"python recorder thread did not stop within {timeout_s:.1f}s"
                )

    def _reencode_python_outputs_to_h264(self) -> None:
        enabled = os.environ.get(
            "OPENFORGE_PREVIEW_RECORDER_REENCODE_H264",
            "1",
        ).strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return

        for cam in self._cam_names:
            video_path = self._output_dir / f"{cam}.mp4"
            if not video_path.exists() or video_path.stat().st_size <= 0:
                continue
            source_path = self._output_dir / f"{cam}.pre_h264.mp4"
            tmp_path = self._output_dir / f"{cam}.h264.tmp.mp4"
            log_path = self._output_dir / f"{cam}.h264_reencode.log"
            try:
                if source_path.exists():
                    source_path.unlink()
                video_path.rename(source_path)
                with log_path.open("w", encoding="utf-8") as log_file:
                    proc = subprocess.run(
                        [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "warning",
                            "-y",
                            "-i",
                            str(source_path),
                            "-c:v",
                            "libx264",
                            "-preset",
                            os.environ.get(
                                "OPENFORGE_PREVIEW_RECORDER_H264_PRESET",
                                "veryfast",
                            ),
                            "-crf",
                            os.environ.get("OPENFORGE_PREVIEW_RECORDER_H264_CRF", "28"),
                            "-vf",
                            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                            "-pix_fmt",
                            "yuv420p",
                            str(tmp_path),
                        ],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=log_file,
                        text=True,
                        timeout=max(20.0, self._stop_timeout_s() * 4.0),
                    )
                if (
                    proc.returncode != 0
                    or not tmp_path.exists()
                    or tmp_path.stat().st_size <= 0
                ):
                    if tmp_path.exists():
                        tmp_path.unlink()
                    source_path.rename(video_path)
                    self._thread_errors[cam] = (
                        f"H.264 re-encode failed for {cam}; see {log_path}"
                    )
                    continue
                tmp_path.rename(video_path)
            except FileNotFoundError:
                if source_path.exists() and not video_path.exists():
                    source_path.rename(video_path)
                self._thread_errors[cam] = "ffmpeg not found for H.264 re-encode"
            except Exception as exc:
                if source_path.exists() and not video_path.exists():
                    source_path.rename(video_path)
                self._thread_errors[cam] = (
                    f"H.264 re-encode error: {type(exc).__name__}: {exc}"
                )

    def _wait_processes(self, timeout_s: float) -> list[tuple[str, subprocess.Popen]]:
        deadline = time.monotonic() + max(0.0, timeout_s)
        remaining: list[tuple[str, subprocess.Popen]] = []
        for cam, proc in self._processes.items():
            if proc.poll() is not None:
                continue
            wait_s = max(0.05, deadline - time.monotonic())
            try:
                proc.wait(timeout=wait_s)
            except subprocess.TimeoutExpired:
                remaining.append((cam, proc))
        return remaining

    def _request_ffmpeg_quit(self, cam: str, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        pipe = proc.stdin
        if pipe is None:
            return
        try:
            pipe.write(b"q\n")
            pipe.flush()
            pipe.close()
        except (BrokenPipeError, OSError, ValueError) as exc:
            print(f"[preview-recorder] Could not send graceful quit to {cam} ffmpeg: {exc}")

    def _terminate_processes(self) -> None:
        timeout_s = self._stop_timeout_s()
        for cam, proc in self._processes.items():
            self._request_ffmpeg_quit(cam, proc)

        remaining = self._wait_processes(timeout_s)
        if remaining:
            for cam, proc in remaining:
                if proc.poll() is None:
                    print(f"[preview-recorder] Sending SIGINT to stalled {cam} ffmpeg process")
                    proc.send_signal(signal.SIGINT)
            remaining = self._wait_processes(timeout_s)

        if remaining:
            for cam, proc in remaining:
                if proc.poll() is None:
                    print(f"[preview-recorder] Terminating stalled {cam} ffmpeg process")
                    proc.terminate()
            remaining = self._wait_processes(max(2.0, timeout_s / 2.0))

        for cam, proc in remaining:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=3)
                print(f"[preview-recorder] Killed stalled {cam} ffmpeg process")

    def _close_logs(self) -> None:
        for log_file in self._logs.values():
            try:
                log_file.close()
            except Exception:
                pass

    def _validate_outputs(self) -> dict[str, object]:
        videos = []
        for cam in self._cam_names:
            path = self._output_dir / f"{cam}.mp4"
            log_path = (
                self._output_dir / f"{cam}.preview_recorder.log"
                if self._backend in {"python", "cv2", "opencv"}
                else self._output_dir / f"{cam}.ffmpeg.log"
            )
            exists = path.exists()
            size = path.stat().st_size if exists else 0
            problem = None
            ffprobe = None
            if not exists:
                problem = "missing mp4"
            elif size <= 0:
                problem = "zero-byte mp4"
            else:
                ffprobe = self._ffprobe_video(path)
                if not ffprobe["ok"]:
                    problem = str(ffprobe["problem"])
                elif cam in self._processes and self._processes[cam].returncode == -9:
                    problem = "ffmpeg was killed with SIGKILL before clean finalization"
                elif cam in self._thread_errors:
                    problem = self._thread_errors[cam]
                elif self._require_h264_outputs() and (
                    ffprobe.get("codec_name") != "h264"
                    or ffprobe.get("pix_fmt") != "yuv420p"
                ):
                    problem = (
                        "unexpected preview video codec; expected "
                        f"h264/yuv420p, got {ffprobe.get('codec_name')}/"
                        f"{ffprobe.get('pix_fmt')}"
                    )
            videos.append(
                {
                    "camera": cam,
                    "path": str(path),
                    "exists": exists,
                    "size_bytes": size,
                    "log_path": str(log_path),
                    "backend": self._backend,
                    "ffmpeg_returncode": (
                        self._processes[cam].returncode if cam in self._processes else None
                    ),
                    "ffprobe": ffprobe,
                    "ok": problem is None,
                    "problem": problem,
                }
            )
        return {
            "schema": "openforge.preview_recording_result.v1",
            "base_url": self._base_url,
            "output_dir": str(self._output_dir),
            "videos": videos,
        }

    def _ffprobe_video(self, path: Path) -> dict[str, object]:
        result: dict[str, object] = {
            "ok": False,
            "problem": None,
            "width": None,
            "height": None,
            "duration_s": None,
            "nb_frames": None,
            "codec_name": None,
            "profile": None,
            "pix_fmt": None,
        }
        try:
            proc = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,profile,pix_fmt,width,height,duration,nb_frames:format=duration",
                    "-of",
                    "json",
                    str(path),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=8,
            )
        except FileNotFoundError:
            result["problem"] = "ffprobe not found; cannot validate mp4"
            return result
        except subprocess.TimeoutExpired:
            result["problem"] = "ffprobe timed out while validating mp4"
            return result

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            result["problem"] = f"ffprobe failed: {stderr or f'return code {proc.returncode}'}"
            return result

        try:
            data = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError as exc:
            result["problem"] = f"ffprobe returned invalid JSON: {exc}"
            return result

        streams = data.get("streams") or []
        if not streams:
            result["problem"] = "ffprobe found no video stream"
            return result
        stream = streams[0]
        width = _positive_int(stream.get("width"))
        height = _positive_int(stream.get("height"))
        duration = _positive_float(stream.get("duration"))
        if duration is None:
            duration = _positive_float((data.get("format") or {}).get("duration"))
        nb_frames = _positive_int(stream.get("nb_frames"))
        result.update(
            {
                "width": width,
                "height": height,
                "duration_s": duration,
                "nb_frames": nb_frames,
                "codec_name": stream.get("codec_name"),
                "profile": stream.get("profile"),
                "pix_fmt": stream.get("pix_fmt"),
            }
        )
        if width is None or height is None:
            result["problem"] = "ffprobe found a video stream without valid dimensions"
            return result
        if duration is None and nb_frames is None:
            result["problem"] = "ffprobe found a video stream without duration or frame count"
            return result
        result["ok"] = True
        return result


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
