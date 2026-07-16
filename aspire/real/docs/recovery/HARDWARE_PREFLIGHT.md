# Non-Motion Hardware Preflight

Run `bash tools/non_motion_preflight.sh` from the real-workspace root. It does not start
services, query arm RPCs, change CAN link state, or transmit CAN frames. It
also deliberately omits device serial numbers and physical USB paths.

The captured `preflight-2026-07-02.txt` establishes:

- four Intel RealSense D405 devices, all on firmware `5.12.14.100`;
- librealsense Python binding `2.56.5.9235`;
- pyzed wheel `5.2`, but no loadable native ZED SDK (`libsl_zed.so` missing);
- `damiao-motor` `1.0.7b1`, i2rt `0.0.1`, and python-can `4.5.0`;
- all four stable CAN interfaces present and up at the configured
  1,000,000-bit/s policy;
- RTX 5090 with driver `580.82.09` and CUDA toolkit `12.8.93`.

Arm motor firmware remains unresolved. Reading it requires protocol traffic to
the motor controllers, which is outside this script's strict non-motion/read-
only boundary. Schedule a separate supervised maintenance procedure that
documents the exact vendor command and proves it cannot enable or command a
motor before querying firmware.
