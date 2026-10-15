# Example station profile

[`yam-example/station.yaml`](yam-example/station.yaml) is the packaged reference
profile and the default station ID. It includes example camera calibration,
gripper measurements, arm endpoints, and joint limits. Configure and calibrate
these values for your station before using it.

Supply your own profiles through `--station-config-root`. Keep `station.yaml`,
gripper measurements, and the hashed calibration bundle together. Robot model
paths are relative to the installed `cap_harness` package.

See [the real YAM guide](../../../../docs/yam-real.md) and the
[camera calibration workflow](../../yam_real/calibration/README.md).
