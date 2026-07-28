# Franka Emika Panda Model Provenance

The 67 files in `assets/` are byte-identical to the Franka Emika Panda model
distributed at `google-deepmind/mujoco_menagerie` commit
`4a7015530bd7a4161103ae8f0905a96481e4cc1a` under
`franka_emika_panda/assets/`. That model documents its derivation from Franka's
public URDF and is distributed under Apache License 2.0.

`panda_scene.xml` is a modified derivative of the corresponding upstream
`panda.xml`. ASPIRE changes include fingertip collision/friction parameters,
task objects and floor geometry, actuator/control parameters, and collision
exclusions. The XML carries a prominent modification notice as required by
Apache-2.0.

See the repository-root `LICENSE` for the complete Apache License 2.0 text,
`LICENSES/THIRD_PARTY_LICENSES.md` for the immutable upstream link, and
`LICENSES/THIRD_PARTY_NOTICES.md` for the wider dependency inventory.
