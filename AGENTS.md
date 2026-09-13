# Project architecture

This project follows a business-capability-first robotics application architecture.

- `app/` contains application entrypoints and presentation-layer code.
- `component/` contains reusable business capabilities.
- `util/` contains infrastructure helpers and external-system adapters.
- `config/` contains configuration.
- `tests/` mirrors production code where practical.
- UI and API are independent presentation surfaces over shared business capabilities.
- Prefer simple, explicit Python architecture over excessive abstraction.

The dependency direction is:

```text
app -> component -> util
```

Do not add generic root folders such as `controllers/`, `repositories/`, `managers/`,
`handlers/`, or `services/`. Prefer domain-oriented modules.

## Current safety boundary

The current milestone is simulation-only.

- Pose recording, pose composition, action composition, and action playback operate only
  on MuJoCo state.
- Tianyi is a wheeled-base robot. The pose workflow authors its head and two seven-joint
  arms. The wheeled base remains locked. The leg mechanism and waist are calibration-only.
- Pose Recorder may expose simulation-only controls for motor 31 waist yaw, motor 32 waist
  pitch, motor 51 hip pitch, and motor 52 knee pitch. These four joints must never enter a
  base, arm, composed-pose, or action file and must remain unavailable through the public
  authored-joint API. Pose/action changes preserve the current session calibration.
- The confirmed reset calibration is motor 31 `0.0`, motor 32 `0.24`, motor 51 `0.14`,
  and motor 52 `-0.4` radians.
- Do not add `rclpy`, ROS 2 publishers, motor-command topics, SSH robot control, or any
  physical-robot execution path during this milestone.
- Do not describe kinematic MuJoCo playback as physical validation.
- A later hardware milestone must introduce a separately reviewed execution boundary.

## Model contract

- The pinned body-only Tianyi 2.0 model under `asset/tianyi2/` is canonical.
- Asset provenance and hashes must remain recorded in `model_metadata.json` and
  `UPSTREAM.md`.
- Runtime code must validate URDF/MJCF joint names, axes, limits, actuators, and meshes
  before using the model.
- Keep the body-only URDF/MJCF authoritative for all 21 simulated body-joint limits.
  The pinned hand-inclusive URDF is the Viser visualization source only. Its three
  lower-body limit differences are normalized to the canonical body contract during
  visual-URDF generation; the differences must remain recorded in generated metadata.
- Keep `tianyi2_pos.xml` as the pinned simulation source. Offline MuJoCo rendering uses
  the generated `tianyi2_visual.xml`; regenerate it only through
  `util/build_tianyi_visual_assets.py` and update its generated metadata together.
- Interactive Viser rendering uses `tianyi2_visual_optimized.urdf`, generated only by
  `util/build_tianyi_visual_urdf.py`. It includes both Inspire hands, preserves their
  source link and joint hierarchy, fixes all finger joints at the source zero pose, uses
  the canonical body joint contract, references only `meshes_visual_optimized/`, and
  contains no collision elements.
- Inspire-hand joints are visualization-only in this milestone. Never expose hand
  controls in poses, actions, the UI, or the simulation API.
- Use one process-wide Viser runtime. Load the robot geometry once, synchronize revisions
  from `SimulationService` at visualization rate, and never recreate meshes per UI edit.
- A `base` or `composed` pose contains exactly the three head joints and both arms.
- A `left_arm` or `right_arm` pose contains exactly the corresponding seven arm joints.

## Coordinate and camera convention

- The robot faces `+X`; its anatomical left is `+Y`, and up is `+Z`.
- `left` and `right` always mean the robot's anatomical left and right.
- The Front camera is positioned in front of the robot and looks toward its face.
- In the unmirrored Front image, the robot's left arm appears on the image's right.
- Back, Left, Right, Front-left, and Front-right are named relative to the robot, using
  the same convention as `g1-action-recorder`.
