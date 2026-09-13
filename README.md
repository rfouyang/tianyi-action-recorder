# Tianyi Action Recorder

Simulation-first tools for authoring X-Humanoid Tianyi 2.0 poses and actions.

The intended workspaces are Pose Recorder, Pose Composer, Pose Retargeting, Action
Composer, and Action Player. The current implementation establishes the pinned body-only URDF/MuJoCo asset
contract, typed Tianyi joint and pose definitions, a shared fixed-base MuJoCo state
service, and live Pose Recorder and Pose Composer API/UI surfaces. Further capabilities will be added
incrementally after each step is reviewed.

No ROS 2 packages are required, no motor-command topics are created, and no physical
robot commands are sent.

## Environment

```bash
uv python install 3.10
uv sync
uv run python -m unittest discover -s tests -v
uv run ruff check .
uv lock --check
```

## Validate the pinned model

```bash
uv run python util/tianyi_asset_helper.py
```

The validator checks the pinned file hashes, URDF/MJCF joint agreement, joint axes and
limits, complete MuJoCo position-actuator coverage, and every referenced mesh.

## Validate the typed pose contract

```bash
uv run python -m unittest tests.test_tianyi_joint_schema tests.test_pose_models -v
```

`TianyiJointSchema` defines the exact fixed-base MuJoCo joint order, authored and locked
groups, pose subsets, and position limits. `PoseDefinition` is the immutable schema-1
JSON model used by later pose recording and composition steps.

## Fixed-base simulation state

`SimulationService` owns the mutable MuJoCo data behind a synchronized API. It supports
partial head/arm edits, exact pose application and capture, typed pose application, and
default-pose reset. Startup and full reset use the validated `base/concierge_init` pose.
Every mutation keeps the 21 joint positions and position-actuator targets aligned. Pose
and action changes preserve the current four-joint lower-body calibration; a full
simulation reset restores its confirmed locked-joint baseline.

## Run the pose workbench

```bash
MUJOCO_GL=egl uv run tianyi-recorder
```

Open <http://127.0.0.1:7900> for the browser UI or
<http://127.0.0.1:7900/docs> for the interactive API documentation. The server builds a
local Tailwind CSS 4 and daisyUI 5 `wireframe` bundle when its sources change; the page
does not depend on a CDN. The shared Viser scene runs at <http://127.0.0.1:7901> and is
embedded in the recorder.

The Pose Recorder can edit the head and arms, switch among the exact 17-joint `base` and
seven-joint arm shapes, reset a selected group to `base/concierge_init`, save validated
JSON poses under `data/poses/`, and change among six robot-relative Viser camera presets. A separate
calibration-only group exposes motor IDs 31, 32, 51, and 52 in MuJoCo/Viser. Those four
values are never recorded in pose or action files.

The Pose Composer starts from one required 17-joint base pose and optionally replaces
either seven-joint anatomical arm. Preview applies the unsaved composition to MuJoCo and
Viser; save persists a `composed` pose with exact source provenance. The head always comes
from the base pose, and Inspire-hand joints never enter the pose contract.

Pose Retargeting opens a local G1 pose JSON through the browser's native file chooser.
It accepts G1 `base`, `composed`, `left_arm`, and `right_arm` pose shapes, validates them
against the pinned G1 contract, and previews the constrained two-arm result in Tianyi
MuJoCo. In the Pose Retargeting workspace, Viser places the original G1 source and the
retargeted Tianyi result side by side with source/result labels and wider versions of the
same robot-relative camera presets. Moving to another workspace hides G1 and centers
Tianyi again. Only one seven-joint Tianyi `left_arm` or `right_arm` pose can be saved at a
time. The reviewed G1 MJCF, URDF, meshes, license, provenance, hashes, and legacy
`retarget_baseline.json` source sample are vendored under `asset/g1/`; no sibling
repository is required at runtime. The legacy sample is not used as a solver reference.

Retargeting uses the source pose's shoulder-to-elbow and elbow-to-hand directions, scaled
to Tianyi's link lengths and anchored at its model-defined shoulder. A functional wrist
frame combines the target forearm direction with the G1 source palm roll. The
robot-specific G1 and Tianyi `concierge_init` arm postures are not correspondence
references. Tianyi's configured head is preserved, and an arm absent from a partial G1
source remains at the configured Tianyi default.

The Action Composer fixes both the first and final keyframes to
`base/concierge_init`. Authors add only complete `base` or `composed` intermediate poses;
the start/finish boundary is enforced by the action model, API, and UI.

The independent client API includes:

- `GET /api/tianyi/system/health`
- `GET /api/tianyi/simulation/state`
- `PUT /api/tianyi/simulation/joints`
- `POST /api/tianyi/simulation/reset`
- `WS /api/tianyi/simulation/ws`
- `GET /api/tianyi/poses`
- `POST /api/tianyi/poses/preview`
- `POST /api/tianyi/poses/compositions/preview`
- `POST /api/tianyi/poses/compositions`
- `POST /api/tianyi/pose-retargeting/preview`
- `POST /api/tianyi/pose-retargeting`

## Shared API and Viser backend

The `tianyi-3d` alias starts the same combined FastAPI, UI, and Viser host:

```bash
MUJOCO_GL=egl uv run tianyi-3d
```

Both REST and WebSocket updates call the same `SimulationService`. The Viser manager loads
`tianyi2_visual_optimized.urdf` and the pinned G1 visual URDF once, then follows newer
Tianyi MuJoCo state revisions at up to 30 Hz. G1 remains hidden except for a solved
retarget comparison and never becomes an editable simulation state. The Tianyi Viser
model includes both Inspire hands with all finger joints fixed at their zero pose and
exposes no hand controls. No route publishes ROS 2 or physical-robot commands.

## Optimized visualization model

The pinned upstream URDF and `tianyi2_pos.xml` remain unchanged and authoritative. The
browser renderer uses the reproducibly derived `tianyi2_visual.xml`: high-resolution STL
visuals are converted to smooth-normal OBJ meshes, decimated to approximately 314,000
triangles, and duplicate collision geoms are removed from the render-only scene. This is
a 75% triangle reduction from the cleaned source visuals while retaining much more detail
than the upstream convex collision meshes.

Regenerate the checked-in visualization assets with Blender:

```bash
blender -b -noaudio --factory-startup \
  --python util/build_tianyi_visual_assets.py --python-exit-code 1
```

Hashes, triangle counts, and generator provenance are recorded in
`asset/tianyi2/visual_model_metadata.json` and validated at application startup.

Generate and validate the collision-free URDF that will be loaded once by Viser:

```bash
uv run python util/build_tianyi_visual_urdf.py
uv run python util/tianyi_visual_urdf_helper.py
```

`tianyi2_visual_optimized.urdf` contains 51 links, 50 joints, both Inspire hands, and 51
optimized visual meshes. The 21 body joints use the canonical MuJoCo-aligned contract;
the 24 source hand joints are fixed at their zero-pose origins in the visual derivative.
Hand joints remain visualization-only. Source and generated hashes are recorded in
`asset/tianyi2/visual_urdf_metadata.json`.

## Current scope

- Canonical body model: 21 revolute joints.
- Complete `base` and `composed` poses contain 17 joints: three head joints and both
  seven-joint arms.
- Partial `left_arm` and `right_arm` poses contain only the corresponding seven joints.
- Tianyi's wheeled base remains locked in MuJoCo. Leg and waist controls are available
  only in Pose Recorder's simulation calibration group and never enter authored files.
- Confirmed reset positions are ID 31 waist yaw `0.0`, ID 32 waist pitch `0.24`, ID 51
  hip pitch `0.14`, and ID 52 knee pitch `-0.4` radians.
- Inspire hands are visible but not controllable; mobile-base motion remains outside this
  milestone.

## Camera convention

Camera names are robot-relative and match `g1-action-recorder`. Front places the camera
in front of the robot looking toward its face. Left and right always mean the robot's
anatomical left and right, so the robot's left arm appears on the image's right in an
unmirrored Front view. See `doc/coordinate_conventions.md` for the complete contract.
