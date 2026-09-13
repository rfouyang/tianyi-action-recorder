# Tianyi Action Recorder current state

Updated: 2026-09-13

## Milestone status

The current milestone is a simulation-only pose and action authoring application for the
X-Humanoid Tianyi 2.0 Pro. The application provides a FastAPI API, a daisyUI `wireframe`
browser interface, MuJoCo state and playback, and one shared Viser visualization runtime.

There is no ROS 2 publisher, physical robot command path, SSH control, or hardware
validation in this milestone. MuJoCo and Viser results must not be treated as physical
robot validation.

## Robot and pose contract

Tianyi is treated as a wheeled-base robot. Only the head and two arms are authored:

- `base` and `composed` poses contain exactly three head joints and both seven-joint arms,
  for 17 authored joints total.
- `left_arm` contains exactly the robot's anatomical left seven-joint arm.
- `right_arm` contains exactly the robot's anatomical right seven-joint arm.
- Inspire hand joints are visualized but are fixed and never recorded.
- The wheeled base, leg mechanism, and waist are not recorded in poses or actions.

The confirmed fixed lower-body calibration is:

```text
motor 51 / first_leg_pitch_joint  =  0.14
motor 52 / second_leg_pitch_joint = -0.40
motor 31 / waist_yaw_joint        =  0.00
motor 32 / waist_pitch_joint      =  0.24
```

These four values are visible in the Pose Recorder calibration group for inspection and
temporary MuJoCo adjustment, but capture and persistence always exclude them.

`base/concierge_init` is the default Tianyi pose. It is also the mandatory first and final
pose of every action definition.

## Coordinate convention

- Robot front is `+X`.
- Robot anatomical left is `+Y`.
- Up is `+Z`.
- Left and right always refer to the robot, not the observer.
- The Front camera faces the robot's face; robot-left appears on the image's right.
- Back, Left, Right, Front-left, and Front-right use the same robot-relative convention as
  `g1-action-recorder`.

## Implemented workspaces

### Pose Recorder

- Edits head, robot-left arm, robot-right arm, or the complete 17-joint base pose in live
  MuJoCo state.
- Loads an existing pose into the recorder for editing.
- Saves only exact `base`, `left_arm`, or `right_arm` pose shapes.
- Resets editable groups from `base/concierge_init`.
- Mirrors robot-left to robot-right and robot-right to robot-left using the reviewed
  sagittal-reflection sign rules.
- Shows the four calibration-only joints without recording them.

### Pose Composer

- Starts from a required 17-joint base pose.
- Optionally replaces either anatomical arm with a saved seven-joint arm pose.
- Previews the composed result in MuJoCo/Viser.
- Saves a 17-joint `composed` pose with source provenance.

### Pose Retargeting

- Uses the browser's native file chooser to load a G1 pose JSON.
- Accepts G1 `base`, `composed`, `left_arm`, and `right_arm` shapes.
- Uses only G1 assets vendored under `asset/g1/`; there is no sibling-repository runtime
  dependency.
- Displays the original G1 and retargeted Tianyi side by side in the same Viser scene.
- Loads both robot geometries once and updates only configurations and scene placement.
- Uses wider versions of the same robot-relative camera presets during comparison.
- Hides G1 and recenters Tianyi when leaving Pose Retargeting.
- Saves only one selected Tianyi `left_arm` or `right_arm` pose.

Retargeting does not use either robot's custom `concierge_init` arm posture as a
correspondence reference. It transfers the actual G1 source shoulder-to-elbow and
elbow-to-hand directions, scales them to Tianyi link lengths, anchors them at Tianyi's
model-defined shoulder, and derives a functional wrist frame from forearm direction and
G1 palm roll. Tianyi's configured head is preserved. For a partial one-arm G1 source, the
unprovided Tianyi arm stays at its configured default.

The solve report includes hand and elbow position error, upper-arm and forearm direction
error, wrist orientation error, iteration count, and reached joint limits. Convergence now
requires hand error at or below 20 mm, elbow error at or below 30 mm, and wrist orientation
error at or below 5 degrees.

The geometry solver was checked against all ten
`g1-action-recorder/data/poses/composed/concierge_speak_*.json` poses. Across their 20
arms, the observed mean errors were:

```text
upper-arm direction  0.84 degrees
forearm direction    2.53 degrees
hand position        3.7 mm
elbow position       16.8 mm
```

All 20 arms converged without reaching a Tianyi joint limit. A final user visual review in
the live side-by-side viewer remains the current acceptance checkpoint.

### Action Composer and Player

- Enforces `base/concierge_init` as both the first and final keyframe.
- Accepts only complete `base` or `composed` intermediate poses.
- Supports editable transition durations and explicit per-keyframe holds.
- Preserves `hold_seconds = 0` as no hold.
- Compiles and persists simulation trajectories.
- Plays, pauses, resumes, and stops trajectories in shared MuJoCo state.
- Keeps the current calibration-only lower-body values unchanged during playback.

## Models and visualization

- `asset/tianyi2/tianyi2_pos.xml` is the pinned Tianyi simulation source.
- `asset/tianyi2/tianyi2_visual.xml` is the generated offline-rendering model.
- `asset/tianyi2/tianyi2_visual_optimized.urdf` is the collision-free Viser model and
  references only `meshes_visual_optimized/`.
- The reviewed G1 MJCF, URDF, meshes, license, hashes, provenance, and legacy source sample
  are under `asset/g1/`.
- Runtime helpers validate model hashes, joint names, joint order, limits, actuators, and
  visualization metadata before use.
- One process-wide `ViserManager` owns both visual robots and synchronizes Tianyi from
  `SimulationService` revisions at up to 30 Hz.

## Architecture

The dependency direction remains:

```text
app -> component -> util
```

- `app/` contains the FastAPI, WebSocket, UI panel, and Viser presentation surfaces.
- `component/` contains pose, composition, retargeting, simulation, action, playback, and
  rendering capabilities.
- `util/` contains file, archive, asset-validation, and generated-visual-model helpers.
- `config/` contains central path and runtime settings.
- `tests/` mirrors the production capabilities.

The REST API and browser UI are independent presentation surfaces over the same business
components and `SimulationService`.

## Verification

The current checked-in state passes:

```bash
uv run ruff check app component util tests
node --check app/ui_tianyi_3d/static/js/app.js
node --check app/ui_tianyi_3d/static/js/panels/pose_retargeting.js
MUJOCO_GL=egl uv run python -m unittest discover -s tests -q
```

Result: 88 tests passed. A real Viser startup loaded and validated both Tianyi and G1 URDF
models. The combined application serves the UI/API on port 7900 and Viser on port 7901.

## Run locally

```bash
MUJOCO_GL=egl uv run tianyi-recorder
```

Open <http://127.0.0.1:7900> and use <http://127.0.0.1:7900/docs> for the API schema.

## Next checkpoint

Visually review several `concierge_speak_*.json` sources in Pose Retargeting from Front and
45-degree camera views. Any subsequent tuning should be driven by those model-to-model
comparisons while preserving the simulation-only boundary and exact pose schemas.
