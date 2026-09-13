# FastAPI and Viser backend runtime

`app/tianyi_3d_main.py` owns the process lifecycle for the new presentation surfaces. It
constructs one `TianyiApplication`, starts one shared `ViserManager`, exposes them through
FastAPI application state, and shuts both down cleanly.

The dependency flow is:

```text
REST client ───────┐
                   ├─> SimulationService ─> MuJoCo revision ─> ViserManager ─> browser
WebSocket client ──┘
```

## API boundaries

`app/api_tianyi_3d/` is independent of the browser UI. Domain routers are assembled under
`/api/tianyi`, application access is centralized in `dependencies.py`, and Pydantic
transport models reject extra command fields.

The public simulation API accepts only partial authored-joint updates. `SimulationService`
rejects its four calibration-only leg-mechanism/waist joints. Pose Recorder uses a separate
browser-only WebSocket command to adjust them in the current MuJoCo session. Pose capture,
composition, and persistence continue to accept only the exact 17/7/7 authored shapes.
The REST state exposes the current calibration as `locked_joint_positions` for inspection.

## Viser lifecycle

`ViserManager` validates and loads `tianyi2_visual_optimized.urdf` and the pinned G1 visual
URDF once. Startup rejects a different actuated-joint order in either model. A background
thread waits for newer simulation revisions and changes only the Tianyi robot root pose
and joint configuration; it does not rebuild scene geometry. Retarget preview requests
update the read-only G1 joint configuration from the uploaded source pose.

The retarget solver does not treat either robot's custom `concierge_init` arm pose as a
kinematic correspondence. It scales the G1 source shoulder-to-elbow and elbow-to-hand
vectors to Tianyi link lengths, anchors them at the Tianyi shoulder, and constructs a
functional wrist target from forearm direction and G1 palm roll.

The standard viewer layout centers Tianyi and hides G1. While Pose Retargeting is active
and a source has been solved, the scene places the original G1 on Front screen-left and
the retargeted Tianyi on Front screen-right, exposes labels for both, and widens the
selected camera preset. Leaving the workspace hides the G1 model and restores the
single-robot framing.

The camera contract matches `g1-action-recorder`: Tianyi faces `+X`, robot-left is `+Y`,
and up is `+Z`. Front, Back, Left, Right, Front-left 45 degrees, and Front-right 45 degrees
are robot-relative presets with smooth orbit transitions. Free mouse orbit, pan, and zoom
remain Viser-native behavior.

The backend remains simulation-only. Viser is a kinematic visualization consumer and is
not a controller or physical validation mechanism.
