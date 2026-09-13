# Tianyi 2.0 body model contract

The simulation-first milestone uses the pinned body-only Tianyi 2.0 URDF and MuJoCo
position-control model under `asset/tianyi2/` as its state authority. The separate
hand-inclusive URDF is used only by the Viser visualization derivative.

## Counts

- 21 body joints and 21 matching MuJoCo position actuators.
- 2 leg mechanism joints.
- 2 waist joints.
- 3 head joints.
- 14 arm joints.
- 17 joints in a complete `base` or `composed` pose: three head joints and both arms.
- 7 joints in a partial `left_arm` or `right_arm` pose.
- 4 calibration-only body joints: two leg mechanism joints and two waist joints.
- 25 referenced mesh names, with visual and convex copies where required.
- 24 visualization-only Inspire finger joints. The source defines 12 independent and 12
  mimic joints; the generated Viser model fixes all 24 at zero and excludes them from pose
  and action schemas.

Tianyi's wheeled base is not an authored or actuated part of this MuJoCo workflow. The
body model stays fixed while poses update the head and two arms. A composed pose copies
the head and both arms from its base, then replaces either or both arm subsets.

## Validation boundary

`TianyiAssetHelper.validate()` verifies:

1. Pinned URDF and MJCF SHA-256 values.
2. Exact agreement between URDF and MJCF movable-joint names.
3. Agreement of joint axes and position limits.
4. One MuJoCo actuator for every body joint.
5. Presence of every referenced visual and collision mesh.
6. Agreement between validated counts and `model_metadata.json`.

This establishes model integrity for MuJoCo authoring. It does not validate the model
against a physical robot or current firmware.

The Viser validator separately checks the complete hand-inclusive source. The generated
visual URDF takes all body joints from the canonical body-only URDF, then preserves the
hand link and joint hierarchy while fixing every finger joint at its zero-pose origin.
This resolves three lower-body range differences without changing the MuJoCo or pose
contract.

## Simulation-state invariant

`SimulationService` uses the body-only MJCF as a fixed-base kinematic authoring model.
It exposes all 21 body positions in canonical model order. The public authored-joint API
accepts edits only for the 17 head and arm joints. Pose Recorder has a separate UI-only
calibration command for the remaining four joints. These values never enter pose or action
schemas. The confirmed reset calibration is:

- motor 31, `waist_yaw_joint`: `0.0` rad;
- motor 32, `waist_pitch_joint`: `0.24` rad;
- motor 51, `first_leg_pitch_joint` (Hip Pitch): `0.14` rad;
- motor 52, `second_leg_pitch_joint` (Knee Pitch): `-0.4` rad.

These motor-ID meanings follow the Tianyi 2.0 Pro SDK table. All four confirmed values are
inside the pinned URDF limits. Session calibration survives authored pose updates and
composition previews. Reset calibration or a full simulation reset restores the confirmed
values. MuJoCo position-actuator targets remain equal to the corresponding joint positions.
