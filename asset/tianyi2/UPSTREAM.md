# Upstream provenance

## Canonical body description

- Repository: https://github.com/Open-X-Humanoid/TienKung_URDF
- Commit: `5c221783fb92fcc4af891ef1dc0502963caf2266`
- Source directory: `tianyi2_urdf/`
- URDF: `urdf/tianyi2.0_URDF.urdf`
- Inspire-hand visualization URDF: `urdf/tianyi2.0_urdf_with_hands.urdf`
- MuJoCo position model: `tianyi2_pos.xml`
- License: OpenAtom Open Hardware License Version 1.0; copied to `LICENSE`

The body URDF, hand-inclusive URDF, MuJoCo XML, and referenced source meshes are copied
without modification. File hashes are recorded in `model_metadata.json` and generated
visual metadata and checked at runtime.

## Visualization and control boundary

The hand-inclusive URDF is used only to visualize both Inspire hands in Viser. It contains
24 movable finger joints: 12 independent joints and 12 mimic followers. The generated
Viser derivative fixes all 24 at their zero-pose origins; no hand joint is part of a pose,
action, UI control, or simulation API command.

The hand URDF differs from the canonical body model at
`first_leg_pitch_joint`, `second_leg_pitch_joint`, and `waist_pitch_joint`. These are all
locked during authoring. `util/build_tianyi_visual_urdf.py` replaces the complete body
joint definitions with the body-only URDF contract while preserving the complete
Inspire-hand link and joint hierarchy. The exact source and canonical ranges are recorded in
`visual_urdf_metadata.json`.

Hand control, ROS 2 packages, robot control nodes, the Slamtec mobile-base interface, and
unrelated models from the upstream repository are not vendored.
