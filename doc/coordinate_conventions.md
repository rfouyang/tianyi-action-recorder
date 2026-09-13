# Coordinate and camera conventions

Tianyi uses the same robot-relative naming convention as `g1-action-recorder`.

## Robot coordinates

- `+X`: the direction the robot faces.
- `+Y`: the robot's anatomical left.
- `+Z`: upward.
- Left and right always refer to the robot, never the camera operator.

## Camera presets

1. **Front**: the camera is in front of the robot and looks toward the robot's face.
2. **Back**: the camera is behind the robot and looks toward its back.
3. **Left**: the camera views the robot from its anatomical left side.
4. **Right**: the camera views the robot from its anatomical right side.
5. **Front-left**: the camera is 45 degrees between Front and robot-left.
6. **Front-right**: the camera is 45 degrees between Front and robot-right.

Images are not mirrored. Consequently, the robot's left arm appears on the right side
of the Front image, and its right arm appears on the left side.

These names describe camera placement relative to the robot and must remain consistent
across static MuJoCo previews, animated action previews, Viser presets, UI labels, and
tests.
