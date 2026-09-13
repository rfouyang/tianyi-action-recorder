# Optimized visualization models

The upstream `tianyi2_pos.xml` references only the 2.4 MB convex mesh set and applies the
same low-detail mesh to both collision and visual geoms. The original body visual STL set
contains more than 1.26 million triangles, which is unnecessarily heavy for repeated
browser preview rendering. The complete body-and-hand set contains about 1.54 million
triangles.

The checked-in `tianyi2_visual.xml` is a render-only derivative with these changes:

1. Original visual STL files are imported by Blender and exported as smooth-normal OBJ.
2. Each mesh is decimated toward 12,000 triangles without changing link transforms.
3. Collision geoms are removed from the render scene, reducing model geoms from 51 to 26.
4. Neutral studio lighting and a gradient sky replace the upstream black horizon.
5. All 21 joints, limits, inertials, and position actuators remain intact.

The resulting mesh set has about 314,000 triangles, a 75% reduction from the cleaned
source visuals. `TianyiVisualAssetHelper` verifies the generated MJCF, every optimized
mesh hash, triangle metadata, and the preserved fixed-base 21-joint model contract.

This derivative is for visualization only. `SimulationService` continues to use the
pinned upstream position-control MJCF as its state authority.

## Viser URDF

`tianyi2_visual_optimized.urdf` is generated from the pinned hand-inclusive URDF after the
OBJ mesh set exists. It includes 51 visual meshes and both Inspire hands. All 24 finger
joints remain in the kinematic tree but are generated as fixed joints at their upstream
zero-pose origins, so Viser exposes no finger degrees of freedom.

The generator replaces the 24 body-tree joint definitions with the canonical body-only
URDF definitions. This normalizes the three lower-body range differences while leaving
every hand joint untouched. All 51 STL visual references become optimized OBJ references,
and all collision elements are removed.

Generate and validate it with:

```bash
uv run python util/build_tianyi_visual_urdf.py
uv run python util/tianyi_visual_urdf_helper.py
```

`visual_urdf_metadata.json` binds the derivative to both source URDFs and
`visual_urdf_mesh_metadata.json` with SHA-256 hashes. Validation checks 51 links, 50 total
joints, the exact 21 movable body joints, zero movable hand joints, 51 unique optimized
mesh references, the complete hand hierarchy, and zero collisions. The optimized
body-and-hand mesh set contains about 506,000 triangles.
