# Declarative World & Per-Robot Trajectory Format (`unrust-it-io`)

This specification documents the **Schema v1** file formats implemented by [`crates/unrust-it-io`](../crates/unrust-it-io/Cargo.toml) for:

1. **Workcell Worlds (`WorldSpec`)** — human-editable `.toml` (primary) and structured `.json` definitions compiling directly into [`WorldScene`](../crates/unrust-it/src/scene.rs).
2. **Per-Robot Trajectory Sets (`TrajectorySetSpec`)** — ordered multi-robot trajectories composed of primary **frames** and arbitrarily nested **subframes**, with semantic roles, reference-frame inheritance, TCP overrides, optional joint seeds, and CBF-filtered Studio execution.

---

## 1. Conventions & TOML Ordering Note

- **Formats**: Both `WorldSpec` and `TrajectorySetSpec` auto-detect `.toml` and `.json` from the file extension (`FileFormat::from_path`).
- **Units**: Lengths in **meters**, time in **seconds**, angles in **radians**, masses in **kilograms**, and RGBA colors normalized in `0.0..=1.0`.
- **Pose Representation (`PoseSpec`)**:
  - `xyz = [x, y, z]` in meters.
  - `rpy = [roll, pitch, yaw]` in radians (extrinsic XYZ / intrinsic ZYX Tait–Bryan angles matching URDF `<origin rpy="..."/>`).
- **TOML Section Ordering**: If hand-editing or templating `.toml` files directly, always place scalar fields (`uid`, `id`, `label`, `role`, `time_offset_s`, etc.) *before* child `[table]` or `[[array_of_tables]]` headers (e.g., `[pose]`, `[[subframes]]`) to satisfy TOML syntax rules (`ValueAfterTable`).

---

## 2. Workcell World Format (`WorldSpec`)

### 2.1 Top-Level Fields

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `schema_version` | `u32` | `1` | Must equal `1`. |
| `name` | `String` | *required* | Non-empty workcell identifier. |
| `description` | `String` | `""` | Optional notes. |
| `esdf` | `EsdfConfigSpec?` | `origin=[-1,-1,0], voxel=0.025, dims=[80,80,64], trunc=1.0` | Optional volumetric ESDF grid carved from `obstacles`. |
| `collision_rules` | `CollisionRulesSpec` | margins/activations + overrides | Body/tool margins, activation distances, `acm_overrides`, `pair_overrides`. |
| `planner` | `PlannerConfigSpec?` | `None` | Optional persisted planner/CBF/MPPI defaults. |
| `robots` | `Vec<RobotSpec>` | `[]` | Mounted robot arms in assembly order. |
| `obstacles` | `Vec<ObstacleSpec>` | `[]` | Static obstacles, pedestals, fixtures, and named `workpiece` shapes. |

### 2.2 Robot Entries (`[[robots]]`)

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `uid` | `String` | *required* | Unique stable UID referenced by `robot_uid` in trajectory files (e.g. `"arm_a"`). |
| `label` | `String` | `uid` | Display name in Studio UI. |
| `prefix` | `String` | `""` | Namespace prefix prepended to URDF/SRDF links, joints, and frames (e.g. `"a_"`). |
| `priority` | `u8` | `0` | Tier priority for CBF resolution (`0` = highest). |
| `active_group` | `String?` | `None` | Optional SRDF `<group>` name (defaults to whole robot). |
| `tcp_frame_name` | `String?` | `None` | Registers an explicit TCP fixed frame if set alongside `tcp_offset`. |
| `tcp_parent_frame` | `String?` | robot EE frame | Parent frame for custom TCP attachment. |
| `color_rgba` | `[f32; 4]?` | palette default | Display RGBA color. |
| `q_start` | `Vec<f64>` | clamped zeros | Initial joint angles (must match robot DOF when non-empty). |
| `source` | `RobotSourceSpec` | *required* | `{ kind = "builtin4_dof" \| "builtin6_dof" }`, `{ kind = "files", urdf_path, srdf_path? }`, or `{ kind = "inline", urdf_xml, srdf_xml? }`. Relative paths resolve against the loaded world file's parent directory. |
| `base_pose` | `PoseSpec` | identity | World mounting pose. |
| `tcp_offset` | `PoseSpec?` | `None` | Fixed rigid transform from `tcp_parent_frame` to tool TCP. |
| `tool_payload` | `ToolPayloadSpec?` | `None` | Attached collision payload (`name`, `parent_link?`, `mass`, `safety_margin?`, `activation_distance?`, `touch_links`, `local_pose`, `shape`). |

### 2.3 Obstacles & Shapes (`[[obstacles]]`)

- **`role`**: `"static"` (default), `"workpiece"`, `"pedestal"`, or `"fixture"`. Workpieces carry `uid` referenced by `ReferenceFrame::Workpiece { obstacle_uid }`.
- **`carve_esdf`**: `bool` (default `true`).
- **`shape`** (`ShapeSpec`, tagged with `kind`, `+Z`-aligned cylinders/capsules):
  - `{ kind = "box", half_extents = [hx, hy, hz] }`
  - `{ kind = "sphere", radius = r }`
  - `{ kind = "cylinder", radius = r, half_height = hh }`
  - `{ kind = "capsule", radius = r, half_height = hh }`
  - `{ kind = "segment_capsule", a = [x0, y0, z0], b = [x1, y1, z1], radius = r }`

---

## 3. Per-Robot Trajectory Set Format (`TrajectorySetSpec`)

### 3.1 Hierarchy & Schema Fields

```text
TrajectorySetSpec
 └── robots: Vec<RobotTrajectoriesSpec>        (per-robot trajectory bundle)
      └── trajectories: Vec<TrajectorySpec>    (ordered trajectories per robot)
           └── frames: Vec<TrajectoryFrameSpec> (ordered primary frames)
                └── subframes: Vec<SubframeSpec> (recursive child subframes)
```

#### Root `TrajectorySetSpec`
| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `schema_version` | `u32` | `1` | Must equal `1`. |
| `name` | `String` | *required* | Non-empty trajectory set name. |
| `description` | `String` | `""` | Optional batch notes. |
| `world_name` | `String?` | `None` | Advisory workcell name. |
| `default_reference_frame` | `ReferenceFrame?` | `world` | Default reference frame inherited downward unless overridden. |
| `robots` | `Vec<RobotTrajectoriesSpec>` | `[]` | Ordered list per robot (`robot_uid` must be unique within file). |

#### Per-Robot `[[robots]]` (`RobotTrajectoriesSpec`)
| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `robot_uid` | `String` | *required* | Must match a compiled robot's `uid` in the active workcell. |
| `group_name` | `String?` | `None` | Optional planning group validation override. |
| `tcp_frame` | `String?` | robot default TCP | TCP frame override for this robot's trajectories. |
| `reference_frame` | `ReferenceFrame?` | inherit root | Reference frame override for all frames under this robot. |
| `trajectories` | `Vec<TrajectorySpec>` | `[]` | Ordered trajectories for this robot. |

#### Trajectory `[[robots.trajectories]]` (`TrajectorySpec`)
| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `id` | `String` | *required* | Unique trajectory ID within this robot. |
| `label` | `String` | `id` | Display label in the Studio card. |
| `enabled` | `bool` | `true` | Only enabled trajectories feed IK & playback; disabled trajectories remain dimmed in 3D preview. |
| `speed_scale` | `f64` | `1.0` | Positive finite speed multiplier (`> 0.0`). |
| `tcp_frame` | `String?` | inherit robot | TCP frame override for this trajectory. |
| `reference_frame` | `ReferenceFrame?` | inherit robot | Reference frame override for this trajectory. |
| `frames` | `Vec<TrajectoryFrameSpec>` | `[]` | Ordered primary target frames (`id` unique within trajectory). |

#### Primary Frame `[[robots.trajectories.frames]]` (`TrajectoryFrameSpec`)
| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `id` | `String` | *required* | Unique frame ID within the trajectory. |
| `label` | `String` | `id` | UI label. |
| `role` | `SubframeRole` | `"waypoint"` | `"approach"`, `"waypoint"`, `"contact"`, `"retract"`, or `"reference_only"`. |
| `time_s` | `f64?` | `None` | Optional target timestamp in seconds from trajectory start. |
| `execute_ik` | `bool?` | `!is_reference_only` | Explicit opt-in/out for IK solving (`reference_only` defaults to `false`). |
| `tcp_frame` | `String?` | inherit trajectory | Target TCP frame name (`frame_graph().find_by_name`). |
| `joint_seed` | `Vec<f64>` | `[]` | Optional joint-space seed of exact robot DOF length. |
| `color_rgba` | `[f32; 4]?` | role default | Custom RGBA marker color. |
| `reference_frame` | `ReferenceFrame?` | inherit trajectory | Coordinate frame in which `pose` is expressed. |
| `pose` | `PoseSpec` | identity | 3D frame pose (`xyz`, `rpy`) in the effective reference frame. |
| `subframes` | `Vec<SubframeSpec>` | `[]` | Nested child subframes relative to this frame (`world_T_child = world_T_parent * local_pose`). |

#### Recursive Subframe (`SubframeSpec`)
Carries `id` (unique among siblings), `label`, `role`, `time_offset_s` (seconds relative to parent), `execute_ik`, `tcp_frame`, `color_rgba`, `pose` (relative to parent), and nested `subframes`.

---

## 4. Reference Frames & Subframe Execution Semantics

### 4.1 `ReferenceFrame` Inheritance Order
Effective reference frame resolves in order:
`frame.reference_frame` -> `trajectory.reference_frame` -> `robot.reference_frame` -> `set.default_reference_frame` -> `world`.

Supported variants (tagged via `kind`):
1. `{ kind = "world" }` — global workcell origin.
2. `{ kind = "robot_base" }` — robot's mounted `base_pose`.
3. `{ kind = "model_frame", name = "..." }` — any named link/TCP/sensor frame evaluated at `q_ref`.
4. `{ kind = "workpiece", obstacle_uid = "..." }` — world pose of the named obstacle in `WorldScene` / `BuiltWorkcell`.

TCP frame names follow the same top-down inheritance (`subframe.tcp_frame` -> `frame.tcp_frame` -> `trajectory.tcp_frame` -> `robot.tcp_frame` -> robot default TCP), supporting either full prefixed names (`"a_torch_tcp"`) or unprefixed names automatically matched with `robot.prefix`.

### 4.2 Pre/Post Ordering Around `time_offset_s`
When `resolve_trajectory_set` flattens a frame node and its `subframes` tree into an ordered waypoint stream (`ResolvedTrajectory::waypoints`), each node resolves as:
1. All child subframes with **`time_offset_s < 0.0`**, sorted stably ascending (e.g. `-0.4s` lead-in **Approach** runs *before* its parent frame). Each child recursively resolves its own nested subframes.
2. The **parent frame itself**.
3. All child subframes with **`time_offset_s >= 0.0`**, sorted stably ascending while preserving declared file order on ties (e.g. `0.0s` / `+0.4s` **Retract** runs *after* the parent frame).
4. Any node whose resolved `execute_ik == false` (such as `role = "reference_only"`) stays in `waypoints` for **3D coordinate-triad & parent-connector visualization**, while `executable_waypoints_for_robot()` omits it from IK & motion execution.

---

## 5. Complete Dual-Arm Examples

### 5.1 TOML Example (`weld_and_scan.toml`)

```toml
schema_version = 1
name = "dual_cell_seam_and_scan"
description = "Arm A welds workpiece seam while Arm B sweeps an inspection camera"
world_name = "dual_cell"

[default_reference_frame]
kind = "workpiece"
obstacle_uid = "workpiece"

[[robots]]
robot_uid = "arm_a"
tcp_frame = "a_torch_tcp"

[[robots.trajectories]]
id = "seam_1"
label = "Front Seam Pass"
enabled = true
speed_scale = 0.85

[[robots.trajectories.frames]]
id = "spot_start"
label = "Seam Start"
role = "contact"
time_s = 1.0

[robots.trajectories.frames.pose]
xyz = [0.0, -0.05, 0.08]
rpy = [0.0, 0.0, 0.0]

[[robots.trajectories.frames.subframes]]
id = "lead_in"
label = "Approach Standoff"
role = "approach"
time_offset_s = -0.35

[robots.trajectories.frames.subframes.pose]
xyz = [0.0, 0.0, 0.06]
rpy = [0.0, 0.0, 0.0]

[[robots.trajectories.frames.subframes.subframes]]
id = "sensor_mark"
label = "Fixture Fiducial"
role = "reference_only"
time_offset_s = 0.0

[robots.trajectories.frames.subframes.subframes.pose]
xyz = [0.02, 0.0, 0.0]
rpy = [0.0, 0.0, 0.0]

[[robots.trajectories.frames.subframes]]
id = "lead_out"
label = "Retract"
role = "retract"
time_offset_s = 0.35

[robots.trajectories.frames.subframes.pose]
xyz = [0.0, 0.0, 0.08]
rpy = [0.0, 0.0, 0.0]

[[robots]]
robot_uid = "arm_b"

[robots.reference_frame]
kind = "robot_base"

[[robots.trajectories]]
id = "scan_sweep"
label = "Optical Sweep"
enabled = true
speed_scale = 1.0

[[robots.trajectories.frames]]
id = "look_1"
label = "View Pose 1"
role = "waypoint"
time_s = 0.8
joint_seed = [0.0, 0.25, -0.25, 0.1]

[robots.trajectories.frames.pose]
xyz = [0.35, -0.20, 0.45]
rpy = [0.0, 0.0, 0.0]
```

### 5.2 JSON Equivalent (`weld_and_scan.json`)

```json
{
  "schema_version": 1,
  "name": "dual_cell_seam_and_scan",
  "description": "Arm A welds workpiece seam while Arm B sweeps an inspection camera",
  "world_name": "dual_cell",
  "default_reference_frame": { "kind": "workpiece", "obstacle_uid": "workpiece" },
  "robots": [
    {
      "robot_uid": "arm_a",
      "tcp_frame": "a_torch_tcp",
      "trajectories": [
        {
          "id": "seam_1",
          "label": "Front Seam Pass",
          "enabled": true,
          "speed_scale": 0.85,
          "frames": [
            {
              "id": "spot_start",
              "label": "Seam Start",
              "role": "contact",
              "time_s": 1.0,
              "pose": { "xyz": [0.0, -0.05, 0.08], "rpy": [0.0, 0.0, 0.0] },
              "subframes": [
                {
                  "id": "lead_in",
                  "label": "Approach Standoff",
                  "role": "approach",
                  "time_offset_s": -0.35,
                  "pose": { "xyz": [0.0, 0.0, 0.06], "rpy": [0.0, 0.0, 0.0] },
                  "subframes": [
                    {
                      "id": "sensor_mark",
                      "label": "Fixture Fiducial",
                      "role": "reference_only",
                      "time_offset_s": 0.0,
                      "pose": { "xyz": [0.02, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0] }
                    }
                  ]
                },
                {
                  "id": "lead_out",
                  "label": "Retract",
                  "role": "retract",
                  "time_offset_s": 0.35,
                  "pose": { "xyz": [0.0, 0.0, 0.08], "rpy": [0.0, 0.0, 0.0] }
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```
