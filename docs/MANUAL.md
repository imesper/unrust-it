# `unrust-it` User & Developer Manual

This manual covers building the workspace, using the `unrust-it` and `unrust-it-urdf` robotics APIs directly in Rust applications, operating the `apps/visualizer` Desktop Engineering Studio and CLI tools, and tuning performance for multi-arm real-time deployment.

---

## 1. Installation & Crate Features

### 1.1 Adding Dependencies

```toml
[dependencies]
unrust-it = { path = "crates/unrust-it" }
unrust-it-urdf = { path = "crates/unrust-it-urdf" }
nalgebra = "0.34"
parry3d = { package = "parry3d-f64", version = "0.26" }
```

### 1.2 Feature Flags Across Crates

| Crate | Feature | Default | Behavior |
| :--- | :--- | :---: | :--- |
| `unrust-it` | `gpu` | Yes | Enables `wgpu` + `pollster` WGSL compute shaders (`WgpuBatchBackend`, `GpuRobotContext`). Disable (`default-features = false`) for pure CPU SIMD (`wide::f32x8` + `rayon`) headless targets. |
| `unrust-it-urdf` | *(none)* | — | Pure CPU XML URDF/SRDF parser (`roxmltree` + `parry3d-f64`). |
| `unrust-it-visualizer` | `gui` | Yes | Enables native Slint 1.19 + `winit` + zero-copy `wgpu 30` Desktop Engineering Studio (implies `render`). |
| `unrust-it-visualizer` | `render` | via `gui` | Pure-`wgpu 30` 3D `SceneRenderer` + `OffscreenViewport` without Slint windowing. |
| `unrust-it-visualizer` | `gui-readback` | No | Slint UI presenting `OffscreenViewport` RGBA8 readback buffers (useful on headless/software backends without `unstable-wgpu-30`). |
| `unrust-it-visualizer` | `gpu-compute` | No | Uses GPU `WgpuBatchBackend` inside Studio planners instead of `CpuSimdBatchBackend`. |

---

## 2. Core Robotics Library Tutorial

### 2.1 Loading N Arms from URDF & SRDF

Mount any number of robots into a **single** `MultiRobotModel` and `CollisionEnvironment` using unique `RobotId(k)` values and non-overlapping `prefix` strings (so link, joint, and frame names never collide):

```rust,no_run
use nalgebra::Isometry3;
use unrust_it::{CollisionEnvironment, MultiRobotModel, RobotId};
use unrust_it_urdf::presets::{ARM_6DOF_SRDF, ARM_6DOF_URDF};
use unrust_it_urdf::{add_urdf_file_to_scene, add_urdf_to_scene, apply_srdf_file, apply_srdf_str};

fn load_three_arms() -> Result<(MultiRobotModel, CollisionEnvironment), Box<dyn std::error::Error>> {
    let mut model = MultiRobotModel::new();
    let mut env = CollisionEnvironment::new();

    // From XML strings / built-in presets:
    for idx in 0..3 {
        let angle = (idx as f64) * std::f64::consts::TAU / 3.0;
        let base_pose = Isometry3::translation(0.56 * angle.cos(), 0.56 * angle.sin(), 0.0);
        let prefix = format!("arm{idx}_");

        add_urdf_to_scene(&mut model, &mut env, ARM_6DOF_URDF, RobotId(idx as u16), &prefix, base_pose)?;
        apply_srdf_str(&mut model, &mut env, ARM_6DOF_SRDF, &prefix)?;
    }

    // Or from filesystem URDF/SRDF files:
    // add_urdf_file_to_scene(&mut model, &mut env, "robots/arm.urdf", RobotId(3), "arm3_", base_pose)?;
    // apply_srdf_file(&mut model, &mut env, "robots/arm.srdf", "arm3_")?;

    Ok((model, env))
}
```

Configuration vectors $\mathbf{q}$ concatenate the actuated DOFs of all loaded arms in mount order (`handle.q_indices` records exact global indices for each robot).

### 2.2 Adding Obstacles, Workpieces & Carving an `EsdfGridF32`

Add exact `parry3d` obstacles to `CollisionEnvironment`, update poses dynamically when objects move, and carve static/workpiece geometry into a 3D `EsdfGridF32` for fast `f32` MPPI & IK rollouts:

```rust,no_run
use nalgebra::Isometry3;
use parry3d::shape::{Cuboid, SharedShape};
use parry3d::math::Vector as ParryVec;
use unrust_it::{CollisionEnvironment, EsdfGridF32};

fn populate_workpiece(env: &mut CollisionEnvironment) -> EsdfGridF32 {
    let box_shape = SharedShape::new(Cuboid::new(ParryVec::new(0.06, 0.06, 0.05)));
    let obs_id = env.add_obstacle(box_shape, Isometry3::translation(0.0, 0.0, 0.42));

    // Later, move an existing obstacle in-place without cloning:
    env.set_obstacle_pose(obs_id, Isometry3::translation(0.02, -0.01, 0.42));

    // Build and carve an EsdfGridF32 (25 mm voxels over a 1.6m x 1.6m x 1.2m volume):
    let mut esdf = EsdfGridF32::new([-0.8, -0.8, 0.0], 0.025, [64, 64, 48], 1.0);
    esdf.carve_collision_environment(env);
    esdf
}
```

> **Axis Convention Reminder**: `parry3d::shape::Cylinder` and `Capsule` are **Y-aligned** locally, whereas URDF and Z-up robotics tools treat cylinders/capsules as **Z-aligned**. Rotate local `parry3d` cylinder/capsule sub-poses by `+std::f64::consts::FRAC_PI_2` around `+X` (or let `unrust-it-urdf` handle URDF shapes automatically).

### 2.3 Attaching Compound Tools, `touch_links`, TCP Frames & Per-Pair Margins

For grippers, screwdrivers, or inspection probes that must approach a workpiece within millimeters while arm bodies maintain larger clearance:

```rust,no_run
use nalgebra::Isometry3;
use parry3d::math::Vector as ParryVec;
use parry3d::shape::{Cuboid, SharedShape};
use unrust_it::{AttachedPayload, CollisionEnvironment, LinkGeometry, MultiRobotModel};

fn attach_gripper(
    model: &mut MultiRobotModel,
    env: &mut CollisionEnvironment,
    flange_name: &str,
) {
    let flange = model.link_by_name(flange_name).expect("flange exists");
    let forearm = model.link(flange).parent_link.expect("parent link");

    // Compound tool shapes in mount frame:
    let body = SharedShape::new(Cuboid::new(ParryVec::new(0.02, 0.03, 0.015)));
    let geom = LinkGeometry::from_convex_shapes(vec![
        (Isometry3::translation(0.0, 0.0, 0.015), body),
    ]);

    let payload = AttachedPayload::new("arm0_gripper", flange, Isometry3::translation(0.0, 0.0, 0.03), geom)
        .with_mass(0.65)
        .with_touch_links(vec![flange, forearm]) // exempt mounting links from self-collision
        .with_safety_margin(0.004)               // 4 mm tool clearance
        .with_activation_distance(0.015);        // 15 mm CBF activation band

    env.attach_payload(payload);

    // Optional TCP frame rigidly attached to flange for IK & servo targets:
    let _tcp_frame = model.add_fixed_frame("arm0_tcp", flange, Isometry3::translation(0.0, 0.0, 0.135));

    // Per-pair custom override (e.g., tight 3 mm approach against obstacle 0):
    // env.set_payload_obstacle_margin("arm0_gripper", obs_id, 0.003);
    // env.set_payload_obstacle_activation("arm0_gripper", obs_id, 0.015);
}
```

### 2.4 Batched Collision-Free IK & Two-Stage Planning + TOPP-RA

1. Compile the `f32` batch model: `let flat = FlatRobotModelF32::compile(&model, &env);`.
2. Solve multi-target collision-free IK across parallel seeds (`BatchIkSolver::new(32).solve_collision_free(&CpuSimdBatchBackend, &model, &flat, &env, Some(&esdf), &q_seed, &targets)`).
3. Plan close-proximity coordinated motions with `TwoStageApproachPlanner`:
   - Pass an `ApproachTarget::with_standoff(TaskPose::new(tcp, final_pose), Vector3::new(0.0, 0.0, -0.04))` per arm.
   - Stage 1 solves coarse standoff trajectories on Tier-1 spheres & `EsdfGridF32` (via MPPI or RRT-Connect).
   - Stage 2 refines with `ReactiveServoController` under exact Tier-2 `GeometryTier::Convex` GJK/EPA witnesses, arm priorities, and tangential curl.
4. Fit time-optimal kinodynamic trajectories with `CubicSplinePath::from_waypoints(&waypoints)` -> `ToppraTrajectory::parameterize(&spline, &limits, num_grid)` or online `JerkLimitedTrajectory::generate(...)`.

### 2.5 Running the 1 kHz Reactive CBF Loop (`ReactiveServoController`)

```rust,no_run
use unrust_it::{CollisionEnvironment, MultiRobotModel, ReactiveServoConfig, ReactiveServoController, ServoTarget};

fn control_loop_step(
    model: &MultiRobotModel,
    env: &CollisionEnvironment,
    servo: &mut ReactiveServoController,
    q: &mut [f64],
    dq: &mut [f64],
    targets: &[ServoTarget],
    dt: f64,
) -> Result<(), unrust_it::ControlError> {
    let mut dq_cmd = vec![0.0; model.dof()]; // allocate once outside real-time loop in production!
    servo.step(model, env, q, dq, targets, None, &mut dq_cmd)?;
    for k in 0..q.len() {
        dq[k] = dq_cmd[k];
        q[k] += dq_cmd[k] * dt;
    }
    Ok(())
}
```

For dual-arm cooperative manipulation carrying one rigid object, pass `Some(&ClosedChainGraspConstraint::new_rigid(...))` as the `manifold` argument to enforce relative end-effector loop closure inside the servo step.

---

## 3. Desktop Engineering Studio Guide (`apps/visualizer`)

### 3.1 Launching GUI & CLI Modes

```bash
# Interactive Desktop Engineering Studio (Slint 1.19 + zero-copy wgpu 30)
cargo run --release -p unrust-it-visualizer

# Headless CLI exports (zero windowing initialized)
cargo run -p unrust-it-visualizer -- --export-html out.html
cargo run -p unrust-it-visualizer -- --export-json out.json

# Serve interactive HTML + JSON API on localhost
cargo run -p unrust-it-visualizer -- --serve 127.0.0.1:8080
```

### 3.2 Workspace Panels & Controls

- **Top Toolbar (`top_toolbar.slint`)**:
  - View mode switcher: **`3D Viewport`**, **`ACM Matrix`**, or **`Split (3D | ACM)`**.
  - Scenario preset loader, Solver status badge (`Idle`, `Running`, `Converged`, `Warning`, `Failed`), active GPU/CPU backend label, and **`Re-Plan Now`** trigger.
- **Left Dock — Scene & N-Arm Manager (`scene_panel.slint`)**:
  - Hierarchical collapsible Scene Tree (`Robots -> Arm -> Links & Tool`, `Obstacles -> Obstacles & Workpieces`).
  - `+ 6-DOF Arm`, `+ 4-DOF Arm`, `Import URDF` file picker, `+ Sphere Obs`, `+ Workpiece Box`, and `Arrange Ring`.
  - **Selected Arm / Tool Inspector**: Base X/Y/Z/Yaw sliders, `Face Origin`, Tool Preset switcher (`None`, `ParallelGripper`, `Screwdriver`, `InspectionProbe`), Tool Safety Margin / CBF Activation sliders, and **`TOUCH LINKS (SELF-EXEMPT)`** switches per link.
  - **Selected Obstacle Inspector** *(appears when an obstacle/workpiece is selected via Tree or 3D Pick)*: toggle `Workpiece Role (WP)`, `Carve Into ESDF Grid`, Pos X/Y/Z, Obstacle Margin, CBF Activation, and `Remove Obstacle`.
- **Center — 3D Viewport (`viewport_panel.slint`)**:
  - **Left-drag**: Orbit camera · **Right/Middle/Shift+Left drag**: Pan · **Scroll**: Zoom · **Double-click**: 3D raycast pick (selects clicked link, tool, or obstacle).
  - Camera preset buttons (`Iso`, `Top`, `Front`, `Side`, `Focus WP`) and **`Layers`** popover (`Links`, `Tools`, `Tier-1 Spheres`, `3D ESDF Slice Heatmap` with `X`/`Y`/`Z` axis + offset + opacity, `MPPI Rollouts & Trails`, `Distance Witnesses` with X-Ray & `Max Witnesses` slider, `TCP / TF Frames`, `Target & Standoff Ghosts`, `Floor Grid`).
  - Top HUD displays scenario title, live safety pill (`SAFE` / `TIGHT` / `COLLISION`), and live `wgpu 30` frame timing (`ms`).
- **Center / Split — Frozen-Header ACM Matrix (`acm_matrix_panel.slint`)**:
  - Slice tabs: `Inter-Arm`, `Self`, `Tool & WP`, `Link-Obs`, `All Filtered`, plus quick bulk buttons (`Inter-Arm 40mm`, `Tool-Tool 4mm`, `Tool-WP 3mm`, `Reset SRDF`).
  - Live text filter (`Filter links / tools / obstacles...`) and `Active / Custom Only` switch.
  - Synchronized frozen column/row headers with per-cell rule code (`4mm`, `DEF`, `ADJ`, `NEV`, `TCH`, `DYN`) and live distance `d(t)` in mm.
  - **Single-click cell**: inspects pair in the pinned bottom strip AND highlights both bodies + draws a cyan 3D connector rod in Split/3D view. **Double-click cell**: toggles pair Enabled/Disabled.
  - Bottom Pair Strip allows editing exact `Margin (mm)` and `Act (mm)` values inline via keyboard + `Apply`.
- **Right Dock — Planner, `Targets & IK`, & `Joints / Jog` (`planner_panel.slint`)**:
  - **`Plan & CBF`**: Preset selector, arm priority weights, CBF safety margin / activation / $\gamma$ / tangential curl, MPPI rollouts/horizon/iterations, and diagnostics (`Solve Time`, `Pose Error`, `Min Body Clr`, `Min Tool Clr`).
  - **`Targets & IK`**: Per-arm target pose sliders (`X/Y/Z`, `Roll/Pitch/Yaw`, `Standoff`), `Copy Current TCP Pose -> Target`, and **`Solve IK Preview (Batch 32 Seeds)`** (runs `BatchIkSolver` live and updates robot configuration in the viewport).
  - **`Joints / Jog`**: Interactive per-DOF joint sliders for live FK and exact witness inspection.
- **Bottom Dock — Timeline & Telemetry Plots (`timeline_dock.slint`)**:
  - Transport controls (`|<`, `<`, `Play/Pause`, `>`, `>|`), speed (`0.1x`–`2.0x`), **`Loop`** toggle, Stage-1/Stage-2 annotated scrubber with near-collision alert ticks, and 4 real-time RGBA8 telemetry plots (`Min Clearance`, `Tool Clearance`, `Pose Error`, `Joint Velocities`).

### 3.3 Built-In Studio Presets

| Preset | Robots | Focus Demonstrated |
| :--- | :--- | :--- |
| **`3-Arm Close Proximity Tool Work`** | Three 6-DOF arms | Gripper, screwdriver, and probe converging to 4 mm / 3 mm around central workpiece via `TwoStageApproachPlanner` |
| **`3-Arm Crossing Priority CBF`** | Three 6-DOF arms | Symmetric central crossing deadlock broken via arm priorities + tangential curl QP-CBF |
| **`Dual-Arm MPPI + ESDF Avoidance`** | Two 6-DOF arms | GPU/SIMD `BatchMppiOptimizer` navigation with candidate rollout ribbons around obstacles |
| **`Dual-Arm RRT-Connect + TOPP-RA`** | Two 6-DOF arms | Global joint-space RRT-Connect routing + time-optimal TOPP-RA parameterization |
| **`Dual-Arm Closed-Chain Manifold CBF`** | Two 4-DOF arms | Holonomic `ClosedChainGraspConstraint` rigid grasp maintenance amidst moving obstacle sweeps |

---

## 4. Performance Best Practices & Troubleshooting

- **Reuse Scratchpads Per Thread**: Create one `TransformBuffer` and one `CollisionWorkspace` per planning/worker thread. Never allocate new buffers or `Vec<SignedDistanceWitness>` vectors inside 1 kHz control ticks (`witnesses.clear()` preserves capacity).
- **Choose the Right Collision Tier**:
  - Use `GeometryTier::AnalyticSsv` or `CpuSimdBatchBackend` + `EsdfGridF32` during high-volume coarse sampling (RRT edges, MPPI rollouts, 32-seed IK).
  - Use `GeometryTier::Convex` during final Stage-2 millimeter convergence and safety verification.
- **Debug Build vs. Release Build**: `parry3d-f64`, `wide::f32x8`, and BVH traversals run **10–30x faster** with `--release`. Always benchmark and run live Studio sessions in `--release`.
- **Self-Collision False Positives After Attaching a Tool**: Check that `AttachedPayload::with_touch_links(...)` includes both the flange and nearby wrist links whose bounding volumes touch the mounted adapter bracket.
- **No GPU Adapter in Headless CI**: Build `unrust-it` with `--no-default-features` or `unrust-it-visualizer` with `--no-default-features --features render` / `gui-readback`.
