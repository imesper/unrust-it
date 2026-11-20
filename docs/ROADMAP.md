# `unrust-it` Roadmap, Phases & Milestones

This document tracks all completed engineering phases and commits on `main`, alongside planned milestones for core RT optimization, declarative world modeling, sensor calibration/perception, mobile manipulators, and ecosystem bindings.

---

## 1. Master Phase Matrix

| Phase | Focus Area | Status | Key Commits |
| :---: | :--- | :---: | :--- |
| **0** | Workspace & crate scaffold | **Complete** | `29d0f63` .. `72cf3f9` |
| **1** | Multi-arm kinematics, bitset ACM, `f64` collision, manifolds, TOPP-RA, 1 kHz CBF servo | **Complete** | `8a29e0e`, `e204ff0` |
| **2A/2B** | CuRobo-style `f32` SIMD/WGSL batching, `EsdfGridF32`, `GpuRobotContext`, Batch IK & MPPI, URDF/SRDF | **Complete** | `31eae0d`, `fc4197c`, `d728dfd`, `335efd1` |
| **2C** | Compound tool payloads, per-pair safety margins, priority QP-CBF + curl, Two-Stage planner | **Complete** | `3f475f6` |
| **2D** | Declarative scene engine, pure-`wgpu 30` 3D renderer, Slint 1.19 Studio UI, interactive inspectors | **Complete** | `556a732`, `9f8067a`, `1eff888`, `dbcf530` |
| **3** | Core 1 kHz zero-alloc polish, `WorldScene`, `unrust-it-io` (TOML/JSON), Dual-Rate Pipeline | **In Progress (3A Complete)** | — |
| **4** | `unrust-it-perception`: Calibration ingestion, RGB-D/LiDAR/Radar self-filtering, dynamic ESDF + CV-EKF | **Planned** | — |
| **5** | Mobile manipulators (`nq`/`nv`), CAD mesh ingestion (VHACD), ROS 2 / Zenoh / Python / C ABI | **Planned** | — |

---

## 2. Completed Phases & Milestones

### Phase 0 — Initial Repository & Cargo Workspace
- **Commits**: `29d0f63`, `de64dcf`, `85fcc1b`, `d5d1650`, `c2edd7c`, `72cf3f9`
- Established workspace build layout (`crates/unrust-it`), error types, `Configuration`, `Robot`, initial `RrtPlanner` skeleton, and runnable examples (`simple_reach`, `dual_arm_reach`).

### Phase 1 — `f64` Multi-Arm Robotics Kernel
- **Commit `8a29e0e`**: Multi-robot kinematic forest (`MultiRobotModel`, `FrameGraph`, `PlanningGroup`), analytical $6 \times N_{\text{dof}}$ geometric Jacobians, screw-axis distance projection, multi-target DLS IK, hardware-`CTZ` bitset `AllowedCollisionMatrix`, multi-tier collision checking (`GeometryTier::AnalyticSsv` + `GeometryTier::Convex`), BVH broad-phase, Lipschitz segment checks, and `SignedDistanceWitness::compute_distance_jacobian`.
- **Commit `e204ff0`**: Holonomic `ConstraintManifold` & `ClosedChainGraspConstraint` (Newton-Raphson projection), $C^2$ `CubicSplinePath`, `ToppraTrajectory`, online `JerkLimitedTrajectory`, and 1 kHz `ReactiveServoController` with CBF inequalities and velocity-damper joint bounds.

### Phase 2A / 2B — Batched `f32` SIMD/GPU Engine & URDF/SRDF Loader
- **Commit `31eae0d`**: `FlatRobotModelF32`, `EsdfGridF32`, and `CpuSimdBatchBackend` (`wide::f32x8` + `rayon`) for $B \times H$ kinematics and collision rollouts matching `f64` reference checks.
- **Commit `fc4197c`**: `WgpuBatchBackend` and WGSL compute kernels (`rollout.wgsl`: `fk_main`, `collision_main`) adhering to WebGPU's 8-storage-buffer ceiling.
- **Commit `d728dfd`**: Persistent resident `GpuRobotContext` (`update_esdf`, `forward_kinematics_resident`, `collision_cost_resident`), parallel multi-seed `BatchIkSolver`, `BatchMppiOptimizer`, and the new `crates/unrust-it-urdf` crate (`load_urdf_str`, `add_urdf_to_scene`, `apply_srdf_str`) with automatic multi-tier decomposition.
- **Commit `335efd1`**: Initial `apps/visualizer` binary & headless 3D HTML/JSON exporter.

### Phase 2C — Close-Proximity Tool Manipulation & Two-Stage Planner
- **Commit `3f475f6`**:
  - Compound multi-primitive `AttachedPayload`s (`PayloadId`, `touch_links`, per-tool safety margins & activations) reflected into both `f64` `CollisionEnvironment` and `f32` `FlatRobotModelF32` / `rollout.wgsl` (`SPHERE_FLAG_PAYLOAD`).
  - Per-pair custom margins & activations (`CollisionRule::CustomMargin`, `PayloadPairRule`).
  - Per-arm priority weighting and tangential curl deadlock breaking in `ReactiveServoController`.
  - `TwoStageApproachPlanner` pairing coarse `BatchIkSolver` + `BatchMppiOptimizer` / `RrtPlanner` with Stage 2 exact `GeometryTier::Convex` CBF convergence down to millimeter-scale tool-to-workpiece standoff (verified in `crates/unrust-it-urdf/tests/three_arm_proximity.rs`).

### Phase 2D — Native Desktop Engineering Studio (`apps/visualizer`)
- **Milestone 2D-1 (`556a732`)**: Added GUI-free kernel/URDF enablers (`BatchMppiOptimizer::optimize_with_trace`, `EsdfGridF32::carve_collision_environment`, `CollisionEnvironment::set_obstacle_pose`, file loaders & `presets` in `unrust-it-urdf`), plus the declarative `WorkcellSpec -> BuiltWorkcell` compiler and 5 multi-arm presets.
- **Milestone 2D-2 (`9f8067a`)**: Built the Slint-free pure-`wgpu 30` `SceneRenderer` (`OrbitCamera`, `meshes.rs`, `extract.rs`, `solid.wgsl`, `lines.wgsl`, `esdf_slice.wgsl`) and `OffscreenViewport` headless CI validation across all presets.
- **Milestone 2D-3 (`1eff888`)**: Built the Slint `~1.19` desktop shell (`ui/main.slint`, `state.slint`, panels, widgets), zero-copy `GraphicsAPI::WGPU30` viewport bridge (plus `gui-readback` fallback), latest-generation coalescing `PlannerWorker`, and 4 RGBA8 telemetry plot views (`plots.rs`).
- **Milestone 2D-4 (`dbcf530`)**: Completed 3D raycast picking (`pick_entity_along_ray`), ACM & entity 3D pair/rod highlighting (`SceneHighlights`), Obstacle & Tool/Touch-Link inspectors (`custom_tool_touch_link_names`), frozen-header 2D `Flickable` ACM Matrix with search/active filtering & SRDF reset, `Targets & IK` tab with 32-seed `BatchIkSolver` preview, ESDF X/Y/Z axis & opacity controls, Timeline `Loop` toggle, and live `wgpu 30` frame-time telemetry.

---

## 3. Planned Phases & Milestones

### Phase 3 — Core 1 kHz RT Ergonomics, World Schema & Dual-Rate Pipeline

#### Milestone 3A — Core Hot-Path Polish & Subgroup Planning ✅ Complete
- **Deliverables**:
  - Eliminate remaining per-tick heap allocations inside `ReactiveServoController::step()`, `ToppraTrajectory::sample_into()`, and payload pair lookups (integer-indexed lookup instead of `String` formatting).
  - Add `ServoTarget::JointReference { q_ref, dq_ref, kp }` so `ReactiveServoController` operates directly as a 1 kHz CBF safety filter over any trajectory.
  - Connect `GpuRobotContext` resident buffers directly into `BatchRolloutBackend` for zero-reallocation GPU MPPI/IK iterations.
  - Enable masked subgroup planning over `PlanningGroup` subsets in `RrtPlanner` and `TwoStageApproachPlanner`.
- **Exit Criteria**: Zero heap allocations inside 1 kHz `step()` and trajectory sampling loops (verified with an allocation-counting test); joint-trajectory CBF filter benchmark & tests pass.

#### Milestone 3B — Unified `WorldScene` Owner
- **Deliverables**:
  - Named obstacle handles (`ObstacleHandle`), add/update/remove APIs, dirty-region `EsdfGridF32` re-carving, and synchronized `CollisionEnvironment` + ESDF state inside one owner.
- **Exit Criteria**: Adding, moving, or removing obstacles updates both Tier-2 BVH queries and `EsdfGridF32` distances atomically without manual caller synchronization.

#### Milestone 3C — `crates/unrust-it-io` (Declarative `WorldSpec` & Per-Robot Trajectory Import)
- **Deliverables**:
  - New `unrust-it-io` crate with `serde`-backed `WorldSpec` (`[[robots]]`, `[[payloads]]`, `[[obstacles]]`, `[[sensors]]`, `[[collision.pair_rules]]`, `[esdf]`, `[planner]`).
  - TOML (`.unrust.toml`) primary human-editable format + automatic JSON (`.unrust.json`) support -> `WorldSpec::compile() -> CompiledWorld`.
  - **Per-Robot Trajectory & Frame/Subframe Import Schema (`format TBD`)**: Design and finalize the multi-robot trajectory list schema (ordered trajectories per robot with frames, nested subframes, timing/blend metadata, and Cartesian/joint representations) inside `crates/unrust-it-io`.
  - File -> Open / Save `.unrust.toml` & `.unrust.json` integration plus a dedicated **`Import Trajectories`** file-picker button in `apps/visualizer` to load per-robot trajectory sequences with frames & subframes for playback, collision verification, and CBF-filtered execution.
- **Exit Criteria**: Round-trip TOML/JSON serialization tests, multi-arm URDF path resolution relative to config file directory, trajectory-list import round-trip parsing & frame/subframe transform resolution tests, and Studio import/export validation.

#### Milestone 3D — Composable Dual-Rate Pipeline (`unrust_it::pipeline`)
- **Deliverables**:
  - Pipeline stage traits (`PerceptionIngest`, `WorldScene`, `TrajectoryMonitor`, `GlobalPlannerStage`, `TimeParameterizerStage`, `ReactiveSafetyFilter`, `CommandSink`).
  - `SyncPipeline::tick(dt)` deterministic single-thread runner and `SplitPipeline` lock-free `ArcSwap`/triple-buffer 1 kHz hard-RT thread + 1–50 Hz background planner with mid-flight horizon invalidation & automatic replanning.
- **Exit Criteria**: Determinism test for `SyncPipeline` and simulated dynamic obstacle insertion triggering automatic background replan under `SplitPipeline`.

---

### Phase 4 — Sensors, Calibration & Dynamic Perception (`crates/unrust-it-perception`)

#### Milestone 4A — Calibration Bundle & Eye-to-Hand / Eye-in-Hand Mounting
- **Deliverables**:
  - `CalibrationBundle` parsers for **TOML**, **JSON**, and **OpenCV/ROS XML** (via `roxmltree`, with optional YAML).
  - Intrinsics models (Pinhole, Brown-Conrady, Kannala-Brandt, LiDAR beam tables, Radar FOV/Doppler, time offset `dt_offset_s`).
  - Automatic mounting into `MultiRobotModel::frame_graph` as world-fixed (Eye-to-Hand) or link-attached (Eye-in-Hand) frames, plus load-time SoA `RayLutF32` precomputation.
- **Exit Criteria**: Calibration round-trip accuracy tests against analytical distortion projections and Eye-in-Hand FK validation.

#### Milestone 4B — Zero-Copy Observations, Deskewing & SIMD Self-Filter
- **Deliverables**:
  - Borrowed `SensorObservation<'a>` types for RGB-D (`DepthU16`/`DepthF32`/`OrganizedCloud`), 2D/3D LiDAR (with 256-slot 1 kHz `JointStateHistory` temporal deskewing), and mmWave Radar (with eye-in-hand $J_v(q)\dot{q}$ Doppler compensation).
  - 3-stage branchless `wide::f32x8` SIMD self-filter tagging robot hits as `FreeOnlyHitRobot`.
- **Exit Criteria**: Zero-allocation filter benchmark (< 0.25 ms for typical VGA depth decimated cloud / LiDAR sweep) and synthetic self-occlusion test confirming robot arms are stripped while nearby workspace obstacles are preserved.

#### Milestone 4C — Dual-Layer Real-Time World Fusion & Studio Overlay
- **Deliverables**:
  - 30 Hz `f32` layer: `i8` log-odds `DynamicVoxelGrid` with deduplicated 3D DDA clearing, TTL decay, Radar velocity-swept predictive capsules, parallel 3D EDT / WGSL JFA -> `min(static_esdf, dyn_esdf)` uploaded to `GpuRobotContext`.
  - 1 kHz `f64` layer: near-field ROI clustering ($\le 48$ convex primitives) + 6-state CV-EKF multi-target tracker via lock-free triple buffer -> velocity-compensated CBF inequality.
  - Studio viewport overlays for live sensor clouds, self-filtered voxels, and EKF velocity vectors.
- **Exit Criteria**: Closed-loop dynamic obstacle avoidance integration test (moving obstacle intercepted by Radar/RGB-D triggers velocity-compensated CBF braking + background MPPI replanning).

---

### Phase 5 — Mobile Manipulators, CAD Meshes & Ecosystem

#### Milestone 5A — Mobile Bases & Mobile Manipulators (`nq` / `nv` Separation)
- **Deliverables**:
  - Separate $n_q$ and $n_v$ in `MultiRobotModel`, `TransformBuffer` screw columns, `JointType::Planar { drive: Holonomic | DiffDrive | Ackermann }`, and `JointType::Floating`.
  - Mounting URDF manipulators onto mobile chassis links via `parent_link`.
  - `rollout.wgsl` joint kinds 3 & 4 + raise static link array cap from `64` to `128`.
- **Exit Criteria**: Mobile manipulator planning test (holonomic and differential-drive base + 6-DOF arm reaching around obstacles cleanly).

#### Milestone 5B — CAD Mesh Ingestion & Convex Decomposition
- **Deliverables**:
  - STL / OBJ / GLTF mesh loader with automatic convex hull & VHACD decomposition into Tier-1 spheres and Tier-2 `parry3d::ConvexPolyhedron` shapes for `<mesh filename="..."/>` URDF geometries and arbitrary workpieces.
- **Exit Criteria**: URDF with mesh links compiles to multi-tier collision geometry and renders natively in `apps/visualizer`.

#### Milestone 5C — Middleware & Foreign-Language Ecosystem
- **Deliverables**:
  - Thin optional transport crates (`unrust-it-ros2` / Zenoh adapters), Python bindings (`PyO3`), C ABI headers for industrial PLC/C++ integration, and criterion benchmark suite tracking latency regressions.
- **Exit Criteria**: Documented end-to-end Python & C++ FFI examples and CI benchmark baseline.

---

## 4. Verification & Commit Policy

Every milestone on `main` is verified before committing with the following workspace gate (requiring exit code `0` and zero warnings):

1. `cargo clippy --workspace --all-targets -- -D warnings`
2. `cargo test --workspace`
3. `cargo test --workspace --no-default-features`
4. `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`
5. `cargo test --doc --workspace`
6. When `apps/visualizer` changes:
   - `cargo check -p unrust-it-visualizer --no-default-features --features render`
   - `cargo check -p unrust-it-visualizer --no-default-features --features gui-readback`
   - `cargo run -p unrust-it-visualizer -- --export-json target/ci_smoke.json --export-html target/ci_smoke.html`
