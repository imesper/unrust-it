# `unrust-it`

**High-performance open-source Rust path planning, dynamic manipulation, collision avoidance, and interactive 3D studio for N-arm ($K \ge 3$) robotic systems.**

---

## Highlights

- **N-Arm Generic Architecture**: Unified configuration vector $\mathbf{q} \in \mathbb{R}^{N_{\text{dof}}}$ across $K \ge 1$ manipulators with precomputed ancestor DOF bitmasks and Lipschitz lever-arm bounds.
- **Dual-Precision Compute**:
  - **Exact `f64` Kernel**: Analytical $6 \times N_{\text{dof}}$ geometric Jacobians, screw-axis distance projections, `parry3d-f64` GJK/EPA signed distances, closed-chain grasp manifolds, TOPP-RA / jerk-limited trajectories, and 1 kHz priority QP-CBF reactive control.
  - **Batched `f32` Rollout Engine**: CuRobo-style $B \times H$ forward kinematics and 3D ESDF evaluation over link sphere trees via `wide::f32x8` + `rayon` (`CpuSimdBatchBackend`) and `wgpu` WGSL compute shaders (`WgpuBatchBackend` + resident `GpuRobotContext`).
- **Compound Tool Payloads & Per-Pair Safety Rules**: Multi-primitive `AttachedPayload`s, hardware-`CTZ` bitset `AllowedCollisionMatrix`, SRDF self-collision exemptions (`touch_links`), and per-pair custom margins/activations (e.g., 3–4 mm tool clearance vs 40 mm inter-arm clearance).
- **Two-Stage Close-Proximity Planner**: Global coarse stage (`BatchIkSolver` + `BatchMppiOptimizer` or `RrtPlanner`) refined into final millimeters by a 1 kHz priority-weighted `ReactiveServoController` with tangential curl deadlock breaking.
- **Desktop Engineering Studio (`apps/visualizer`)**: Native **`eframe` / `egui 0.37` + zero-copy `wgpu 30` + `egui_plot 0.38`** 3D Studio with 3D raycast picking, frozen-header interactive ACM Matrix, Scene/Tool/Obstacle inspectors, 32-seed `Targets & IK` solver preview, interactive telemetry plots, plus headless CLI HTML/JSON exports.

---

## Workspace Layout

| Crate | Path | Role & Status |
| :--- | :--- | :--- |
| **`unrust-it`** | [`crates/unrust-it`](crates/unrust-it/Cargo.toml) | Pure Rust kinematics, collision, manifolds, trajectories, 1 kHz CBF control, planners, and CPU SIMD / `wgpu` batched rollouts (**Implemented**, GUI/IO-free) |
| **`unrust-it-urdf`** | [`crates/unrust-it-urdf`](crates/unrust-it-urdf/Cargo.toml) | URDF & MoveIt SRDF parser with automatic multi-tier sphere/capsule/convex collision decomposition & built-in presets (**Implemented**, GUI-free) |
| **`unrust-it-visualizer`** | [`apps/visualizer`](apps/visualizer/Cargo.toml) | Native `eframe`/`egui 0.37` + zero-copy `wgpu 30` Desktop Engineering Studio, offscreen CI renderer, and headless JSON/HTML scenario exporter (**Implemented**) |
| **`unrust-it-io`** | *Planned (Phase 3)* | Unified `serde` declarative `WorldSpec` loader/compiler (`.unrust.toml` + `.unrust.json`) shared across runtime and Studio |
| **`unrust-it-perception`** | *Planned (Phase 4)* | Sensor calibration (`CalibrationBundle`), RGB-D / LiDAR / mmWave Radar ingestion, zero-alloc `f32x8` self-filtering, and dynamic voxel-ESDF + CV-EKF obstacle fusion |

Core robotics crates (`unrust-it`, `unrust-it-urdf`) intentionally contain **no GUI, windowing, or runtime-framework dependencies**.

---

## Documentation

| Document | Scope |
| :--- | :--- |
| **[Architecture](docs/ARCHITECTURE.md)** | Current dual-precision architecture, zero-alloc workspace separation, Phase 2 friction audit, and planned Phase 3–5 world modeling, sensor calibration/fusion, dual-rate pipeline, and mobile manipulator designs |
| **[Roadmap & Milestones](docs/ROADMAP.md)** | Completed Phases 0–2D (with exact commits), planned Phase 3–5 milestones with exit criteria, and CI verification policy |
| **[Code Flow Reference](docs/CODE_FLOW.md)** | Module-by-module inventory and step-by-step data & execution flows from URDF parsing through Two-Stage planning, 1 kHz CBF ticks, WGSL rollouts, and the `egui` + `wgpu 30` Studio loop |
| **[User & Developer Manual](docs/MANUAL.md)** | Setup, feature flags, core library tutorials, `apps/visualizer` Studio UI guide, preset descriptions, performance tuning, and troubleshooting |

---

## Quick Start

```bash
# Run all workspace unit, integration, and doc tests
cargo test --workspace
cargo test --workspace --no-default-features

# Run runnable core examples
cargo run -p unrust-it --example simple_reach
cargo run -p unrust-it --example dual_arm_reach

# Launch the interactive Desktop Engineering Studio (eframe/egui 0.37 + wgpu 30)
cargo run --release -p unrust-it-visualizer

# Headless CLI exports / standalone web viewer (no GUI required)
cargo run -p unrust-it-visualizer -- --export-html visualizer.html
cargo run -p unrust-it-visualizer -- --export-json scenarios.json
cargo run -p unrust-it-visualizer -- --serve 127.0.0.1:8080

# Build local API Rustdoc
cargo doc --workspace --no-deps --open
```

---

## License

Licensed under the **GNU General Public License v3.0 or later** ([`LICENSE`](LICENSE)).
