//! Declarative TOML/JSON workcell world definitions and per-robot ordered
//! trajectory / frame / subframe import and export for [`unrust_it`].
//!
//! Keep `serde`, `toml`, and `serde_json` dependencies isolated to this crate so
//! core planning/control (`unrust-it`) and URDF/SRDF parsing (`unrust-it-urdf`)
//! remain zero-overhead and dependency-minimal.

pub mod error;
pub mod pose;
pub mod trajectory_io;
pub mod world;

pub use error::IoError;
pub use pose::{FileFormat, PoseSpec, ShapeSpec};
pub use trajectory_io::{
    resolve_trajectory_set, resolve_trajectory_set_with_model, ReferenceFrame,
    ResolvedRobotTrajectories, ResolvedTrajectory, ResolvedTrajectorySet, ResolvedWaypoint,
    RobotResolveInfo, RobotTrajectoriesSpec, SubframeRole, SubframeSpec, TrajectoryFrameSpec,
    TrajectorySetSpec, TrajectorySpec, TRAJECTORY_SCHEMA_VERSION,
};
pub use world::{
    AcmOverrideSpec, AcmRuleValueSpec, CollisionRulesSpec, CompiledRobot, CompiledWorld,
    CompiledWorldObstacle, EntityRefSpec, EsdfConfigSpec, ObstacleRoleSpec, ObstacleSpec,
    PairOverrideSpec, PlannerConfigSpec, RobotSourceSpec, RobotSpec, Stage1StrategySpec,
    ToolPayloadSpec, WorldSpec, WORLD_SCHEMA_VERSION,
};
