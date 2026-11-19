//! Declarative workcell model, compilation, presets, and trajectory evaluation.

pub mod compile;
pub mod eval;
pub mod presets;
pub mod spec;

pub use compile::{BuiltWorkcell, CompiledArm, CompiledObstacle, SceneCompileError};
pub use eval::{
    evaluate_frame_snapshot, run_planner_sync, FrameSnapshot, PlanMetrics, PlaybackTrajectory,
    RolloutRibbon, TrajectoryFrame, TrajectoryStage,
};
pub use presets::{all_preset_ids, preset_workcell};
pub use spec::{
    arm_palette_color, AcmRuleSpec, ArmSpec, ArmTargetSpec, CollisionRulesSpec,
    ComputeBackendChoice, CustomToolSpec, DirtyFlags, EntityRef, EsdfSpec, ObstacleRole,
    ObstacleSpec, PairRuleOverrideSpec, PlannerSpec, PoseSpec, PresetId, ShapePrimitiveSpec,
    Stage1Strategy, ToolPreset, UrdfSource, WorkcellSpec,
};
