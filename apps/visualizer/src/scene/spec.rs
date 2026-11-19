//! Declarative workcell, robot arm, tool payload, obstacle, collision rule,
//! and planner specifications for `unrust-it` desktop visualization and batch execution.

use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
use parry3d::math::Vector as ParryVector;
use parry3d::shape::{Ball, Capsule, Cuboid, Cylinder, SharedShape};
use std::f64::consts::FRAC_PI_2;
use std::path::PathBuf;
use unrust_it::{MppiConfig, ReactiveServoConfig};

/// 3D rigid pose encoded as translation `(x, y, z)` in meters and extrinsic XYZ Euler angles `(r, p, y)` in radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoseSpec {
    pub xyz: [f64; 3],
    pub rpy: [f64; 3],
}

impl Default for PoseSpec {
    fn default() -> Self {
        Self {
            xyz: [0.0, 0.0, 0.0],
            rpy: [0.0, 0.0, 0.0],
        }
    }
}

impl PoseSpec {
    /// Creates a pure translation pose with zero rotation.
    pub fn translation(x: f64, y: f64, z: f64) -> Self {
        Self {
            xyz: [x, y, z],
            rpy: [0.0, 0.0, 0.0],
        }
    }

    /// Creates a pose translated to `(x, y, z)` with a yaw rotation about `+Z` in radians.
    pub fn from_xyz_yaw(x: f64, y: f64, z: f64, yaw: f64) -> Self {
        Self {
            xyz: [x, y, z],
            rpy: [0.0, 0.0, yaw],
        }
    }

    /// Converts to a `nalgebra::Isometry3<f64>`.
    pub fn to_isometry(&self) -> Isometry3<f64> {
        Isometry3::from_parts(
            Translation3::new(self.xyz[0], self.xyz[1], self.xyz[2]),
            UnitQuaternion::from_euler_angles(self.rpy[0], self.rpy[1], self.rpy[2]),
        )
    }

    /// Converts from a `nalgebra::Isometry3<f64>`.
    pub fn from_isometry(iso: &Isometry3<f64>) -> Self {
        let t = iso.translation.vector;
        let (r, p, y) = iso.rotation.euler_angles();
        Self {
            xyz: [t.x, t.y, t.z],
            rpy: [r, p, y],
        }
    }

    /// Constructs a right-handed frame at `tip` whose local `+Z` points along `z_axis`.
    pub fn from_z_approach(tip: Vector3<f64>, z_axis: Vector3<f64>) -> Self {
        let z = z_axis.normalize();
        let ref_up = if z.z.abs() < 0.9 {
            Vector3::z()
        } else {
            Vector3::x()
        };
        let x = ref_up.cross(&z).normalize();
        let y = z.cross(&x).normalize();
        let rot_mat = nalgebra::Matrix3::from_columns(&[x, y, z]);
        let rot =
            UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(rot_mat));
        Self::from_isometry(&Isometry3::from_parts(Translation3::from(tip), rot))
    }
}

/// Source for loading a robot arm's URDF kinematics and optional MoveIt SRDF defaults.
#[derive(Debug, Clone, PartialEq)]
pub enum UrdfSource {
    Builtin4Dof,
    Builtin6Dof,
    InlineXml { urdf: String, srdf: Option<String> },
    FilePath { urdf_path: PathBuf, srdf_path: Option<PathBuf> },
}

/// Geometric shape primitive in right-handed Z-up local coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShapePrimitiveSpec {
    Sphere { radius: f64 },
    Cuboid { half_extents: [f64; 3] },
    CylinderZ { half_height: f64, radius: f64 },
    CapsuleZ { half_height: f64, radius: f64 },
}

impl ShapePrimitiveSpec {
    /// Converts this declarative primitive into a `(local_pose, parry3d::shape::SharedShape)` pair.
    ///
    /// Note that `parry3d::shape::Cylinder` is Y-aligned locally, so `CylinderZ` applies a
    /// `+FRAC_PI_2` X rotation to match Z-up URDF conventions.
    pub fn to_parry_shape_and_pose(&self) -> (Isometry3<f64>, SharedShape) {
        match *self {
            Self::Sphere { radius } => (Isometry3::identity(), SharedShape::new(Ball::new(radius))),
            Self::Cuboid { half_extents } => (
                Isometry3::identity(),
                SharedShape::new(Cuboid::new(ParryVector::new(
                    half_extents[0],
                    half_extents[1],
                    half_extents[2],
                ))),
            ),
            Self::CylinderZ { half_height, radius } => (
                Isometry3::rotation(Vector3::x() * FRAC_PI_2),
                SharedShape::new(Cylinder::new(half_height, radius)),
            ),
            Self::CapsuleZ { half_height, radius } => (
                Isometry3::identity(),
                SharedShape::new(Capsule::new_z(half_height, radius)),
            ),
        }
    }
}

/// Definition of a custom multi-shape attached end-of-arm tool (`AttachedPayload`).
#[derive(Debug, Clone, PartialEq)]
pub struct CustomToolSpec {
    pub name: String,
    pub mount_offset: PoseSpec,
    pub tcp_offset_in_flange: PoseSpec,
    pub primitives: Vec<(PoseSpec, ShapePrimitiveSpec)>,
    pub mass: f64,
    pub safety_margin: Option<f64>,
    pub activation_distance: Option<f64>,
    pub touch_link_depth: usize,
}

/// Preconfigured or custom tool attachment on an arm's flange.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolPreset {
    None,
    ParallelGripper,
    Screwdriver,
    InspectionProbe,
    Custom(CustomToolSpec),
}

impl ToolPreset {
    /// Expands built-in tool presets into a normalized [`CustomToolSpec`], if a tool is attached.
    pub fn expand(&self, arm_uid: &str, default_tcp_z: f64) -> Option<CustomToolSpec> {
        match self {
            Self::None => None,
            Self::ParallelGripper => Some(CustomToolSpec {
                name: format!("{arm_uid}_gripper"),
                mount_offset: PoseSpec::translation(0.0, 0.0, 0.03),
                tcp_offset_in_flange: PoseSpec::translation(0.0, 0.0, 0.104),
                primitives: vec![
                    (
                        PoseSpec::translation(0.0, 0.0, 0.012),
                        ShapePrimitiveSpec::Cuboid {
                            half_extents: [0.018, 0.026, 0.012],
                        },
                    ),
                    (
                        PoseSpec::translation(0.0, 0.020, 0.049),
                        ShapePrimitiveSpec::Cuboid {
                            half_extents: [0.005, 0.004, 0.025],
                        },
                    ),
                    (
                        PoseSpec::translation(0.0, -0.020, 0.049),
                        ShapePrimitiveSpec::Cuboid {
                            half_extents: [0.005, 0.004, 0.025],
                        },
                    ),
                ],
                mass: 0.65,
                safety_margin: Some(0.004),
                activation_distance: Some(0.015),
                touch_link_depth: 3,
            }),
            Self::Screwdriver => Some(CustomToolSpec {
                name: format!("{arm_uid}_screwdriver"),
                mount_offset: PoseSpec::translation(0.0, 0.0, 0.03),
                tcp_offset_in_flange: PoseSpec::translation(0.0, 0.0, 0.143),
                primitives: vec![
                    (
                        PoseSpec::translation(0.0, 0.0, 0.022),
                        ShapePrimitiveSpec::CylinderZ {
                            half_height: 0.022,
                            radius: 0.015,
                        },
                    ),
                    (
                        PoseSpec::translation(0.0, 0.0, 0.076),
                        ShapePrimitiveSpec::CapsuleZ {
                            half_height: 0.032,
                            radius: 0.005,
                        },
                    ),
                ],
                mass: 0.55,
                safety_margin: Some(0.004),
                activation_distance: Some(0.015),
                touch_link_depth: 3,
            }),
            Self::InspectionProbe => Some(CustomToolSpec {
                name: format!("{arm_uid}_probe"),
                mount_offset: PoseSpec::translation(0.0, 0.0, 0.03),
                tcp_offset_in_flange: PoseSpec::translation(0.0, 0.0, 0.138),
                primitives: vec![
                    (
                        PoseSpec::translation(0.0, 0.0, 0.016),
                        ShapePrimitiveSpec::Cuboid {
                            half_extents: [0.013, 0.013, 0.016],
                        },
                    ),
                    (
                        PoseSpec::translation(0.0, 0.0, 0.068),
                        ShapePrimitiveSpec::CapsuleZ {
                            half_height: 0.036,
                            radius: 0.004,
                        },
                    ),
                ],
                mass: 0.35,
                safety_margin: Some(0.004),
                activation_distance: Some(0.015),
                touch_link_depth: 3,
            }),
            Self::Custom(spec) => {
                let mut cloned = spec.clone();
                if cloned.tcp_offset_in_flange == PoseSpec::default() {
                    cloned.tcp_offset_in_flange = PoseSpec::translation(0.0, 0.0, default_tcp_z);
                }
                Some(cloned)
            }
        }
    }
}

/// Goal / reference target assigned to an arm inside a workcell scenario.
#[derive(Debug, Clone, PartialEq)]
pub enum ArmTargetSpec {
    None,
    JointGoal(Vec<f64>),
    ApproachPose {
        final_pose: PoseSpec,
        standoff_local: [f64; 3],
    },
    DirectPose(PoseSpec),
    Twist {
        linear: [f64; 3],
        angular: [f64; 3],
    },
}

/// Declarative specification of one robot arm in the workcell.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmSpec {
    pub uid: String,
    pub label: String,
    pub urdf_source: UrdfSource,
    pub base_pose: PoseSpec,
    pub flange_link_raw: String,
    pub default_tcp_z: f64,
    pub tool: ToolPreset,
    pub color_rgba: [f32; 4],
    pub q_start: Vec<f64>,
    pub priority: f64,
    pub target: ArmTargetSpec,
}

/// Semantic role of a cell obstacle.
#[derive(Debug, Clone, PartialEq)]
pub enum ObstacleRole {
    StaticObstacle,
    Workpiece,
    Pedestal,
    DynamicSweep {
        center: [f64; 3],
        amp_xyz: [f64; 3],
        freq_hz: [f64; 3],
        phase_rad: [f64; 3],
    },
}

impl ObstacleRole {
    /// Evaluates the animated translation at trajectory phase `phase_0_to_1` if dynamic.
    pub fn evaluate_pose(&self, static_pose: &PoseSpec, phase_0_to_1: f64) -> Isometry3<f64> {
        match self {
            Self::DynamicSweep {
                center,
                amp_xyz,
                freq_hz,
                phase_rad,
            } => {
                let tau = std::f64::consts::TAU;
                let x = center[0] + amp_xyz[0] * (tau * freq_hz[0] * phase_0_to_1 + phase_rad[0]).cos();
                let y = center[1] + amp_xyz[1] * (tau * freq_hz[1] * phase_0_to_1 + phase_rad[1]).cos();
                let z = center[2] + amp_xyz[2] * (tau * freq_hz[2] * phase_0_to_1 + phase_rad[2]).sin();
                Isometry3::from_parts(
                    Translation3::new(x, y, z),
                    static_pose.to_isometry().rotation,
                )
            }
            _ => static_pose.to_isometry(),
        }
    }
}

/// Declarative specification of a workcell obstacle or fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct ObstacleSpec {
    pub uid: String,
    pub label: String,
    pub shape: ShapePrimitiveSpec,
    pub pose: PoseSpec,
    pub role: ObstacleRole,
    pub color_rgba: [f32; 4],
    pub safety_margin: Option<f64>,
    pub activation_distance: Option<f64>,
    pub carve_esdf: bool,
}

/// Stable string-based entity reference for ACM and pair-rule overrides.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntityRef {
    Link(String),
    Payload(String),
    Obstacle(String),
}

/// Allowed Collision Matrix override rule between two named links.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AcmRuleSpec {
    Enabled,
    Disabled,
    CustomMargin(f64),
}

/// Per-pair margin / activation / enable override between two named entities.
#[derive(Debug, Clone, PartialEq)]
pub struct PairRuleOverrideSpec {
    pub a: EntityRef,
    pub b: EntityRef,
    pub enabled: bool,
    pub margin: Option<f64>,
    pub activation_distance: Option<f64>,
}

/// Cell-wide default safety margins, tool margins, and per-pair overrides.
#[derive(Debug, Clone, PartialEq)]
pub struct CollisionRulesSpec {
    pub default_body_margin: f64,
    pub default_body_activation: f64,
    pub self_collision_margin: Option<f64>,
    pub tool_tool_margin: f64,
    pub tool_tool_activation: f64,
    pub tool_workpiece_margin: f64,
    pub tool_workpiece_activation: f64,
    pub tool_other_link_margin: f64,
    pub tool_other_link_activation: f64,
    pub acm_overrides: Vec<(String, String, AcmRuleSpec)>,
    pub pair_overrides: Vec<PairRuleOverrideSpec>,
}

impl Default for CollisionRulesSpec {
    fn default() -> Self {
        Self {
            default_body_margin: 0.04,
            default_body_activation: 0.16,
            self_collision_margin: None,
            tool_tool_margin: 0.004,
            tool_tool_activation: 0.015,
            tool_workpiece_margin: 0.003,
            tool_workpiece_activation: 0.015,
            tool_other_link_margin: 0.015,
            tool_other_link_activation: 0.06,
            acm_overrides: Vec::new(),
            pair_overrides: Vec::new(),
        }
    }
}

/// 3D ESDF voxel volume parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EsdfSpec {
    pub origin: [f32; 3],
    pub voxel_size: f32,
    pub dims: [usize; 3],
    pub truncation: f32,
}

impl Default for EsdfSpec {
    fn default() -> Self {
        Self {
            origin: [-0.8, -0.8, 0.0],
            voxel_size: 0.025,
            dims: [64, 64, 48],
            truncation: 1.0,
        }
    }
}

/// Coarse Stage 1 strategy in [`PlannerSpec::TwoStageProximity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage1Strategy {
    MppiWithRrtFallback,
    MppiOnly,
    RrtOnly,
}

/// Selectable motion planner / controller algorithm and hyperparameters.
#[derive(Debug, Clone)]
pub enum PlannerSpec {
    TwoStageProximity {
        strategy: Stage1Strategy,
        mppi: MppiConfig,
        rrt_step_size: f64,
        rrt_max_workspace_step: f64,
        rrt_max_iterations: usize,
        ik_num_seeds: usize,
        ik_max_iterations: usize,
        ik_tolerance: f64,
        servo: ReactiveServoConfig,
        approach_pose_gain: f64,
        max_refinement_steps: usize,
        final_pos_tol: f64,
        final_rot_tol: f64,
        waypoint_stride: usize,
    },
    ThreeArmCrossing {
        start_poses: Vec<PoseSpec>,
        ik_num_seeds: usize,
        servo: ReactiveServoConfig,
        num_steps: usize,
        record_stride: usize,
    },
    RrtConnectToppra {
        step_size: f64,
        max_workspace_step: f64,
        max_iterations: usize,
        seed: u64,
        toppra_grid_points: usize,
        num_playback_frames: usize,
    },
    BatchMppiEsdf {
        mppi: MppiConfig,
        num_visual_ribbons: usize,
        toppra_grid_points: usize,
        num_playback_frames: usize,
    },
    ClosedChainCbfServo {
        arm_a_uid: String,
        arm_b_uid: String,
        active_grasp_dims: [bool; 6],
        servo: ReactiveServoConfig,
        num_steps: usize,
        linear_twist_amp: [f64; 3],
    },
    InteractiveSandbox,
}

/// Batched rollout compute backend preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackendChoice {
    CpuSimd,
    AutoGpu,
}

/// Built-in showcase & regression presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PresetId {
    ThreeArmCloseProximity,
    ThreeArmCrossing,
    DualArmMppiEsdf,
    DualArmRrtToppra,
    DualArmClosedChainCbf,
}

impl PresetId {
    /// All built-in presets in menu presentation order.
    pub const ALL: &'static [PresetId] = &[
        PresetId::ThreeArmCloseProximity,
        PresetId::ThreeArmCrossing,
        PresetId::DualArmMppiEsdf,
        PresetId::DualArmRrtToppra,
        PresetId::DualArmClosedChainCbf,
    ];

    /// Short display title for UI selectors and headers.
    pub const fn title(&self) -> &'static str {
        match self {
            Self::ThreeArmCloseProximity => "3-Arm Close Proximity (18 DOF + 3 Tools)",
            Self::ThreeArmCrossing => "3-Arm Symmetric Crossing (Deadlock-Free CBF)",
            Self::DualArmMppiEsdf => "Dual-Arm Batched MPPI + 3D ESDF",
            Self::DualArmRrtToppra => "Dual-Arm RRT-Connect + TOPP-RA",
            Self::DualArmClosedChainCbf => "Dual-Arm Closed-Chain + CBF Servo",
        }
    }

    /// Descriptive subtitle explaining the robotics showcase.
    pub const fn subtitle(&self) -> &'static str {
        match self {
            Self::ThreeArmCloseProximity => {
                "Stage 1 MPPI/RRT -> Stage 2 Convex QP-CBF convergence with 4mm tool-tool & 3mm tool-workpiece margins"
            }
            Self::ThreeArmCrossing => {
                "Priority-weighted QP-CBF + base/velocity-aligned tangential curl across a tight 3-arm crossing zone"
            }
            Self::DualArmMppiEsdf => {
                "Parallel f32x8 rollout optimization around a 3D Euclidean Signed Distance Field + TOPP-RA"
            }
            Self::DualArmRrtToppra => {
                "Bidirectional RRT-Connect global planning + C2 spline + time-optimal path parameterization"
            }
            Self::DualArmClosedChainCbf => {
                "Cooperative bimanual closed-chain grasp stabilization with 1 kHz CBF obstacle avoidance"
            }
        }
    }
}

/// Tiered invalidation mask classifying edits made in the UI or programmatically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirtyFlags(u8);

impl DirtyFlags {
    pub const NONE: Self = Self(0);
    pub const POSE: Self = Self(1 << 0);
    pub const PLAN: Self = Self(1 << 1);
    pub const RULES: Self = Self(1 << 2);
    pub const OBSTACLES: Self = Self(1 << 3);
    pub const TOPO: Self = Self(1 << 4);

    #[inline]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline]
    pub const fn contains(self, flag: Self) -> bool {
        (self.0 & flag.0) == flag.0
    }

    #[inline]
    pub const fn needs_full_compile(self) -> bool {
        self.contains(Self::TOPO) || self.contains(Self::OBSTACLES)
    }

    #[inline]
    pub const fn needs_rules_recompile(self) -> bool {
        self.needs_full_compile() || self.contains(Self::RULES)
    }
}

/// Complete declarative workcell specification shared across UI, CLI, and batch workflows.
#[derive(Debug, Clone)]
pub struct WorkcellSpec {
    pub preset_id: Option<PresetId>,
    pub title: String,
    pub subtitle: String,
    pub arms: Vec<ArmSpec>,
    pub obstacles: Vec<ObstacleSpec>,
    pub rules: CollisionRulesSpec,
    pub esdf: EsdfSpec,
    pub planner: PlannerSpec,
    pub compute_backend: ComputeBackendChoice,
}

/// Default 8-color categorical arm palette (`[R, G, B, A]`).
pub fn arm_palette_color(index: usize) -> [f32; 4] {
    const PALETTE: [[f32; 4]; 8] = [
        [0.220, 0.741, 0.973, 1.0], // Sky #38bdf8
        [0.655, 0.545, 0.980, 1.0], // Violet #a78bfa
        [0.204, 0.827, 0.600, 1.0], // Emerald #34d399
        [0.984, 0.573, 0.235, 1.0], // Tangerine #fb923c
        [0.957, 0.447, 0.714, 1.0], // Pink #f472b6
        [0.980, 0.800, 0.082, 1.0], // Yellow #facc15
        [0.176, 0.831, 0.749, 1.0], // Teal #2dd4bf
        [0.973, 0.443, 0.443, 1.0], // Coral #f87171
    ];
    PALETTE[index % PALETTE.len()]
}
