//! Per-robot ordered trajectory, frame, and hierarchical subframe definitions
//! and world-space resolution.

use crate::error::IoError;
use crate::pose::{FileFormat, PoseSpec};
use crate::world::CompiledWorld;
use nalgebra::Isometry3;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use unrust_it::{FrameId, MultiRobotModel};

/// Current supported schema version for [`TrajectorySetSpec`] files.
pub const TRAJECTORY_SCHEMA_VERSION: u32 = 1;

fn default_trajectory_schema_version() -> u32 {
    TRAJECTORY_SCHEMA_VERSION
}

fn default_true() -> bool {
    true
}

fn default_speed_scale() -> f64 {
    1.0
}

/// Parent coordinate frame in which a [`TrajectoryFrameSpec`] pose is expressed.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReferenceFrame {
    /// Global workcell `world` frame.
    #[default]
    World,
    /// Base mounting pose of the target robot (`robot.base_pose`).
    RobotBase,
    /// Any named frame in `MultiRobotModel::frame_graph()` evaluated at `q_ref`.
    ModelFrame {
        /// Exact or robot-prefixed frame name in the model's frame graph.
        name: String,
    },
    /// World pose of a named obstacle or workpiece in `WorldScene`.
    Workpiece {
        /// Obstacle UID from `WorldSpec::obstacles`.
        obstacle_uid: String,
    },
}

/// Semantic role of a frame or subframe inside a robot trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubframeRole {
    /// Lead-in approach waypoint preceding contact/work.
    Approach,
    /// Standard pass-through or stop waypoint along the path.
    #[default]
    Waypoint,
    /// Process contact, seam, grasp, or precision dwell frame.
    Contact,
    /// Lead-out departure waypoint after contact/work.
    Retract,
    /// Fixture/sensor/inspection annotation rendered in 3D without executing IK unless overridden.
    ReferenceOnly,
}

impl SubframeRole {
    /// Default value for `execute_ik` when omitted on a frame or subframe.
    #[inline]
    pub const fn default_execute_ik(self) -> bool {
        !matches!(self, Self::ReferenceOnly)
    }

    /// Default highlight RGBA color for visualizing this role in 3D.
    #[inline]
    pub const fn default_color(self) -> [f32; 4] {
        match self {
            Self::Approach => [0.22, 0.78, 0.88, 0.95],
            Self::Waypoint => [0.32, 0.62, 0.96, 0.95],
            Self::Contact => [0.95, 0.56, 0.18, 0.98],
            Self::Retract => [0.28, 0.82, 0.54, 0.95],
            Self::ReferenceOnly => [0.74, 0.78, 0.86, 0.80],
        }
    }
}

/// Recursive child subframe attached relative to its parent frame (or parent subframe).
///
/// Children whose `time_offset_s < 0` execute before their parent frame, while children
/// with `time_offset_s >= 0` execute after their parent frame (preserving declared order
/// on equal offsets).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubframeSpec {
    /// Unique identifier within its parent frame/subframe.
    pub id: String,
    /// Optional human-readable display label (defaults to `id`).
    #[serde(default)]
    pub label: String,
    /// Semantic role (approach, waypoint, contact, retract, reference_only).
    #[serde(default)]
    pub role: SubframeRole,
    /// Signed time offset relative to parent frame timestamp in seconds.
    #[serde(default)]
    pub time_offset_s: f64,
    /// Optional explicit override whether this subframe participates in IK/execution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execute_ik: Option<bool>,
    /// Optional TCP frame name override for this subframe.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tcp_frame: Option<String>,
    /// Optional RGBA marker color override `[r, g, b, a]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_rgba: Option<[f32; 4]>,
    /// Pose relative to the parent frame/subframe.
    #[serde(default)]
    pub pose: PoseSpec,
    /// Nested child subframes relative to this subframe.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub subframes: Vec<SubframeSpec>,
}

/// Top-level frame along a robot's [`TrajectorySpec`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryFrameSpec {
    /// Unique identifier within the enclosing [`TrajectorySpec`].
    pub id: String,
    /// Optional display label (defaults to `id`).
    #[serde(default)]
    pub label: String,
    /// Semantic role (defaults to `waypoint`).
    #[serde(default)]
    pub role: SubframeRole,
    /// Optional nominal target timestamp in seconds from start of trajectory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_s: Option<f64>,
    /// Optional explicit override whether this frame participates in IK/execution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execute_ik: Option<bool>,
    /// Optional TCP frame name override for this frame.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tcp_frame: Option<String>,
    /// Optional robot joint-space IK seed for this frame.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub joint_seed: Vec<f64>,
    /// Optional RGBA marker color override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_rgba: Option<[f32; 4]>,
    /// Optional reference frame override for this frame's `pose`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_frame: Option<ReferenceFrame>,
    /// Frame pose expressed in the effective reference frame.
    #[serde(default)]
    pub pose: PoseSpec,
    /// Ordered child subframes attached relative to this frame.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub subframes: Vec<SubframeSpec>,
}

/// Ordered sequence of frames (and subframes) defining one trajectory for a robot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectorySpec {
    /// Unique trajectory ID within the parent [`RobotTrajectoriesSpec`].
    pub id: String,
    /// Human-readable trajectory name (defaults to `id`).
    #[serde(default)]
    pub label: String,
    /// Whether this trajectory is active for execution (disabled trajectories can still preview).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Positive speed scaling multiplier (`1.0` = nominal limits).
    #[serde(default = "default_speed_scale")]
    pub speed_scale: f64,
    /// Optional TCP frame override inherited by all frames in this trajectory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tcp_frame: Option<String>,
    /// Optional default reference frame inherited by frames in this trajectory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_frame: Option<ReferenceFrame>,
    /// Ordered primary frames of this trajectory.
    #[serde(default)]
    pub frames: Vec<TrajectoryFrameSpec>,
}

/// Collection of ordered trajectories assigned to one robot UID.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RobotTrajectoriesSpec {
    /// Target robot UID matching [`crate::RobotSpec::uid`].
    pub robot_uid: String,
    /// Optional planning group override (validated if specified).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_name: Option<String>,
    /// Optional default TCP frame override for this robot's trajectories.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tcp_frame: Option<String>,
    /// Optional default reference frame for this robot's trajectories.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_frame: Option<ReferenceFrame>,
    /// Ordered list of trajectories belonging to this robot.
    #[serde(default)]
    pub trajectories: Vec<TrajectorySpec>,
}

/// Root file schema describing trajectories (with frames & subframes) for all robots
/// in a workcell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectorySetSpec {
    /// Schema version (`1`).
    #[serde(default = "default_trajectory_schema_version")]
    pub schema_version: u32,
    /// Human-readable identifier for this trajectory batch.
    pub name: String,
    /// Optional notes or job description.
    #[serde(default)]
    pub description: String,
    /// Optional advisory workcell name associated with this file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub world_name: Option<String>,
    /// Default coordinate frame inherited across all robots/trajectories/frames unless overridden.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_reference_frame: Option<ReferenceFrame>,
    /// Per-robot ordered trajectory lists.
    #[serde(default)]
    pub robots: Vec<RobotTrajectoriesSpec>,
}

impl TrajectorySetSpec {
    /// Creates an empty [`TrajectorySetSpec`] with schema version 1.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            schema_version: TRAJECTORY_SCHEMA_VERSION,
            name: name.into(),
            description: String::new(),
            world_name: None,
            default_reference_frame: None,
            robots: Vec::new(),
        }
    }

    /// Parses a [`TrajectorySetSpec`] from a TOML string.
    pub fn from_toml_str(s: &str) -> Result<Self, IoError> {
        let spec: Self = toml::from_str(s)?;
        spec.validate()?;
        Ok(spec)
    }

    /// Serializes this [`TrajectorySetSpec`] to pretty TOML.
    pub fn to_toml_string(&self) -> Result<String, IoError> {
        Ok(toml::to_string_pretty(self)?)
    }

    /// Parses a [`TrajectorySetSpec`] from a JSON string.
    pub fn from_json_str(s: &str) -> Result<Self, IoError> {
        let spec: Self = serde_json::from_str(s)?;
        spec.validate()?;
        Ok(spec)
    }

    /// Serializes this [`TrajectorySetSpec`] to pretty JSON.
    pub fn to_json_string_pretty(&self) -> Result<String, IoError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Loads and validates a [`TrajectorySetSpec`] from `.toml` or `.json`.
    pub fn load_file(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let path = path.as_ref();
        let format = FileFormat::from_path(path)?;
        let content =
            std::fs::read_to_string(path).map_err(|err| IoError::io_with_path(path, err))?;
        match format {
            FileFormat::Toml => Self::from_toml_str(&content),
            FileFormat::Json => Self::from_json_str(&content),
        }
    }

    /// Validates and saves this [`TrajectorySetSpec`] to `.toml` or `.json`.
    pub fn save_file(&self, path: impl AsRef<Path>) -> Result<(), IoError> {
        self.validate()?;
        let path = path.as_ref();
        let format = FileFormat::from_path(path)?;
        let text = match format {
            FileFormat::Toml => self.to_toml_string()?,
            FileFormat::Json => self.to_json_string_pretty()?,
        };
        std::fs::write(path, text).map_err(|err| IoError::io_with_path(path, err))
    }

    /// Validates schema version, duplicate IDs, speed scales, poses, and joint seeds.
    pub fn validate(&self) -> Result<(), IoError> {
        if self.schema_version != TRAJECTORY_SCHEMA_VERSION {
            return Err(IoError::UnsupportedSchemaVersion {
                expected: TRAJECTORY_SCHEMA_VERSION,
                found: self.schema_version,
            });
        }
        if self.name.trim().is_empty() {
            return Err(IoError::Validation(
                "trajectory set name cannot be empty".into(),
            ));
        }

        let mut seen_robots = HashSet::new();
        for robot in &self.robots {
            if robot.robot_uid.trim().is_empty() {
                return Err(IoError::Validation("robot_uid cannot be empty".into()));
            }
            if !seen_robots.insert(robot.robot_uid.as_str()) {
                return Err(IoError::DuplicateId {
                    kind: "robot_trajectories",
                    id: robot.robot_uid.clone(),
                });
            }

            let mut seen_trajs = HashSet::new();
            for traj in &robot.trajectories {
                if traj.id.trim().is_empty() {
                    return Err(IoError::Validation(format!(
                        "empty trajectory id for robot `{}`",
                        robot.robot_uid
                    )));
                }
                if !seen_trajs.insert(traj.id.as_str()) {
                    return Err(IoError::DuplicateId {
                        kind: "trajectory",
                        id: format!("{}/{}", robot.robot_uid, traj.id),
                    });
                }
                if !(traj.speed_scale.is_finite() && traj.speed_scale > 0.0) {
                    return Err(IoError::Validation(format!(
                        "speed_scale must be finite and > 0 for `{}/{}`",
                        robot.robot_uid, traj.id
                    )));
                }

                let mut seen_frames = HashSet::new();
                for frame in &traj.frames {
                    if frame.id.trim().is_empty() {
                        return Err(IoError::Validation(format!(
                            "empty frame id in `{}/{}`",
                            robot.robot_uid, traj.id
                        )));
                    }
                    if !seen_frames.insert(frame.id.as_str()) {
                        return Err(IoError::DuplicateId {
                            kind: "frame",
                            id: format!("{}/{}/{}", robot.robot_uid, traj.id, frame.id),
                        });
                    }
                    if let Some(t) = frame.time_s {
                        if !t.is_finite() {
                            return Err(IoError::Validation(format!(
                                "non-finite time_s in `{}/{}/{}`",
                                robot.robot_uid, traj.id, frame.id
                            )));
                        }
                    }
                    frame
                        .pose
                        .validate(&format!("{}/{}/{}", robot.robot_uid, traj.id, frame.id))?;
                    validate_subframes_recursive(
                        &frame.subframes,
                        &format!("{}/{}/{}", robot.robot_uid, traj.id, frame.id),
                    )?;
                }
            }
        }
        Ok(())
    }
}

fn validate_subframes_recursive(subframes: &[SubframeSpec], prefix: &str) -> Result<(), IoError> {
    let mut seen = HashSet::new();
    for sf in subframes {
        if sf.id.trim().is_empty() {
            return Err(IoError::Validation(format!(
                "empty subframe id under `{prefix}`"
            )));
        }
        if !seen.insert(sf.id.as_str()) {
            return Err(IoError::DuplicateId {
                kind: "subframe",
                id: format!("{prefix}/{}", sf.id),
            });
        }
        if !sf.time_offset_s.is_finite() {
            return Err(IoError::Validation(format!(
                "non-finite time_offset_s on `{prefix}/{}`",
                sf.id
            )));
        }
        sf.pose.validate(&format!("{prefix}/{}", sf.id))?;
        validate_subframes_recursive(&sf.subframes, &format!("{prefix}/{}", sf.id))?;
    }
    Ok(())
}

/// Flattened, world-space resolved waypoint produced by [`resolve_trajectory_set`].
#[derive(Debug, Clone)]
pub struct ResolvedWaypoint {
    /// Owning robot UID.
    pub robot_uid: String,
    /// Owning trajectory ID.
    pub trajectory_id: String,
    /// Enclosing primary frame ID.
    pub primary_frame_id: String,
    /// Unique hierarchical node path (`"frame"` or `"frame/sub1/sub2"`).
    pub node_id: String,
    /// Display label.
    pub label: String,
    /// Semantic role.
    pub role: SubframeRole,
    /// Tree depth (`0` for primary frame, `>= 1` for nested subframes).
    pub depth: usize,
    /// Parent hierarchical node path (`None` when `depth == 0`).
    pub parent_node_id: Option<String>,
    /// Resolved nominal timestamp in seconds if provided on the ancestor frame.
    pub time_s: Option<f64>,
    /// Whether this node participates in IK/motion execution (`false` for `ReferenceOnly` by default).
    pub execute_ik: bool,
    /// Resolved target TCP frame ID in `MultiRobotModel::frame_graph()`.
    pub tcp_frame_id: FrameId,
    /// Resolved target TCP frame name.
    pub tcp_frame_name: String,
    /// World-space target pose `world_T_target`.
    pub world_pose: Isometry3<f64>,
    /// Local pose relative to the direct parent (`reference_frame` or parent subframe).
    pub local_pose: Isometry3<f64>,
    /// World pose of the direct parent coordinate frame (for rendering hierarchy connectors).
    pub parent_world_pose: Isometry3<f64>,
    /// Optional per-robot joint seed specified on a primary frame.
    pub joint_seed: Option<Vec<f64>>,
    /// RGBA visualization tint.
    pub color_rgba: [f32; 4],
}

/// Resolved trajectory containing flattened ordered waypoints in world space.
#[derive(Debug, Clone)]
pub struct ResolvedTrajectory {
    /// Owning robot UID.
    pub robot_uid: String,
    /// Trajectory ID.
    pub id: String,
    /// Display label.
    pub label: String,
    /// Whether this trajectory is enabled for planning/execution.
    pub enabled: bool,
    /// Positive speed scaling multiplier.
    pub speed_scale: f64,
    /// Default TCP frame ID for this trajectory.
    pub tcp_frame_id: FrameId,
    /// Default TCP frame name for this trajectory.
    pub tcp_frame_name: String,
    /// Expanded waypoints ordered cleanly with pre-parent (`time_offset_s < 0`) and
    /// post-parent (`time_offset_s >= 0`) subframes.
    pub waypoints: Vec<ResolvedWaypoint>,
}

/// Resolved per-robot bundle of trajectories.
#[derive(Debug, Clone)]
pub struct ResolvedRobotTrajectories {
    /// Target robot UID.
    pub robot_uid: String,
    /// Target robot group name.
    pub group_name: String,
    /// Resolved trajectories in file order.
    pub trajectories: Vec<ResolvedTrajectory>,
}

/// Fully validated and world-resolved trajectory set ready for IK/motion planning and 3D rendering.
#[derive(Debug, Clone)]
pub struct ResolvedTrajectorySet {
    /// Name of the imported trajectory set.
    pub name: String,
    /// Per-robot resolved trajectories.
    pub robots: Vec<ResolvedRobotTrajectories>,
}

impl ResolvedTrajectorySet {
    /// Returns all resolved waypoints across all robots (including reference-only frames for 3D preview).
    pub fn all_waypoints(&self) -> impl Iterator<Item = &ResolvedWaypoint> {
        self.robots
            .iter()
            .flat_map(|r| r.trajectories.iter())
            .flat_map(|t| t.waypoints.iter())
    }

    /// Returns only executable (`execute_ik == true`) waypoints for a given robot UID across enabled trajectories.
    pub fn executable_waypoints_for_robot<'a>(
        &'a self,
        robot_uid: &'a str,
    ) -> impl Iterator<Item = &'a ResolvedWaypoint> + 'a {
        self.robots
            .iter()
            .filter(move |r| r.robot_uid == robot_uid)
            .flat_map(|r| r.trajectories.iter())
            .filter(|t| t.enabled)
            .flat_map(|t| t.waypoints.iter())
            .filter(|w| w.execute_ik)
    }
}

/// Lightweight per-robot descriptor used by [`resolve_trajectory_set_with_model`].
#[derive(Debug, Clone)]
pub struct RobotResolveInfo {
    /// Stable robot UID.
    pub uid: String,
    /// Link/joint/frame namespace prefix applied to this robot.
    pub prefix: String,
    /// Number of actuated DOFs belonging to this robot.
    pub dof: usize,
    /// Default planning group name.
    pub default_group_name: String,
    /// Robot base mounting pose in world coordinates.
    pub base_pose: Isometry3<f64>,
    /// Default TCP frame ID.
    pub tcp_frame_id: FrameId,
    /// Default TCP frame name.
    pub tcp_frame_name: String,
    /// Optional SRDF planning group names registered for this robot.
    pub srdf_group_names: Vec<String>,
}

/// Resolves every frame and recursive subframe in `set` against `world` at configuration `q_ref`,
/// validating robot UIDs, TCP frames, model reference frames, workpiece UIDs, and joint seeds.
pub fn resolve_trajectory_set(
    set: &TrajectorySetSpec,
    world: &CompiledWorld,
    q_ref: &[f64],
) -> Result<ResolvedTrajectorySet, IoError> {
    let robots: Vec<RobotResolveInfo> = world
        .robots
        .iter()
        .map(|r| RobotResolveInfo {
            uid: r.uid.clone(),
            prefix: r.prefix.clone(),
            dof: r.handle.q_indices.len(),
            default_group_name: r.planning_group.name.clone(),
            base_pose: r.base_pose,
            tcp_frame_id: r.tcp_frame_id,
            tcp_frame_name: r.tcp_frame_name.clone(),
            srdf_group_names: r
                .srdf
                .as_ref()
                .map(|s| s.groups.iter().map(|g| g.name.clone()).collect())
                .unwrap_or_default(),
        })
        .collect();

    let mut obstacle_poses: HashMap<String, Isometry3<f64>> = world
        .obstacles
        .iter()
        .map(|o| (o.uid.clone(), o.base_pose))
        .collect();
    for (name, _, obs) in world.scene.obstacles() {
        obstacle_poses.entry(name.to_string()).or_insert(obs.pose);
    }

    resolve_trajectory_set_with_model(set, world.scene.model(), &robots, &obstacle_poses, q_ref)
}

/// Resolves every frame and recursive subframe in `set` against an existing
/// [`MultiRobotModel`], robot table, and obstacle pose map at configuration `q_ref`.
pub fn resolve_trajectory_set_with_model(
    set: &TrajectorySetSpec,
    model: &MultiRobotModel,
    robots: &[RobotResolveInfo],
    obstacle_poses: &HashMap<String, Isometry3<f64>>,
    q_ref: &[f64],
) -> Result<ResolvedTrajectorySet, IoError> {
    set.validate()?;
    let expected_full_dof = model.dof();
    if q_ref.len() != expected_full_dof {
        return Err(IoError::Validation(format!(
            "q_ref length mismatch: expected {expected_full_dof}, got {}",
            q_ref.len()
        )));
    }
    let mut tf_buf = model.create_transform_buffer();
    model.forward_kinematics(q_ref, &mut tf_buf)?;

    let robot_map: HashMap<&str, &RobotResolveInfo> =
        robots.iter().map(|r| (r.uid.as_str(), r)).collect();

    let mut resolved_robots = Vec::with_capacity(set.robots.len());

    for r_spec in &set.robots {
        let robot_info = robot_map
            .get(r_spec.robot_uid.as_str())
            .copied()
            .ok_or_else(|| IoError::UnknownRobot(r_spec.robot_uid.clone()))?;

        let group_name = if let Some(ref gname) = r_spec.group_name {
            let prefixed = format!("{}{gname}", robot_info.prefix);
            let exists = model.group_by_name(gname).is_some()
                || model.group_by_name(&prefixed).is_some()
                || robot_info
                    .srdf_group_names
                    .iter()
                    .any(|g| g == gname || g == &prefixed);
            if !exists {
                return Err(IoError::UnknownGroup {
                    robot_uid: r_spec.robot_uid.clone(),
                    group: gname.clone(),
                });
            }
            gname.clone()
        } else {
            robot_info.default_group_name.clone()
        };

        let robot_tcp = resolve_tcp_frame(
            model,
            &robot_info.uid,
            &robot_info.prefix,
            r_spec.tcp_frame.as_deref(),
            robot_info.tcp_frame_id,
            &robot_info.tcp_frame_name,
        )?;
        let robot_ref = r_spec
            .reference_frame
            .as_ref()
            .or(set.default_reference_frame.as_ref())
            .cloned()
            .unwrap_or_default();

        let mut resolved_trajs = Vec::with_capacity(r_spec.trajectories.len());
        let robot_dof = robot_info.dof;

        for traj in &r_spec.trajectories {
            let traj_tcp = resolve_tcp_frame(
                model,
                &robot_info.uid,
                &robot_info.prefix,
                traj.tcp_frame.as_deref(),
                robot_tcp.0,
                &robot_tcp.1,
            )?;
            let traj_ref = traj
                .reference_frame
                .as_ref()
                .unwrap_or(&robot_ref)
                .clone();

            let mut waypoints = Vec::new();
            for frame in &traj.frames {
                if !frame.joint_seed.is_empty() && frame.joint_seed.len() != robot_dof {
                    return Err(IoError::JointCountMismatch {
                        robot_uid: robot_info.uid.clone(),
                        expected: robot_dof,
                        got: frame.joint_seed.len(),
                    });
                }

                let frame_tcp = resolve_tcp_frame(
                    model,
                    &robot_info.uid,
                    &robot_info.prefix,
                    frame.tcp_frame.as_deref(),
                    traj_tcp.0,
                    &traj_tcp.1,
                )?;
                let eff_ref = frame.reference_frame.as_ref().unwrap_or(&traj_ref);
                let ref_world_pose = resolve_reference_pose(
                    model,
                    obstacle_poses,
                    robot_info.uid.as_str(),
                    &robot_info.prefix,
                    robot_info.base_pose,
                    &tf_buf,
                    eff_ref,
                )?;

                let local_pose = frame.pose.to_isometry();
                let world_pose = ref_world_pose * local_pose;

                let self_wp = ResolvedWaypoint {
                    robot_uid: robot_info.uid.clone(),
                    trajectory_id: traj.id.clone(),
                    primary_frame_id: frame.id.clone(),
                    node_id: frame.id.clone(),
                    label: if frame.label.trim().is_empty() {
                        frame.id.clone()
                    } else {
                        frame.label.clone()
                    },
                    role: frame.role,
                    depth: 0,
                    parent_node_id: None,
                    time_s: frame.time_s,
                    execute_ik: frame
                        .execute_ik
                        .unwrap_or_else(|| frame.role.default_execute_ik()),
                    tcp_frame_id: frame_tcp.0,
                    tcp_frame_name: frame_tcp.1.clone(),
                    world_pose,
                    local_pose,
                    parent_world_pose: ref_world_pose,
                    joint_seed: if frame.joint_seed.is_empty() {
                        None
                    } else {
                        Some(frame.joint_seed.clone())
                    },
                    color_rgba: frame
                        .color_rgba
                        .unwrap_or_else(|| frame.role.default_color()),
                };

                expand_node_and_subframes(
                    model,
                    &robot_info.uid,
                    &robot_info.prefix,
                    &traj.id,
                    &frame.id,
                    self_wp,
                    &frame.subframes,
                    &mut waypoints,
                )?;
            }

            resolved_trajs.push(ResolvedTrajectory {
                robot_uid: robot_info.uid.clone(),
                id: traj.id.clone(),
                label: if traj.label.trim().is_empty() {
                    traj.id.clone()
                } else {
                    traj.label.clone()
                },
                enabled: traj.enabled,
                speed_scale: traj.speed_scale,
                tcp_frame_id: traj_tcp.0,
                tcp_frame_name: traj_tcp.1,
                waypoints,
            });
        }

        resolved_robots.push(ResolvedRobotTrajectories {
            robot_uid: robot_info.uid.clone(),
            group_name,
            trajectories: resolved_trajs,
        });
    }

    Ok(ResolvedTrajectorySet {
        name: set.name.clone(),
        robots: resolved_robots,
    })
}

#[allow(clippy::too_many_arguments)]
fn expand_node_and_subframes(
    model: &MultiRobotModel,
    robot_uid: &str,
    prefix: &str,
    trajectory_id: &str,
    primary_frame_id: &str,
    self_wp: ResolvedWaypoint,
    subframes: &[SubframeSpec],
    out: &mut Vec<ResolvedWaypoint>,
) -> Result<(), IoError> {
    if subframes.is_empty() {
        out.push(self_wp);
        return Ok(());
    }

    let mut pre: Vec<&SubframeSpec> = subframes
        .iter()
        .filter(|s| s.time_offset_s < 0.0)
        .collect();
    let mut post: Vec<&SubframeSpec> = subframes
        .iter()
        .filter(|s| s.time_offset_s >= 0.0)
        .collect();
    pre.sort_by(|a, b| a.time_offset_s.total_cmp(&b.time_offset_s));
    post.sort_by(|a, b| a.time_offset_s.total_cmp(&b.time_offset_s));

    let parent_world = self_wp.world_pose;
    let parent_node = self_wp.node_id.clone();
    let parent_depth = self_wp.depth;
    let parent_time = self_wp.time_s;
    let parent_tcp = (self_wp.tcp_frame_id, self_wp.tcp_frame_name.clone());

    for sf in pre {
        expand_subframe(
            model,
            robot_uid,
            prefix,
            trajectory_id,
            primary_frame_id,
            &parent_node,
            parent_depth,
            parent_world,
            parent_time,
            &parent_tcp,
            sf,
            out,
        )?;
    }

    out.push(self_wp);

    for sf in post {
        expand_subframe(
            model,
            robot_uid,
            prefix,
            trajectory_id,
            primary_frame_id,
            &parent_node,
            parent_depth,
            parent_world,
            parent_time,
            &parent_tcp,
            sf,
            out,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn expand_subframe(
    model: &MultiRobotModel,
    robot_uid: &str,
    prefix: &str,
    trajectory_id: &str,
    primary_frame_id: &str,
    parent_node: &str,
    parent_depth: usize,
    parent_world: Isometry3<f64>,
    parent_time: Option<f64>,
    parent_tcp: &(FrameId, String),
    sf: &SubframeSpec,
    out: &mut Vec<ResolvedWaypoint>,
) -> Result<(), IoError> {
    let tcp = resolve_tcp_frame(
        model,
        robot_uid,
        prefix,
        sf.tcp_frame.as_deref(),
        parent_tcp.0,
        &parent_tcp.1,
    )?;
    let local_pose = sf.pose.to_isometry();
    let world_pose = parent_world * local_pose;
    let node_id = format!("{parent_node}/{}", sf.id);

    let wp = ResolvedWaypoint {
        robot_uid: robot_uid.to_string(),
        trajectory_id: trajectory_id.to_string(),
        primary_frame_id: primary_frame_id.to_string(),
        node_id,
        label: if sf.label.trim().is_empty() {
            sf.id.clone()
        } else {
            sf.label.clone()
        },
        role: sf.role,
        depth: parent_depth + 1,
        parent_node_id: Some(parent_node.to_string()),
        time_s: parent_time.map(|t| t + sf.time_offset_s),
        execute_ik: sf.execute_ik.unwrap_or_else(|| sf.role.default_execute_ik()),
        tcp_frame_id: tcp.0,
        tcp_frame_name: tcp.1,
        world_pose,
        local_pose,
        parent_world_pose: parent_world,
        joint_seed: None,
        color_rgba: sf.color_rgba.unwrap_or_else(|| sf.role.default_color()),
    };

    expand_node_and_subframes(
        model,
        robot_uid,
        prefix,
        trajectory_id,
        primary_frame_id,
        wp,
        &sf.subframes,
        out,
    )
}

fn resolve_tcp_frame(
    model: &MultiRobotModel,
    robot_uid: &str,
    prefix: &str,
    override_name: Option<&str>,
    fallback_id: FrameId,
    fallback_name: &str,
) -> Result<(FrameId, String), IoError> {
    let Some(name) = override_name else {
        return Ok((fallback_id, fallback_name.to_string()));
    };
    let graph = model.frame_graph();
    let prefixed = format!("{prefix}{name}");
    if let Some(fid) = graph.find_by_name(name) {
        return Ok((fid, name.to_string()));
    }
    if let Some(fid) = graph.find_by_name(&prefixed) {
        return Ok((fid, prefixed));
    }
    Err(IoError::UnknownFrame {
        robot_uid: robot_uid.to_string(),
        frame: name.to_string(),
    })
}

fn resolve_reference_pose(
    model: &MultiRobotModel,
    obstacle_poses: &HashMap<String, Isometry3<f64>>,
    robot_uid: &str,
    prefix: &str,
    robot_base_pose: Isometry3<f64>,
    tf_buf: &unrust_it::TransformBuffer,
    reference: &ReferenceFrame,
) -> Result<Isometry3<f64>, IoError> {
    match reference {
        ReferenceFrame::World => Ok(Isometry3::identity()),
        ReferenceFrame::RobotBase => Ok(robot_base_pose),
        ReferenceFrame::ModelFrame { name } => {
            let graph = model.frame_graph();
            let prefixed = format!("{prefix}{name}");
            let fid = graph
                .find_by_name(name)
                .or_else(|| graph.find_by_name(&prefixed))
                .ok_or_else(|| IoError::UnknownFrame {
                    robot_uid: robot_uid.to_string(),
                    frame: name.clone(),
                })?;
            Ok(*tf_buf.frame_pose(fid))
        }
        ReferenceFrame::Workpiece { obstacle_uid } => obstacle_poses
            .get(obstacle_uid)
            .copied()
            .ok_or_else(|| IoError::UnknownObstacle(obstacle_uid.clone())),
    }
}
