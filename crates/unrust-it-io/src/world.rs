//! Declarative workcell world definitions (`WorldSpec`) and compilation into
//! an authoritative [`unrust_it::WorldScene`].

use crate::error::IoError;
use crate::pose::{FileFormat, PoseSpec, ShapeSpec};
use nalgebra::Isometry3;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::path::{Path, PathBuf};
use unrust_it::{
    AllowedCollisionMatrix, AttachedPayload, CollisionEnvironment, CollisionRule, Configuration,
    EsdfConfig, FlatRobotModelF32, FrameId, LinkGeometry, LinkId, MultiRobotModel, ObstacleHandle,
    ObstacleId, ObstacleOptions, PlanningGroup, RobotId, WorldScene,
};
use unrust_it_urdf::presets::{ARM_4DOF_SRDF, ARM_4DOF_URDF, ARM_6DOF_SRDF, ARM_6DOF_URDF};
use unrust_it_urdf::{add_urdf_to_scene, apply_srdf_str, SrdfData, UrdfRobotHandle};

/// Current supported schema version for [`WorldSpec`] files.
pub const WORLD_SCHEMA_VERSION: u32 = 1;

fn default_world_schema_version() -> u32 {
    WORLD_SCHEMA_VERSION
}

fn default_true() -> bool {
    true
}

fn default_body_margin() -> f64 {
    0.012
}

fn default_body_activation() -> f64 {
    0.16
}

fn default_tool_tool_margin() -> f64 {
    0.008
}

fn default_tool_tool_activation() -> f64 {
    0.10
}

fn default_tool_workpiece_margin() -> f64 {
    0.005
}

fn default_tool_workpiece_activation() -> f64 {
    0.06
}

fn default_tool_other_link_margin() -> f64 {
    0.010
}

fn default_tool_other_link_activation() -> f64 {
    0.14
}

fn default_payload_mass() -> f64 {
    0.5
}

fn default_esdf_voxel_size() -> f32 {
    0.025
}

fn default_esdf_dims() -> [usize; 3] {
    [80, 80, 64]
}

fn default_esdf_origin() -> [f32; 3] {
    [-1.0, -1.0, 0.0]
}

fn default_esdf_truncation() -> f32 {
    1.0
}

fn default_dt() -> f64 {
    0.015
}

fn default_max_speed_scale() -> f64 {
    0.85
}

fn default_safety_margin() -> f64 {
    0.012
}

fn default_cbf_gamma() -> f64 {
    5.0
}

fn default_tan_gain() -> f64 {
    0.25
}

fn default_mppi_samples() -> usize {
    128
}

fn default_mppi_horizon() -> usize {
    20
}

fn default_mppi_iters() -> usize {
    3
}

fn default_mppi_lambda() -> f64 {
    1.0
}

fn default_two_stage_threshold() -> f64 {
    0.14
}

fn default_rrt_step_size() -> f64 {
    0.20
}

fn default_rrt_max_iters() -> usize {
    1800
}

fn default_toppra_dt() -> f64 {
    0.01
}

/// Complete declarative description of a multi-robot robotic workcell.
///
/// Supports round-trip serialization in human-editable **TOML** and structured
/// **JSON**, and compiles directly into an authoritative [`WorldScene`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldSpec {
    /// File schema version (`1`).
    #[serde(default = "default_world_schema_version")]
    pub schema_version: u32,
    /// Human-readable workcell identifier.
    pub name: String,
    /// Optional notes describing the workcell or setup.
    #[serde(default)]
    pub description: String,
    /// Optional volumetric ESDF grid configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub esdf: Option<EsdfConfigSpec>,
    /// Global & per-pair collision margin/activation rules.
    #[serde(default)]
    pub collision_rules: CollisionRulesSpec,
    /// Optional planner/controller parameter defaults persisted alongside the world.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planner: Option<PlannerConfigSpec>,
    /// Ordered list of robots mounted into the workcell.
    #[serde(default)]
    pub robots: Vec<RobotSpec>,
    /// Static fixtures, workpieces, pedestals, and obstacles in the workcell.
    #[serde(default)]
    pub obstacles: Vec<ObstacleSpec>,
    /// Optional base directory used to resolve relative URDF/SRDF paths when loading from disk.
    #[serde(skip)]
    base_dir: Option<PathBuf>,
}

impl PartialEq for WorldSpec {
    fn eq(&self, other: &Self) -> bool {
        self.schema_version == other.schema_version
            && self.name == other.name
            && self.description == other.description
            && self.esdf == other.esdf
            && self.collision_rules == other.collision_rules
            && self.planner == other.planner
            && self.robots == other.robots
            && self.obstacles == other.obstacles
    }
}

impl WorldSpec {
    /// Creates an empty [`WorldSpec`] with schema version 1 and default collision rules.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            schema_version: WORLD_SCHEMA_VERSION,
            name: name.into(),
            description: String::new(),
            esdf: Some(EsdfConfigSpec::default()),
            collision_rules: CollisionRulesSpec::default(),
            planner: None,
            robots: Vec::new(),
            obstacles: Vec::new(),
            base_dir: None,
        }
    }

    /// Associates a base directory used when resolving relative URDF/SRDF file paths.
    pub fn with_base_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.base_dir = Some(dir.into());
        self
    }

    /// Returns the base directory associated with this world spec, if any.
    pub fn base_dir(&self) -> Option<&Path> {
        self.base_dir.as_deref()
    }

    /// Parses a [`WorldSpec`] from a TOML string.
    pub fn from_toml_str(s: &str) -> Result<Self, IoError> {
        let spec: Self = toml::from_str(s)?;
        spec.validate()?;
        Ok(spec)
    }

    /// Serializes this [`WorldSpec`] as pretty TOML.
    pub fn to_toml_string(&self) -> Result<String, IoError> {
        Ok(toml::to_string_pretty(self)?)
    }

    /// Parses a [`WorldSpec`] from a JSON string.
    pub fn from_json_str(s: &str) -> Result<Self, IoError> {
        let spec: Self = serde_json::from_str(s)?;
        spec.validate()?;
        Ok(spec)
    }

    /// Serializes this [`WorldSpec`] as pretty-printed JSON.
    pub fn to_json_string_pretty(&self) -> Result<String, IoError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Loads and validates a [`WorldSpec`] from `.toml` or `.json`, recording its parent
    /// directory for resolving relative URDF/SRDF file references.
    pub fn load_file(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let path = path.as_ref();
        let format = FileFormat::from_path(path)?;
        let content =
            std::fs::read_to_string(path).map_err(|err| IoError::io_with_path(path, err))?;
        let mut spec = match format {
            FileFormat::Toml => Self::from_toml_str(&content)?,
            FileFormat::Json => Self::from_json_str(&content)?,
        };
        let parent = path.parent().map(|p| {
            if p.as_os_str().is_empty() {
                PathBuf::from(".")
            } else {
                p.to_path_buf()
            }
        });
        spec.base_dir = parent;
        Ok(spec)
    }

    /// Saves this [`WorldSpec`] to `.toml` or `.json` inferred from the destination extension.
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

    /// Validates schema version, UID uniqueness, poses, and shape dimensions.
    pub fn validate(&self) -> Result<(), IoError> {
        if self.schema_version != WORLD_SCHEMA_VERSION {
            return Err(IoError::UnsupportedSchemaVersion {
                expected: WORLD_SCHEMA_VERSION,
                found: self.schema_version,
            });
        }
        if self.name.trim().is_empty() {
            return Err(IoError::Validation("world name cannot be empty".into()));
        }

        let mut robot_uids = HashSet::new();
        for robot in &self.robots {
            if robot.uid.trim().is_empty() {
                return Err(IoError::Validation("robot uid cannot be empty".into()));
            }
            if !robot_uids.insert(robot.uid.as_str()) {
                return Err(IoError::DuplicateId {
                    kind: "robot",
                    id: robot.uid.clone(),
                });
            }
            robot.base_pose.validate(&format!("robot `{}`", robot.uid))?;
            if let Some(ref offset) = robot.tcp_offset {
                offset.validate(&format!("robot `{}` tcp_offset", robot.uid))?;
            }
            if let Some(ref tool) = robot.tool_payload {
                tool.local_pose
                    .validate(&format!("robot `{}` tool `{}`", robot.uid, tool.name))?;
                tool.shape.validate()?;
            }
        }

        let mut obs_uids = HashSet::new();
        for obs in &self.obstacles {
            if obs.uid.trim().is_empty() {
                return Err(IoError::Validation("obstacle uid cannot be empty".into()));
            }
            if !obs_uids.insert(obs.uid.as_str()) {
                return Err(IoError::DuplicateId {
                    kind: "obstacle",
                    id: obs.uid.clone(),
                });
            }
            obs.pose.validate(&format!("obstacle `{}`", obs.uid))?;
            obs.shape.validate()?;
        }

        Ok(())
    }

    /// Compiles this [`WorldSpec`] into a live [`CompiledWorld`] backed by [`WorldScene`],
    /// resolving any relative URDF/SRDF paths against `self.base_dir`.
    pub fn compile(&self) -> Result<CompiledWorld, IoError> {
        self.compile_with_base_dir(self.base_dir.as_deref())
    }

    /// Compiles this [`WorldSpec`] resolving relative URDF/SRDF file paths against `base_dir`.
    pub fn compile_with_base_dir(&self, base_dir: Option<&Path>) -> Result<CompiledWorld, IoError> {
        self.validate()?;

        let mut model = MultiRobotModel::new();
        let mut env = CollisionEnvironment::new();

        struct PendingRobotMeta {
            uid: String,
            label: String,
            robot_id: RobotId,
            prefix: String,
            handle: UrdfRobotHandle,
            srdf: Option<SrdfData>,
            planning_group: PlanningGroup,
            q_start_spec: Vec<f64>,
            base_pose: Isometry3<f64>,
            tcp_frame_id: FrameId,
            tcp_frame_name: String,
            payload_name: Option<String>,
            color_rgba: [f32; 4],
            priority: u8,
        }

        let mut pending = Vec::with_capacity(self.robots.len());

        for (idx, robot) in self.robots.iter().enumerate() {
            let robot_id = RobotId(idx as u16);
            let (urdf_xml, srdf_xml) = robot.source.load_xml(base_dir)?;
            let base_iso = robot.base_pose.to_isometry();
            let handle = add_urdf_to_scene(
                &mut model,
                &mut env,
                &urdf_xml,
                robot_id,
                &robot.prefix,
                base_iso,
            )?;

            let srdf = match srdf_xml {
                Some(ref xml) => Some(apply_srdf_str(&mut model, &mut env, xml, &robot.prefix)?),
                None => None,
            };

            let mut planning_group = if let Some(ref group_name) = robot.active_group {
                let prefixed = format!("{}{group_name}", robot.prefix);
                srdf.as_ref()
                    .and_then(|s| {
                        s.groups
                            .iter()
                            .find(|g| g.name == *group_name || g.name == prefixed)
                            .cloned()
                    })
                    .or_else(|| model.group_by_name(group_name).cloned())
                    .or_else(|| model.group_by_name(&prefixed).cloned())
                    .ok_or_else(|| IoError::UnknownGroup {
                        robot_uid: robot.uid.clone(),
                        group: group_name.clone(),
                    })?
            } else {
                handle.planning_group.clone()
            };

            // Determine parent frame for TCP (either explicit `tcp_parent_frame`, planning group's
            // default end effector, or fallback last link of this robot).
            let default_parent_frame = planning_group
                .end_effectors
                .first()
                .copied()
                .or_else(|| handle.planning_group.end_effectors.first().copied())
                .or_else(|| handle.links.last().map(|&lid| model.link(lid).frame))
                .ok_or_else(|| {
                    IoError::Validation(format!("robot `{}` has no links", robot.uid))
                })?;

            let parent_frame_id = if let Some(ref pname) = robot.tcp_parent_frame {
                let prefixed = format!("{}{pname}", robot.prefix);
                model
                    .frame_graph()
                    .find_by_name(pname)
                    .or_else(|| model.frame_graph().find_by_name(&prefixed))
                    .ok_or_else(|| IoError::UnknownFrame {
                        robot_uid: robot.uid.clone(),
                        frame: pname.clone(),
                    })?
            } else {
                default_parent_frame
            };

            let (tcp_frame_id, tcp_frame_name) =
                if robot.tcp_offset.is_some() || robot.tcp_frame_name.is_some() {
                    let tcp_name = robot
                        .tcp_frame_name
                        .clone()
                        .unwrap_or_else(|| format!("{}_tcp", robot.uid));
                    let offset = robot
                        .tcp_offset
                        .map(|p| p.to_isometry())
                        .unwrap_or_else(Isometry3::identity);
                    let fid = model.add_fixed_frame(tcp_name.clone(), parent_frame_id, offset)?;
                    planning_group.end_effectors = vec![fid];
                    model.add_planning_group(planning_group.clone());
                    (fid, tcp_name)
                } else {
                    let name = model
                        .frame_graph()
                        .node(parent_frame_id)
                        .map(|n| n.name.clone())
                        .unwrap_or_else(|| format!("{}_ee", robot.uid));
                    (parent_frame_id, name)
                };

            let payload_name = if let Some(ref tool) = robot.tool_payload {
                let parent_link = if let Some(ref plink) = tool.parent_link {
                    let prefixed = format!("{}{plink}", robot.prefix);
                    model
                        .link_by_name(plink)
                        .or_else(|| model.link_by_name(&prefixed))
                        .ok_or_else(|| IoError::UnknownLink(plink.clone()))?
                } else {
                    *handle.links.last().ok_or_else(|| {
                        IoError::Validation(format!("robot `{}` has no links", robot.uid))
                    })?
                };

                let (shape_pose, shape) = tool.shape.to_parry_shape_and_pose()?;
                let local_pose = tool.local_pose.to_isometry() * shape_pose;

                let mut touch_links = Vec::with_capacity(tool.touch_links.len() + 2);
                if tool.touch_links.is_empty() {
                    touch_links.push(parent_link);
                    if let Some(prev) = handle
                        .links
                        .len()
                        .checked_sub(2)
                        .and_then(|i| handle.links.get(i))
                    {
                        touch_links.push(*prev);
                    }
                } else {
                    for tlink in &tool.touch_links {
                        let prefixed = format!("{}{tlink}", robot.prefix);
                        let lid = model
                            .link_by_name(tlink)
                            .or_else(|| model.link_by_name(&prefixed))
                            .ok_or_else(|| IoError::UnknownLink(tlink.clone()))?;
                        touch_links.push(lid);
                    }
                }

                env.attach_payload(AttachedPayload {
                    id: tool.name.clone(),
                    parent_link,
                    local_pose,
                    geometry: LinkGeometry::from_shape(Isometry3::identity(), shape),
                    mass: tool.mass,
                    touch_links,
                    safety_margin: tool.safety_margin,
                    activation_distance: tool.activation_distance,
                });
                Some(tool.name.clone())
            } else {
                None
            };

            let label = if robot.label.trim().is_empty() {
                robot.uid.clone()
            } else {
                robot.label.clone()
            };
            let color_rgba = robot
                .color_rgba
                .unwrap_or_else(|| default_robot_palette(idx));

            pending.push(PendingRobotMeta {
                uid: robot.uid.clone(),
                label,
                robot_id,
                prefix: robot.prefix.clone(),
                handle,
                srdf,
                planning_group,
                q_start_spec: robot.q_start.clone(),
                base_pose: base_iso,
                tcp_frame_id,
                tcp_frame_name,
                payload_name,
                color_rgba,
                priority: robot.priority,
            });
        }

        // Build full-system `q_start` once all robots have been added and DOF count is fixed.
        let mut q_start_raw = vec![0.0_f64; model.dof()];
        let dof_limits = model.dof_limits();
        for (idx, &lim) in dof_limits.iter().enumerate() {
            q_start_raw[idx] = 0.0_f64.clamp(lim.position.min, lim.position.max);
        }

        let mut compiled_robots = Vec::with_capacity(pending.len());
        let mut robot_by_uid = HashMap::with_capacity(pending.len());

        for (idx, meta) in pending.into_iter().enumerate() {
            let expected = meta.handle.q_indices.len();
            let mut local_q_start = Vec::with_capacity(expected);
            if meta.q_start_spec.is_empty() {
                for &g_idx in &meta.handle.q_indices {
                    local_q_start.push(q_start_raw[g_idx]);
                }
            } else {
                if meta.q_start_spec.len() != expected {
                    return Err(IoError::JointCountMismatch {
                        robot_uid: meta.uid,
                        expected,
                        got: meta.q_start_spec.len(),
                    });
                }
                for (&g_idx, &val) in meta.handle.q_indices.iter().zip(&meta.q_start_spec) {
                    q_start_raw[g_idx] = val;
                    local_q_start.push(val);
                }
            }

            let q_range = if let (Some(&first), Some(&last)) =
                (meta.handle.q_indices.first(), meta.handle.q_indices.last())
            {
                first..(last + 1)
            } else {
                0..0
            };

            robot_by_uid.insert(meta.uid.clone(), idx);
            compiled_robots.push(CompiledRobot {
                uid: meta.uid,
                label: meta.label,
                robot_id: meta.robot_id,
                prefix: meta.prefix,
                handle: meta.handle,
                srdf: meta.srdf,
                planning_group: meta.planning_group,
                q_range,
                q_start: local_q_start,
                base_pose: meta.base_pose,
                tcp_frame_id: meta.tcp_frame_id,
                tcp_frame_name: meta.tcp_frame_name,
                payload_name: meta.payload_name,
                color_rgba: meta.color_rgba,
                priority: meta.priority,
            });
        }

        // Configure ACM and non-obstacle link/payload collision rules before wrapping in `WorldScene`.
        let mut acm = env
            .acm()
            .cloned()
            .unwrap_or_else(|| AllowedCollisionMatrix::from_model(&model));
        acm.set_inter_arm_margin(&model, self.collision_rules.default_body_margin);
        if let Some(self_margin) = self.collision_rules.self_collision_margin {
            acm.set_self_collision_margin(&model, self_margin);
        }
        for ov in &self.collision_rules.acm_overrides {
            let la = model
                .link_by_name(&ov.link_a)
                .ok_or_else(|| IoError::UnknownLink(ov.link_a.clone()))?;
            let lb = model
                .link_by_name(&ov.link_b)
                .ok_or_else(|| IoError::UnknownLink(ov.link_b.clone()))?;
            acm.set_rule(la, lb, ov.rule.into());
        }
        env.set_acm(acm);

        for a in 0..model.num_links() {
            for b in (a + 1)..model.num_links() {
                let la = LinkId(a as u16);
                let lb = LinkId(b as u16);
                if model.link(la).robot != model.link(lb).robot {
                    env.set_link_pair_activation(
                        la,
                        lb,
                        self.collision_rules.default_body_activation,
                    );
                }
            }
        }

        for a in 0..compiled_robots.len() {
            let Some(ref pa) = compiled_robots[a].payload_name else {
                continue;
            };
            for rb in compiled_robots.iter().skip(a + 1) {
                if let Some(ref pb) = rb.payload_name {
                    env.set_payload_pair_margin(pa, pb, self.collision_rules.tool_tool_margin);
                    env.set_payload_pair_activation(
                        pa,
                        pb,
                        self.collision_rules.tool_tool_activation,
                    );
                }
            }
            for link in model.links() {
                if link.robot != compiled_robots[a].robot_id {
                    env.set_payload_link_margin(
                        pa,
                        link.id,
                        self.collision_rules.tool_other_link_margin,
                    );
                    env.set_payload_link_activation(
                        pa,
                        link.id,
                        self.collision_rules.tool_other_link_activation,
                    );
                }
            }
        }

        // Apply non-obstacle pair overrides on `env`.
        for ov in &self.collision_rules.pair_overrides {
            apply_non_obstacle_pair_override(&model, &mut env, ov)?;
        }

        let mut scene = match &self.esdf {
            Some(esdf_spec) => WorldScene::with_esdf(model, env, esdf_spec.to_esdf_config())?,
            None => WorldScene::new(model, env),
        };

        let mut compiled_obstacles = Vec::with_capacity(self.obstacles.len());
        let mut obstacle_by_uid = HashMap::with_capacity(self.obstacles.len());

        for obs in &self.obstacles {
            let base_pose = obs.pose.to_isometry();
            let (shape_pose, shape) = obs.shape.to_parry_shape_and_pose()?;
            let world_pose = base_pose * shape_pose;

            let handle = scene.add_obstacle_with_options(
                &obs.uid,
                shape,
                world_pose,
                ObstacleOptions {
                    carve_esdf: obs.carve_esdf,
                    safety_margin: obs.safety_margin,
                    activation_distance: obs.activation_distance,
                },
            )?;
            obstacle_by_uid.insert(obs.uid.clone(), handle);

            let label = if obs.label.trim().is_empty() {
                obs.uid.clone()
            } else {
                obs.label.clone()
            };
            let color_rgba = obs.color_rgba.unwrap_or_else(|| obs.role.default_color());

            compiled_obstacles.push(CompiledWorldObstacle {
                uid: obs.uid.clone(),
                label,
                role: obs.role,
                handle,
                obstacle_id: handle.obstacle_id(),
                shape: obs.shape,
                base_pose,
                color_rgba,
                carve_esdf: obs.carve_esdf,
            });
        }

        // Apply tool-to-workpiece rules on `WorldScene`.
        for robot in &compiled_robots {
            let Some(ref pname) = robot.payload_name else {
                continue;
            };
            for obs in &compiled_obstacles {
                if matches!(obs.role, ObstacleRoleSpec::Workpiece) {
                    scene.set_payload_obstacle_margin(
                        pname,
                        obs.handle,
                        self.collision_rules.tool_workpiece_margin,
                    )?;
                    scene.set_payload_obstacle_activation(
                        pname,
                        obs.handle,
                        self.collision_rules.tool_workpiece_activation,
                    )?;
                }
            }
        }

        // Apply obstacle-involving pair overrides via `WorldScene`.
        for ov in &self.collision_rules.pair_overrides {
            apply_obstacle_pair_override(&mut scene, &obstacle_by_uid, ov)?;
        }

        let flat_model = FlatRobotModelF32::compile_with_default_margin(
            scene.model(),
            scene.env(),
            self.collision_rules.default_body_margin,
        );

        Ok(CompiledWorld {
            name: self.name.clone(),
            scene,
            robots: compiled_robots,
            robot_by_uid,
            obstacles: compiled_obstacles,
            obstacle_by_uid,
            q_start: Configuration::new(q_start_raw),
            flat_model,
        })
    }
}

/// Serialized robot entry inside a [`WorldSpec`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RobotSpec {
    /// Unique stable robot identifier referenced by trajectory files (e.g. `"arm_a"`).
    pub uid: String,
    /// Human-readable display label (defaults to `uid` when empty).
    #[serde(default)]
    pub label: String,
    /// Prefix applied to URDF/SRDF links, joints, and frames (e.g. `"a_"`).
    #[serde(default)]
    pub prefix: String,
    /// Priority tier (`0` = primary/highest priority in CBF resolution).
    #[serde(default)]
    pub priority: u8,
    /// Optional SRDF planning group name to select instead of the full-robot group.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_group: Option<String>,
    /// Optional custom TCP fixed-frame name (registered when `tcp_offset` or this field is set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tcp_frame_name: Option<String>,
    /// Optional parent frame name for custom TCP attachment (defaults to robot EE link frame).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tcp_parent_frame: Option<String>,
    /// Optional RGBA display color `[r, g, b, a]` in `0.0..=1.0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_rgba: Option<[f32; 4]>,
    /// Optional initial joint angles in radians (length must equal robot DOF when non-empty).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub q_start: Vec<f64>,
    /// URDF / SRDF source location or built-in preset.
    pub source: RobotSourceSpec,
    /// Base mounting pose of the robot in world coordinates.
    #[serde(default)]
    pub base_pose: PoseSpec,
    /// Optional fixed transform from `tcp_parent_frame` (or tip link) to tool TCP.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tcp_offset: Option<PoseSpec>,
    /// Optional collision payload rigidly mounted on the end effector.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_payload: Option<ToolPayloadSpec>,
}

/// URDF / SRDF source specification for a robot in [`WorldSpec`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RobotSourceSpec {
    /// Built-in 4-DOF manipulator fixture bundled with `unrust-it-urdf`.
    Builtin4Dof,
    /// Built-in 6-DOF manipulator fixture bundled with `unrust-it-urdf`.
    Builtin6Dof,
    /// External URDF (and optional SRDF) files loaded from disk.
    Files {
        /// Path to the URDF XML file (relative paths resolve against the world file directory).
        urdf_path: PathBuf,
        /// Optional path to a MoveIt SRDF XML file.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        srdf_path: Option<PathBuf>,
    },
    /// Embedded URDF (and optional SRDF) XML strings.
    Inline {
        /// Inline URDF XML string.
        urdf_xml: String,
        /// Optional inline SRDF XML string.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        srdf_xml: Option<String>,
    },
}

impl RobotSourceSpec {
    fn load_xml(&self, base_dir: Option<&Path>) -> Result<(String, Option<String>), IoError> {
        match self {
            Self::Builtin4Dof => Ok((ARM_4DOF_URDF.to_string(), Some(ARM_4DOF_SRDF.to_string()))),
            Self::Builtin6Dof => Ok((ARM_6DOF_URDF.to_string(), Some(ARM_6DOF_SRDF.to_string()))),
            Self::Inline { urdf_xml, srdf_xml } => Ok((urdf_xml.clone(), srdf_xml.clone())),
            Self::Files {
                urdf_path,
                srdf_path,
            } => {
                let resolved_urdf = resolve_rel_path(urdf_path, base_dir);
                let urdf = std::fs::read_to_string(&resolved_urdf)
                    .map_err(|e| IoError::io_with_path(&resolved_urdf, e))?;
                let srdf = match srdf_path {
                    Some(p) => {
                        let resolved_srdf = resolve_rel_path(p, base_dir);
                        Some(
                            std::fs::read_to_string(&resolved_srdf)
                                .map_err(|e| IoError::io_with_path(&resolved_srdf, e))?,
                        )
                    }
                    None => None,
                };
                Ok((urdf, srdf))
            }
        }
    }
}

fn resolve_rel_path(path: &Path, base_dir: Option<&Path>) -> PathBuf {
    if path.is_relative() {
        if let Some(dir) = base_dir {
            return dir.join(path);
        }
    }
    path.to_path_buf()
}

/// Tool / payload collision geometry attached to a robot link.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolPayloadSpec {
    /// Globally unique payload identifier (e.g. `"a_torch"`).
    pub name: String,
    /// Optional parent link name (defaults to the robot's tip link).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_link: Option<String>,
    /// Payload mass in kg.
    #[serde(default = "default_payload_mass")]
    pub mass: f64,
    /// Optional safety margin override in meters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_margin: Option<f64>,
    /// Optional CBF activation distance override in meters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_distance: Option<f64>,
    /// Links allowed to touch this payload (defaults to parent link and its predecessor).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub touch_links: Vec<String>,
    /// Pose of the payload relative to `parent_link`.
    #[serde(default)]
    pub local_pose: PoseSpec,
    /// Collision primitive geometry.
    pub shape: ShapeSpec,
}

/// Semantic category of an environment obstacle in a workcell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObstacleRoleSpec {
    /// General static collision geometry (walls, fences, tables).
    #[default]
    Static,
    /// Primary shared workpiece being serviced or inspected.
    Workpiece,
    /// Robot mounting pedestal or base riser.
    Pedestal,
    /// Jigs, clamps, or process fixtures.
    Fixture,
}

impl ObstacleRoleSpec {
    /// Default RGBA tint used for visualizers when an obstacle omits `color_rgba`.
    pub fn default_color(self) -> [f32; 4] {
        match self {
            Self::Static => [0.48, 0.52, 0.58, 1.0],
            Self::Workpiece => [0.76, 0.56, 0.34, 1.0],
            Self::Pedestal => [0.34, 0.38, 0.44, 1.0],
            Self::Fixture => [0.42, 0.62, 0.55, 1.0],
        }
    }
}

/// Static obstacle, fixture, or workpiece inside a [`WorldSpec`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObstacleSpec {
    /// Unique stable identifier (e.g. `"workpiece"`). Referenced by [`crate::ReferenceFrame::Workpiece`].
    pub uid: String,
    /// Human-readable display label (defaults to `uid`).
    #[serde(default)]
    pub label: String,
    /// Semantic role in the workcell.
    #[serde(default)]
    pub role: ObstacleRoleSpec,
    /// Whether this obstacle is carved into the workcell ESDF grid.
    #[serde(default = "default_true")]
    pub carve_esdf: bool,
    /// Optional RGBA color override `[r, g, b, a]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_rgba: Option<[f32; 4]>,
    /// Optional safety margin override (m).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_margin: Option<f64>,
    /// Optional CBF activation distance override (m).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_distance: Option<f64>,
    /// World pose of the obstacle.
    #[serde(default)]
    pub pose: PoseSpec,
    /// Collision primitive geometry.
    pub shape: ShapeSpec,
}

/// Volumetric 3D ESDF configuration in [`WorldSpec`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EsdfConfigSpec {
    /// World-space minimum corner `[ox, oy, oz]` in meters.
    #[serde(default = "default_esdf_origin")]
    pub origin: [f32; 3],
    /// Isotropic voxel edge length in meters.
    #[serde(default = "default_esdf_voxel_size")]
    pub voxel_size: f32,
    /// Grid resolution `[nx, ny, nz]`.
    #[serde(default = "default_esdf_dims")]
    pub dims: [usize; 3],
    /// Positive truncation distance in meters.
    #[serde(default = "default_esdf_truncation")]
    pub truncation: f32,
}

impl Default for EsdfConfigSpec {
    fn default() -> Self {
        Self {
            origin: default_esdf_origin(),
            voxel_size: default_esdf_voxel_size(),
            dims: default_esdf_dims(),
            truncation: default_esdf_truncation(),
        }
    }
}

impl EsdfConfigSpec {
    /// Converts this serializable spec into [`unrust_it::EsdfConfig`].
    pub fn to_esdf_config(self) -> EsdfConfig {
        EsdfConfig {
            origin: self.origin,
            voxel_size: self.voxel_size,
            dims: self.dims,
            truncation: self.truncation,
        }
    }
}

/// Reference to a collision entity used in pair overrides.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum EntityRefSpec {
    /// A robot link by full prefixed name.
    Link(String),
    /// An attached tool or payload by ID.
    Payload(String),
    /// An environment obstacle by UID.
    Obstacle(String),
}

/// Override rule for a pair of links in the [`AllowedCollisionMatrix`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AcmRuleValueSpec {
    /// Enforce default collision checking.
    Enabled,
    /// Disable collision checking dynamically.
    Disabled,
    /// Enforce collision checking with a custom margin in meters.
    CustomMargin {
        /// Margin in meters.
        margin: f64,
    },
}

impl From<AcmRuleValueSpec> for CollisionRule {
    fn from(value: AcmRuleValueSpec) -> Self {
        match value {
            AcmRuleValueSpec::Enabled => Self::Enabled,
            AcmRuleValueSpec::Disabled => Self::DisabledDynamic,
            AcmRuleValueSpec::CustomMargin { margin } => Self::CustomMargin(margin),
        }
    }
}

/// Per-link-pair ACM rule override inside [`CollisionRulesSpec`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcmOverrideSpec {
    /// First link name.
    pub link_a: String,
    /// Second link name.
    pub link_b: String,
    /// Rule applied between `link_a` and `link_b`.
    pub rule: AcmRuleValueSpec,
}

/// Pairwise safety-margin, activation, or enable override between two entities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairOverrideSpec {
    /// Whether collision checking is enabled for this pair.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Optional custom margin override in meters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub margin: Option<f64>,
    /// Optional custom CBF activation distance override in meters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_distance: Option<f64>,
    /// First entity.
    pub a: EntityRefSpec,
    /// Second entity.
    pub b: EntityRefSpec,
}

/// Global and pairwise collision rule configuration inside [`WorldSpec`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollisionRulesSpec {
    /// Default inter-robot link-pair safety margin (meters).
    #[serde(default = "default_body_margin")]
    pub default_body_margin: f64,
    /// Default inter-robot link-pair CBF activation distance (meters).
    #[serde(default = "default_body_activation")]
    pub default_body_activation: f64,
    /// Optional intra-robot self-collision margin override (meters).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub self_collision_margin: Option<f64>,
    /// Margin between attached tools on different robots (meters).
    #[serde(default = "default_tool_tool_margin")]
    pub tool_tool_margin: f64,
    /// CBF activation distance between attached tools (meters).
    #[serde(default = "default_tool_tool_activation")]
    pub tool_tool_activation: f64,
    /// Margin between attached tools and `workpiece`-role obstacles (meters).
    #[serde(default = "default_tool_workpiece_margin")]
    pub tool_workpiece_margin: f64,
    /// CBF activation distance between attached tools and workpieces (meters).
    #[serde(default = "default_tool_workpiece_activation")]
    pub tool_workpiece_activation: f64,
    /// Margin between one robot's tool and another robot's links (meters).
    #[serde(default = "default_tool_other_link_margin")]
    pub tool_other_link_margin: f64,
    /// CBF activation distance between one robot's tool and another robot's links (meters).
    #[serde(default = "default_tool_other_link_activation")]
    pub tool_other_link_activation: f64,
    /// Explicit link-pair ACM overrides.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub acm_overrides: Vec<AcmOverrideSpec>,
    /// Explicit entity-pair margin/activation/enable overrides.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pair_overrides: Vec<PairOverrideSpec>,
}

impl Default for CollisionRulesSpec {
    fn default() -> Self {
        Self {
            default_body_margin: default_body_margin(),
            default_body_activation: default_body_activation(),
            self_collision_margin: None,
            tool_tool_margin: default_tool_tool_margin(),
            tool_tool_activation: default_tool_tool_activation(),
            tool_workpiece_margin: default_tool_workpiece_margin(),
            tool_workpiece_activation: default_tool_workpiece_activation(),
            tool_other_link_margin: default_tool_other_link_margin(),
            tool_other_link_activation: default_tool_other_link_activation(),
            acm_overrides: Vec::new(),
            pair_overrides: Vec::new(),
        }
    }
}

/// Stage-1 coarse planning strategy saved in [`PlannerConfigSpec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage1StrategySpec {
    /// Automatic selection based on proximity / difficulty.
    #[default]
    Auto,
    /// Force bidirectional RRT-Connect.
    RrtConnect,
    /// Force multi-seed sampling IK + direct interpolation.
    SamplingIk,
}

/// Optional planner and CBF controller parameter block stored in [`WorldSpec`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlannerConfigSpec {
    /// Simulation / servo integration timestep (seconds).
    #[serde(default = "default_dt")]
    pub dt: f64,
    /// Joint speed scaling factor (`0.0..=1.0`).
    #[serde(default = "default_max_speed_scale")]
    pub max_speed_scale: f64,
    /// Global CBF/planner safety margin (meters).
    #[serde(default = "default_safety_margin")]
    pub safety_margin: f64,
    /// CBF decay gain gamma.
    #[serde(default = "default_cbf_gamma")]
    pub cbf_gamma: f64,
    /// Tangential slide gain used to prevent deadlocks.
    #[serde(default = "default_tan_gain")]
    pub tan_gain: f64,
    /// Number of MPPI rollout samples `K`.
    #[serde(default = "default_mppi_samples")]
    pub mppi_samples: usize,
    /// MPPI rollout horizon steps `H`.
    #[serde(default = "default_mppi_horizon")]
    pub mppi_horizon: usize,
    /// Number of MPPI iterations.
    #[serde(default = "default_mppi_iters")]
    pub mppi_iters: usize,
    /// MPPI softmax temperature `lambda`.
    #[serde(default = "default_mppi_lambda")]
    pub mppi_lambda: f64,
    /// Stage-1 coarse planner mode.
    #[serde(default)]
    pub stage1_strategy: Stage1StrategySpec,
    /// Enable two-stage close-proximity approach planner.
    #[serde(default = "default_true")]
    pub two_stage_enable: bool,
    /// Activation distance threshold for stage-2 precision handoff (meters).
    #[serde(default = "default_two_stage_threshold")]
    pub two_stage_threshold: f64,
    /// RRT step size in configuration space (radians).
    #[serde(default = "default_rrt_step_size")]
    pub rrt_step_size: f64,
    /// Maximum RRT-Connect iterations.
    #[serde(default = "default_rrt_max_iters")]
    pub rrt_max_iters: usize,
    /// TOPP-RA output sample timestep (seconds).
    #[serde(default = "default_toppra_dt")]
    pub toppra_dt: f64,
    /// Prefer GPU backend when available.
    #[serde(default)]
    pub use_gpu: bool,
}

impl Default for PlannerConfigSpec {
    fn default() -> Self {
        Self {
            dt: default_dt(),
            max_speed_scale: default_max_speed_scale(),
            safety_margin: default_safety_margin(),
            cbf_gamma: default_cbf_gamma(),
            tan_gain: default_tan_gain(),
            mppi_samples: default_mppi_samples(),
            mppi_horizon: default_mppi_horizon(),
            mppi_iters: default_mppi_iters(),
            mppi_lambda: default_mppi_lambda(),
            stage1_strategy: Stage1StrategySpec::Auto,
            two_stage_enable: true,
            two_stage_threshold: default_two_stage_threshold(),
            rrt_step_size: default_rrt_step_size(),
            rrt_max_iters: default_rrt_max_iters(),
            toppra_dt: default_toppra_dt(),
            use_gpu: false,
        }
    }
}

/// Metadata for a compiled robot inside [`CompiledWorld`].
#[derive(Debug, Clone)]
pub struct CompiledRobot {
    /// Stable UID from [`RobotSpec::uid`].
    pub uid: String,
    /// Display label.
    pub label: String,
    /// Internal [`RobotId`] assigned during compilation.
    pub robot_id: RobotId,
    /// Link/joint/frame name prefix.
    pub prefix: String,
    /// URDF loader handle containing link/joint/DOF indices.
    pub handle: UrdfRobotHandle,
    /// Optional parsed SRDF data.
    pub srdf: Option<SrdfData>,
    /// Active [`PlanningGroup`] for this robot.
    pub planning_group: PlanningGroup,
    /// Contiguous range in the full-system `q` vector.
    pub q_range: Range<usize>,
    /// Local initial configuration values.
    pub q_start: Vec<f64>,
    /// Base pose in world coordinates.
    pub base_pose: Isometry3<f64>,
    /// Default TCP coordinate frame ID.
    pub tcp_frame_id: FrameId,
    /// Default TCP coordinate frame name.
    pub tcp_frame_name: String,
    /// Attached collision payload ID if configured.
    pub payload_name: Option<String>,
    /// Display RGBA color.
    pub color_rgba: [f32; 4],
    /// Priority tier (`0` = highest).
    pub priority: u8,
}

/// Metadata for a compiled obstacle inside [`CompiledWorld`].
#[derive(Debug, Clone)]
pub struct CompiledWorldObstacle {
    /// Stable UID from [`ObstacleSpec::uid`].
    pub uid: String,
    /// Display label.
    pub label: String,
    /// Semantic role.
    pub role: ObstacleRoleSpec,
    /// Generational [`ObstacleHandle`] inside [`WorldScene`].
    pub handle: ObstacleHandle,
    /// Raw [`ObstacleId`].
    pub obstacle_id: ObstacleId,
    /// Original declarative primitive shape.
    pub shape: ShapeSpec,
    /// World base pose before any internal primitive alignment rotation.
    pub base_pose: Isometry3<f64>,
    /// Display RGBA color.
    pub color_rgba: [f32; 4],
    /// Whether this obstacle was carved into ESDF.
    pub carve_esdf: bool,
}

/// Compiled runtime multi-robot workcell produced by [`WorldSpec::compile`].
#[derive(Debug, Clone)]
pub struct CompiledWorld {
    /// Workcell name.
    pub name: String,
    /// Authoritative synchronized [`WorldScene`] owning `MultiRobotModel`,
    /// `CollisionEnvironment`, and optional `EsdfGridF32`.
    pub scene: WorldScene,
    /// Compiled robots in mounting order.
    pub robots: Vec<CompiledRobot>,
    /// Lookup map `robot_uid -> index in self.robots`.
    pub robot_by_uid: HashMap<String, usize>,
    /// Compiled static obstacles and workpieces.
    pub obstacles: Vec<CompiledWorldObstacle>,
    /// Lookup map `obstacle_uid -> ObstacleHandle`.
    pub obstacle_by_uid: HashMap<String, ObstacleHandle>,
    /// Initial full-system joint configuration.
    pub q_start: Configuration,
    /// Pre-compiled `f32` batched rollout representation.
    pub flat_model: FlatRobotModelF32,
}

impl CompiledWorld {
    /// Looks up a compiled robot by UID.
    pub fn robot(&self, uid: &str) -> Option<&CompiledRobot> {
        self.robot_by_uid.get(uid).and_then(|&i| self.robots.get(i))
    }

    /// Looks up a compiled obstacle metadata record by UID.
    pub fn obstacle_meta(&self, uid: &str) -> Option<&CompiledWorldObstacle> {
        self.obstacles.iter().find(|o| o.uid == uid)
    }

    /// Re-compiles [`Self::flat_model`] from the current [`WorldScene`] state.
    pub fn refresh_flat_model(&mut self, default_margin: f64) {
        self.flat_model = FlatRobotModelF32::compile_with_default_margin(
            self.scene.model(),
            self.scene.env(),
            default_margin,
        );
    }
}

fn default_robot_palette(index: usize) -> [f32; 4] {
    const PALETTE: [[f32; 4]; 6] = [
        [0.24, 0.54, 0.88, 1.0],
        [0.88, 0.48, 0.20, 1.0],
        [0.24, 0.72, 0.50, 1.0],
        [0.72, 0.42, 0.84, 1.0],
        [0.86, 0.72, 0.22, 1.0],
        [0.22, 0.74, 0.82, 1.0],
    ];
    PALETTE[index % PALETTE.len()]
}

fn apply_non_obstacle_pair_override(
    model: &MultiRobotModel,
    env: &mut CollisionEnvironment,
    ov: &PairOverrideSpec,
) -> Result<(), IoError> {
    match (&ov.a, &ov.b) {
        (EntityRefSpec::Link(a), EntityRefSpec::Link(b)) => {
            let la = model
                .link_by_name(a)
                .ok_or_else(|| IoError::UnknownLink(a.clone()))?;
            let lb = model
                .link_by_name(b)
                .ok_or_else(|| IoError::UnknownLink(b.clone()))?;
            if let Some(mut acm) = env.acm().cloned() {
                if !ov.enabled {
                    acm.set_rule(la, lb, CollisionRule::DisabledDynamic);
                } else if let Some(m) = ov.margin {
                    acm.set_rule(la, lb, CollisionRule::CustomMargin(m));
                } else {
                    acm.set_rule(la, lb, CollisionRule::Enabled);
                }
                env.set_acm(acm);
            }
            if let Some(act) = ov.activation_distance {
                env.set_link_pair_activation(la, lb, act);
            }
        }
        (EntityRefSpec::Payload(pa), EntityRefSpec::Payload(pb)) => {
            env.set_payload_pair_enabled(pa, pb, ov.enabled);
            if let Some(m) = ov.margin {
                env.set_payload_pair_margin(pa, pb, m);
            }
            if let Some(act) = ov.activation_distance {
                env.set_payload_pair_activation(pa, pb, act);
            }
        }
        (EntityRefSpec::Payload(p), EntityRefSpec::Link(l))
        | (EntityRefSpec::Link(l), EntityRefSpec::Payload(p)) => {
            let lid = model
                .link_by_name(l)
                .ok_or_else(|| IoError::UnknownLink(l.clone()))?;
            env.set_payload_link_enabled(p, lid, ov.enabled);
            if let Some(m) = ov.margin {
                env.set_payload_link_margin(p, lid, m);
            }
            if let Some(act) = ov.activation_distance {
                env.set_payload_link_activation(p, lid, act);
            }
        }
        _ => {}
    }
    Ok(())
}

fn apply_obstacle_pair_override(
    scene: &mut WorldScene,
    obstacle_by_uid: &HashMap<String, ObstacleHandle>,
    ov: &PairOverrideSpec,
) -> Result<(), IoError> {
    match (&ov.a, &ov.b) {
        (EntityRefSpec::Payload(p), EntityRefSpec::Obstacle(o))
        | (EntityRefSpec::Obstacle(o), EntityRefSpec::Payload(p)) => {
            let handle = obstacle_by_uid
                .get(o)
                .copied()
                .ok_or_else(|| IoError::UnknownObstacle(o.clone()))?;
            scene.set_payload_obstacle_enabled(p, handle, ov.enabled)?;
            if let Some(m) = ov.margin {
                scene.set_payload_obstacle_margin(p, handle, m)?;
            }
            if let Some(act) = ov.activation_distance {
                scene.set_payload_obstacle_activation(p, handle, act)?;
            }
        }
        (EntityRefSpec::Link(l), EntityRefSpec::Obstacle(o))
        | (EntityRefSpec::Obstacle(o), EntityRefSpec::Link(l)) => {
            let lid = scene
                .model()
                .link_by_name(l)
                .ok_or_else(|| IoError::UnknownLink(l.clone()))?;
            let handle = obstacle_by_uid
                .get(o)
                .copied()
                .ok_or_else(|| IoError::UnknownObstacle(o.clone()))?;
            scene.set_link_obstacle_enabled(lid, handle, ov.enabled)?;
            if let Some(m) = ov.margin {
                scene.set_link_obstacle_margin(lid, handle, m)?;
            }
            if let Some(act) = ov.activation_distance {
                scene.set_link_obstacle_activation(lid, handle, act)?;
            }
        }
        _ => {}
    }
    Ok(())
}
