//! Deterministic compilation from declarative [`WorkcellSpec`] into immutable runtime
//! robotics structures ([`BuiltWorkcell`]).

use crate::scene::spec::{
    AcmRuleSpec, EntityRef, ObstacleRole, PairRuleOverrideSpec, ShapePrimitiveSpec, UrdfSource,
    WorkcellSpec,
};
use nalgebra::Isometry3;
use std::collections::HashMap;
use std::ops::Range;
use thiserror::Error;
use unrust_it::{
    AttachedPayload, CollisionEnvironment, CollisionRule, Configuration, EsdfGridF32,
    FlatRobotModelF32, FrameId, LinkGeometry, LinkId, MultiRobotModel, ObstacleId, PayloadId,
    RobotId,
};
use unrust_it_urdf::presets::{ARM_4DOF_SRDF, ARM_4DOF_URDF, ARM_6DOF_SRDF, ARM_6DOF_URDF};
use unrust_it_urdf::{add_urdf_to_scene, apply_srdf_str, UrdfError, UrdfRobotHandle};

/// Errors produced when compiling a declarative [`WorkcellSpec`].
#[derive(Debug, Error)]
pub enum SceneCompileError {
    #[error("URDF/SRDF load error for arm `{arm}`: {source}")]
    Urdf { arm: String, source: UrdfError },
    #[error("IO error reading URDF/SRDF for arm `{arm}`: {source}")]
    Io {
        arm: String,
        source: std::io::Error,
    },
    #[error("arm `{0}` has no links")]
    EmptyArmLinks(String),
    #[error("robot frame registration error: {0}")]
    Robot(#[from] unrust_it::RobotError),
    #[error("unknown link `{0}` in collision rule override")]
    UnknownLink(String),
}

/// Compiled metadata for one robot arm inside a [`BuiltWorkcell`].
#[derive(Debug, Clone)]
pub struct CompiledArm {
    pub uid: String,
    pub label: String,
    pub robot_id: RobotId,
    pub prefix: String,
    pub color_rgba: [f32; 4],
    pub handle: UrdfRobotHandle,
    pub flange_link: LinkId,
    pub tcp_frame: FrameId,
    pub payload_id: Option<PayloadId>,
    pub payload_name: Option<String>,
    pub q_range: Range<usize>,
    pub joint_names: Vec<String>,
    pub priority: f64,
}

/// Compiled metadata for one obstacle or workpiece inside a [`BuiltWorkcell`].
#[derive(Debug, Clone)]
pub struct CompiledObstacle {
    pub uid: String,
    pub label: String,
    pub obstacle_id: ObstacleId,
    pub role: ObstacleRole,
    pub shape: ShapePrimitiveSpec,
    pub color_rgba: [f32; 4],
    pub base_pose: Isometry3<f64>,
}

/// Immutable compiled robotics workcell produced by [`WorkcellSpec::compile`].
pub struct BuiltWorkcell {
    pub revision: u64,
    pub model: MultiRobotModel,
    pub env: CollisionEnvironment,
    pub flat_model: FlatRobotModelF32,
    pub esdf: EsdfGridF32,
    pub arms: Vec<CompiledArm>,
    pub obstacles: Vec<CompiledObstacle>,
    pub obstacle_by_uid: HashMap<String, ObstacleId>,
    pub payload_by_name: HashMap<String, PayloadId>,
    pub q_start: Configuration,
}

impl WorkcellSpec {
    /// Compiles this declarative specification into an immutable [`BuiltWorkcell`].
    pub fn compile(&self, revision: u64) -> Result<BuiltWorkcell, SceneCompileError> {
        let mut model = MultiRobotModel::new();
        let mut env = CollisionEnvironment::new();
        let mut arms = Vec::with_capacity(self.arms.len());
        let mut payload_by_name = HashMap::new();
        let mut q_start_raw = Vec::new();

        for (idx, arm_spec) in self.arms.iter().enumerate() {
            let prefix = format!("{}_", arm_spec.uid);
            let robot_id = RobotId(idx as u16);
            let base_pose = arm_spec.base_pose.to_isometry();

            let (urdf_xml, srdf_xml) = resolve_urdf_srdf(&arm_spec.uid, &arm_spec.urdf_source)?;
            let handle =
                add_urdf_to_scene(&mut model, &mut env, &urdf_xml, robot_id, &prefix, base_pose)
                    .map_err(|source| SceneCompileError::Urdf {
                        arm: arm_spec.uid.clone(),
                        source,
                    })?;

            if let Some(srdf) = srdf_xml {
                apply_srdf_str(&mut model, &mut env, &srdf, &prefix).map_err(|source| {
                    SceneCompileError::Urdf {
                        arm: arm_spec.uid.clone(),
                        source,
                    }
                })?;
            }

            let flange_name = format!("{prefix}{}", arm_spec.flange_link_raw);
            let flange_link = model
                .link_by_name(&flange_name)
                .or_else(|| handle.links.last().copied())
                .ok_or_else(|| SceneCompileError::EmptyArmLinks(arm_spec.uid.clone()))?;
            let flange_frame = model.link(flange_link).frame;

            let (payload_id, payload_name, tcp_offset) =
                if let Some(tool) = arm_spec.tool.expand(&arm_spec.uid, arm_spec.default_tcp_z) {
                    let touch_links = if let Some(explicit) = &tool.touch_link_names {
                        explicit
                            .iter()
                            .filter_map(|name| {
                                model
                                    .link_by_name(name)
                                    .or_else(|| model.link_by_name(&format!("{prefix}{name}")))
                            })
                            .collect::<Vec<_>>()
                    } else {
                        let mut auto = Vec::with_capacity(tool.touch_link_depth.max(1));
                        let mut cur = Some(flange_link);
                        for _ in 0..tool.touch_link_depth.max(1) {
                            if let Some(lid) = cur {
                                auto.push(lid);
                                cur = model.link(lid).parent_link;
                            }
                        }
                        auto
                    };

                    let mut shapes = Vec::with_capacity(tool.primitives.len());
                    for (sub_pose, sub_prim) in &tool.primitives {
                        let (prim_pose, parry_shape) = sub_prim.to_parry_shape_and_pose();
                        shapes.push((sub_pose.to_isometry() * prim_pose, parry_shape));
                    }
                    if shapes.is_empty() {
                        (None, None, tool.tcp_offset_in_flange.to_isometry())
                    } else {
                        let geom = LinkGeometry::from_convex_shapes(shapes);
                        let mut payload = AttachedPayload::new(
                            tool.name.clone(),
                            flange_link,
                            tool.mount_offset.to_isometry(),
                            geom,
                        )
                        .with_mass(tool.mass)
                        .with_touch_links(touch_links);
                        if let Some(margin) = tool.safety_margin {
                            payload = payload.with_safety_margin(margin);
                        }
                        if let Some(act) = tool.activation_distance {
                            payload = payload.with_activation_distance(act);
                        }

                        let pid = env.attach_payload(payload);
                        payload_by_name.insert(tool.name.clone(), pid);
                        (
                            Some(pid),
                            Some(tool.name),
                            tool.tcp_offset_in_flange.to_isometry(),
                        )
                    }
                } else {
                    (
                        None,
                        None,
                        Isometry3::translation(0.0, 0.0, arm_spec.default_tcp_z),
                    )
                };

            let tcp_frame = model.add_fixed_frame(format!("{prefix}tcp"), flange_frame, tcp_offset)?;

            let q_start_idx = *handle.q_indices.first().unwrap_or(&0);
            let q_len = handle.q_indices.len();
            let q_range = q_start_idx..(q_start_idx + q_len);
            let joint_names: Vec<String> = handle
                .actuated_joints
                .iter()
                .map(|&jid| model.joint(jid).name.clone())
                .collect();

            for i in 0..q_len {
                let raw_q = arm_spec.q_start.get(i).copied().unwrap_or(0.0);
                let global_dof = q_start_idx + i;
                let clamped = model.dof_limits()[global_dof].position.clamp(raw_q);
                q_start_raw.push(clamped);
            }

            arms.push(CompiledArm {
                uid: arm_spec.uid.clone(),
                label: arm_spec.label.clone(),
                robot_id,
                prefix,
                color_rgba: arm_spec.color_rgba,
                handle,
                flange_link,
                tcp_frame,
                payload_id,
                payload_name,
                q_range,
                joint_names,
                priority: arm_spec.priority,
            });
        }

        // Configure baseline inter-arm & self-collision margins + activations
        if let Some(mut acm) = env.acm().cloned() {
            acm.set_inter_arm_margin(&model, self.rules.default_body_margin);
            if let Some(self_margin) = self.rules.self_collision_margin {
                acm.set_self_collision_margin(&model, self_margin);
            }
            for (link_a, link_b, rule) in &self.rules.acm_overrides {
                let la = model
                    .link_by_name(link_a)
                    .ok_or_else(|| SceneCompileError::UnknownLink(link_a.clone()))?;
                let lb = model
                    .link_by_name(link_b)
                    .ok_or_else(|| SceneCompileError::UnknownLink(link_b.clone()))?;
                let coll_rule = match *rule {
                    AcmRuleSpec::Enabled => CollisionRule::Enabled,
                    AcmRuleSpec::Disabled => CollisionRule::DisabledDynamic,
                    AcmRuleSpec::CustomMargin(m) => CollisionRule::CustomMargin(m),
                };
                acm.set_rule(la, lb, coll_rule);
            }
            env.set_acm(acm);
        }

        for a in 0..model.num_links() {
            for b in (a + 1)..model.num_links() {
                let la = LinkId(a as u16);
                let lb = LinkId(b as u16);
                if model.link(la).robot != model.link(lb).robot {
                    env.set_link_pair_activation(la, lb, self.rules.default_body_activation);
                }
            }
        }

        // Add obstacles and carve static/workpiece/pedestal shapes into 3D ESDF
        let mut esdf = EsdfGridF32::new(
            self.esdf.origin,
            self.esdf.voxel_size,
            self.esdf.dims,
            self.esdf.truncation,
        );
        let mut obstacles = Vec::with_capacity(self.obstacles.len());
        let mut obstacle_by_uid = HashMap::with_capacity(self.obstacles.len());

        for obs_spec in &self.obstacles {
            let base_pose = obs_spec.pose.to_isometry();
            let (shape_pose, shape) = obs_spec.shape.to_parry_shape_and_pose();
            let world_pose = base_pose * shape_pose;
            let id = env.add_obstacle(shape.clone(), world_pose);
            if let Some(margin) = obs_spec.safety_margin {
                env.set_obstacle_margin(id, margin);
            }
            if let Some(act) = obs_spec.activation_distance {
                env.set_obstacle_activation(id, act);
            }
            if obs_spec.carve_esdf && !matches!(obs_spec.role, ObstacleRole::DynamicSweep { .. }) {
                esdf.carve_shape(&shape, &world_pose);
            }
            obstacle_by_uid.insert(obs_spec.uid.clone(), id);
            obstacles.push(CompiledObstacle {
                uid: obs_spec.uid.clone(),
                label: obs_spec.label.clone(),
                obstacle_id: id,
                role: obs_spec.role.clone(),
                shape: obs_spec.shape,
                color_rgba: obs_spec.color_rgba,
                base_pose,
            });
        }

        // Apply baseline tool-to-tool, tool-to-workpiece, and tool-to-other-arm-link rules
        for a in 0..arms.len() {
            let Some(ref pname_a) = arms[a].payload_name else {
                continue;
            };
            for arm_b in arms.iter().skip(a + 1) {
                if let Some(ref pname_b) = arm_b.payload_name {
                    env.set_payload_pair_margin(pname_a, pname_b, self.rules.tool_tool_margin);
                    env.set_payload_pair_activation(
                        pname_a,
                        pname_b,
                        self.rules.tool_tool_activation,
                    );
                }
            }
            for obs in &obstacles {
                if matches!(obs.role, ObstacleRole::Workpiece) {
                    env.set_payload_obstacle_margin(
                        pname_a,
                        obs.obstacle_id,
                        self.rules.tool_workpiece_margin,
                    );
                    env.set_payload_obstacle_activation(
                        pname_a,
                        obs.obstacle_id,
                        self.rules.tool_workpiece_activation,
                    );
                }
            }
            for link in model.links() {
                if link.robot != arms[a].robot_id {
                    env.set_payload_link_margin(
                        pname_a,
                        link.id,
                        self.rules.tool_other_link_margin,
                    );
                    env.set_payload_link_activation(
                        pname_a,
                        link.id,
                        self.rules.tool_other_link_activation,
                    );
                }
            }
        }

        // Apply explicit pair-rule overrides last so user edits always take precedence
        for ov in &self.rules.pair_overrides {
            apply_pair_override(&model, &mut env, &obstacle_by_uid, ov);
        }

        let flat_model =
            FlatRobotModelF32::compile_with_default_margin(&model, &env, self.rules.default_body_margin);

        Ok(BuiltWorkcell {
            revision,
            model,
            env,
            flat_model,
            esdf,
            arms,
            obstacles,
            obstacle_by_uid,
            payload_by_name,
            q_start: Configuration::new(q_start_raw),
        })
    }
}

fn resolve_urdf_srdf(
    arm_uid: &str,
    source: &UrdfSource,
) -> Result<(String, Option<String>), SceneCompileError> {
    match source {
        UrdfSource::Builtin4Dof => Ok((ARM_4DOF_URDF.to_string(), Some(ARM_4DOF_SRDF.to_string()))),
        UrdfSource::Builtin6Dof => Ok((ARM_6DOF_URDF.to_string(), Some(ARM_6DOF_SRDF.to_string()))),
        UrdfSource::InlineXml { urdf, srdf } => Ok((urdf.clone(), srdf.clone())),
        UrdfSource::FilePath {
            urdf_path,
            srdf_path,
        } => {
            let urdf = std::fs::read_to_string(urdf_path).map_err(|source| SceneCompileError::Io {
                arm: arm_uid.to_string(),
                source,
            })?;
            let srdf = match srdf_path {
                Some(p) => Some(std::fs::read_to_string(p).map_err(|source| {
                    SceneCompileError::Io {
                        arm: arm_uid.to_string(),
                        source,
                    }
                })?),
                None => None,
            };
            Ok((urdf, srdf))
        }
    }
}

fn apply_pair_override(
    model: &MultiRobotModel,
    env: &mut CollisionEnvironment,
    obstacle_by_uid: &HashMap<String, ObstacleId>,
    ov: &PairRuleOverrideSpec,
) {
    match (&ov.a, &ov.b) {
        (EntityRef::Link(a), EntityRef::Link(b)) => {
            if let (Some(la), Some(lb)) = (model.link_by_name(a), model.link_by_name(b)) {
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
        }
        (EntityRef::Payload(pa), EntityRef::Payload(pb)) => {
            env.set_payload_pair_enabled(pa, pb, ov.enabled);
            if let Some(m) = ov.margin {
                env.set_payload_pair_margin(pa, pb, m);
            }
            if let Some(act) = ov.activation_distance {
                env.set_payload_pair_activation(pa, pb, act);
            }
        }
        (EntityRef::Payload(p), EntityRef::Obstacle(o))
        | (EntityRef::Obstacle(o), EntityRef::Payload(p)) => {
            if let Some(&oid) = obstacle_by_uid.get(o) {
                env.set_payload_obstacle_enabled(p, oid, ov.enabled);
                if let Some(m) = ov.margin {
                    env.set_payload_obstacle_margin(p, oid, m);
                }
                if let Some(act) = ov.activation_distance {
                    env.set_payload_obstacle_activation(p, oid, act);
                }
            }
        }
        (EntityRef::Payload(p), EntityRef::Link(l))
        | (EntityRef::Link(l), EntityRef::Payload(p)) => {
            if let Some(lid) = model.link_by_name(l) {
                env.set_payload_link_enabled(p, lid, ov.enabled);
                if let Some(m) = ov.margin {
                    env.set_payload_link_margin(p, lid, m);
                }
                if let Some(act) = ov.activation_distance {
                    env.set_payload_link_activation(p, lid, act);
                }
            }
        }
        (EntityRef::Link(l), EntityRef::Obstacle(o))
        | (EntityRef::Obstacle(o), EntityRef::Link(l)) => {
            if let (Some(lid), Some(&oid)) = (model.link_by_name(l), obstacle_by_uid.get(o)) {
                env.set_link_obstacle_enabled(lid, oid, ov.enabled);
                if let Some(m) = ov.margin {
                    env.set_link_obstacle_margin(lid, oid, m);
                }
                if let Some(act) = ov.activation_distance {
                    env.set_link_obstacle_activation(lid, oid, act);
                }
            }
        }
        (EntityRef::Obstacle(_), EntityRef::Obstacle(_)) => {}
    }
}
