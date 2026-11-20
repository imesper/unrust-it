//! Bridge converting between visualizer `WorkcellSpec`/`BuiltWorkcell` and `unrust-it-io`
//! `WorldSpec`/`TrajectorySetSpec`, plus multi-robot frame/subframe IK + optional CBF evaluation.

use crate::scene::compile::BuiltWorkcell;
use crate::scene::eval::{
    build_trajectory_frame, compute_tcp_trails, summarize_metrics, verify_path_convex,
    PlaybackTrajectory, TrajectoryStage,
};
use crate::scene::spec::{
    arm_palette_color, AcmRuleSpec, ArmSpec, ArmTargetSpec, CollisionRulesSpec,
    ComputeBackendChoice, CustomToolSpec, EntityRef, EsdfSpec, ObstacleRole, ObstacleSpec,
    PairRuleOverrideSpec, PlannerSpec, PoseSpec, ShapePrimitiveSpec, ToolPreset, UrdfSource,
    WorkcellSpec,
};
use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use unrust_it::{
    BatchIkSolver, Configuration, CpuSimdBatchBackend, ReactiveServoConfig,
    ReactiveServoController, TaskPose,
};
use unrust_it_io::{
    resolve_trajectory_set_with_model, AcmOverrideSpec, AcmRuleValueSpec,
    CollisionRulesSpec as IoCollisionRulesSpec, EntityRefSpec, EsdfConfigSpec, IoError,
    ObstacleRoleSpec, ObstacleSpec as IoObstacleSpec, PairOverrideSpec,
    PlannerConfigSpec as IoPlannerConfigSpec, PoseSpec as IoPoseSpec, ReferenceFrame,
    ResolvedTrajectorySet, RobotResolveInfo, RobotSourceSpec, RobotSpec, RobotTrajectoriesSpec,
    ShapeSpec, SubframeRole, SubframeSpec, ToolPayloadSpec, TrajectoryFrameSpec,
    TrajectorySetSpec, TrajectorySpec, WorldSpec,
};

/// Converts an interactive [`WorkcellSpec`] into a serializable [`unrust_it_io::WorldSpec`].
pub fn workcell_spec_to_world_spec(spec: &WorkcellSpec) -> WorldSpec {
    let name = if spec.title.trim().is_empty() {
        "workcell".to_string()
    } else {
        spec.title.clone()
    };
    let mut world = WorldSpec::new(name);
    world.description = spec.subtitle.clone();
    world.esdf = Some(EsdfConfigSpec {
        origin: spec.esdf.origin,
        voxel_size: spec.esdf.voxel_size,
        dims: spec.esdf.dims,
        truncation: spec.esdf.truncation,
    });
    world.planner = Some(IoPlannerConfigSpec::default());

    world.collision_rules = IoCollisionRulesSpec {
        default_body_margin: spec.rules.default_body_margin,
        default_body_activation: spec.rules.default_body_activation,
        self_collision_margin: spec.rules.self_collision_margin,
        tool_tool_margin: spec.rules.tool_tool_margin,
        tool_tool_activation: spec.rules.tool_tool_activation,
        tool_workpiece_margin: spec.rules.tool_workpiece_margin,
        tool_workpiece_activation: spec.rules.tool_workpiece_activation,
        tool_other_link_margin: spec.rules.tool_other_link_margin,
        tool_other_link_activation: spec.rules.tool_other_link_activation,
        acm_overrides: spec
            .rules
            .acm_overrides
            .iter()
            .map(|(a, b, r)| AcmOverrideSpec {
                link_a: a.clone(),
                link_b: b.clone(),
                rule: match *r {
                    AcmRuleSpec::Enabled => AcmRuleValueSpec::Enabled,
                    AcmRuleSpec::Disabled => AcmRuleValueSpec::Disabled,
                    AcmRuleSpec::CustomMargin(margin) => AcmRuleValueSpec::CustomMargin { margin },
                },
            })
            .collect(),
        pair_overrides: spec
            .rules
            .pair_overrides
            .iter()
            .map(|ov| PairOverrideSpec {
                enabled: ov.enabled,
                margin: ov.margin,
                activation_distance: ov.activation_distance,
                a: entity_ref_to_io(&ov.a),
                b: entity_ref_to_io(&ov.b),
            })
            .collect(),
    };

    for arm in &spec.arms {
        let prefix = format!("{}_", arm.uid);
        let source = match &arm.urdf_source {
            UrdfSource::Builtin4Dof => RobotSourceSpec::Builtin4Dof,
            UrdfSource::Builtin6Dof => RobotSourceSpec::Builtin6Dof,
            UrdfSource::InlineXml { urdf, srdf } => RobotSourceSpec::Inline {
                urdf_xml: urdf.clone(),
                srdf_xml: srdf.clone(),
            },
            UrdfSource::FilePath {
                urdf_path,
                srdf_path,
            } => RobotSourceSpec::Files {
                urdf_path: urdf_path.clone(),
                srdf_path: srdf_path.clone(),
            },
        };

        let expanded_tool = arm.tool.expand(&arm.uid, arm.default_tcp_z);
        let tcp_offset = Some(match &expanded_tool {
            Some(t) => pose_to_io(&t.tcp_offset_in_flange),
            None => IoPoseSpec::from_xyz([0.0, 0.0, arm.default_tcp_z]),
        });
        let tool_payload = expanded_tool.and_then(|t| {
            let (prim_pose, prim_shape) = *t.primitives.first()?;
            let local_iso = t.mount_offset.to_isometry() * prim_pose.to_isometry();
            Some(ToolPayloadSpec {
                name: t.name,
                parent_link: Some(format!("{prefix}{}", arm.flange_link_raw)),
                mass: t.mass,
                safety_margin: t.safety_margin,
                activation_distance: t.activation_distance,
                touch_links: t.touch_link_names.unwrap_or_default(),
                local_pose: IoPoseSpec::from_isometry(&local_iso),
                shape: shape_to_io(&prim_shape),
            })
        });

        world.robots.push(RobotSpec {
            uid: arm.uid.clone(),
            label: arm.label.clone(),
            prefix,
            priority: arm.priority.round().clamp(0.0, 255.0) as u8,
            active_group: None,
            tcp_frame_name: Some(format!("{}_tcp", arm.uid)),
            tcp_parent_frame: Some(format!("{}_{}", arm.uid, arm.flange_link_raw)),
            color_rgba: Some(arm.color_rgba),
            q_start: arm.q_start.clone(),
            source,
            base_pose: pose_to_io(&arm.base_pose),
            tcp_offset,
            tool_payload,
        });
    }

    for obs in &spec.obstacles {
        let role = match obs.role {
            ObstacleRole::StaticObstacle => ObstacleRoleSpec::Static,
            ObstacleRole::Workpiece => ObstacleRoleSpec::Workpiece,
            ObstacleRole::Pedestal => ObstacleRoleSpec::Pedestal,
            ObstacleRole::DynamicSweep { .. } => ObstacleRoleSpec::Fixture,
        };
        world.obstacles.push(IoObstacleSpec {
            uid: obs.uid.clone(),
            label: obs.label.clone(),
            role,
            carve_esdf: obs.carve_esdf,
            color_rgba: Some(obs.color_rgba),
            safety_margin: obs.safety_margin,
            activation_distance: obs.activation_distance,
            pose: pose_to_io(&obs.pose),
            shape: shape_to_io(&obs.shape),
        });
    }

    world
}

/// Converts a loaded [`unrust_it_io::WorldSpec`] into an editable visualizer [`WorkcellSpec`],
/// resolving any relative URDF/SRDF paths against `world.base_dir()`.
pub fn world_spec_to_workcell_spec(world: &WorldSpec) -> WorkcellSpec {
    let base_dir = world.base_dir();
    let resolve_path = |p: &Path| {
        if p.is_relative() {
            if let Some(dir) = base_dir {
                return dir.join(p);
            }
        }
        p.to_path_buf()
    };

    let arms = world
        .robots
        .iter()
        .enumerate()
        .map(|(idx, r)| {
            let urdf_source = match &r.source {
                RobotSourceSpec::Builtin4Dof => UrdfSource::Builtin4Dof,
                RobotSourceSpec::Builtin6Dof => UrdfSource::Builtin6Dof,
                RobotSourceSpec::Inline { urdf_xml, srdf_xml } => UrdfSource::InlineXml {
                    urdf: urdf_xml.clone(),
                    srdf: srdf_xml.clone(),
                },
                RobotSourceSpec::Files {
                    urdf_path,
                    srdf_path,
                } => UrdfSource::FilePath {
                    urdf_path: resolve_path(urdf_path),
                    srdf_path: srdf_path.as_deref().map(resolve_path),
                },
            };

            let default_tcp_z = r.tcp_offset.map_or(0.10, |o| o.xyz[2]);
            let tcp_offset_in_flange = r
                .tcp_offset
                .map_or_else(|| PoseSpec::translation(0.0, 0.0, default_tcp_z), |p| pose_from_io(&p));

            let tool = match &r.tool_payload {
                Some(tp) => {
                    let (rot_fix, prim_shape) = shape_from_io(&tp.shape);
                    let mount_iso = tp.local_pose.to_isometry() * rot_fix;
                    ToolPreset::Custom(Box::new(CustomToolSpec {
                        name: tp.name.clone(),
                        mount_offset: PoseSpec::from_isometry(&mount_iso),
                        tcp_offset_in_flange,
                        primitives: vec![(PoseSpec::default(), prim_shape)],
                        mass: tp.mass,
                        safety_margin: tp.safety_margin,
                        activation_distance: tp.activation_distance,
                        touch_link_depth: 2,
                        touch_link_names: (!tp.touch_links.is_empty()).then(|| tp.touch_links.clone()),
                    }))
                }
                None => ToolPreset::None,
            };

            let flange_link_raw = r
                .tcp_parent_frame
                .as_deref()
                .map(|f| f.strip_prefix(&r.prefix).unwrap_or(f).to_string())
                .unwrap_or_else(|| "flange_link".to_string());

            ArmSpec {
                uid: r.uid.clone(),
                label: if r.label.trim().is_empty() {
                    r.uid.clone()
                } else {
                    r.label.clone()
                },
                urdf_source,
                base_pose: pose_from_io(&r.base_pose),
                flange_link_raw,
                default_tcp_z,
                tool,
                color_rgba: r.color_rgba.unwrap_or_else(|| arm_palette_color(idx)),
                q_start: r.q_start.clone(),
                priority: if r.priority == 0 {
                    1.0
                } else {
                    r.priority as f64
                },
                target: ArmTargetSpec::None,
            }
        })
        .collect();

    let obstacles = world
        .obstacles
        .iter()
        .map(|o| {
            let (rot_fix, shape) = shape_from_io(&o.shape);
            let pose = PoseSpec::from_isometry(&(o.pose.to_isometry() * rot_fix));
            let role = match o.role {
                ObstacleRoleSpec::Static | ObstacleRoleSpec::Fixture => ObstacleRole::StaticObstacle,
                ObstacleRoleSpec::Workpiece => ObstacleRole::Workpiece,
                ObstacleRoleSpec::Pedestal => ObstacleRole::Pedestal,
            };
            ObstacleSpec {
                uid: o.uid.clone(),
                label: if o.label.trim().is_empty() {
                    o.uid.clone()
                } else {
                    o.label.clone()
                },
                shape,
                pose,
                role,
                color_rgba: o.color_rgba.unwrap_or_else(|| o.role.default_color()),
                safety_margin: o.safety_margin,
                activation_distance: o.activation_distance,
                carve_esdf: o.carve_esdf,
            }
        })
        .collect();

    let cr = &world.collision_rules;
    let rules = CollisionRulesSpec {
        default_body_margin: cr.default_body_margin,
        default_body_activation: cr.default_body_activation,
        self_collision_margin: cr.self_collision_margin,
        tool_tool_margin: cr.tool_tool_margin,
        tool_tool_activation: cr.tool_tool_activation,
        tool_workpiece_margin: cr.tool_workpiece_margin,
        tool_workpiece_activation: cr.tool_workpiece_activation,
        tool_other_link_margin: cr.tool_other_link_margin,
        tool_other_link_activation: cr.tool_other_link_activation,
        acm_overrides: cr
            .acm_overrides
            .iter()
            .map(|ov| {
                let rule = match ov.rule {
                    AcmRuleValueSpec::Enabled => AcmRuleSpec::Enabled,
                    AcmRuleValueSpec::Disabled => AcmRuleSpec::Disabled,
                    AcmRuleValueSpec::CustomMargin { margin } => AcmRuleSpec::CustomMargin(margin),
                };
                (ov.link_a.clone(), ov.link_b.clone(), rule)
            })
            .collect(),
        pair_overrides: cr
            .pair_overrides
            .iter()
            .map(|ov| PairRuleOverrideSpec {
                a: entity_ref_from_io(&ov.a),
                b: entity_ref_from_io(&ov.b),
                enabled: ov.enabled,
                margin: ov.margin,
                activation_distance: ov.activation_distance,
            })
            .collect(),
    };

    let esdf = world.esdf.map_or_else(EsdfSpec::default, |e| EsdfSpec {
        origin: e.origin,
        voxel_size: e.voxel_size,
        dims: e.dims,
        truncation: e.truncation,
    });

    WorkcellSpec {
        preset_id: None,
        title: world.name.clone(),
        subtitle: world.description.clone(),
        arms,
        obstacles,
        rules,
        esdf,
        planner: PlannerSpec::InteractiveSandbox,
        compute_backend: ComputeBackendChoice::CpuSimd,
    }
}

/// Resolves a [`TrajectorySetSpec`] against a visualizer [`WorkcellSpec`] & [`BuiltWorkcell`].
pub fn resolve_trajectory_set_for_workcell(
    set: &TrajectorySetSpec,
    spec: &WorkcellSpec,
    workcell: &BuiltWorkcell,
    q_ref: &[f64],
) -> Result<ResolvedTrajectorySet, IoError> {
    let robots: Vec<RobotResolveInfo> = workcell
        .arms
        .iter()
        .enumerate()
        .map(|(idx, arm)| {
            let base_pose = spec
                .arms
                .get(idx)
                .map_or_else(Isometry3::identity, |a| a.base_pose.to_isometry());
            RobotResolveInfo {
                uid: arm.uid.clone(),
                prefix: arm.prefix.clone(),
                dof: arm.handle.q_indices.len(),
                default_group_name: arm.handle.planning_group.name.clone(),
                base_pose,
                tcp_frame_id: arm.tcp_frame,
                tcp_frame_name: format!("{}_tcp", arm.uid),
                srdf_group_names: Vec::new(),
            }
        })
        .collect();

    let obstacle_poses: HashMap<String, Isometry3<f64>> = workcell
        .obstacles
        .iter()
        .map(|o| (o.uid.clone(), o.base_pose))
        .collect();

    resolve_trajectory_set_with_model(set, &workcell.model, &robots, &obstacle_poses, q_ref)
}

/// Converts resolved imported frames and recursive subframes into 3D viewport markers.
#[cfg(feature = "render")]
pub fn build_imported_frame_markers(
    resolved: &ResolvedTrajectorySet,
    selected_node: Option<&str>,
) -> Vec<crate::render::ImportedFrameMarker> {
    let mut markers = Vec::new();
    for robot in &resolved.robots {
        for traj in &robot.trajectories {
            let mut prev_exec_pos: Option<[f64; 3]> = None;
            for wp in &traj.waypoints {
                let full_key = format!("{}/{}/{}", wp.robot_uid, wp.trajectory_id, wp.node_id);
                let selected = selected_node.is_some_and(|s| s == full_key);
                let parent_world_pose = if wp.depth > 0 {
                    Some(wp.parent_world_pose)
                } else {
                    None
                };
                let prev_sequence_pos = if wp.execute_ik && traj.enabled {
                    let prev = prev_exec_pos;
                    let p = wp.world_pose.translation.vector;
                    prev_exec_pos = Some([p.x, p.y, p.z]);
                    prev
                } else {
                    None
                };
                let mut color = wp.color_rgba;
                if !traj.enabled {
                    color[3] *= 0.45;
                }
                markers.push(crate::render::ImportedFrameMarker {
                    world_pose: wp.world_pose,
                    parent_world_pose,
                    prev_sequence_pos,
                    depth: wp.depth,
                    execute_ik: wp.execute_ik,
                    selected,
                    color_rgba: color,
                });
            }
        }
    }
    markers
}

/// Synthesizes a valid multi-robot sample [`TrajectorySetSpec`] anchored near each arm's current
/// FK TCP pose at `q_ref`, complete with `Approach`, `Contact`, `Retract`, and `ReferenceOnly` subframes.
pub fn sample_trajectory_set_for_workcell(
    spec: &WorkcellSpec,
    workcell: &BuiltWorkcell,
    q_ref: &[f64],
) -> TrajectorySetSpec {
    let mut tf_buf = workcell.model.create_transform_buffer();
    let _ = workcell.model.forward_kinematics(q_ref, &mut tf_buf);

    let mut set = TrajectorySetSpec::new(format!("{}_sample_trajectories", spec.title));
    set.description =
        "Sample per-robot trajectories with hierarchical frames and subframes generated from current workcell FK."
            .to_string();
    set.world_name = Some(spec.title.clone());
    set.default_reference_frame = Some(ReferenceFrame::World);

    for (idx, arm) in workcell.arms.iter().enumerate() {
        let tcp_iso = *tf_buf.frame_pose(arm.tcp_frame);
        let sign = if idx % 2 == 0 { 1.0 } else { -1.0 };

        let mut p1_iso = tcp_iso;
        p1_iso.translation.y += 0.035 * sign;
        p1_iso.translation.z += 0.025;

        let mut p2_iso = tcp_iso;
        p2_iso.translation.x += 0.025;
        p2_iso.translation.y -= 0.035 * sign;
        p2_iso.translation.z += 0.035;

        set.robots.push(RobotTrajectoriesSpec {
            robot_uid: arm.uid.clone(),
            group_name: None,
            tcp_frame: Some(format!("{}_tcp", arm.uid)),
            reference_frame: Some(ReferenceFrame::World),
            trajectories: vec![TrajectorySpec {
                id: "pass_1".to_string(),
                label: format!("{} Sample Pass", arm.label),
                enabled: true,
                speed_scale: 1.0,
                tcp_frame: None,
                reference_frame: None,
                frames: vec![
                    TrajectoryFrameSpec {
                        id: "frame_a".to_string(),
                        label: "Touch Point A".to_string(),
                        role: SubframeRole::Contact,
                        time_s: Some(0.8),
                        execute_ik: Some(true),
                        tcp_frame: None,
                        joint_seed: Vec::new(),
                        color_rgba: None,
                        reference_frame: None,
                        pose: IoPoseSpec::from_isometry(&p1_iso),
                        subframes: vec![
                            SubframeSpec {
                                id: "approach".to_string(),
                                label: "Approach".to_string(),
                                role: SubframeRole::Approach,
                                time_offset_s: -0.35,
                                execute_ik: Some(true),
                                tcp_frame: None,
                                color_rgba: None,
                                pose: IoPoseSpec::from_xyz([0.0, 0.0, -0.04]),
                                subframes: vec![SubframeSpec {
                                    id: "standoff_ref".to_string(),
                                    label: "Sensor Standoff".to_string(),
                                    role: SubframeRole::ReferenceOnly,
                                    time_offset_s: 0.0,
                                    execute_ik: Some(false),
                                    tcp_frame: None,
                                    color_rgba: None,
                                    pose: IoPoseSpec::from_xyz([0.02, 0.0, 0.0]),
                                    subframes: Vec::new(),
                                }],
                            },
                            SubframeSpec {
                                id: "retract".to_string(),
                                label: "Retract".to_string(),
                                role: SubframeRole::Retract,
                                time_offset_s: 0.35,
                                execute_ik: Some(true),
                                tcp_frame: None,
                                color_rgba: None,
                                pose: IoPoseSpec::from_xyz([0.0, 0.0, -0.04]),
                                subframes: Vec::new(),
                            },
                        ],
                    },
                    TrajectoryFrameSpec {
                        id: "frame_b".to_string(),
                        label: "Via Point B".to_string(),
                        role: SubframeRole::Waypoint,
                        time_s: Some(1.6),
                        execute_ik: Some(true),
                        tcp_frame: None,
                        joint_seed: Vec::new(),
                        color_rgba: None,
                        reference_frame: None,
                        pose: IoPoseSpec::from_isometry(&p2_iso),
                        subframes: Vec::new(),
                    },
                ],
            }],
        });
    }

    set
}

struct ArmWaypointKeyframes {
    times: Vec<f64>,
    qs: Vec<Vec<f64>>,
}

/// Evaluates all enabled imported trajectories for each robot using masked subgroup IK,
/// smooth joint interpolation, and optional 1 kHz `ReactiveServoController::step_joint_reference`
/// CBF safety filtering, producing a [`PlaybackTrajectory`] for 3D playback & plots.
pub fn evaluate_imported_trajectory_set(
    spec: &WorkcellSpec,
    workcell: &BuiltWorkcell,
    resolved: &ResolvedTrajectorySet,
    q_start: &[f64],
    apply_cbf_filter: bool,
) -> Result<PlaybackTrajectory, String> {
    let t0 = Instant::now();
    let total_dof = workcell.model.dof();
    if q_start.len() != total_dof {
        return Err(format!(
            "q_start length mismatch: expected {total_dof}, got {}",
            q_start.len()
        ));
    }

    let limits = workcell.model.dof_limits();
    let clamp_full_q = |q: &mut [f64]| {
        for (v, &lim) in q.iter_mut().zip(limits.iter()) {
            *v = lim.position.clamp(*v);
        }
    };

    let ik_solver = BatchIkSolver {
        num_seeds: 24,
        max_iterations: 80,
        tolerance: 2e-3,
        damping: 0.05,
        activation_distance: 0.08,
    };

    let mut tf_buf = workcell.model.create_transform_buffer();
    let mut schedules: Vec<ArmWaypointKeyframes> = Vec::with_capacity(workcell.arms.len());
    let mut max_pos_err_m = 0.0_f64;
    let mut max_rot_err_rad = 0.0_f64;
    let mut max_duration = 0.0_f64;

    for arm in &workcell.arms {
        let mut mask = vec![false; total_dof];
        for &d in &arm.handle.q_indices {
            mask[d] = true;
        }

        let arm_init: Vec<f64> = arm.handle.q_indices.iter().map(|&d| q_start[d]).collect();
        let mut q_cur_full = q_start.to_vec();
        let mut times = vec![0.0_f64];
        let mut qs = vec![arm_init.clone()];
        let mut prev_t = 0.0_f64;
        let mut prev_pos: Option<Vector3<f64>> = {
            if workcell.model.forward_kinematics(&q_cur_full, &mut tf_buf).is_ok() {
                Some(tf_buf.frame_pose(arm.tcp_frame).translation.vector)
            } else {
                None
            }
        };

        for wp in resolved.executable_waypoints_for_robot(&arm.uid) {
            if let Some(ref seed) = wp.joint_seed {
                for (&d, &v) in arm.handle.q_indices.iter().zip(seed.iter()) {
                    q_cur_full[d] = limits[d].position.clamp(v);
                }
            }

            let target = [TaskPose::new(wp.tcp_frame_id, wp.world_pose)];
            let solved = ik_solver
                .solve_collision_free_masked(
                    &CpuSimdBatchBackend,
                    &workcell.model,
                    &workcell.flat_model,
                    &workcell.env,
                    Some(&workcell.esdf),
                    &q_cur_full,
                    Some(&mask),
                    &target,
                )
                .or_else(|_| {
                    workcell.model.solve_ik_dls_masked(
                        &target,
                        &q_cur_full,
                        Some(&mask),
                        90,
                        2e-3,
                        0.05,
                        &mut tf_buf,
                    )
                });

            if let Ok(cfg) = solved {
                q_cur_full = cfg.0;
                clamp_full_q(&mut q_cur_full);
            }

            if workcell.model.forward_kinematics(&q_cur_full, &mut tf_buf).is_ok() {
                let actual = tf_buf.frame_pose(wp.tcp_frame_id);
                let dp = (wp.world_pose.translation.vector - actual.translation.vector).norm();
                let dr = (wp.world_pose.rotation * actual.rotation.inverse()).angle();
                max_pos_err_m = max_pos_err_m.max(dp);
                max_rot_err_rad = max_rot_err_rad.max(dr);
            }

            let dt_seg = if let Some(t_abs) = wp.time_s {
                (t_abs - prev_t).max(0.25)
            } else if let Some(p_prev) = prev_pos {
                let dist = (wp.world_pose.translation.vector - p_prev).norm();
                (dist / 0.25).clamp(0.35, 1.5)
            } else {
                0.5
            };
            prev_t += dt_seg;
            prev_pos = Some(wp.world_pose.translation.vector);
            times.push(prev_t);
            qs.push(arm.handle.q_indices.iter().map(|&d| q_cur_full[d]).collect());
        }

        max_duration = max_duration.max(prev_t);
        schedules.push(ArmWaypointKeyframes { times, qs });
    }

    if max_duration <= 0.0 {
        return Err("No enabled executable waypoints in imported trajectory set".to_string());
    }

    let dt = 0.025_f64;
    let num_steps = ((max_duration / dt).ceil() as usize).clamp(16, 240);
    let eff_dt = max_duration / (num_steps as f64);

    let mut servo_cfg = ReactiveServoConfig {
        safety_margin: spec.rules.default_body_margin.max(0.004),
        cbf_activation_distance: spec.rules.default_body_activation.max(0.04),
        ..Default::default()
    };
    for arm in &workcell.arms {
        servo_cfg = servo_cfg.with_arm_priority(arm.robot_id, arm.priority);
    }
    let mut servo = ReactiveServoController::new(&workcell.model, &workcell.env, servo_cfg);

    let mut ws = workcell.env.create_workspace(&workcell.model);
    let mut frames = Vec::with_capacity(num_steps + 1);
    let mut path_configs = Vec::with_capacity(num_steps + 1);

    let mut q_cur = q_start.to_vec();
    let mut dq_cur = vec![0.0_f64; total_dof];
    let mut dq_cmd = vec![0.0_f64; total_dof];
    let mut q_ref = vec![0.0_f64; total_dof];
    let mut dq_ref = vec![0.0_f64; total_dof];

    for step in 0..=num_steps {
        let t = (step as f64) * eff_dt;
        for (arm_idx, arm) in workcell.arms.iter().enumerate() {
            sample_arm_schedule(&schedules[arm_idx], t, &arm.handle.q_indices, &mut q_ref, &mut dq_ref);
        }

        if apply_cbf_filter && step > 0 {
            if servo
                .step_joint_reference(
                    &workcell.model,
                    &workcell.env,
                    &q_cur,
                    &dq_cur,
                    &q_ref,
                    &dq_ref,
                    8.0,
                    None,
                    &mut dq_cmd,
                )
                .is_ok()
            {
                for d in 0..total_dof {
                    let v_max = limits[d].max_velocity;
                    dq_cur[d] = dq_cmd[d].clamp(-v_max, v_max);
                    q_cur[d] = limits[d].position.clamp(q_cur[d] + dq_cur[d] * eff_dt);
                }
            } else {
                q_cur.copy_from_slice(&q_ref);
                dq_cur.copy_from_slice(&dq_ref);
            }
        } else {
            q_cur.copy_from_slice(&q_ref);
            dq_cur.copy_from_slice(&dq_ref);
        }

        path_configs.push(Configuration::new(q_cur.clone()));
        let frame = build_trajectory_frame(
            workcell,
            &workcell.env,
            t,
            &q_cur,
            &dq_cur,
            Vec::new(),
            0.0,
            0.0,
            TrajectoryStage::SingleStage,
            &mut ws,
        );
        frames.push(frame);
    }

    let tcp_trails = compute_tcp_trails(workcell, &frames);
    let convex_verified = verify_path_convex(workcell, &path_configs, 0.04, &mut ws);
    let mut metrics = summarize_metrics(
        &frames,
        t0.elapsed().as_secs_f64() * 1000.0,
        max_pos_err_m * 1000.0,
        max_rot_err_rad.to_degrees(),
        None,
        convex_verified,
    );
    metrics.backend_label = if apply_cbf_filter {
        "Imported + QP-CBF"
    } else {
        "Imported + Masked IK"
    };

    Ok(PlaybackTrajectory {
        workcell_revision: workcell.revision,
        frames,
        tcp_trails,
        rollout_ribbons: Vec::new(),
        metrics,
    })
}

fn sample_arm_schedule(
    sched: &ArmWaypointKeyframes,
    t: f64,
    q_indices: &[usize],
    out_q: &mut [f64],
    out_dq: &mut [f64],
) {
    if sched.qs.is_empty() {
        return;
    }
    if sched.qs.len() == 1 || t <= sched.times[0] {
        for (&d, &val) in q_indices.iter().zip(sched.qs[0].iter()) {
            out_q[d] = val;
            out_dq[d] = 0.0;
        }
        return;
    }
    let last_idx = sched.qs.len() - 1;
    if t >= sched.times[last_idx] {
        for (&d, &val) in q_indices.iter().zip(sched.qs[last_idx].iter()) {
            out_q[d] = val;
            out_dq[d] = 0.0;
        }
        return;
    }

    let seg = sched
        .times
        .windows(2)
        .position(|w| t >= w[0] && t <= w[1])
        .unwrap_or(last_idx - 1);
    let t0 = sched.times[seg];
    let t1 = sched.times[seg + 1];
    let span = (t1 - t0).max(1e-6);
    let s = ((t - t0) / span).clamp(0.0, 1.0);
    // Cubic smoothstep s^2 * (3 - 2s) with zero endpoint velocity at each waypoint
    let h = s * s * (3.0 - 2.0 * s);
    let dh_dt = (6.0 * s * (1.0 - s)) / span;

    for (j, &d) in q_indices.iter().enumerate() {
        let q0 = sched.qs[seg][j];
        let q1 = sched.qs[seg + 1][j];
        out_q[d] = q0 + h * (q1 - q0);
        out_dq[d] = dh_dt * (q1 - q0);
    }
}

fn pose_to_io(p: &PoseSpec) -> IoPoseSpec {
    IoPoseSpec {
        xyz: p.xyz,
        rpy: p.rpy,
    }
}

fn pose_from_io(p: &IoPoseSpec) -> PoseSpec {
    PoseSpec {
        xyz: p.xyz,
        rpy: p.rpy,
    }
}

fn shape_to_io(s: &ShapePrimitiveSpec) -> ShapeSpec {
    match *s {
        ShapePrimitiveSpec::Sphere { radius } => ShapeSpec::Sphere { radius },
        ShapePrimitiveSpec::Cuboid { half_extents } => ShapeSpec::Box { half_extents },
        ShapePrimitiveSpec::CylinderZ {
            half_height,
            radius,
        } => ShapeSpec::Cylinder {
            radius,
            half_height,
        },
        ShapePrimitiveSpec::CapsuleZ {
            half_height,
            radius,
        } => ShapeSpec::Capsule {
            radius,
            half_height,
        },
    }
}

fn shape_from_io(s: &ShapeSpec) -> (Isometry3<f64>, ShapePrimitiveSpec) {
    match *s {
        ShapeSpec::Sphere { radius } => (Isometry3::identity(), ShapePrimitiveSpec::Sphere { radius }),
        ShapeSpec::Box { half_extents } => {
            (Isometry3::identity(), ShapePrimitiveSpec::Cuboid { half_extents })
        }
        ShapeSpec::Cylinder {
            radius,
            half_height,
        } => (
            Isometry3::identity(),
            ShapePrimitiveSpec::CylinderZ {
                half_height,
                radius,
            },
        ),
        ShapeSpec::Capsule {
            radius,
            half_height,
        } => (
            Isometry3::identity(),
            ShapePrimitiveSpec::CapsuleZ {
                half_height,
                radius,
            },
        ),
        ShapeSpec::SegmentCapsule { a, b, radius } => {
            let va = Vector3::from(a);
            let vb = Vector3::from(b);
            let delta = vb - va;
            let mid = 0.5 * (va + vb);
            let len = delta.norm();
            let rot = if len > 1e-9 {
                UnitQuaternion::rotation_between(&Vector3::z(), &delta).unwrap_or_else(UnitQuaternion::identity)
            } else {
                UnitQuaternion::identity()
            };
            (
                Isometry3::from_parts(Translation3::from(mid), rot),
                ShapePrimitiveSpec::CapsuleZ {
                    half_height: 0.5 * len,
                    radius,
                },
            )
        }
    }
}

fn entity_ref_to_io(e: &EntityRef) -> EntityRefSpec {
    match e {
        EntityRef::Link(s) => EntityRefSpec::Link(s.clone()),
        EntityRef::Payload(s) => EntityRefSpec::Payload(s.clone()),
        EntityRef::Obstacle(s) => EntityRefSpec::Obstacle(s.clone()),
    }
}

fn entity_ref_from_io(e: &EntityRefSpec) -> EntityRef {
    match e {
        EntityRefSpec::Link(s) => EntityRef::Link(s.clone()),
        EntityRefSpec::Payload(s) => EntityRef::Payload(s.clone()),
        EntityRefSpec::Obstacle(s) => EntityRef::Obstacle(s.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{preset_workcell, PresetId};

    #[test]
    fn workcell_spec_world_spec_roundtrip_compiles() {
        let orig = preset_workcell(PresetId::DualArmMppiEsdf);
        let io_world = workcell_spec_to_world_spec(&orig);
        let toml_text = io_world.to_toml_string().expect("serialize world to TOML");
        let parsed = WorldSpec::from_toml_str(&toml_text).expect("parse world TOML");
        let rt_workcell_spec = world_spec_to_workcell_spec(&parsed);
        let compiled = rt_workcell_spec
            .compile(1)
            .expect("compile round-tripped workcell");

        assert_eq!(compiled.arms.len(), orig.arms.len());
        assert_eq!(compiled.obstacles.len(), orig.obstacles.len());
        assert!(compiled.model.dof() > 0);
    }

    #[test]
    fn sample_imported_trajectories_evaluate_with_and_without_cbf() {
        let spec = preset_workcell(PresetId::DualArmMppiEsdf);
        let workcell = spec.compile(1).expect("compile dual-arm preset");

        let sample = sample_trajectory_set_for_workcell(&spec, &workcell, workcell.q_start.as_slice());
        let toml_str = sample
            .to_toml_string()
            .expect("serialize sample trajectory set to TOML");
        let reparsed = TrajectorySetSpec::from_toml_str(&toml_str).expect("reparse TOML");

        let resolved =
            resolve_trajectory_set_for_workcell(&reparsed, &spec, &workcell, workcell.q_start.as_slice())
                .expect("resolve sample trajectories");

        let traj_raw = evaluate_imported_trajectory_set(
            &spec,
            &workcell,
            &resolved,
            workcell.q_start.as_slice(),
            false,
        )
        .expect("evaluate raw imported IK trajectory");
        assert!(traj_raw.frames.len() > 5);
        assert!(traj_raw.metrics.min_clearance_mm.is_finite());

        let traj_cbf = evaluate_imported_trajectory_set(
            &spec,
            &workcell,
            &resolved,
            workcell.q_start.as_slice(),
            true,
        )
        .expect("evaluate CBF-filtered imported trajectory");
        assert!(traj_cbf.frames.len() > 5);
        assert!(traj_cbf.metrics.min_clearance_mm.is_finite());
        assert_eq!(traj_cbf.metrics.backend_label, "Imported + QP-CBF");
    }
}
