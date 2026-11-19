use approx::assert_relative_eq;
use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
use parry3d::math::Vector as ParryVector;
use parry3d::shape::{Capsule, Cuboid, Cylinder, SharedShape};
use std::f64::consts::{FRAC_PI_2, PI};
use unrust_it::{
    ApproachTarget, AttachedPayload, BatchIkSolver, CoarsePlannerMode, CollidingPair,
    CollisionEnvironment, Configuration, CpuSimdBatchBackend, EsdfGridF32, FrameId, GeometryTier,
    LinkId, MppiConfig, MultiRobotModel, ReactiveServoConfig, ReactiveServoController, RobotId,
    ServoTarget, TaskPose, TwoStageApproachPlanner, TwoStagePlanConfig,
};
use unrust_it_urdf::presets::{ARM_6DOF_SRDF, ARM_6DOF_URDF};
use unrust_it_urdf::{add_urdf_to_scene, apply_srdf_str, UrdfRobotHandle};

struct ThreeArmCell {
    model: MultiRobotModel,
    env: CollisionEnvironment,
    esdf: EsdfGridF32,
    tcp_frames: [FrameId; 3],
    handles: [UrdfRobotHandle; 3],
    workpiece_id: Option<unrust_it::ObstacleId>,
}

fn pose_from_z_approach(tip: Vector3<f64>, z_axis: Vector3<f64>) -> Isometry3<f64> {
    let z = z_axis.normalize();
    let ref_up = if z.z.abs() < 0.9 {
        Vector3::z()
    } else {
        Vector3::x()
    };
    let x = ref_up.cross(&z).normalize();
    let y = z.cross(&x).normalize();
    let rot_mat = nalgebra::Matrix3::from_columns(&[x, y, z]);
    let rot = UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(
        rot_mat,
    ));
    Isometry3::from_parts(Translation3::from(tip), rot)
}

fn build_three_arm_tool_workcell(with_workpiece: bool) -> ThreeArmCell {
    let mut model = MultiRobotModel::new();
    let mut env = CollisionEnvironment::new();

    let r_base = 0.56;
    let angles = [0.0_f64, 2.0 * PI / 3.0, 4.0 * PI / 3.0];
    let prefixes = ["arm0_", "arm1_", "arm2_"];
    let mut handles_vec = Vec::with_capacity(3);

    for i in 0..3 {
        let th = angles[i];
        let pos = Vector3::new(r_base * th.cos(), r_base * th.sin(), 0.0);
        let yaw = th + PI; // face inward toward workspace origin
        let base_pose = Isometry3::from_parts(
            Translation3::from(pos),
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw),
        );
        let handle = add_urdf_to_scene(
            &mut model,
            &mut env,
            ARM_6DOF_URDF,
            RobotId(i as u16),
            prefixes[i],
            base_pose,
        )
        .expect("URDF should load");
        apply_srdf_str(&mut model, &mut env, ARM_6DOF_SRDF, prefixes[i])
            .expect("SRDF should apply");
        handles_vec.push(handle);
    }
    let handles: [UrdfRobotHandle; 3] = handles_vec.try_into().unwrap();

    // Per-pair body safety margin: 40 mm between arms
    let mut acm = env.acm().expect("ACM initialized by URDF").clone();
    acm.set_inter_arm_margin(&model, 0.04);
    env.set_acm(acm);

    // Optional central workpiece + pedestal obstacles + matching ESDF
    let mut esdf = EsdfGridF32::new([-0.8, -0.8, 0.0], 0.025, [64, 64, 48], 1.0);
    let workpiece_id = if with_workpiece {
        // Pedestal from z=0.0 to z=0.39 (parry3d Cylinder is Y-aligned -> rotate +pi/2 around X)
        let z_align = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), FRAC_PI_2);
        env.add_obstacle(
            SharedShape::new(Cylinder::new(0.195, 0.032)),
            Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.195), z_align),
        );
        esdf.add_aabb_obstacle([-0.032, -0.032, 0.0], [0.032, 0.032, 0.39]);

        // Workpiece: 80 x 80 x 50 mm cuboid centered at (0, 0, 0.42), top at z=0.445, +X face at x=0.040
        let wp_id = env.add_obstacle(
            SharedShape::new(Cuboid::new(ParryVector::new(0.04, 0.04, 0.025))),
            Isometry3::translation(0.0, 0.0, 0.42),
        );
        env.set_obstacle_margin(wp_id, 0.003);
        env.set_obstacle_activation(wp_id, 0.02);
        esdf.add_aabb_obstacle([-0.04, -0.04, 0.395], [0.04, 0.04, 0.445]);
        Some(wp_id)
    } else {
        None
    };

    // Attach 3 distinct compound tools on each arm's flange_link
    let z_align = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), FRAC_PI_2);
    let mut tcp_frames_vec = Vec::with_capacity(3);
    let payload_ids = ["arm0_gripper", "arm1_screwdriver", "arm2_probe"];

    for (i, prefix) in prefixes.iter().enumerate() {
        let flange_id = model.link_by_name(&format!("{prefix}flange_link")).unwrap();
        let wrist2_id = model.link_by_name(&format!("{prefix}wrist2_link")).unwrap();
        let wrist1_id = model.link_by_name(&format!("{prefix}wrist1_link")).unwrap();
        let flange_frame = model.link(flange_id).frame;
        let touch = vec![flange_id, wrist2_id, wrist1_id];

        let (payload, tcp_z_in_flange) = match i {
            0 => {
                // Parallel gripper: palm + 2 slim fingers (total tool length = 0.074 past mount z=0.03)
                let p = AttachedPayload::from_compound(
                    payload_ids[0],
                    flange_id,
                    Isometry3::translation(0.0, 0.0, 0.03),
                    vec![
                        (
                            Isometry3::translation(0.0, 0.0, 0.012),
                            SharedShape::new(Cuboid::new(ParryVector::new(0.018, 0.026, 0.012))),
                        ),
                        (
                            Isometry3::translation(0.0, 0.020, 0.049),
                            SharedShape::new(Cuboid::new(ParryVector::new(0.005, 0.004, 0.025))),
                        ),
                        (
                            Isometry3::translation(0.0, -0.020, 0.049),
                            SharedShape::new(Cuboid::new(ParryVector::new(0.005, 0.004, 0.025))),
                        ),
                    ],
                )
                .with_touch_links(touch)
                .with_safety_margin(0.004)
                .with_activation_distance(0.015);
                (p, 0.104)
            }
            1 => {
                // Precision screwdriver: housing cylinder + narrow 5mm capsule shaft (tip surface at z=0.143)
                let p = AttachedPayload::from_compound(
                    payload_ids[1],
                    flange_id,
                    Isometry3::translation(0.0, 0.0, 0.03),
                    vec![
                        (
                            Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.022), z_align),
                            SharedShape::new(Cylinder::new(0.022, 0.015)),
                        ),
                        (
                            Isometry3::translation(0.0, 0.0, 0.076),
                            SharedShape::new(Capsule::new_z(0.032, 0.005)),
                        ),
                    ],
                )
                .with_touch_links(touch)
                .with_safety_margin(0.004)
                .with_activation_distance(0.015);
                (p, 0.143)
            }
            _ => {
                // Inspection probe: compact body + slim 4mm capsule stylus (tip surface at z=0.138)
                let p = AttachedPayload::from_compound(
                    payload_ids[2],
                    flange_id,
                    Isometry3::translation(0.0, 0.0, 0.03),
                    vec![
                        (
                            Isometry3::translation(0.0, 0.0, 0.016),
                            SharedShape::new(Cuboid::new(ParryVector::new(0.013, 0.013, 0.016))),
                        ),
                        (
                            Isometry3::translation(0.0, 0.0, 0.068),
                            SharedShape::new(Capsule::new_z(0.036, 0.004)),
                        ),
                    ],
                )
                .with_touch_links(touch)
                .with_safety_margin(0.004)
                .with_activation_distance(0.015);
                (p, 0.138)
            }
        };

        env.attach_payload(payload);
        let tcp_frame = model
            .add_fixed_frame(
                format!("{prefix}tcp"),
                flange_frame,
                Isometry3::translation(0.0, 0.0, tcp_z_in_flange),
            )
            .unwrap();
        tcp_frames_vec.push(tcp_frame);
    }

    // Configure pair-specific safety margins and activation distances:
    // - Tool <-> Tool: 4 mm margin, 15 mm activation
    // - Tool <-> Workpiece: 3 mm margin, 15 mm activation
    // - Tool <-> Other arm links: 15 mm margin, 60 mm activation
    for a in 0..3 {
        for b in (a + 1)..3 {
            env.set_payload_pair_margin(payload_ids[a], payload_ids[b], 0.004);
            env.set_payload_pair_activation(payload_ids[a], payload_ids[b], 0.015);
        }
        if let Some(wp) = workpiece_id {
            env.set_payload_obstacle_margin(payload_ids[a], wp, 0.003);
            env.set_payload_obstacle_activation(payload_ids[a], wp, 0.015);
        }
        for link in model.links() {
            if link.robot != RobotId(a as u16) {
                env.set_payload_link_margin(payload_ids[a], link.id, 0.015);
                env.set_payload_link_activation(payload_ids[a], link.id, 0.06);
            }
        }
    }

    // Also set larger activation distance for inter-arm body links
    for a in 0..model.num_links() {
        for b in (a + 1)..model.num_links() {
            let la = LinkId(a as u16);
            let lb = LinkId(b as u16);
            if model.link(la).robot != model.link(lb).robot {
                env.set_link_pair_activation(la, lb, 0.16);
            }
        }
    }

    ThreeArmCell {
        model,
        env,
        esdf,
        tcp_frames: tcp_frames_vec.try_into().unwrap(),
        handles,
        workpiece_id,
    }
}

#[test]
fn three_arm_urdf_cell_two_stage_close_proximity_tool_convergence() {
    let cell = build_three_arm_tool_workcell(true);
    let model = &cell.model;
    let env = &cell.env;
    let _ = &cell.handles;

    assert_eq!(model.dof(), 18);
    assert_eq!(env.attached_payloads().len(), 3);

    // Symmetric open ready posture for all 3 arms (elbows bent back, tips well clear of center)
    let arm_ready = [0.0, -0.35, 1.35, 0.0, 0.70, 0.0];
    let mut q_start_vec = Vec::with_capacity(18);
    for _ in 0..3 {
        q_start_vec.extend_from_slice(&arm_ready);
    }
    let q_start = Configuration::new(q_start_vec);

    let mut ws = env.create_workspace(model);
    assert!(
        env.is_state_collision_free(model, q_start.as_slice(), GeometryTier::Convex, &mut ws)
            .unwrap(),
        "initial 3-arm posture must be collision-free"
    );

    // Final tight-proximity 6-DOF targets around the central workpiece (80x80x50 mm at z=0.42):
    // - Arm 0 gripper approaches +X face at ~12 mm gap
    // - Arm 1 screwdriver approaches top face at ~12 mm gap
    // - Arm 2 probe approaches top face at ~12 mm gap, ~13.8 mm surface-to-surface from screwdriver
    let targets = [
        ApproachTarget::with_standoff(
            TaskPose::new(
                cell.tcp_frames[0],
                pose_from_z_approach(
                    Vector3::new(0.052, 0.0, 0.438),
                    Vector3::new(-0.85, 0.0, -0.52),
                ),
            ),
            Vector3::new(0.0, 0.0, -0.06),
        ),
        ApproachTarget::with_standoff(
            TaskPose::new(
                cell.tcp_frames[1],
                pose_from_z_approach(
                    Vector3::new(0.0, 0.0095, 0.457),
                    Vector3::new(0.22, -0.38, -0.90),
                ),
            ),
            Vector3::new(0.0, 0.0, -0.06),
        ),
        ApproachTarget::with_standoff(
            TaskPose::new(
                cell.tcp_frames[2],
                pose_from_z_approach(
                    Vector3::new(0.0, -0.0095, 0.457),
                    Vector3::new(0.22, 0.38, -0.90),
                ),
            ),
            Vector3::new(0.0, 0.0, -0.06),
        ),
    ];

    let mut servo_cfg = ReactiveServoConfig {
        dt: 0.01,
        safety_margin: 0.04,
        cbf_activation_distance: 0.16,
        cbf_gamma: 8.0,
        damping: 0.02,
        posture_gain: 0.05,
        tangential_curl_gain: 0.35,
        ..ReactiveServoConfig::default()
    }
    .with_arm_priority(RobotId(0), 3.0)
    .with_arm_priority(RobotId(1), 2.0)
    .with_arm_priority(RobotId(2), 1.0);
    servo_cfg.qp_max_iterations = 35;

    let mut ik_solver = BatchIkSolver::new(48);
    ik_solver.max_iterations = 120;
    ik_solver.tolerance = 5e-4;

    let plan_cfg = TwoStagePlanConfig {
        coarse_mode: CoarsePlannerMode::MppiWithRrtFallback(MppiConfig {
            num_rollouts: 48,
            horizon: 16,
            num_iterations: 20,
            noise_std: 0.20,
            temperature: 0.12,
            step_size: 0.75,
            activation_distance: 0.08,
            collision_weight: 160.0,
            velocity_weight: 2.0,
            acceleration_weight: 5.0,
        }),
        ik_solver,
        rrt_max_workspace_step: 0.025,
        servo_config: servo_cfg,
        approach_pose_gain: 6.0,
        max_refinement_steps: 450,
        final_position_tolerance: 1.8e-3,
        final_orientation_tolerance: 1.8e-2,
        waypoint_stride: 4,
        ..TwoStagePlanConfig::default()
    };

    let planner = TwoStageApproachPlanner::new(plan_cfg);
    let result = planner
        .plan(
            &CpuSimdBatchBackend,
            model,
            env,
            Some(&cell.esdf),
            q_start,
            &targets,
        )
        .expect("3-arm two-stage planner should converge cleanly to tight proximity");

    assert!(
        result.max_position_error < 2.0e-3,
        "expected < 2 mm position error, got {:.4} m",
        result.max_position_error
    );
    assert!(
        result.max_orientation_error < 2.0e-2,
        "expected < 0.02 rad orientation error, got {:.4} rad",
        result.max_orientation_error
    );

    // Validate EVERY waypoint and segment in full_path is strictly collision-free in GeometryTier::Convex
    for (idx, cfg) in result.full_path.iter().enumerate() {
        assert!(
            env.is_state_collision_free(model, cfg.as_slice(), GeometryTier::Convex, &mut ws)
                .unwrap(),
            "waypoint {idx} in full_path must be collision-free"
        );
    }
    for (idx, seg) in result.full_path.windows(2).enumerate() {
        assert!(
            env.is_segment_collision_free(
                model,
                seg[0].as_slice(),
                seg[1].as_slice(),
                0.02,
                GeometryTier::Convex,
                &mut ws,
            )
            .unwrap(),
            "segment {idx} in full_path must be continuously collision-free"
        );
    }

    // Verify exact final signed distances for tool-to-workpiece, tool-to-tool, and body-to-body pairs
    let final_q = result.full_path.last().unwrap().as_slice();
    let mut final_witnesses = Vec::new();
    env.query_signed_distances(
        model,
        final_q,
        0.5,
        GeometryTier::Convex,
        &mut ws,
        &mut final_witnesses,
    )
    .unwrap();

    let wp_id = cell.workpiece_id.unwrap();
    let min_tool_workpiece = final_witnesses
        .iter()
        .filter(|w| {
            matches!(
                w.pair,
                CollidingPair::PayloadToObstacle { obstacle, .. } if obstacle == wp_id
            )
        })
        .map(|w| w.signed_distance)
        .fold(f64::INFINITY, f64::min);

    let min_tool_tool = final_witnesses
        .iter()
        .filter(|w| matches!(w.pair, CollidingPair::PayloadToPayload { .. }))
        .map(|w| w.signed_distance)
        .fold(f64::INFINITY, f64::min);

    let min_inter_arm_body = final_witnesses
        .iter()
        .filter(|w| match w.pair {
            CollidingPair::LinkToLink { link_a, link_b } => {
                model.link(link_a).robot != model.link(link_b).robot
            }
            _ => false,
        })
        .map(|w| w.signed_distance)
        .fold(f64::INFINITY, f64::min);

    assert!(
        min_tool_workpiece > 0.003 && min_tool_workpiece <= 0.016,
        "expected tool-to-workpiece clearance in (3mm, 16mm], got {min_tool_workpiece:.4} m"
    );
    assert!(
        min_tool_tool > 0.004 && min_tool_tool <= 0.022,
        "expected tool-to-tool clearance in (4mm, 22mm], got {min_tool_tool:.4} m"
    );
    assert!(
        min_inter_arm_body >= 0.04,
        "expected inter-arm body clearance >= 40mm, got {min_inter_arm_body:.4} m"
    );
}

#[test]
fn three_arm_symmetric_crossing_deadlock_free_cbf() {
    let cell = build_three_arm_tool_workcell(false);
    let model = &cell.model;
    let env = &cell.env;

    // Solve clean collision-free start & crossing goal poses where all 3 arms pass through a tight
    // central zone (< 80 mm) while remaining reachable and satisfying 40 mm body & 4 mm tool margins.
    let start_tips = [
        Vector3::new(0.07, -0.04, 0.46),
        Vector3::new(-0.02, 0.08, 0.46),
        Vector3::new(-0.06, -0.06, 0.46),
    ];
    let goal_tips = [
        Vector3::new(0.01, 0.06, 0.40),
        Vector3::new(0.04, -0.03, 0.48),
        Vector3::new(-0.06, 0.02, 0.44),
    ];
    let sector_angles = [0.0_f64, 2.0 * PI / 3.0, 4.0 * PI / 3.0];
    let mut start_poses = Vec::with_capacity(3);
    let mut goal_poses = Vec::with_capacity(3);

    for i in 0..3 {
        let sector = sector_angles[i];
        let z_inward = Vector3::new(-0.42 * sector.cos(), -0.42 * sector.sin(), -0.91);
        start_poses.push(TaskPose::new(
            cell.tcp_frames[i],
            pose_from_z_approach(start_tips[i], z_inward),
        ));
        goal_poses.push(TaskPose::new(
            cell.tcp_frames[i],
            pose_from_z_approach(goal_tips[i], z_inward),
        ));
    }

    let arm_ready = [0.0, -0.35, 1.35, 0.0, 0.70, 0.0];
    let mut q_seed = Vec::with_capacity(18);
    for _ in 0..3 {
        q_seed.extend_from_slice(&arm_ready);
    }

    let flat = unrust_it::FlatRobotModelF32::compile(model, env);
    let mut ik = BatchIkSolver::new(48);
    ik.max_iterations = 120;
    let q_start = ik
        .solve_collision_free(&CpuSimdBatchBackend, model, &flat, env, None, &q_seed, &start_poses)
        .expect("start ring poses should have collision-free IK");
    let _q_goal_check = ik
        .solve_collision_free(&CpuSimdBatchBackend, model, &flat, env, None, &q_seed, &goal_poses)
        .expect("goal crossing poses should be feasible and collision-free");

    let cfg = ReactiveServoConfig {
        dt: 0.01,
        safety_margin: 0.04,
        cbf_activation_distance: 0.16,
        cbf_gamma: 7.0,
        damping: 0.025,
        posture_gain: 0.05,
        tangential_curl_gain: 0.55,
        qp_max_iterations: 35,
        cbf_margin_padding: 0.002,
        ..ReactiveServoConfig::default()
    }
    .with_arm_priority(RobotId(0), 3.0)
    .with_arm_priority(RobotId(1), 2.0)
    .with_arm_priority(RobotId(2), 1.0);

    let mut servo = ReactiveServoController::new(model, env, cfg);
    let servo_targets: Vec<ServoTarget> = goal_poses
        .iter()
        .map(|tp| ServoTarget::pose(tp.frame, tp.pose, 4.0))
        .collect();

    let mut q_curr = q_start.0;
    let mut qd_curr = vec![0.0_f64; 18];
    let mut dq_cmd = vec![0.0_f64; 18];
    let mut ws = env.create_workspace(model);
    let mut buf = model.create_transform_buffer();

    for step in 0..500 {
        servo
            .step(model, env, &q_curr, &qd_curr, &servo_targets, None, &mut dq_cmd)
            .unwrap();
        for k in 0..18 {
            q_curr[k] = model.dof_limits()[k]
                .position
                .clamp(q_curr[k] + dq_cmd[k] * 0.01);
            qd_curr[k] = dq_cmd[k];
        }
        if !env
            .is_state_collision_free(model, &q_curr, GeometryTier::Convex, &mut ws)
            .unwrap()
        {
            let mut wits = Vec::new();
            env.query_signed_distances(model, &q_curr, 0.2, GeometryTier::Convex, &mut ws, &mut wits)
                .unwrap();
            wits.sort_by(|a, b| {
                let da = a.signed_distance - env.effective_pair_margin(&a.pair, 0.0);
                let db = b.signed_distance - env.effective_pair_margin(&b.pair, 0.0);
                da.total_cmp(&db)
            });
            let worst = wits.first().map(|w| {
                (
                    w.pair,
                    w.signed_distance,
                    env.effective_pair_margin(&w.pair, 0.0),
                    env.effective_pair_activation(&w.pair, 0.16),
                )
            });
            panic!("3-arm crossing step {step} violated Convex collision freedom; worst={worst:?}");
        }
    }

    model.forward_kinematics(&q_curr, &mut buf).unwrap();
    for (i, goal) in goal_poses.iter().enumerate() {
        let curr = buf.frame_pose(goal.frame);
        let pos_err = (goal.pose.translation.vector - curr.translation.vector).norm();
        let rot_err = (goal.pose.rotation * curr.rotation.inverse()).angle();
        assert!(
            pos_err < 5.0e-3,
            "arm {i} position did not converge in 3-way crossing (pos_err={pos_err:.4} m)"
        );
        assert!(
            rot_err < 5.0e-2,
            "arm {i} orientation did not converge in 3-way crossing (rot_err={rot_err:.4} rad)"
        );
    }
    assert_relative_eq!(q_curr.len() as f64, 18.0);
}
