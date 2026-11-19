//! Planner execution, trajectory playback sampling, and zero-alloc frame snapshot evaluation.

use crate::scene::compile::BuiltWorkcell;
use crate::scene::spec::{ArmTargetSpec, ComputeBackendChoice, PlannerSpec, Stage1Strategy, WorkcellSpec};
use nalgebra::{Isometry3, SVector, Vector3};
use std::sync::Arc;
use std::time::Instant;
use unrust_it::{
    ApproachTarget, BatchIkSolver, BatchMppiOptimizer, BatchRolloutBackend,
    ClosedChainGraspConstraint, CollidingPair, CollisionEnvironment, CollisionWorkspace,
    Configuration, ConstraintManifold, CpuSimdBatchBackend, CubicSplinePath, GeometryTier,
    ObstacleId, PathPlanner, ReactiveServoController, Robot, RrtPlanner, ServoTarget,
    SignedDistanceWitness, Target, TaskPose, ToppraTrajectory, TwoStageApproachPlanner,
    TwoStagePlanConfig,
};

#[cfg(feature = "gpu-compute")]
use unrust_it::WgpuBatchBackend;

/// Stage annotation along a recorded trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrajectoryStage {
    SingleStage,
    Stage1Coarse,
    Stage2Refinement,
}

/// One recorded state frame in a [`PlaybackTrajectory`].
#[derive(Debug, Clone)]
pub struct TrajectoryFrame {
    pub time: f64,
    pub q: Vec<f64>,
    pub qd: Vec<f64>,
    pub dynamic_obstacles: Vec<(ObstacleId, Isometry3<f64>)>,
    pub min_clearance: f64,
    pub min_body_clearance: Option<f64>,
    pub min_tool_clearance: Option<f64>,
    pub min_tool_workpiece_clearance: Option<f64>,
    pub grasp_error_mm: f64,
    pub pose_error_mm: f64,
    pub stage: TrajectoryStage,
}

/// 3D TCP trail polyline for a candidate MPPI rollout.
#[derive(Debug, Clone)]
pub struct RolloutRibbon {
    pub arm_index: usize,
    pub points: Vec<[f32; 3]>,
    pub total_cost: f32,
    pub is_elite: bool,
}

/// Summary metrics for an executed planner run.
#[derive(Debug, Clone)]
pub struct PlanMetrics {
    pub solve_time_ms: f64,
    pub backend_label: &'static str,
    pub num_waypoints: usize,
    pub duration_s: f64,
    pub max_pos_error_mm: f64,
    pub max_rot_error_deg: f64,
    pub min_clearance_mm: f64,
    pub min_tool_clearance_mm: Option<f64>,
    pub stage1_end_time: Option<f64>,
    pub convex_verified: bool,
}

/// Complete multi-arm trajectory + rollout ribbons + diagnostics ready for 3D rendering & plots.
#[derive(Debug, Clone)]
pub struct PlaybackTrajectory {
    pub workcell_revision: u64,
    pub frames: Vec<TrajectoryFrame>,
    pub tcp_trails: Vec<Vec<[f32; 3]>>,
    pub rollout_ribbons: Vec<RolloutRibbon>,
    pub metrics: PlanMetrics,
}

/// Instantaneous kinematic & signed-distance snapshot evaluated for rendering or live UI inspection.
#[derive(Debug, Clone)]
pub struct FrameSnapshot {
    pub q: Vec<f64>,
    pub link_poses: Vec<Isometry3<f64>>,
    pub tcp_poses: Vec<Isometry3<f64>>,
    pub payload_poses: Vec<Isometry3<f64>>,
    pub obstacle_poses: Vec<(ObstacleId, Isometry3<f64>)>,
    pub witnesses: Vec<SignedDistanceWitness>,
    pub all_distances_below_cutoff: Vec<(CollidingPair, f64)>,
    pub min_clearance: f64,
}

/// Evaluates forward kinematics and `GeometryTier::Convex` signed-distance witnesses at `q`.
pub fn evaluate_frame_snapshot(
    workcell: &BuiltWorkcell,
    q: &[f64],
    dynamic_obstacles: &[(ObstacleId, Isometry3<f64>)],
    max_witnesses: usize,
    env_scratch: Option<&mut CollisionEnvironment>,
    ws: &mut CollisionWorkspace,
) -> FrameSnapshot {
    let mut temp_env;
    let env_ref: &CollisionEnvironment = if dynamic_obstacles.is_empty() {
        &workcell.env
    } else if let Some(scratch) = env_scratch {
        for &(id, pose) in dynamic_obstacles {
            scratch.set_obstacle_pose(id, pose);
        }
        scratch
    } else {
        temp_env = workcell.env.clone();
        for &(id, pose) in dynamic_obstacles {
            temp_env.set_obstacle_pose(id, pose);
        }
        &temp_env
    };

    let mut witnesses_raw = Vec::new();
    let _ = env_ref.query_signed_distances(
        &workcell.model,
        q,
        0.50,
        GeometryTier::Convex,
        ws,
        &mut witnesses_raw,
    );

    let link_poses = ws.transforms.link_poses().to_vec();
    let tcp_poses: Vec<Isometry3<f64>> = workcell
        .arms
        .iter()
        .map(|arm| *ws.transforms.frame_pose(arm.tcp_frame))
        .collect();
    let payload_poses: Vec<Isometry3<f64>> = env_ref
        .attached_payloads()
        .iter()
        .map(|p| ws.transforms.link_pose(p.parent_link) * p.local_pose)
        .collect();
    let obstacle_poses: Vec<(ObstacleId, Isometry3<f64>)> = env_ref
        .obstacles
        .iter()
        .enumerate()
        .map(|(idx, obs)| (ObstacleId(idx as u32), obs.pose))
        .collect();

    witnesses_raw.sort_by(|a, b| a.signed_distance.total_cmp(&b.signed_distance));
    let min_clearance = witnesses_raw
        .first()
        .map_or(0.5, |w| w.signed_distance);
    let all_distances_below_cutoff = witnesses_raw
        .iter()
        .map(|w| (w.pair, w.signed_distance))
        .collect();
    witnesses_raw.truncate(max_witnesses);

    FrameSnapshot {
        q: q.to_vec(),
        link_poses,
        tcp_poses,
        payload_poses,
        obstacle_poses,
        witnesses: witnesses_raw,
        all_distances_below_cutoff,
        min_clearance,
    }
}

/// Executes the planner configured in `spec` against `workcell`.
pub fn run_planner_sync(
    workcell: &BuiltWorkcell,
    spec: &WorkcellSpec,
) -> Result<PlaybackTrajectory, String> {
    let t_start = Instant::now();

    let (traj, backend_label) = match (&spec.planner, spec.compute_backend) {
        #[cfg(feature = "gpu-compute")]
        (
            PlannerSpec::TwoStageProximity { .. } | PlannerSpec::BatchMppiEsdf { .. },
            ComputeBackendChoice::AutoGpu,
        ) => {
            if workcell.model.num_links() <= 64 {
                if let Ok(gpu) = WgpuBatchBackend::new() {
                    if gpu.is_hardware_accelerated() {
                        (
                            execute_with_backend(&gpu, workcell, spec, t_start)?,
                            "wgpu GPU Compute",
                        )
                    } else {
                        (
                            execute_with_backend(&CpuSimdBatchBackend, workcell, spec, t_start)?,
                            "CPU SIMD f32x8",
                        )
                    }
                } else {
                    (
                        execute_with_backend(&CpuSimdBatchBackend, workcell, spec, t_start)?,
                        "CPU SIMD f32x8",
                    )
                }
            } else {
                (
                    execute_with_backend(&CpuSimdBatchBackend, workcell, spec, t_start)?,
                    "CPU SIMD f32x8 (>64 links fallback)",
                )
            }
        }
        _ => {
            let label = match spec.compute_backend {
                ComputeBackendChoice::CpuSimd => "CPU SIMD f32x8",
                ComputeBackendChoice::AutoGpu => "CPU SIMD f32x8",
            };
            (
                execute_with_backend(&CpuSimdBatchBackend, workcell, spec, t_start)?,
                label,
            )
        }
    };

    let mut final_traj = traj;
    final_traj.metrics.backend_label = backend_label;
    Ok(final_traj)
}

fn execute_with_backend<B: BatchRolloutBackend>(
    backend: &B,
    workcell: &BuiltWorkcell,
    spec: &WorkcellSpec,
    t_start: Instant,
) -> Result<PlaybackTrajectory, String> {
    match &spec.planner {
        PlannerSpec::TwoStageProximity {
            strategy,
            mppi,
            rrt_step_size,
            rrt_max_workspace_step,
            rrt_max_iterations,
            ik_num_seeds,
            ik_max_iterations,
            ik_tolerance,
            servo,
            approach_pose_gain,
            max_refinement_steps,
            final_pos_tol,
            final_rot_tol,
            waypoint_stride,
        } => {
            let targets = build_approach_targets(workcell, spec)?;
            let mut ik_solver = BatchIkSolver::new(*ik_num_seeds);
            ik_solver.max_iterations = *ik_max_iterations;
            ik_solver.tolerance = *ik_tolerance;

            let coarse_mode = match strategy {
                Stage1Strategy::MppiWithRrtFallback => {
                    unrust_it::CoarsePlannerMode::MppiWithRrtFallback(mppi.clone())
                }
                Stage1Strategy::MppiOnly => unrust_it::CoarsePlannerMode::Mppi(mppi.clone()),
                Stage1Strategy::RrtOnly => unrust_it::CoarsePlannerMode::RrtConnect,
            };

            let mut servo_cfg = servo.clone();
            for arm in &workcell.arms {
                servo_cfg = servo_cfg.with_arm_priority(arm.robot_id, arm.priority);
            }

            let cfg = TwoStagePlanConfig {
                coarse_mode,
                ik_solver,
                rrt_step_size: *rrt_step_size,
                rrt_max_workspace_step: *rrt_max_workspace_step,
                rrt_max_iterations: *rrt_max_iterations,
                servo_config: servo_cfg,
                approach_pose_gain: *approach_pose_gain,
                max_refinement_steps: *max_refinement_steps,
                final_position_tolerance: *final_pos_tol,
                final_orientation_tolerance: *final_rot_tol,
                waypoint_stride: (*waypoint_stride).max(1),
            };
            let res = TwoStageApproachPlanner::new(cfg)
                .plan(
                    backend,
                    &workcell.model,
                    &workcell.env,
                    Some(&workcell.esdf),
                    workcell.q_start.clone(),
                    &targets,
                )
                .map_err(|e| e.to_string())?;

            let s1_len = res.stage1_path.len();
            let s2_len = res.stage2_path.len();
            let mut frames = Vec::with_capacity(s1_len + s2_len);
            let mut ws = workcell.env.create_workspace(&workcell.model);
            let dt = 0.04;
            let mut t_acc = 0.0;

            let stage_iter = res
                .stage1_path
                .iter()
                .map(|c| (c, TrajectoryStage::Stage1Coarse))
                .chain(
                    res.stage2_path
                        .iter()
                        .skip(1)
                        .map(|c| (c, TrajectoryStage::Stage2Refinement)),
                );

            let combined: Vec<(Vec<f64>, TrajectoryStage)> = stage_iter
                .map(|(c, st)| (c.0.clone(), st))
                .collect();

            let mut stage1_end_time = None;
            for (idx, (q, st)) in combined.iter().enumerate() {
                if *st == TrajectoryStage::Stage2Refinement && stage1_end_time.is_none() {
                    stage1_end_time = Some(t_acc);
                }
                let qd = finite_diff_qd(&combined, idx, dt);
                let pose_err_mm =
                    compute_approach_pos_err_mm(workcell, q, &targets, &mut ws.transforms);
                frames.push(build_trajectory_frame(
                    workcell,
                    &workcell.env,
                    t_acc,
                    q,
                    &qd,
                    Vec::new(),
                    0.0,
                    pose_err_mm,
                    *st,
                    &mut ws,
                ));
                t_acc += dt;
            }

            let tcp_trails = compute_tcp_trails(workcell, &frames);
            let convex_verified = verify_path_convex(workcell, &res.full_path, *rrt_max_workspace_step, &mut ws);
            let metrics = summarize_metrics(
                &frames,
                t_start.elapsed().as_secs_f64() * 1000.0,
                res.max_position_error * 1000.0,
                res.max_orientation_error.to_degrees(),
                stage1_end_time,
                convex_verified,
            );
            Ok(PlaybackTrajectory {
                workcell_revision: workcell.revision,
                frames,
                tcp_trails,
                rollout_ribbons: Vec::new(),
                metrics,
            })
        }

        PlannerSpec::ThreeArmCrossing {
            start_poses,
            ik_num_seeds,
            servo,
            num_steps,
            record_stride,
        } => {
            let start_tasks: Vec<TaskPose> = workcell
                .arms
                .iter()
                .zip(start_poses.iter())
                .map(|(arm, p)| TaskPose::new(arm.tcp_frame, p.to_isometry()))
                .collect();
            let goal_tasks = build_direct_task_poses(workcell, spec)?;

            let mut ik = BatchIkSolver::new(*ik_num_seeds);
            ik.max_iterations = 120;
            let q_init = if start_tasks.len() == workcell.arms.len() && !start_tasks.is_empty() {
                ik.solve_collision_free(
                    backend,
                    &workcell.model,
                    &workcell.flat_model,
                    &workcell.env,
                    None,
                    workcell.q_start.as_slice(),
                    &start_tasks,
                )
                .map_err(|e| format!("Start IK failed: {e}"))?
            } else {
                workcell.q_start.clone()
            };

            let mut servo_cfg = servo.clone();
            for arm in &workcell.arms {
                servo_cfg = servo_cfg.with_arm_priority(arm.robot_id, arm.priority);
            }
            let mut controller =
                ReactiveServoController::new(&workcell.model, &workcell.env, servo_cfg.clone());
            let servo_targets: Vec<ServoTarget> = goal_tasks
                .iter()
                .map(|tp| ServoTarget::pose(tp.frame, tp.pose, 4.0))
                .collect();

            let dof = workcell.model.dof();
            let mut q_curr = q_init.0;
            let mut qd_curr = vec![0.0; dof];
            let mut dq_cmd = vec![0.0; dof];
            let mut ws = workcell.env.create_workspace(&workcell.model);
            let mut frames = Vec::new();
            let stride = (*record_stride).max(1);

            for step in 0..*num_steps {
                controller
                    .step(
                        &workcell.model,
                        &workcell.env,
                        &q_curr,
                        &qd_curr,
                        &servo_targets,
                        None,
                        &mut dq_cmd,
                    )
                    .map_err(|e| e.to_string())?;
                for k in 0..dof {
                    q_curr[k] = workcell.model.dof_limits()[k]
                        .position
                        .clamp(q_curr[k] + dq_cmd[k] * servo_cfg.dt);
                    qd_curr[k] = dq_cmd[k];
                }
                if step % stride == 0 || step + 1 == *num_steps {
                    let pose_err_mm = compute_task_pos_err_mm(
                        workcell,
                        &q_curr,
                        &goal_tasks,
                        &mut ws.transforms,
                    );
                    frames.push(build_trajectory_frame(
                        workcell,
                        &workcell.env,
                        (step + 1) as f64 * servo_cfg.dt,
                        &q_curr,
                        &qd_curr,
                        Vec::new(),
                        0.0,
                        pose_err_mm,
                        TrajectoryStage::SingleStage,
                        &mut ws,
                    ));
                }
            }

            let (max_pos_err_m, max_rot_err_rad) =
                compute_task_errors(workcell, &q_curr, &goal_tasks, &mut ws.transforms);
            let configs: Vec<Configuration> = frames
                .iter()
                .map(|f| Configuration::from_slice(&f.q))
                .collect();
            let convex_verified = verify_path_convex(workcell, &configs, 0.025, &mut ws);
            let tcp_trails = compute_tcp_trails(workcell, &frames);
            let metrics = summarize_metrics(
                &frames,
                t_start.elapsed().as_secs_f64() * 1000.0,
                max_pos_err_m * 1000.0,
                max_rot_err_rad.to_degrees(),
                None,
                convex_verified,
            );
            Ok(PlaybackTrajectory {
                workcell_revision: workcell.revision,
                frames,
                tcp_trails,
                rollout_ribbons: Vec::new(),
                metrics,
            })
        }

        PlannerSpec::RrtConnectToppra {
            step_size,
            max_workspace_step,
            max_iterations,
            seed,
            toppra_grid_points,
            num_playback_frames,
        } => {
            let goal_q = assemble_joint_goal(workcell, spec);
            let robot = Robot::from_model(Arc::new(workcell.model.clone()));
            let mut rrt = RrtPlanner::new(workcell.env.clone());
            rrt.step_size = *step_size;
            rrt.max_workspace_step = *max_workspace_step;
            rrt.max_iterations = *max_iterations;
            rrt.seed = *seed;
            let raw_wps = rrt
                .plan_path(&robot, workcell.q_start.clone(), Target::Joint(goal_q))
                .map_err(|e| e.to_string())?;

            let mut dense = Vec::new();
            for seg in raw_wps.windows(2) {
                for s in 0..4 {
                    let a = (s as f64) / 4.0;
                    let q_interp: Vec<f64> = seg[0]
                        .as_slice()
                        .iter()
                        .zip(seg[1].as_slice().iter())
                        .map(|(x, y)| (1.0 - a) * x + a * y)
                        .collect();
                    dense.push(Configuration::new(q_interp));
                }
            }
            if let Some(last) = raw_wps.last() {
                dense.push(last.clone());
            }

            let spline = CubicSplinePath::from_waypoints(&dense)
                .ok_or_else(|| "insufficient waypoints for RRT cubic spline".to_string())?;
            let toppra = ToppraTrajectory::parameterize(
                spline,
                workcell.model.dof_limits(),
                *toppra_grid_points,
            )
            .map_err(|e| e.to_string())?;

            let frames = sample_toppra_frames(workcell, &toppra, *num_playback_frames);
            let configs: Vec<Configuration> = frames
                .iter()
                .map(|f| Configuration::from_slice(&f.q))
                .collect();
            let mut ws = workcell.env.create_workspace(&workcell.model);
            let convex_verified = verify_path_convex(workcell, &configs, *max_workspace_step, &mut ws);
            let tcp_trails = compute_tcp_trails(workcell, &frames);
            let metrics = summarize_metrics(
                &frames,
                t_start.elapsed().as_secs_f64() * 1000.0,
                0.0,
                0.0,
                None,
                convex_verified,
            );
            Ok(PlaybackTrajectory {
                workcell_revision: workcell.revision,
                frames,
                tcp_trails,
                rollout_ribbons: Vec::new(),
                metrics,
            })
        }

        PlannerSpec::BatchMppiEsdf {
            mppi,
            num_visual_ribbons,
            toppra_grid_points,
            num_playback_frames,
        } => {
            let goal_q = assemble_joint_goal(workcell, spec);
            let optimizer = BatchMppiOptimizer::new(mppi.clone());
            let trace = optimizer.optimize_with_trace(
                backend,
                &workcell.model,
                &workcell.flat_model,
                Some(&workcell.esdf),
                workcell.q_start.as_slice(),
                goal_q.as_slice(),
            );

            // Convert authentic MPPI final candidate rollouts into 3D TCP ribbons
            let mut indexed: Vec<(usize, f64, f64)> = trace
                .rollout_costs
                .iter()
                .zip(trace.rollout_weights.iter())
                .enumerate()
                .map(|(idx, (&c, &w))| (idx, c, w))
                .collect();
            indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
            let take_count = (*num_visual_ribbons).min(indexed.len());

            let mut buf = workcell.model.create_transform_buffer();
            let mut rollout_ribbons = Vec::with_capacity(take_count * workcell.arms.len());
            for (rank, &(r_idx, cost, weight)) in indexed.iter().take(take_count).enumerate() {
                let is_elite = rank < 3 || weight > (2.0 / (trace.rollout_costs.len().max(1) as f64));
                let wps = &trace.final_rollouts[r_idx];
                for (arm_idx, arm) in workcell.arms.iter().enumerate() {
                    let mut pts = Vec::with_capacity(wps.len());
                    for wp in wps {
                        let _ = workcell.model.forward_kinematics(wp.as_slice(), &mut buf);
                        let p = buf.frame_pose(arm.tcp_frame).translation.vector;
                        pts.push([p.x as f32, p.y as f32, p.z as f32]);
                    }
                    rollout_ribbons.push(RolloutRibbon {
                        arm_index: arm_idx,
                        points: pts,
                        total_cost: cost as f32,
                        is_elite,
                    });
                }
            }

            let spline = CubicSplinePath::from_waypoints(&trace.nominal)
                .ok_or_else(|| "insufficient waypoints for MPPI cubic spline".to_string())?;
            let toppra = ToppraTrajectory::parameterize(
                spline,
                workcell.model.dof_limits(),
                *toppra_grid_points,
            )
            .map_err(|e| e.to_string())?;

            let frames = sample_toppra_frames(workcell, &toppra, *num_playback_frames);
            let configs: Vec<Configuration> = frames
                .iter()
                .map(|f| Configuration::from_slice(&f.q))
                .collect();
            let mut ws = workcell.env.create_workspace(&workcell.model);
            let convex_verified = verify_path_convex(workcell, &configs, 0.03, &mut ws);
            let tcp_trails = compute_tcp_trails(workcell, &frames);
            let metrics = summarize_metrics(
                &frames,
                t_start.elapsed().as_secs_f64() * 1000.0,
                0.0,
                0.0,
                None,
                convex_verified,
            );
            Ok(PlaybackTrajectory {
                workcell_revision: workcell.revision,
                frames,
                tcp_trails,
                rollout_ribbons,
                metrics,
            })
        }

        PlannerSpec::ClosedChainCbfServo {
            arm_a_uid,
            arm_b_uid,
            active_grasp_dims,
            servo,
            num_steps,
            linear_twist_amp,
        } => {
            let arm_a = workcell
                .arms
                .iter()
                .find(|a| a.uid == *arm_a_uid)
                .ok_or_else(|| format!("unknown closed-chain arm `{arm_a_uid}`"))?;
            let arm_b = workcell
                .arms
                .iter()
                .find(|a| a.uid == *arm_b_uid)
                .ok_or_else(|| format!("unknown closed-chain arm `{arm_b_uid}`"))?;

            let dof = workcell.model.dof();
            let mut q_curr = workcell.q_start.0.clone();
            let mut qd_curr = vec![0.0; dof];
            let mut dq_cmd = vec![0.0; dof];
            let mut env = workcell.env.clone();
            let mut ws = env.create_workspace(&workcell.model);

            workcell
                .model
                .forward_kinematics(&q_curr, &mut ws.transforms)
                .map_err(|e| e.to_string())?;
            let desired_rel = ws
                .transforms
                .relative_pose(arm_a.tcp_frame, arm_b.tcp_frame);
            let closure = ClosedChainGraspConstraint::with_active_dims(
                arm_a.tcp_frame,
                arm_b.tcp_frame,
                desired_rel,
                *active_grasp_dims,
            );

            let mut controller =
                ReactiveServoController::new(&workcell.model, &env, servo.clone());
            let mut frames = Vec::with_capacity(*num_steps);
            let n = (*num_steps).max(2);

            for i in 0..n {
                let phase = (i as f64) / ((n - 1) as f64);
                let t = (i as f64) * servo.dt;

                let mut dyn_obs = Vec::new();
                for obs in &workcell.obstacles {
                    if matches!(obs.role, crate::scene::spec::ObstacleRole::DynamicSweep { .. }) {
                        let pose = obs.role.evaluate_pose(
                            &crate::scene::spec::PoseSpec::from_isometry(&obs.base_pose),
                            phase,
                        );
                        env.set_obstacle_pose(obs.obstacle_id, pose);
                        dyn_obs.push((obs.obstacle_id, pose));
                    }
                }

                let vz = linear_twist_amp[2] * (std::f64::consts::TAU * phase).sin();
                let vx = linear_twist_amp[0] * (std::f64::consts::PI * phase).cos();
                let targets = [
                    ServoTarget::twist(arm_a.tcp_frame, SVector::<f64, 6>::new(vx, 0.0, vz, 0.0, 0.0, 0.0)),
                    ServoTarget::twist(arm_b.tcp_frame, SVector::<f64, 6>::new(vx, 0.0, vz, 0.0, 0.0, 0.0)),
                ];
                controller
                    .step(
                        &workcell.model,
                        &env,
                        &q_curr,
                        &qd_curr,
                        &targets,
                        Some(&closure),
                        &mut dq_cmd,
                    )
                    .map_err(|e| e.to_string())?;

                workcell
                    .model
                    .forward_kinematics(&q_curr, &mut ws.transforms)
                    .map_err(|e| e.to_string())?;
                let actual_rel = ws
                    .transforms
                    .relative_pose(arm_a.tcp_frame, arm_b.tcp_frame);
                let grasp_err_mm =
                    (actual_rel.translation.vector - desired_rel.translation.vector).norm() * 1000.0;

                frames.push(build_trajectory_frame(
                    workcell,
                    &env,
                    t,
                    &q_curr,
                    &dq_cmd,
                    dyn_obs,
                    grasp_err_mm,
                    grasp_err_mm,
                    TrajectoryStage::SingleStage,
                    &mut ws,
                ));

                for d in 0..dof {
                    q_curr[d] = workcell.model.dof_limits()[d]
                        .position
                        .clamp(q_curr[d] + dq_cmd[d] * servo.dt);
                    qd_curr[d] = dq_cmd[d];
                }
                let _ = closure.project(&workcell.model, &mut q_curr, &mut ws.transforms, 1e-6, 15);
            }

            let tcp_trails = compute_tcp_trails(workcell, &frames);
            let metrics = summarize_metrics(
                &frames,
                t_start.elapsed().as_secs_f64() * 1000.0,
                frames.last().map_or(0.0, |f| f.grasp_error_mm),
                0.0,
                None,
                true,
            );
            Ok(PlaybackTrajectory {
                workcell_revision: workcell.revision,
                frames,
                tcp_trails,
                rollout_ribbons: Vec::new(),
                metrics,
            })
        }

        PlannerSpec::InteractiveSandbox => {
            let mut ws = workcell.env.create_workspace(&workcell.model);
            let dof = workcell.model.dof();
            let frame = build_trajectory_frame(
                workcell,
                &workcell.env,
                0.0,
                workcell.q_start.as_slice(),
                &vec![0.0; dof],
                Vec::new(),
                0.0,
                0.0,
                TrajectoryStage::SingleStage,
                &mut ws,
            );
            let frames = vec![frame];
            let tcp_trails = compute_tcp_trails(workcell, &frames);
            let metrics = summarize_metrics(&frames, 0.0, 0.0, 0.0, None, true);
            Ok(PlaybackTrajectory {
                workcell_revision: workcell.revision,
                frames,
                tcp_trails,
                rollout_ribbons: Vec::new(),
                metrics,
            })
        }
    }
}

fn build_approach_targets(
    workcell: &BuiltWorkcell,
    spec: &WorkcellSpec,
) -> Result<Vec<ApproachTarget>, String> {
    let mut out = Vec::with_capacity(workcell.arms.len());
    for (arm_compiled, arm_spec) in workcell.arms.iter().zip(spec.arms.iter()) {
        match &arm_spec.target {
            ArmTargetSpec::ApproachPose {
                final_pose,
                standoff_local,
            } => {
                out.push(ApproachTarget::with_standoff(
                    TaskPose::new(arm_compiled.tcp_frame, final_pose.to_isometry()),
                    Vector3::new(standoff_local[0], standoff_local[1], standoff_local[2]),
                ));
            }
            ArmTargetSpec::DirectPose(pose) => {
                out.push(ApproachTarget::with_standoff(
                    TaskPose::new(arm_compiled.tcp_frame, pose.to_isometry()),
                    Vector3::new(0.0, 0.0, -0.05),
                ));
            }
            _ => {}
        }
    }
    if out.is_empty() {
        return Err("No Cartesian ApproachPose targets configured".to_string());
    }
    Ok(out)
}

fn build_direct_task_poses(
    workcell: &BuiltWorkcell,
    spec: &WorkcellSpec,
) -> Result<Vec<TaskPose>, String> {
    let mut out = Vec::with_capacity(workcell.arms.len());
    for (arm_compiled, arm_spec) in workcell.arms.iter().zip(spec.arms.iter()) {
        match &arm_spec.target {
            ArmTargetSpec::DirectPose(pose) => {
                out.push(TaskPose::new(arm_compiled.tcp_frame, pose.to_isometry()));
            }
            ArmTargetSpec::ApproachPose { final_pose, .. } => {
                out.push(TaskPose::new(
                    arm_compiled.tcp_frame,
                    final_pose.to_isometry(),
                ));
            }
            _ => {}
        }
    }
    if out.is_empty() {
        return Err("No Cartesian pose targets configured".to_string());
    }
    Ok(out)
}

fn assemble_joint_goal(workcell: &BuiltWorkcell, spec: &WorkcellSpec) -> Configuration {
    let mut goal = Vec::with_capacity(workcell.model.dof());
    for (arm_c, arm_s) in workcell.arms.iter().zip(spec.arms.iter()) {
        let len = arm_c.q_range.len();
        for i in 0..len {
            let v = match &arm_s.target {
                ArmTargetSpec::JointGoal(q) => q.get(i).copied().unwrap_or(0.0),
                _ => workcell.q_start.0[arm_c.q_range.start + i],
            };
            goal.push(
                workcell.model.dof_limits()[arm_c.q_range.start + i]
                    .position
                    .clamp(v),
            );
        }
    }
    Configuration::new(goal)
}

fn sample_toppra_frames(
    workcell: &BuiltWorkcell,
    toppra: &ToppraTrajectory,
    num_frames: usize,
) -> Vec<TrajectoryFrame> {
    let dur = toppra.duration().max(1e-6);
    let n = num_frames.max(2);
    let mut ws = workcell.env.create_workspace(&workcell.model);
    let mut frames = Vec::with_capacity(n);
    for i in 0..n {
        let t = dur * (i as f64) / ((n - 1) as f64);
        let sample = toppra.sample(t);
        frames.push(build_trajectory_frame(
            workcell,
            &workcell.env,
            t,
            sample.position.as_slice(),
            sample.velocity.as_slice(),
            Vec::new(),
            0.0,
            0.0,
            TrajectoryStage::SingleStage,
            &mut ws,
        ));
    }
    frames
}

#[allow(clippy::too_many_arguments)]
fn build_trajectory_frame(
    workcell: &BuiltWorkcell,
    env: &CollisionEnvironment,
    time: f64,
    q: &[f64],
    qd: &[f64],
    dynamic_obstacles: Vec<(ObstacleId, Isometry3<f64>)>,
    grasp_error_mm: f64,
    pose_error_mm: f64,
    stage: TrajectoryStage,
    ws: &mut CollisionWorkspace,
) -> TrajectoryFrame {
    let mut wits = Vec::new();
    let _ = env.query_signed_distances(&workcell.model, q, 0.6, GeometryTier::Convex, ws, &mut wits);

    let mut min_clearance = f64::INFINITY;
    let mut min_body = None::<f64>;
    let mut min_tool = None::<f64>;
    let mut min_tool_wp = None::<f64>;

    for w in &wits {
        let d = w.signed_distance;
        if d < min_clearance {
            min_clearance = d;
        }
        match w.pair {
            CollidingPair::LinkToLink { .. } | CollidingPair::LinkToObstacle { .. } => {
                min_body = Some(min_body.map_or(d, |cur| cur.min(d)));
            }
            CollidingPair::PayloadToPayload { .. } => {
                min_tool = Some(min_tool.map_or(d, |cur| cur.min(d)));
            }
            CollidingPair::PayloadToObstacle { obstacle, .. } => {
                let is_wp = workcell
                    .obstacles
                    .get(obstacle.0 as usize)
                    .is_some_and(|o| matches!(o.role, crate::scene::spec::ObstacleRole::Workpiece));
                if is_wp {
                    min_tool_wp = Some(min_tool_wp.map_or(d, |cur| cur.min(d)));
                } else {
                    min_tool = Some(min_tool.map_or(d, |cur| cur.min(d)));
                }
            }
            CollidingPair::PayloadToLink { .. } => {
                min_tool = Some(min_tool.map_or(d, |cur| cur.min(d)));
            }
        }
    }
    if !min_clearance.is_finite() {
        min_clearance = 0.6;
    }

    TrajectoryFrame {
        time,
        q: q.to_vec(),
        qd: qd.to_vec(),
        dynamic_obstacles,
        min_clearance,
        min_body_clearance: min_body,
        min_tool_clearance: min_tool,
        min_tool_workpiece_clearance: min_tool_wp,
        grasp_error_mm,
        pose_error_mm,
        stage,
    }
}

fn finite_diff_qd(combined: &[(Vec<f64>, TrajectoryStage)], idx: usize, dt: f64) -> Vec<f64> {
    if combined.len() < 2 || dt <= 0.0 {
        return vec![0.0; combined.first().map_or(0, |c| c.0.len())];
    }
    let next_idx = (idx + 1).min(combined.len() - 1);
    let prev_idx = idx.saturating_sub(1);
    let span = ((next_idx - prev_idx) as f64) * dt;
    combined[next_idx]
        .0
        .iter()
        .zip(combined[prev_idx].0.iter())
        .map(|(a, b)| (a - b) / span)
        .collect()
}

fn compute_approach_pos_err_mm(
    workcell: &BuiltWorkcell,
    q: &[f64],
    targets: &[ApproachTarget],
    buf: &mut unrust_it::TransformBuffer,
) -> f64 {
    if workcell.model.forward_kinematics(q, buf).is_err() {
        return 0.0;
    }
    targets
        .iter()
        .map(|t| {
            let p = buf.frame_pose(t.final_pose.frame);
            (t.final_pose.pose.translation.vector - p.translation.vector).norm() * 1000.0
        })
        .fold(0.0_f64, f64::max)
}

fn compute_task_pos_err_mm(
    workcell: &BuiltWorkcell,
    q: &[f64],
    targets: &[TaskPose],
    buf: &mut unrust_it::TransformBuffer,
) -> f64 {
    compute_task_errors(workcell, q, targets, buf).0 * 1000.0
}

fn compute_task_errors(
    workcell: &BuiltWorkcell,
    q: &[f64],
    targets: &[TaskPose],
    buf: &mut unrust_it::TransformBuffer,
) -> (f64, f64) {
    if workcell.model.forward_kinematics(q, buf).is_err() {
        return (0.0, 0.0);
    }
    let mut max_p = 0.0_f64;
    let mut max_r = 0.0_f64;
    for t in targets {
        let p = buf.frame_pose(t.frame);
        let dp = (t.pose.translation.vector - p.translation.vector).norm();
        let dr = (t.pose.rotation * p.rotation.inverse()).angle();
        max_p = max_p.max(dp);
        max_r = max_r.max(dr);
    }
    (max_p, max_r)
}

fn verify_path_convex(
    workcell: &BuiltWorkcell,
    path: &[Configuration],
    max_workspace_step: f64,
    ws: &mut CollisionWorkspace,
) -> bool {
    if path.is_empty() {
        return false;
    }
    if path.len() == 1 {
        return workcell
            .env
            .is_state_collision_free(
                &workcell.model,
                path[0].as_slice(),
                GeometryTier::Convex,
                ws,
            )
            .unwrap_or(false);
    }
    for seg in path.windows(2) {
        if !workcell
            .env
            .is_segment_collision_free(
                &workcell.model,
                seg[0].as_slice(),
                seg[1].as_slice(),
                max_workspace_step,
                GeometryTier::Convex,
                ws,
            )
            .unwrap_or(false)
        {
            return false;
        }
    }
    true
}

fn compute_tcp_trails(
    workcell: &BuiltWorkcell,
    frames: &[TrajectoryFrame],
) -> Vec<Vec<[f32; 3]>> {
    let mut buf = workcell.model.create_transform_buffer();
    let mut trails = vec![Vec::with_capacity(frames.len()); workcell.arms.len()];
    for f in frames {
        if workcell.model.forward_kinematics(&f.q, &mut buf).is_ok() {
            for (idx, arm) in workcell.arms.iter().enumerate() {
                let p = buf.frame_pose(arm.tcp_frame).translation.vector;
                trails[idx].push([p.x as f32, p.y as f32, p.z as f32]);
            }
        }
    }
    trails
}

fn summarize_metrics(
    frames: &[TrajectoryFrame],
    solve_time_ms: f64,
    max_pos_error_mm: f64,
    max_rot_error_deg: f64,
    stage1_end_time: Option<f64>,
    convex_verified: bool,
) -> PlanMetrics {
    let duration_s = frames.last().map_or(0.0, |f| f.time);
    let min_clearance_mm = frames
        .iter()
        .map(|f| f.min_clearance * 1000.0)
        .fold(f64::INFINITY, f64::min);
    let min_tool_clearance_mm = frames
        .iter()
        .filter_map(|f| {
            match (f.min_tool_clearance, f.min_tool_workpiece_clearance) {
                (Some(a), Some(b)) => Some(a.min(b) * 1000.0),
                (Some(a), None) => Some(a * 1000.0),
                (None, Some(b)) => Some(b * 1000.0),
                (None, None) => None,
            }
        })
        .reduce(f64::min);

    PlanMetrics {
        solve_time_ms,
        backend_label: "CPU SIMD f32x8",
        num_waypoints: frames.len(),
        duration_s,
        max_pos_error_mm,
        max_rot_error_deg,
        min_clearance_mm: if min_clearance_mm.is_finite() {
            min_clearance_mm
        } else {
            0.0
        },
        min_tool_clearance_mm,
        stage1_end_time,
        convex_verified,
    }
}
