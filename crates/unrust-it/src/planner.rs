//! Path-planning traits and a bidirectional RRT-Connect planner for single- and multi-arm systems.

use crate::batch::{
    BatchIkSolver, BatchMppiOptimizer, BatchRolloutBackend, EsdfGridF32, FlatRobotModelF32,
    MppiConfig,
};
use crate::collision::{
    CollisionChecker, CollisionEnvironment, CollisionError, CollisionWorkspace, GeometryTier,
    SignedDistanceWitness,
};
use crate::control::{ControlError, ReactiveServoConfig, ReactiveServoController, ServoTarget};
use crate::kinematics::{KinematicsError, TransformBuffer};
use crate::robot::{
    Configuration, JointType, MultiRobotModel, Robot, RobotError, TaskPose, TaskSpacePoint,
};
use nalgebra::{DMatrix, DVector, Isometry3, Translation3, Vector3};
use std::sync::Arc;
use thiserror::Error;

/// A planning target can be expressed in joint space or task space.
#[derive(Debug, Clone, PartialEq)]
pub enum Target {
    /// A destination directly in C-space.
    Joint(Configuration),
    /// A 3D Cartesian destination to be solved by inverse kinematics.
    Task(TaskSpacePoint),
    /// Simultaneous 6-DOF $SE(3)$ pose targets for one or more coordinate frames (e.g., dual-arm TCPs).
    MultiPose(Vec<TaskPose>),
}

/// Errors returned by a path planner.
#[derive(Debug, Error, PartialEq)]
pub enum PlannerError {
    /// A configuration is outside the robot's C-space.
    #[error(transparent)]
    InvalidConfiguration(#[from] RobotError),
    /// A candidate path intersects an obstacle or the robot itself, or no collision-free path was found.
    #[error("collision detected")]
    CollisionDetected,
    /// Collision geometry could not be evaluated.
    #[error(transparent)]
    Collision(#[from] CollisionError),
    /// Kinematics or IK error during target resolution.
    #[error(transparent)]
    Kinematics(#[from] KinematicsError),
    /// No configuration was found for a task-space target.
    #[error("target is unreachable")]
    TargetUnreachable,
    /// The planner needs an IK model to solve task-space targets.
    #[error("task-space planning requires an inverse-kinematics implementation")]
    MissingInverseKinematics,
}

/// Plans a collision-free sequence through configuration space.
pub trait PathPlanner {
    /// Plans from `start` to a joint-space or task-space target.
    fn plan_path(
        &self,
        robot: &Robot,
        start: Configuration,
        target: Target,
    ) -> Result<Vec<Configuration>, PlannerError>;
}

/// Flat-arena tree used by [`RrtPlanner`] to avoid per-node heap allocations.
struct FlatTree {
    dof: usize,
    /// Contiguous configuration values (`len = num_nodes * dof`).
    configs: Vec<f64>,
    /// Parent node index for each node (`usize::MAX` for root).
    parents: Vec<usize>,
}

impl FlatTree {
    fn with_capacity(dof: usize, capacity: usize, root: &[f64]) -> Self {
        let mut configs = Vec::with_capacity(capacity * dof);
        configs.extend_from_slice(root);
        let mut parents = Vec::with_capacity(capacity);
        parents.push(usize::MAX);
        Self {
            dof,
            configs,
            parents,
        }
    }

    #[inline]
    fn num_nodes(&self) -> usize {
        self.parents.len()
    }

    #[inline]
    fn node_config(&self, idx: usize) -> &[f64] {
        &self.configs[idx * self.dof..(idx + 1) * self.dof]
    }

    fn push_node(&mut self, q: &[f64], parent: usize) -> usize {
        let idx = self.parents.len();
        self.configs.extend_from_slice(q);
        self.parents.push(parent);
        idx
    }

    fn nearest(&self, model: &MultiRobotModel, target_q: &[f64]) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;
        for i in 0..self.num_nodes() {
            let d = model.distance(self.node_config(i), target_q);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        best_idx
    }

    fn trace_to_root(&self, mut node_idx: usize) -> Vec<Configuration> {
        let mut chain = Vec::new();
        while node_idx != usize::MAX {
            chain.push(Configuration::from_slice(self.node_config(node_idx)));
            node_idx = self.parents[node_idx];
        }
        chain.reverse();
        chain
    }
}

/// A bidirectional RRT-Connect planner backed by a flat contiguous C-space arena.
#[derive(Debug, Clone)]
pub struct RrtPlanner {
    /// Static and robot collision checker used by the planner.
    pub environment: CollisionEnvironment,
    /// Maximum C-space step size (radians/meters) per tree extension.
    pub step_size: f64,
    /// Maximum workspace step (meters) for continuous segment collision checking.
    pub max_workspace_step: f64,
    /// Maximum number of RRT-Connect iterations.
    pub max_iterations: usize,
    /// Deterministic PRNG seed for reproducible sampling.
    pub seed: u64,
}

impl RrtPlanner {
    /// Creates an RRT-Connect planner for an environment with default parameters.
    pub fn new(environment: CollisionEnvironment) -> Self {
        Self {
            environment,
            step_size: 0.35,
            max_workspace_step: 0.04,
            max_iterations: 1500,
            seed: 0x5EED_CAFE_1234_5678,
        }
    }

    fn resolve_goal(&self, robot: &Robot, start: &Configuration, target: Target) -> Result<Configuration, PlannerError> {
        match target {
            Target::Joint(goal) => Ok(goal),
            Target::Task(point) => {
                let Some(model) = &robot.model else {
                    return Err(PlannerError::MissingInverseKinematics);
                };
                let ee_frame = model
                    .groups()
                    .first()
                    .and_then(|g| g.end_effectors.first().copied())
                    .ok_or(PlannerError::MissingInverseKinematics)?;
                let mut buf = model.create_transform_buffer();
                model.forward_kinematics(start.as_slice(), &mut buf)?;
                let current_rot = buf.frame_pose(ee_frame).rotation;
                let pose = Isometry3::from_parts(point.position.into(), current_rot);
                model
                    .solve_ik_dls(
                        &[TaskPose::new(ee_frame, pose)],
                        start.as_slice(),
                        120,
                        1e-4,
                        1e-2,
                        &mut buf,
                    )
                    .map_err(|_| PlannerError::TargetUnreachable)
            }
            Target::MultiPose(poses) => {
                let Some(model) = &robot.model else {
                    return Err(PlannerError::MissingInverseKinematics);
                };
                let mut buf = model.create_transform_buffer();
                model
                    .solve_ik_dls(&poses, start.as_slice(), 150, 1e-4, 1e-2, &mut buf)
                    .map_err(|_| PlannerError::TargetUnreachable)
            }
        }
    }
}

impl PathPlanner for RrtPlanner {
    fn plan_path(
        &self,
        robot: &Robot,
        start: Configuration,
        target: Target,
    ) -> Result<Vec<Configuration>, PlannerError> {
        robot.validate_configuration(&start)?;
        let goal = self.resolve_goal(robot, &start, target)?;
        robot.validate_configuration(&goal)?;

        if !self.environment.is_collision_free(robot, &start)?
            || !self.environment.is_collision_free(robot, &goal)?
        {
            return Err(PlannerError::CollisionDetected);
        }

        let Some(model) = &robot.model else {
            return Ok(vec![start, goal]);
        };

        let mut workspace = self.environment.create_workspace(model);
        let tier = GeometryTier::Convex;

        // Fast path: if direct geodesic segment [start, goal] is collision-free, return immediately.
        if self.environment.is_segment_collision_free(
            model,
            start.as_slice(),
            goal.as_slice(),
            self.max_workspace_step,
            tier,
            &mut workspace,
        )? {
            return Ok(vec![start, goal]);
        }

        // Bidirectional RRT-Connect search in flat arenas
        let dof = model.dof();
        let mut tree_a = FlatTree::with_capacity(dof, 512, start.as_slice());
        let mut tree_b = FlatTree::with_capacity(dof, 512, goal.as_slice());
        let mut swapped = false;

        let mut rng_state = self.seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next_u64 = || -> u64 {
            // SplitMix64 deterministic PRNG
            rng_state = rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = rng_state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };

        let mut q_rand = vec![0.0; dof];
        let mut q_step = vec![0.0; dof];

        for _ in 0..self.max_iterations {
            for (k, lim) in model.dof_limits().iter().enumerate() {
                let u = (next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
                let (lo, hi) = if lim.position.min.is_finite() && lim.position.max.is_finite() {
                    (lim.position.min, lim.position.max)
                } else {
                    (-std::f64::consts::PI, std::f64::consts::PI)
                };
                q_rand[k] = lo + u * (hi - lo);
            }

            // Extend tree_a one step toward q_rand
            let near_a = tree_a.nearest(model, &q_rand);
            let q_near_a = tree_a.node_config(near_a);
            let dist_a = model.distance(q_near_a, &q_rand);
            if dist_a <= 1e-9 {
                continue;
            }
            let alpha_a = (self.step_size / dist_a).min(1.0);
            model.interpolate(q_near_a, &q_rand, alpha_a, &mut q_step);

            if !self.environment.is_segment_collision_free(
                model,
                q_near_a,
                &q_step,
                self.max_workspace_step,
                tier,
                &mut workspace,
            )? {
                std::mem::swap(&mut tree_a, &mut tree_b);
                swapped = !swapped;
                continue;
            }

            let new_a = tree_a.push_node(&q_step, near_a);
            let q_target_b = tree_a.node_config(new_a).to_vec();

            // Greedily connect tree_b toward q_target_b
            let mut curr_b = tree_b.nearest(model, &q_target_b);
            loop {
                let q_curr_b = tree_b.node_config(curr_b);
                let dist_b = model.distance(q_curr_b, &q_target_b);
                if dist_b <= self.step_size {
                    if self.environment.is_segment_collision_free(
                        model,
                        q_curr_b,
                        &q_target_b,
                        self.max_workspace_step,
                        tier,
                        &mut workspace,
                    )? {
                        // Connected! Reconstruct and shortcut path.
                        let mut path_a = tree_a.trace_to_root(new_a);
                        let mut path_b = tree_b.trace_to_root(curr_b);
                        if swapped {
                            std::mem::swap(&mut path_a, &mut path_b);
                        }
                        path_b.reverse();
                        path_a.extend(path_b);
                        return Ok(self.shortcut_path(model, path_a, &mut workspace));
                    }
                    break;
                }

                let alpha_b = self.step_size / dist_b;
                model.interpolate(q_curr_b, &q_target_b, alpha_b, &mut q_step);
                if self.environment.is_segment_collision_free(
                    model,
                    q_curr_b,
                    &q_step,
                    self.max_workspace_step,
                    tier,
                    &mut workspace,
                )? {
                    curr_b = tree_b.push_node(&q_step, curr_b);
                } else {
                    break;
                }
            }

            std::mem::swap(&mut tree_a, &mut tree_b);
            swapped = !swapped;
        }

        Err(PlannerError::CollisionDetected)
    }
}

impl RrtPlanner {
    fn shortcut_path(
        &self,
        model: &MultiRobotModel,
        mut path: Vec<Configuration>,
        workspace: &mut crate::collision::CollisionWorkspace,
    ) -> Vec<Configuration> {
        if path.len() <= 2 {
            return path;
        }
        let mut i = 0;
        while i + 2 < path.len() {
            let mut shortcut_to = None;
            for j in ((i + 2)..path.len()).rev() {
                if self
                    .environment
                    .is_segment_collision_free(
                        model,
                        path[i].as_slice(),
                        path[j].as_slice(),
                        self.max_workspace_step,
                        GeometryTier::Convex,
                        workspace,
                    )
                    .unwrap_or(false)
                {
                    shortcut_to = Some(j);
                    break;
                }
            }
            if let Some(j) = shortcut_to {
                path.drain((i + 1)..j);
            }
            i += 1;
        }
        path
    }
}

/// Target specification for [`TwoStageApproachPlanner`], pairing a tight final 6-DOF pose
/// with either an explicit pre-approach pose or a local standoff offset along the TCP frame.
#[derive(Debug, Clone, PartialEq)]
pub struct ApproachTarget {
    /// Final tight-proximity 6-DOF pose target.
    pub final_pose: TaskPose,
    /// Optional explicit world-frame pre-approach pose. If `None`, computed via `final_pose.pose * standoff_local`.
    pub pre_approach_pose: Option<Isometry3<f64>>,
    /// Standoff translation in `final_pose.frame` local coordinates when `pre_approach_pose` is `None`.
    pub standoff_local: Vector3<f64>,
}

impl ApproachTarget {
    /// Creates an approach target whose pre-approach pose is offset by `standoff_local`
    /// in the target frame's local coordinates relative to `final_pose`.
    pub fn with_standoff(final_pose: TaskPose, standoff_local: Vector3<f64>) -> Self {
        Self {
            final_pose,
            pre_approach_pose: None,
            standoff_local,
        }
    }

    /// Creates an approach target with an explicit pre-approach world pose.
    pub fn with_pre_approach(final_pose: TaskPose, pre_approach_pose: Isometry3<f64>) -> Self {
        Self {
            final_pose,
            pre_approach_pose: Some(pre_approach_pose),
            standoff_local: Vector3::zeros(),
        }
    }

    /// Resolves the Stage 1 pre-approach [`TaskPose`].
    pub fn resolved_pre_approach_pose(&self) -> TaskPose {
        let pose = self
            .pre_approach_pose
            .unwrap_or_else(|| self.final_pose.pose * Translation3::from(self.standoff_local));
        TaskPose::new(self.final_pose.frame, pose)
    }
}

/// Global Stage 1 planner strategy used before Stage 2 CBF/IK refinement.
#[derive(Debug, Clone)]
pub enum CoarsePlannerMode {
    /// Bidirectional RRT-Connect directly in exact `GeometryTier::Convex`.
    RrtConnect,
    /// Batched `f32` MPPI trajectory optimization followed by exact `GeometryTier::Convex` validation.
    Mppi(MppiConfig),
    /// Batched MPPI first, automatically falling back to RRT-Connect if MPPI fails exact validation.
    MppiWithRrtFallback(MppiConfig),
}

/// Configuration for [`TwoStageApproachPlanner`].
#[derive(Debug, Clone)]
pub struct TwoStagePlanConfig {
    /// Stage 1 global planner mode (`start -> q_pre_approach`).
    pub coarse_mode: CoarsePlannerMode,
    /// Multi-seed collision-free IK solver used to resolve Stage 1 pre-approach configurations.
    pub ik_solver: BatchIkSolver,
    /// RRT-Connect C-space step size when Stage 1 uses RRT.
    pub rrt_step_size: f64,
    /// RRT-Connect continuous collision checking workspace step (meters).
    pub rrt_max_workspace_step: f64,
    /// Maximum RRT-Connect iterations.
    pub rrt_max_iterations: usize,
    /// Stage 2 reactive CBF controller tuning (per-pair margins in [`CollisionEnvironment`] override `safety_margin`).
    pub servo_config: ReactiveServoConfig,
    /// Proportional task-space gain during Stage 2 final approach.
    pub approach_pose_gain: f64,
    /// Maximum Stage 2 reactive CBF refinement steps.
    pub max_refinement_steps: usize,
    /// Required final Cartesian position error tolerance (meters) across all target frames.
    pub final_position_tolerance: f64,
    /// Required final orientation error tolerance (radians) across all target frames.
    pub final_orientation_tolerance: f64,
    /// Subsampling stride for storing Stage 2 waypoints (`1` stores every step).
    pub waypoint_stride: usize,
}

impl Default for TwoStagePlanConfig {
    fn default() -> Self {
        Self {
            coarse_mode: CoarsePlannerMode::MppiWithRrtFallback(MppiConfig::default()),
            ik_solver: BatchIkSolver::default(),
            rrt_step_size: 0.35,
            rrt_max_workspace_step: 0.03,
            rrt_max_iterations: 2500,
            servo_config: ReactiveServoConfig {
                dt: 0.01,
                safety_margin: 0.04,
                cbf_activation_distance: 0.18,
                cbf_gamma: 8.0,
                damping: 0.02,
                posture_gain: 0.05,
                tangential_curl_gain: 0.35,
                ..ReactiveServoConfig::default()
            },
            approach_pose_gain: 6.0,
            max_refinement_steps: 450,
            final_position_tolerance: 1.5e-3,
            final_orientation_tolerance: 1.5e-2,
            waypoint_stride: 5,
        }
    }
}

/// Output of [`TwoStageApproachPlanner::plan`].
#[derive(Debug, Clone)]
pub struct TwoStagePlanResult {
    /// Coarse global collision-free path (`start -> q_pre_approach`).
    pub stage1_path: Vec<Configuration>,
    /// Tight-margin Stage 2 refinement path (`q_pre_approach -> q_final`).
    pub stage2_path: Vec<Configuration>,
    /// Combined stitched collision-free path (`start -> q_final`).
    pub full_path: Vec<Configuration>,
    /// Maximum final Cartesian position error across all targets (meters).
    pub max_position_error: f64,
    /// Maximum final orientation error across all targets (radians).
    pub max_orientation_error: f64,
    /// Minimum observed signed distance across Stage 2 in `GeometryTier::Convex` (meters).
    pub min_stage2_clearance: f64,
}

/// Two-stage multi-arm motion planner for tight-proximity manipulation:
///
/// 1. **Stage 1 (Coarse Global)**: Plans `start -> pre_approach` using MPPI/RRT-Connect.
/// 2. **Stage 2 (Exact Refinement)**: Converges `pre_approach -> final_pose` using tight-margin
///    `GeometryTier::Convex` priority-weighted QP-CBF + guarded DLS micro-polish.
#[derive(Debug, Clone)]
pub struct TwoStageApproachPlanner {
    /// Planner configuration.
    pub config: TwoStagePlanConfig,
}

impl TwoStageApproachPlanner {
    /// Creates a new two-stage planner with the given configuration.
    pub fn new(config: TwoStagePlanConfig) -> Self {
        Self { config }
    }

    fn evaluate_pose_errors(
        model: &MultiRobotModel,
        q: &[f64],
        targets: &[ApproachTarget],
        buf: &mut TransformBuffer,
    ) -> Result<(f64, f64), PlannerError> {
        model.forward_kinematics(q, buf)?;
        let mut max_pos = 0.0_f64;
        let mut max_rot = 0.0_f64;
        for t in targets {
            let curr = buf.frame_pose(t.final_pose.frame);
            let pos_err = (t.final_pose.pose.translation.vector - curr.translation.vector).norm();
            let rot_err = (t.final_pose.pose.rotation * curr.rotation.inverse()).angle();
            max_pos = max_pos.max(pos_err);
            max_rot = max_rot.max(rot_err);
        }
        Ok((max_pos, max_rot))
    }

    fn validate_path_convex(
        model: &MultiRobotModel,
        env: &CollisionEnvironment,
        path: &[Configuration],
        max_workspace_step: f64,
        ws: &mut CollisionWorkspace,
    ) -> Result<bool, CollisionError> {
        if path.is_empty() {
            return Ok(false);
        }
        if path.len() == 1 {
            return env.is_state_collision_free(
                model,
                path[0].as_slice(),
                GeometryTier::Convex,
                ws,
            );
        }
        for seg in path.windows(2) {
            if !env.is_segment_collision_free(
                model,
                seg[0].as_slice(),
                seg[1].as_slice(),
                max_workspace_step,
                GeometryTier::Convex,
                ws,
            )? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn plan_stage1_rrt(
        &self,
        model: &MultiRobotModel,
        env: &CollisionEnvironment,
        start: &Configuration,
        goal: &Configuration,
    ) -> Result<Vec<Configuration>, PlannerError> {
        let robot = Robot::from_model(Arc::new(model.clone()));
        let mut rrt = RrtPlanner::new(env.clone());
        rrt.step_size = self.config.rrt_step_size;
        rrt.max_workspace_step = self.config.rrt_max_workspace_step;
        rrt.max_iterations = self.config.rrt_max_iterations;
        rrt.plan_path(&robot, start.clone(), Target::Joint(goal.clone()))
    }

    fn update_min_clearance(
        model: &MultiRobotModel,
        env: &CollisionEnvironment,
        q: &[f64],
        cutoff: f64,
        ws: &mut CollisionWorkspace,
        witnesses: &mut Vec<SignedDistanceWitness>,
        min_clearance: &mut f64,
    ) -> Result<(), PlannerError> {
        env.query_signed_distances(model, q, cutoff, GeometryTier::Convex, ws, witnesses)?;
        for w in witnesses.iter() {
            if w.signed_distance < *min_clearance {
                *min_clearance = w.signed_distance;
            }
        }
        Ok(())
    }

    /// Plans a collision-free two-stage trajectory from `start` to all simultaneous `targets`.
    #[allow(clippy::too_many_arguments)]
    pub fn plan<B: BatchRolloutBackend>(
        &self,
        backend: &B,
        model: &MultiRobotModel,
        env: &CollisionEnvironment,
        esdf: Option<&EsdfGridF32>,
        start: Configuration,
        targets: &[ApproachTarget],
    ) -> Result<TwoStagePlanResult, PlannerError> {
        model.validate_configuration(start.as_slice())?;
        let mut ws = env.create_workspace(model);
        if !env.is_state_collision_free(model, start.as_slice(), GeometryTier::Convex, &mut ws)? {
            return Err(PlannerError::CollisionDetected);
        }

        if targets.is_empty() {
            return Ok(TwoStagePlanResult {
                stage1_path: vec![start.clone()],
                stage2_path: vec![start.clone()],
                full_path: vec![start],
                max_position_error: 0.0,
                max_orientation_error: 0.0,
                min_stage2_clearance: f64::INFINITY,
            });
        }

        // 1. Solve collision-free multi-seed IK for Stage 1 pre-approach poses
        let flat_model = FlatRobotModelF32::compile(model, env);
        let pre_targets: Vec<TaskPose> = targets
            .iter()
            .map(ApproachTarget::resolved_pre_approach_pose)
            .collect();
        let q_pre = self
            .config
            .ik_solver
            .solve_collision_free(
                backend,
                model,
                &flat_model,
                env,
                esdf,
                start.as_slice(),
                &pre_targets,
            )
            .map_err(|_| PlannerError::TargetUnreachable)?;

        // 2. Stage 1: Coarse Global Plan (start -> q_pre) validated in GeometryTier::Convex
        let stage1_path = match &self.config.coarse_mode {
            CoarsePlannerMode::RrtConnect => {
                self.plan_stage1_rrt(model, env, &start, &q_pre)?
            }
            CoarsePlannerMode::Mppi(mppi_cfg) => {
                let mppi = BatchMppiOptimizer::new(mppi_cfg.clone());
                let traj = mppi.optimize(
                    backend,
                    model,
                    &flat_model,
                    esdf,
                    start.as_slice(),
                    q_pre.as_slice(),
                );
                if Self::validate_path_convex(
                    model,
                    env,
                    &traj,
                    self.config.rrt_max_workspace_step,
                    &mut ws,
                )? {
                    traj
                } else {
                    return Err(PlannerError::CollisionDetected);
                }
            }
            CoarsePlannerMode::MppiWithRrtFallback(mppi_cfg) => {
                let mppi = BatchMppiOptimizer::new(mppi_cfg.clone());
                let traj = mppi.optimize(
                    backend,
                    model,
                    &flat_model,
                    esdf,
                    start.as_slice(),
                    q_pre.as_slice(),
                );
                if Self::validate_path_convex(
                    model,
                    env,
                    &traj,
                    self.config.rrt_max_workspace_step,
                    &mut ws,
                )? {
                    traj
                } else {
                    self.plan_stage1_rrt(model, env, &start, &q_pre)?
                }
            }
        };

        // 3. Stage 2: Tight-margin GeometryTier::Convex CBF Refinement (q_pre -> final_pose)
        let dof = model.dof();
        let mut q_curr = q_pre.0.clone();
        let mut dq_curr = vec![0.0_f64; dof];
        let mut dq_cmd = vec![0.0_f64; dof];
        let mut q_cand = vec![0.0_f64; dof];

        let mut servo = ReactiveServoController::new(model, env, self.config.servo_config.clone());
        let servo_targets: Vec<ServoTarget> = targets
            .iter()
            .map(|t| ServoTarget::pose(t.final_pose.frame, t.final_pose.pose, self.config.approach_pose_gain))
            .collect();

        let mut buf = model.create_transform_buffer();
        let mut witnesses = Vec::with_capacity(64);
        let clearance_cutoff = env
            .max_activation_distance(self.config.servo_config.cbf_activation_distance)
            .max(0.2);
        let mut min_stage2_clearance = f64::INFINITY;
        Self::update_min_clearance(
            model,
            env,
            &q_curr,
            clearance_cutoff,
            &mut ws,
            &mut witnesses,
            &mut min_stage2_clearance,
        )?;

        let mut stage2_path = vec![q_pre.clone()];
        let stride = self.config.waypoint_stride.max(1);
        let dt = self.config.servo_config.dt.max(1e-4);

        for step_idx in 0..self.config.max_refinement_steps {
            let (pos_err, rot_err) = Self::evaluate_pose_errors(model, &q_curr, targets, &mut buf)?;
            if pos_err <= self.config.final_position_tolerance
                && rot_err <= self.config.final_orientation_tolerance
            {
                break;
            }

            servo
                .step(
                    model,
                    env,
                    &q_curr,
                    &dq_curr,
                    &servo_targets,
                    None,
                    &mut dq_cmd,
                )
                .map_err(|e| match e {
                    ControlError::Collision(c) => PlannerError::Collision(c),
                    ControlError::Kinematics(k) => PlannerError::Kinematics(k),
                    ControlError::Robot(r) => PlannerError::InvalidConfiguration(r),
                })?;

            let mut accepted = false;
            for scale in [1.0, 0.5, 0.25, 0.125] {
                for (k, lim) in model.dof_limits().iter().enumerate() {
                    let next_val = q_curr[k] + dq_cmd[k] * dt * scale;
                    let joint = model.joint(model.actuated_joint(k));
                    q_cand[k] = if matches!(joint.joint_type, JointType::Continuous { .. }) {
                        (next_val + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
                            - std::f64::consts::PI
                    } else {
                        lim.position.clamp(next_val)
                    };
                }

                if env.is_segment_collision_free(
                    model,
                    &q_curr,
                    &q_cand,
                    0.01,
                    GeometryTier::Convex,
                    &mut ws,
                )? {
                    q_curr.copy_from_slice(&q_cand);
                    for k in 0..dof {
                        dq_curr[k] = dq_cmd[k] * scale;
                    }
                    Self::update_min_clearance(
                        model,
                        env,
                        &q_curr,
                        clearance_cutoff,
                        &mut ws,
                        &mut witnesses,
                        &mut min_stage2_clearance,
                    )?;
                    if (step_idx + 1).is_multiple_of(stride) {
                        stage2_path.push(Configuration::from_slice(&q_curr));
                    }
                    accepted = true;
                    break;
                }
            }

            if !accepted {
                dq_curr.fill(0.0);
                break;
            }
        }

        // 4. Guarded DLS micro-polish for sub-millimeter final alignment without violating Convex collision freedom
        let (mut pos_err, mut rot_err) =
            Self::evaluate_pose_errors(model, &q_curr, targets, &mut buf)?;
        if (pos_err > self.config.final_position_tolerance
            || rot_err > self.config.final_orientation_tolerance)
            && pos_err < 0.05
        {
            let m = 6 * targets.len();
            let mut err_vec = DVector::<f64>::zeros(m);
            let mut j_mat = DMatrix::<f64>::zeros(m, dof);
            for _ in 0..15 {
                if pos_err <= self.config.final_position_tolerance
                    && rot_err <= self.config.final_orientation_tolerance
                {
                    break;
                }
                model.forward_kinematics(&q_curr, &mut buf)?;
                for (i, t) in targets.iter().enumerate() {
                    let curr = buf.frame_pose(t.final_pose.frame);
                    let p_e = t.final_pose.pose.translation.vector - curr.translation.vector;
                    let r_e = (t.final_pose.pose.rotation * curr.rotation.inverse()).scaled_axis();
                    let off = 6 * i;
                    err_vec[off] = p_e.x;
                    err_vec[off + 1] = p_e.y;
                    err_vec[off + 2] = p_e.z;
                    err_vec[off + 3] = r_e.x;
                    err_vec[off + 4] = r_e.y;
                    err_vec[off + 5] = r_e.z;
                    model.geometric_jacobian(
                        &buf,
                        t.final_pose.frame,
                        j_mat.view_mut((off, 0), (6, dof)),
                    )?;
                }

                let mut gram = &j_mat * j_mat.transpose();
                let lam_sq = 1e-6;
                for r in 0..m {
                    gram[(r, r)] += lam_sq;
                }
                let Some(chol) = gram.cholesky() else {
                    break;
                };
                let mut dq = j_mat.transpose() * chol.solve(&err_vec);
                let max_step = 0.02;
                let dq_inf = dq.amax();
                if dq_inf > max_step {
                    dq *= max_step / dq_inf;
                }

                for (k, lim) in model.dof_limits().iter().enumerate() {
                    let next_val = q_curr[k] + dq[k];
                    let joint = model.joint(model.actuated_joint(k));
                    q_cand[k] = if matches!(joint.joint_type, JointType::Continuous { .. }) {
                        (next_val + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
                            - std::f64::consts::PI
                    } else {
                        lim.position.clamp(next_val)
                    };
                }

                if env.is_state_collision_free(model, &q_cand, GeometryTier::Convex, &mut ws)?
                    && env.is_segment_collision_free(
                        model,
                        &q_curr,
                        &q_cand,
                        0.005,
                        GeometryTier::Convex,
                        &mut ws,
                    )?
                {
                    q_curr.copy_from_slice(&q_cand);
                    Self::update_min_clearance(
                        model,
                        env,
                        &q_curr,
                        clearance_cutoff,
                        &mut ws,
                        &mut witnesses,
                        &mut min_stage2_clearance,
                    )?;
                    (pos_err, rot_err) =
                        Self::evaluate_pose_errors(model, &q_curr, targets, &mut buf)?;
                } else {
                    break;
                }
            }
        }

        if stage2_path
            .last()
            .is_none_or(|last| last.as_slice() != q_curr.as_slice())
        {
            stage2_path.push(Configuration::from_slice(&q_curr));
        }

        if pos_err > self.config.final_position_tolerance
            || rot_err > self.config.final_orientation_tolerance
        {
            return Err(PlannerError::TargetUnreachable);
        }

        let mut full_path = stage1_path.clone();
        if stage2_path.len() > 1 {
            full_path.extend(stage2_path.iter().skip(1).cloned());
        }

        Ok(TwoStagePlanResult {
            stage1_path,
            stage2_path,
            full_path,
            max_position_error: pos_err,
            max_orientation_error: rot_err,
            min_stage2_clearance,
        })
    }
}
