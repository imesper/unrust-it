//! Path-planning traits and a bidirectional RRT-Connect planner for single- and multi-arm systems.

use crate::collision::{CollisionChecker, CollisionEnvironment, CollisionError, GeometryTier};
use crate::kinematics::KinematicsError;
use crate::robot::{Configuration, MultiRobotModel, Robot, RobotError, TaskPose, TaskSpacePoint};
use nalgebra::Isometry3;
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
