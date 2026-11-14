//! Path-planning traits and a minimal RRT starting point.

use crate::collision::{CollisionChecker, CollisionEnvironment, CollisionError};
use crate::robot::{Configuration, Robot, RobotError, TaskSpacePoint};
use thiserror::Error;

/// A planning target can be expressed in joint space or task space.
#[derive(Debug, Clone, PartialEq)]
pub enum Target {
    /// A destination directly in C-space.
    Joint(Configuration),
    /// A Cartesian destination to be solved by inverse kinematics.
    Task(TaskSpacePoint),
}

/// Errors returned by a path planner.
#[derive(Debug, Error)]
pub enum PlannerError {
    /// A configuration is outside the robot's C-space.
    #[error(transparent)]
    InvalidConfiguration(#[from] RobotError),
    /// A candidate path intersects an obstacle or the robot itself.
    #[error("collision detected")]
    CollisionDetected,
    /// Collision geometry could not be evaluated.
    #[error(transparent)]
    Collision(#[from] CollisionError),
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

/// A minimal Rapidly-exploring Random Tree (RRT) planner skeleton.
///
/// The current implementation validates a joint-space request and returns its
/// endpoints. Sampling, steering, and tree backtracking are the next steps for
/// a production planner.
#[derive(Debug, Clone)]
pub struct RrtPlanner {
    /// Static and robot collision checker used by the planner.
    pub environment: CollisionEnvironment,
}

impl RrtPlanner {
    /// Creates an RRT planner for an environment.
    pub fn new(environment: CollisionEnvironment) -> Self {
        Self { environment }
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
        let Target::Joint(goal) = target else {
            return Err(PlannerError::MissingInverseKinematics);
        };
        robot.validate_configuration(&goal)?;
        if !self.environment.is_collision_free(robot, &start)?
            || !self.environment.is_collision_free(robot, &goal)?
        {
            return Err(PlannerError::CollisionDetected);
        }
        Ok(vec![start, goal])
    }
}
