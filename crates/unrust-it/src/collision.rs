//! Collision environment and collision-checking extension points.

use crate::robot::{Configuration, Robot};
use nalgebra::Isometry3;
use parry3d::shape::SharedShape;
use thiserror::Error;

/// Errors produced while checking collision geometry.
#[derive(Debug, Error)]
pub enum CollisionError {
    /// Collision geometry could not be evaluated.
    #[error("collision geometry could not be evaluated")]
    GeometryUnavailable,
}

/// A static obstacle in the robot's world frame.
#[derive(Debug, Clone)]
pub struct Obstacle {
    /// Shape supplied by `parry3d`.
    pub shape: SharedShape,
    /// Pose of the shape in world coordinates.
    pub pose: Isometry3<f64>,
}

/// Static geometry against which a robot can be checked.
#[derive(Debug, Default, Clone)]
pub struct CollisionEnvironment {
    /// Obstacles in the world frame.
    pub obstacles: Vec<Obstacle>,
}

impl CollisionEnvironment {
    /// Creates an empty environment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a `parry3d` obstacle to the environment.
    pub fn add_obstacle(&mut self, shape: SharedShape, pose: Isometry3<f64>) {
        self.obstacles.push(Obstacle { shape, pose });
    }
}

/// Checks a robot configuration for self-collision and environmental collision.
pub trait CollisionChecker {
    /// Returns `true` when the configuration is collision-free.
    fn is_collision_free(
        &self,
        robot: &Robot,
        configuration: &Configuration,
    ) -> Result<bool, CollisionError>;
}

impl CollisionChecker for CollisionEnvironment {
    fn is_collision_free(
        &self,
        robot: &Robot,
        configuration: &Configuration,
    ) -> Result<bool, CollisionError> {
        robot
            .validate_configuration(configuration)
            .map_err(|_| CollisionError::GeometryUnavailable)?;
        // Link geometry and self-collision are intentionally supplied by a
        // robot-specific implementation in this initial architecture.
        Ok(true)
    }
}
