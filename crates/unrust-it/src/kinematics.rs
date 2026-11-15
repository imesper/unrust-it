//! Forward and inverse kinematics extension points.

use crate::robot::{Configuration, TaskSpacePoint};
use nalgebra::Isometry3;
use thiserror::Error;

/// Errors an inverse-kinematics implementation may report.
#[derive(Debug, Error)]
pub enum KinematicsError {
    /// No valid configuration can reach the requested pose.
    #[error("target pose is unreachable")]
    TargetUnreachable,
    /// The requested operation is not implemented by a model.
    #[error("kinematics operation is not implemented")]
    NotImplemented,
}

/// Computes link or end-effector poses from a joint-space configuration.
pub trait ForwardKinematics {
    /// Computes the end-effector pose in the robot's world frame.
    fn forward(&self, configuration: &Configuration) -> Result<Isometry3<f64>, KinematicsError>;
}

/// Solves for a joint-space configuration that reaches a task-space target.
pub trait InverseKinematics {
    /// Returns a valid configuration for `target`, if one exists.
    fn inverse(&self, target: &TaskSpacePoint) -> Result<Configuration, KinematicsError>;
}
