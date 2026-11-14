//! Robot descriptions and strongly typed configuration values.

use nalgebra::Point3;
use thiserror::Error;

/// A joint's inclusive limits, expressed in radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointLimit {
    /// Smallest permitted joint value.
    pub min: f64,
    /// Largest permitted joint value.
    pub max: f64,
}

impl JointLimit {
    /// Creates a joint limit and rejects an inverted range.
    pub fn new(min: f64, max: f64) -> Self {
        assert!(min <= max, "joint minimum must not exceed its maximum");
        Self { min, max }
    }

    /// Returns whether a joint value is inside this limit.
    pub fn contains(&self, value: f64) -> bool {
        (self.min..=self.max).contains(&value)
    }
}

/// A point in configuration space (C-space), one value per joint.
#[derive(Debug, Clone, PartialEq)]
pub struct Configuration(pub Vec<f64>);

impl Configuration {
    /// Creates a joint-space configuration in radians.
    pub fn new(values: Vec<f64>) -> Self {
        Self(values)
    }

    /// Returns the number of degrees of freedom represented.
    pub fn dof(&self) -> usize {
        self.0.len()
    }
}

/// A target in task space, kept distinct from a joint configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskSpacePoint {
    /// Cartesian position in the robot's world frame.
    pub position: Point3<f64>,
}

impl TaskSpacePoint {
    /// Creates a Cartesian task-space target.
    pub fn new(position: Point3<f64>) -> Self {
        Self { position }
    }
}

/// Errors raised while constructing or updating a robot.
#[derive(Debug, Error, PartialEq)]
pub enum RobotError {
    /// The configuration has a different number of values than the robot DOF.
    #[error("configuration has {actual} joints, but robot has {expected} degrees of freedom")]
    WrongDof { expected: usize, actual: usize },
    /// A joint value is outside its configured limit.
    #[error("joint {joint} value {value} is outside [{min}, {max}]")]
    JointOutOfBounds {
        joint: usize,
        value: f64,
        min: f64,
        max: f64,
    },
}

/// The robot model used by kinematics, collision checking, and planning.
#[derive(Debug, Clone)]
pub struct Robot {
    /// Per-joint limits defining the valid portion of C-space.
    pub joint_limits: Vec<JointLimit>,
    /// The robot's current joint-space configuration.
    pub configuration: Configuration,
}

impl Robot {
    /// Creates a robot at the midpoint of every joint limit.
    pub fn new(joint_limits: Vec<JointLimit>) -> Result<Self, RobotError> {
        let configuration = Configuration::new(
            joint_limits
                .iter()
                .map(|limit| (limit.min + limit.max) / 2.0)
                .collect(),
        );
        Ok(Self {
            joint_limits,
            configuration,
        })
    }

    /// Updates the current configuration after checking DOF and joint limits.
    pub fn set_configuration(&mut self, configuration: Configuration) -> Result<(), RobotError> {
        self.validate_configuration(&configuration)?;
        self.configuration = configuration;
        Ok(())
    }

    /// Checks whether a configuration is valid for this robot.
    pub fn validate_configuration(&self, configuration: &Configuration) -> Result<(), RobotError> {
        if configuration.dof() != self.joint_limits.len() {
            return Err(RobotError::WrongDof {
                expected: self.joint_limits.len(),
                actual: configuration.dof(),
            });
        }
        for (joint, (value, limit)) in configuration.0.iter().zip(&self.joint_limits).enumerate() {
            if !limit.contains(*value) {
                return Err(RobotError::JointOutOfBounds {
                    joint,
                    value: *value,
                    min: limit.min,
                    max: limit.max,
                });
            }
        }
        Ok(())
    }
}
