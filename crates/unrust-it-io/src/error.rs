//! Error types for declarative world and trajectory I/O.

use std::path::PathBuf;
use thiserror::Error;
use unrust_it::{CollisionError, KinematicsError, RobotError, SceneError};
use unrust_it_urdf::UrdfError;

/// Unified error type returned by `unrust-it-io` parsing, serialization,
/// compilation, and trajectory resolution routines.
#[derive(Debug, Error)]
pub enum IoError {
    /// Filesystem I/O failure.
    #[error("I/O error{}: {source}", path.as_ref().map(|p| format!(" on `{}`", p.display())).unwrap_or_default())]
    Io {
        /// Optional path associated with the failing I/O operation.
        path: Option<PathBuf>,
        /// Underlying OS error.
        #[source]
        source: std::io::Error,
    },
    /// JSON parse or serialization failure.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// TOML parse failure.
    #[error("TOML parse error: {0}")]
    TomlDeserialize(#[from] toml::de::Error),
    /// TOML serialization failure.
    #[error("TOML serialization error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),
    /// URDF or SRDF loading error.
    #[error("URDF/SRDF error: {0}")]
    Urdf(#[from] UrdfError),
    /// Core kinematic model error.
    #[error("robot model error: {0}")]
    Robot(#[from] RobotError),
    /// Forward/inverse kinematics evaluation error.
    #[error("kinematics error: {0}")]
    Kinematics(#[from] KinematicsError),
    /// Collision environment configuration error.
    #[error("collision environment error: {0}")]
    Collision(#[from] CollisionError),
    /// `WorldScene` construction or ESDF carving error.
    #[error("scene compilation error: {0}")]
    Scene(#[from] SceneError),
    /// Schema major version is not supported by this crate.
    #[error("unsupported schema_version `{found}` (expected `{expected}`)")]
    UnsupportedSchemaVersion {
        /// Supported schema version.
        expected: u32,
        /// Version found in input file.
        found: u32,
    },
    /// File extension is neither `.toml` nor `.json`.
    #[error("unsupported file format extension `{0}` (expected `.toml` or `.json`)")]
    UnsupportedFormat(String),
    /// Semantic validation failure on a world or trajectory spec.
    #[error("validation error: {0}")]
    Validation(String),
    /// Reference to a robot UID not present in the compiled world.
    #[error("unknown robot UID `{0}`")]
    UnknownRobot(String),
    /// Reference to an obstacle UID not present in the compiled world.
    #[error("unknown obstacle UID `{0}`")]
    UnknownObstacle(String),
    /// Reference to a link name not present in the kinematic model.
    #[error("unknown link `{0}`")]
    UnknownLink(String),
    /// Reference to a coordinate frame or planning group not present in the model.
    #[error("unknown frame `{frame}` for robot `{robot_uid}`")]
    UnknownFrame {
        /// Robot UID associated with the frame reference (or `"<world>"`).
        robot_uid: String,
        /// Frame name that could not be resolved.
        frame: String,
    },
    /// Reference to an unknown SRDF planning group.
    #[error("unknown planning group `{group}` for robot `{robot_uid}`")]
    UnknownGroup {
        /// Robot UID.
        robot_uid: String,
        /// Group name requested.
        group: String,
    },
    /// Duplicate identifier where uniqueness is required.
    #[error("duplicate {kind} identifier `{id}`")]
    DuplicateId {
        /// Kind of entity (`robot`, `obstacle`, `trajectory`, `frame`, `subframe`).
        kind: &'static str,
        /// Conflicting identifier.
        id: String,
    },
    /// Joint vector length mismatch for a robot instance.
    #[error("joint vector length mismatch for robot `{robot_uid}`: expected {expected}, got {got}")]
    JointCountMismatch {
        /// Robot UID.
        robot_uid: String,
        /// Expected DOF count of the robot.
        expected: usize,
        /// Supplied vector length.
        got: usize,
    },
    /// Invalid or degenerate primitive shape dimension.
    #[error("invalid primitive shape `{0}`")]
    InvalidShape(String),
}

impl IoError {
    pub(crate) fn io_with_path(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: Some(path.into()),
            source,
        }
    }
}
