//! Serializable 3D pose and primitive geometry specifications.

use crate::error::IoError;
use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, Vector3};
use parry3d::shape::{Ball, Capsule, Cuboid, Cylinder, SharedShape};
use serde::{Deserialize, Serialize};
use std::f64::consts::FRAC_PI_2;
use std::path::Path;

/// Supported on-disk interchange formats for declarative specs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileFormat {
    /// Human-authored TOML (`.toml`).
    Toml,
    /// Structured JSON (`.json`).
    Json,
}

impl FileFormat {
    /// Detects [`FileFormat`] from a file extension (case-insensitive).
    pub fn from_path(path: &Path) -> Result<Self, IoError> {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        match ext.as_str() {
            "toml" => Ok(Self::Toml),
            "json" => Ok(Self::Json),
            other => Err(IoError::UnsupportedFormat(other.to_string())),
        }
    }
}

/// Serializable SE(3) pose expressed as translation `xyz` (meters) and
/// URDF-style Euler angles `rpy` `[roll, pitch, yaw]` (radians).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PoseSpec {
    /// Translation `[x, y, z]` in meters.
    #[serde(default)]
    pub xyz: [f64; 3],
    /// Intrinsic ZYX / extrinsic XYZ Euler rotation `[roll, pitch, yaw]` in radians.
    #[serde(default)]
    pub rpy: [f64; 3],
}

impl Default for PoseSpec {
    fn default() -> Self {
        Self {
            xyz: [0.0, 0.0, 0.0],
            rpy: [0.0, 0.0, 0.0],
        }
    }
}

impl PoseSpec {
    /// Identity pose at the origin.
    pub const IDENTITY: Self = Self {
        xyz: [0.0, 0.0, 0.0],
        rpy: [0.0, 0.0, 0.0],
    };

    /// Constructs a pure-translation pose `[x, y, z]` with zero rotation.
    pub const fn from_xyz(xyz: [f64; 3]) -> Self {
        Self {
            xyz,
            rpy: [0.0, 0.0, 0.0],
        }
    }

    /// Constructs a pose from translation `xyz` (m) and Euler `rpy` (rad).
    pub const fn from_xyz_rpy(xyz: [f64; 3], rpy: [f64; 3]) -> Self {
        Self { xyz, rpy }
    }

    /// Extracts a [`PoseSpec`] from a `nalgebra::Isometry3<f64>`.
    pub fn from_isometry(iso: &Isometry3<f64>) -> Self {
        let (r, p, y) = iso.rotation.euler_angles();
        Self {
            xyz: [iso.translation.x, iso.translation.y, iso.translation.z],
            rpy: [r, p, y],
        }
    }

    /// Validates that all pose coordinates are finite numbers.
    pub fn validate(&self, context: &str) -> Result<(), IoError> {
        if self.xyz.iter().any(|v| !v.is_finite()) || self.rpy.iter().any(|v| !v.is_finite()) {
            return Err(IoError::Validation(format!(
                "non-finite pose value in `{context}`"
            )));
        }
        Ok(())
    }

    /// Converts this pose to `nalgebra::Isometry3<f64>`.
    pub fn to_isometry(&self) -> Isometry3<f64> {
        let trans = Translation3::new(self.xyz[0], self.xyz[1], self.xyz[2]);
        let rot = UnitQuaternion::from_euler_angles(self.rpy[0], self.rpy[1], self.rpy[2]);
        Isometry3::from_parts(trans, rot)
    }
}

/// Declarative collision primitive shape specification.
///
/// Cylinders and axial capsules default to local **+Z** alignment to match URDF
/// `<geometry>` conventions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ShapeSpec {
    /// Axis-aligned box centered at the local origin with half-extents `[hx, hy, hz]` (meters).
    Box {
        /// Half-extents along X, Y, Z in meters (must be finite and `> 0`).
        half_extents: [f64; 3],
    },
    /// Sphere centered at the local origin.
    Sphere {
        /// Radius in meters (must be finite and `> 0`).
        radius: f64,
    },
    /// Z-aligned right circular cylinder centered at the local origin.
    Cylinder {
        /// Radius in meters (must be finite and `> 0`).
        radius: f64,
        /// Half-height along local Z in meters (must be finite and `> 0`).
        half_height: f64,
    },
    /// Z-aligned capsule centered at the local origin.
    Capsule {
        /// Hemisphere radius in meters (must be finite and `> 0`).
        radius: f64,
        /// Half-length of the cylindrical shaft along local Z in meters (`>= 0`).
        half_height: f64,
    },
    /// Capsule defined between two arbitrary local-frame endpoints `a` and `b`.
    SegmentCapsule {
        /// Start point `[x, y, z]` in meters.
        a: [f64; 3],
        /// End point `[x, y, z]` in meters.
        b: [f64; 3],
        /// Radius in meters (must be finite and `> 0`).
        radius: f64,
    },
}

impl ShapeSpec {
    /// Validates finite, positive primitive dimensions.
    pub fn validate(&self) -> Result<(), IoError> {
        let valid_pos = |v: f64| v.is_finite() && v > 0.0;
        let valid_nonneg = |v: f64| v.is_finite() && v >= 0.0;
        match *self {
            Self::Box { half_extents } => {
                if half_extents.iter().any(|&v| !valid_pos(v)) {
                    return Err(IoError::InvalidShape(
                        "box half_extents must be finite and > 0".into(),
                    ));
                }
            }
            Self::Sphere { radius } => {
                if !valid_pos(radius) {
                    return Err(IoError::InvalidShape(
                        "sphere radius must be finite and > 0".into(),
                    ));
                }
            }
            Self::Cylinder {
                radius,
                half_height,
            } => {
                if !valid_pos(radius) || !valid_pos(half_height) {
                    return Err(IoError::InvalidShape(
                        "cylinder radius and half_height must be finite and > 0".into(),
                    ));
                }
            }
            Self::Capsule {
                radius,
                half_height,
            } => {
                if !valid_pos(radius) || !valid_nonneg(half_height) {
                    return Err(IoError::InvalidShape(
                        "capsule radius must be > 0 and half_height >= 0".into(),
                    ));
                }
            }
            Self::SegmentCapsule { a, b, radius } => {
                if a.iter().any(|v| !v.is_finite())
                    || b.iter().any(|v| !v.is_finite())
                    || !valid_pos(radius)
                {
                    return Err(IoError::InvalidShape(
                        "segment_capsule endpoints must be finite and radius > 0".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Converts this shape into `(local_offset_pose, parry3d SharedShape)`.
    pub fn to_parry_shape_and_pose(&self) -> Result<(Isometry3<f64>, SharedShape), IoError> {
        self.validate()?;
        Ok(match *self {
            Self::Box { half_extents } => (
                Isometry3::identity(),
                SharedShape::new(Cuboid::new(Vector3::from(half_extents).into())),
            ),
            Self::Sphere { radius } => (Isometry3::identity(), SharedShape::new(Ball::new(radius))),
            Self::Cylinder {
                radius,
                half_height,
            } => (
                // `parry3d::Cylinder` aligns along local +Y; rotate +90 deg about X to match +Z.
                Isometry3::rotation(Vector3::x() * FRAC_PI_2),
                SharedShape::new(Cylinder::new(half_height, radius)),
            ),
            Self::Capsule {
                radius,
                half_height,
            } => (
                Isometry3::identity(),
                SharedShape::new(Capsule::new_z(half_height, radius)),
            ),
            Self::SegmentCapsule { a, b, radius } => (
                Isometry3::identity(),
                SharedShape::new(Capsule::new(
                    Point3::from(a).into(),
                    Point3::from(b).into(),
                    radius,
                )),
            ),
        })
    }
}
