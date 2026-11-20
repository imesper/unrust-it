//! Multi-tier collision detection, bitset Allowed Collision Matrix (ACM),
//! BVH broad-phase, continuous collision checking, and differentiable signed-distance queries.

use crate::kinematics::{KinematicsError, TransformBuffer};
use crate::robot::{
    Configuration, JointType, LinkId, MultiRobotModel, ObstacleId, PayloadId, Robot, RobotError,
};
use nalgebra::{Isometry3, Point3, Quaternion, Translation3, UnitQuaternion, Vector3};
use parry3d::bounding_volume::{Aabb, BoundingVolume};
use parry3d::math::{Pose as ParryPose, Rotation as ParryRotation, Vector as ParryVector};
use parry3d::partitioning::{Bvh, BvhBuildStrategy};
use parry3d::shape::{SharedShape, TypedShape};
use std::collections::HashMap;
use std::ops::Range;
use thiserror::Error;
use wide::f64x4;

type ShapeWitness = (f64, Point3<f64>, Point3<f64>, Vector3<f64>);

/// Errors produced while evaluating collision or signed-distance queries.
#[derive(Debug, Error, PartialEq)]
pub enum CollisionError {
    /// Collision geometry or kinematic configuration could not be evaluated.
    #[error("collision geometry could not be evaluated")]
    GeometryUnavailable,
    /// Invalid robot configuration passed to a collision query.
    #[error(transparent)]
    InvalidConfiguration(#[from] RobotError),
    /// Forward kinematics evaluation failed during collision checking.
    #[error(transparent)]
    Kinematics(#[from] KinematicsError),
}

/// Converts a `nalgebra::Isometry3<f64>` into a `parry3d` `Pose` (`f64`).
#[inline]
pub fn isometry_to_parry(iso: &Isometry3<f64>) -> ParryPose {
    let t = iso.translation.vector;
    let q = iso.rotation.coords;
    ParryPose::from_parts(
        ParryVector::new(t.x, t.y, t.z),
        ParryRotation::from_xyzw(q.x, q.y, q.z, q.w),
    )
}

/// Converts a `parry3d` `Pose` (`f64`) into a `nalgebra::Isometry3<f64>`.
#[inline]
pub fn parry_to_isometry(pose: &ParryPose) -> Isometry3<f64> {
    let t = Translation3::new(pose.translation.x, pose.translation.y, pose.translation.z);
    let q = UnitQuaternion::from_quaternion(Quaternion::new(
        pose.rotation.w,
        pose.rotation.x,
        pose.rotation.y,
        pose.rotation.z,
    ));
    Isometry3::from_parts(t, q)
}

/// Converts a `parry3d` `Vector` (`f64`) into a `nalgebra::Point3<f64>`.
#[inline]
pub fn parry_to_point(v: ParryVector) -> Point3<f64> {
    Point3::new(v.x, v.y, v.z)
}

/// Converts a `parry3d` `Vector` (`f64`) into a `nalgebra::Vector3<f64>`.
#[inline]
pub fn parry_to_vector(v: ParryVector) -> Vector3<f64> {
    Vector3::new(v.x, v.y, v.z)
}

/// Geometric fidelity level requested by a planner or reactive controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryTier {
    /// Tier 1: Fast analytic Swept-Sphere Volumes (SoA spheres & capsules, SIMD-accelerated, $C^\infty$ smooth normals).
    AnalyticSsv,
    /// Tier 2: Exact convex shapes & primitives via `parry3d` GJK/EPA.
    Convex,
}

/// A sphere primitive in a link's local coordinate frame (Tier 1 SSV).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingSphere {
    /// Center of the sphere in local link coordinates.
    pub center: Point3<f64>,
    /// Sphere radius in meters.
    pub radius: f64,
}

impl BoundingSphere {
    /// Creates a new bounding sphere.
    pub fn new(center: Point3<f64>, radius: f64) -> Self {
        assert!(radius >= 0.0, "sphere radius must be non-negative");
        Self { center, radius }
    }
}

/// A line-swept sphere (capsule) primitive in a link's local coordinate frame (Tier 1 SSV).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkCapsule {
    /// First segment endpoint in local link coordinates.
    pub endpoint_a: Point3<f64>,
    /// Second segment endpoint in local link coordinates.
    pub endpoint_b: Point3<f64>,
    /// Capsule radius in meters.
    pub radius: f64,
}

impl LinkCapsule {
    /// Creates a new link capsule.
    pub fn new(endpoint_a: Point3<f64>, endpoint_b: Point3<f64>, radius: f64) -> Self {
        assert!(radius >= 0.0, "capsule radius must be non-negative");
        Self {
            endpoint_a,
            endpoint_b,
            radius,
        }
    }

    /// Transforms this capsule into world coordinates given `link_pose`.
    #[inline]
    pub fn transformed(&self, link_pose: &Isometry3<f64>) -> Self {
        Self {
            endpoint_a: link_pose.transform_point(&self.endpoint_a),
            endpoint_b: link_pose.transform_point(&self.endpoint_b),
            radius: self.radius,
        }
    }

    /// Computes analytical closest witness points `(p_self, p_other)`, unit normal (from `other` to `self`),
    /// and signed distance between two world-space capsules.
    pub fn signed_distance_to_capsule(
        &self,
        other: &Self,
    ) -> (f64, Point3<f64>, Point3<f64>, Vector3<f64>) {
        let (c_a, c_b) = closest_points_segments(
            &self.endpoint_a,
            &self.endpoint_b,
            &other.endpoint_a,
            &other.endpoint_b,
        );
        let diff = c_a - c_b;
        let center_dist = diff.norm();
        let normal = if center_dist > 1e-12 {
            diff / center_dist
        } else {
            Vector3::z()
        };
        let witness_a = c_a - normal * self.radius;
        let witness_b = c_b + normal * other.radius;
        let signed_dist = center_dist - (self.radius + other.radius);
        (signed_dist, witness_a, witness_b, normal)
    }
}

/// Computes the closest points `(c1, c2)` between 3D line segments `[p1, q1]` and `[p2, q2]`.
#[inline]
pub fn closest_points_segments(
    p1: &Point3<f64>,
    q1: &Point3<f64>,
    p2: &Point3<f64>,
    q2: &Point3<f64>,
) -> (Point3<f64>, Point3<f64>) {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.dot(&d1);
    let e = d2.dot(&d2);
    let f = d2.dot(&r);

    let (s, t) = if a <= 1e-12 && e <= 1e-12 {
        (0.0, 0.0)
    } else if a <= 1e-12 {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = d1.dot(&r);
        if e <= 1e-12 {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = d1.dot(&d2);
            let denom = a * e - b * b;
            let mut s = if denom != 0.0 {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
            (s, t)
        }
    };

    (p1 + d1 * s, p2 + d2 * t)
}

#[inline]
fn aabb_is_valid(aabb: &Aabb) -> bool {
    aabb.mins.x <= aabb.maxs.x && aabb.mins.y <= aabb.maxs.y && aabb.mins.z <= aabb.maxs.z
}

fn sample_spheres_along_segment_capped(
    endpoint_a: Point3<f64>,
    endpoint_b: Point3<f64>,
    radius: f64,
    max_spheres: usize,
) -> Vec<BoundingSphere> {
    let r = radius.max(1e-4);
    let seg_vec = endpoint_b - endpoint_a;
    let len = seg_vec.norm();
    if len <= 1e-6 {
        return vec![BoundingSphere::new(endpoint_a, r)];
    }
    let num_spheres = ((len / r.max(1e-3)).ceil() as usize).clamp(1, max_spheres.max(2)) + 1;
    let mut spheres = Vec::with_capacity(num_spheres);
    for i in 0..num_spheres {
        let alpha = (i as f64) / ((num_spheres - 1) as f64);
        spheres.push(BoundingSphere::new(endpoint_a + seg_vec * alpha, r));
    }
    spheres
}

fn decompose_cuboid_endpoints(
    local_pose: &Isometry3<f64>,
    hx: f64,
    hy: f64,
    hz: f64,
) -> (Point3<f64>, Point3<f64>, f64) {
    let hx = hx.abs().max(1e-4);
    let hy = hy.abs().max(1e-4);
    let hz = hz.abs().max(1e-4);
    if hx >= hy && hx >= hz {
        let r_cross = hy.hypot(hz);
        let span = (hx - 0.5 * r_cross).max(0.0);
        let sphere_r = if span <= 1e-9 {
            (hx * hx + hy * hy + hz * hz).sqrt()
        } else {
            r_cross.hypot(hx - span)
        };
        (
            local_pose.transform_point(&Point3::new(-span, 0.0, 0.0)),
            local_pose.transform_point(&Point3::new(span, 0.0, 0.0)),
            sphere_r,
        )
    } else if hy >= hx && hy >= hz {
        let r_cross = hx.hypot(hz);
        let span = (hy - 0.5 * r_cross).max(0.0);
        let sphere_r = if span <= 1e-9 {
            (hx * hx + hy * hy + hz * hz).sqrt()
        } else {
            r_cross.hypot(hy - span)
        };
        (
            local_pose.transform_point(&Point3::new(0.0, -span, 0.0)),
            local_pose.transform_point(&Point3::new(0.0, span, 0.0)),
            sphere_r,
        )
    } else {
        let r_cross = hx.hypot(hy);
        let span = (hz - 0.5 * r_cross).max(0.0);
        let sphere_r = if span <= 1e-9 {
            (hx * hx + hy * hy + hz * hz).sqrt()
        } else {
            r_cross.hypot(hz - span)
        };
        (
            local_pose.transform_point(&Point3::new(0.0, 0.0, -span)),
            local_pose.transform_point(&Point3::new(0.0, 0.0, span)),
            sphere_r,
        )
    }
}

/// Automatically decomposes a `parry3d` `SharedShape` at `local_pose` into tight Tier-1
/// bounding spheres suitable for SIMD and GPU batched rollout evaluation.
pub fn decompose_shape_to_spheres(
    local_pose: &Isometry3<f64>,
    shape: &SharedShape,
) -> Vec<BoundingSphere> {
    match shape.as_typed_shape() {
        TypedShape::Ball(b) => {
            vec![BoundingSphere::new(
                local_pose.transform_point(&Point3::origin()),
                b.radius.max(1e-4),
            )]
        }
        TypedShape::Capsule(c) => {
            let a = local_pose.transform_point(&parry_to_point(c.segment.a));
            let b = local_pose.transform_point(&parry_to_point(c.segment.b));
            sample_spheres_along_segment_capped(a, b, c.radius.max(1e-4), 12)
        }
        TypedShape::Cylinder(c) => {
            // parry3d::shape::Cylinder is always aligned along local Y.
            let hh = c.half_height.max(0.0);
            let a = local_pose.transform_point(&Point3::new(0.0, -hh, 0.0));
            let b = local_pose.transform_point(&Point3::new(0.0, hh, 0.0));
            sample_spheres_along_segment_capped(a, b, c.radius.max(1e-4), 12)
        }
        TypedShape::Cuboid(b) => {
            let (p_a, p_b, sphere_r) = decompose_cuboid_endpoints(
                local_pose,
                b.half_extents.x,
                b.half_extents.y,
                b.half_extents.z,
            );
            sample_spheres_along_segment_capped(p_a, p_b, sphere_r, 12)
        }
        TypedShape::Compound(comp) => {
            let mut out = Vec::new();
            for (sub_pose, sub_shape) in comp.shapes() {
                let combined = local_pose * parry_to_isometry(sub_pose);
                out.extend(decompose_shape_to_spheres(&combined, sub_shape));
            }
            if out.is_empty() {
                let parry_pose = isometry_to_parry(local_pose);
                let aabb = shape.compute_aabb(&parry_pose);
                let center = parry_to_point(aabb.center());
                let radius = parry_to_vector(aabb.half_extents()).norm().max(1e-4);
                out.push(BoundingSphere::new(center, radius));
            }
            out
        }
        _ => {
            let parry_pose = isometry_to_parry(local_pose);
            let aabb = shape.compute_aabb(&parry_pose);
            let center = parry_to_point(aabb.center());
            let radius = parry_to_vector(aabb.half_extents()).norm().max(1e-4);
            vec![BoundingSphere::new(center, radius)]
        }
    }
}

/// Multi-tier collision geometry attached to a single robot link or tool payload.
#[derive(Debug, Clone)]
pub struct LinkGeometry {
    /// Local-frame bounding AABB enclosing all representations of this link.
    pub local_aabb: Aabb,
    /// Optional Tier-1 bounding capsule.
    pub bounding_capsule: Option<LinkCapsule>,
    /// Tier-1 multi-sphere decomposition in link-local coordinates.
    pub local_spheres: Vec<BoundingSphere>,
    /// Tier-2 `parry3d` shapes with local pose offsets relative to the link frame.
    pub convex_shapes: Vec<(Isometry3<f64>, SharedShape)>,
}

impl LinkGeometry {
    fn single_shape_capsule(local_pose: &Isometry3<f64>, shape: &SharedShape) -> Option<LinkCapsule> {
        match shape.as_typed_shape() {
            TypedShape::Ball(b) => {
                let p = local_pose.transform_point(&Point3::origin());
                Some(LinkCapsule::new(p, p, b.radius.max(1e-4)))
            }
            TypedShape::Capsule(c) => Some(LinkCapsule::new(
                local_pose.transform_point(&parry_to_point(c.segment.a)),
                local_pose.transform_point(&parry_to_point(c.segment.b)),
                c.radius.max(1e-4),
            )),
            TypedShape::Cylinder(c) => {
                let hh = c.half_height.max(0.0);
                Some(LinkCapsule::new(
                    local_pose.transform_point(&Point3::new(0.0, -hh, 0.0)),
                    local_pose.transform_point(&Point3::new(0.0, hh, 0.0)),
                    c.radius.max(1e-4),
                ))
            }
            TypedShape::Cuboid(b) => {
                let (p_a, p_b, sphere_r) = decompose_cuboid_endpoints(
                    local_pose,
                    b.half_extents.x,
                    b.half_extents.y,
                    b.half_extents.z,
                );
                Some(LinkCapsule::new(p_a, p_b, sphere_r))
            }
            _ => {
                let aabb = shape.compute_aabb(&isometry_to_parry(local_pose));
                let center = parry_to_point(aabb.center());
                let radius = parry_to_vector(aabb.half_extents()).norm().max(1e-4);
                Some(LinkCapsule::new(center, center, radius))
            }
        }
    }

    fn capsule_from_aabb(local_aabb: &Aabb) -> Option<LinkCapsule> {
        if !aabb_is_valid(local_aabb) {
            return None;
        }
        let center = parry_to_point(local_aabb.center());
        let radius = parry_to_vector(local_aabb.half_extents()).norm().max(1e-4);
        Some(LinkCapsule::new(center, center, radius))
    }

    /// Creates a link geometry from a single `parry3d` shape and automatically populates tight Tier-1 spheres/capsule.
    pub fn from_shape(local_pose: Isometry3<f64>, shape: SharedShape) -> Self {
        Self::from_convex_shapes(vec![(local_pose, shape)])
    }

    /// Creates a multi-part compound geometry from multiple posed convex shapes,
    /// decomposing each sub-shape into tight Tier-1 spheres.
    pub fn from_convex_shapes(shapes: Vec<(Isometry3<f64>, SharedShape)>) -> Self {
        let mut local_aabb = Aabb::new_invalid();
        let mut local_spheres = Vec::new();
        for (pose, shape) in &shapes {
            let aabb = shape.compute_aabb(&isometry_to_parry(pose));
            if aabb_is_valid(&local_aabb) {
                local_aabb.merge(&aabb);
            } else {
                local_aabb = aabb;
            }
            local_spheres.extend(decompose_shape_to_spheres(pose, shape));
        }

        let bounding_capsule = if shapes.len() == 1 {
            Self::single_shape_capsule(&shapes[0].0, &shapes[0].1)
        } else {
            Self::capsule_from_aabb(&local_aabb)
        };

        Self {
            local_aabb,
            bounding_capsule,
            local_spheres,
            convex_shapes: shapes,
        }
    }

    /// Creates a compound geometry from explicit convex sub-shapes and custom Tier-1 spheres.
    ///
    /// If `local_spheres` is empty, each convex sub-shape is automatically decomposed into spheres.
    pub fn from_compound(
        shapes: Vec<(Isometry3<f64>, SharedShape)>,
        local_spheres: Vec<BoundingSphere>,
    ) -> Self {
        if local_spheres.is_empty() {
            return Self::from_convex_shapes(shapes);
        }
        let mut local_aabb = Aabb::new_invalid();
        for (pose, shape) in &shapes {
            let aabb = shape.compute_aabb(&isometry_to_parry(pose));
            if aabb_is_valid(&local_aabb) {
                local_aabb.merge(&aabb);
            } else {
                local_aabb = aabb;
            }
        }
        if !aabb_is_valid(&local_aabb) {
            for s in &local_spheres {
                let half = ParryVector::splat(s.radius);
                let c = ParryVector::new(s.center.x, s.center.y, s.center.z);
                let aabb = Aabb::from_half_extents(c, half);
                if aabb_is_valid(&local_aabb) {
                    local_aabb.merge(&aabb);
                } else {
                    local_aabb = aabb;
                }
            }
        }
        let bounding_capsule = Self::capsule_from_aabb(&local_aabb);
        Self {
            local_aabb,
            bounding_capsule,
            local_spheres,
            convex_shapes: shapes,
        }
    }

    /// Appends another posed convex shape to this compound geometry and updates its AABB and Tier-1 spheres.
    pub fn add_shape(&mut self, local_pose: Isometry3<f64>, shape: SharedShape) {
        let aabb = shape.compute_aabb(&isometry_to_parry(&local_pose));
        if aabb_is_valid(&self.local_aabb) {
            self.local_aabb.merge(&aabb);
        } else {
            self.local_aabb = aabb;
        }
        self.local_spheres
            .extend(decompose_shape_to_spheres(&local_pose, &shape));
        self.convex_shapes.push((local_pose, shape));
        self.bounding_capsule = if self.convex_shapes.len() == 1 {
            Self::single_shape_capsule(&self.convex_shapes[0].0, &self.convex_shapes[0].1)
        } else {
            Self::capsule_from_aabb(&self.local_aabb)
        };
    }

    /// Returns the maximum radial extent from the link/payload local origin across all bounding AABB corners.
    #[inline]
    pub fn max_extent_from_origin(&self) -> f64 {
        if !aabb_is_valid(&self.local_aabb) {
            return 0.0;
        }
        let mx = self.local_aabb.mins.x.abs().max(self.local_aabb.maxs.x.abs());
        let my = self.local_aabb.mins.y.abs().max(self.local_aabb.maxs.y.abs());
        let mz = self.local_aabb.mins.z.abs().max(self.local_aabb.maxs.z.abs());
        (mx * mx + my * my + mz * mz).sqrt()
    }

    /// Creates a link geometry from a capsule (generating both Tier-1 spheres/capsule and a Tier-2 `parry3d` capsule).
    pub fn from_capsule(endpoint_a: Point3<f64>, endpoint_b: Point3<f64>, radius: f64) -> Self {
        let local_spheres = sample_spheres_along_segment_capped(endpoint_a, endpoint_b, radius, 8);

        let parry_capsule = SharedShape::new(parry3d::shape::Capsule::new(
            ParryVector::new(endpoint_a.x, endpoint_a.y, endpoint_a.z),
            ParryVector::new(endpoint_b.x, endpoint_b.y, endpoint_b.z),
            radius,
        ));
        let local_aabb = parry_capsule.compute_local_aabb();
        Self {
            local_aabb,
            bounding_capsule: Some(LinkCapsule::new(endpoint_a, endpoint_b, radius)),
            local_spheres,
            convex_shapes: vec![(Isometry3::identity(), parry_capsule)],
        }
    }

    /// Computes the world-space AABB of this link geometry given `link_pose`.
    #[inline]
    pub fn world_aabb(&self, link_pose: &Isometry3<f64>) -> Aabb {
        let pose = isometry_to_parry(link_pose);
        self.local_aabb.transform_by(&pose)
    }
}

/// An object or end-of-arm tool dynamically grasped by or attached to a robot link.
#[derive(Debug, Clone)]
pub struct AttachedPayload {
    /// Unique identifier for the payload or tool.
    pub id: String,
    /// Link to which this payload is rigidly attached.
    pub parent_link: LinkId,
    /// Pose of the payload relative to `parent_link`.
    pub local_pose: Isometry3<f64>,
    /// Multi-tier collision geometry of the payload (compound Tier-2 shapes + Tier-1 spheres).
    pub geometry: LinkGeometry,
    /// Payload mass in kilograms (for dynamic manipulation).
    pub mass: f64,
    /// Links allowed to touch this payload without triggering collision (e.g., flange, wrist, fingers).
    pub touch_links: Vec<LinkId>,
    /// Optional payload-specific safety margin override in meters (e.g., 3-5 mm for tool tips).
    pub safety_margin: Option<f64>,
    /// Optional payload-specific CBF/MPPI activation distance override in meters.
    pub activation_distance: Option<f64>,
}

impl AttachedPayload {
    /// Creates an attached payload or tool from a [`LinkGeometry`].
    pub fn new(
        id: impl Into<String>,
        parent_link: LinkId,
        local_pose: Isometry3<f64>,
        geometry: LinkGeometry,
    ) -> Self {
        Self {
            id: id.into(),
            parent_link,
            local_pose,
            geometry,
            mass: 0.0,
            touch_links: vec![parent_link],
            safety_margin: None,
            activation_distance: None,
        }
    }

    /// Creates an attached payload from a single `parry3d` shape.
    pub fn from_shape(
        id: impl Into<String>,
        parent_link: LinkId,
        local_pose: Isometry3<f64>,
        shape: SharedShape,
    ) -> Self {
        Self::new(
            id,
            parent_link,
            local_pose,
            LinkGeometry::from_shape(Isometry3::identity(), shape),
        )
    }

    /// Creates an attached compound tool or grasped object from multiple posed `parry3d` shapes.
    pub fn from_compound(
        id: impl Into<String>,
        parent_link: LinkId,
        local_pose: Isometry3<f64>,
        shapes: Vec<(Isometry3<f64>, SharedShape)>,
    ) -> Self {
        Self::new(
            id,
            parent_link,
            local_pose,
            LinkGeometry::from_convex_shapes(shapes),
        )
    }

    /// Sets the payload mass in kilograms.
    pub fn with_mass(mut self, mass: f64) -> Self {
        self.mass = mass;
        self
    }

    /// Sets the links allowed to touch this payload without triggering collisions.
    pub fn with_touch_links(mut self, touch_links: Vec<LinkId>) -> Self {
        self.touch_links = touch_links;
        self
    }

    /// Sets a default safety margin override (meters) for this payload against external bodies.
    pub fn with_safety_margin(mut self, margin: f64) -> Self {
        self.safety_margin = Some(margin);
        self
    }

    /// Sets a default activation distance override (meters) for this payload in CBF/batch checks.
    pub fn with_activation_distance(mut self, activation: f64) -> Self {
        self.activation_distance = Some(activation);
        self
    }
}

/// Structure-of-Arrays (SoA) batch storage of world-space spheres across all links and attached payloads.
///
/// Enables branchless 4-wide `f64x4` SIMD distance and collision sweeps.
#[derive(Debug, Clone, Default)]
pub struct WorldSphereSoA {
    /// World X coordinates of spheres.
    pub cx: Vec<f64>,
    /// World Y coordinates of spheres.
    pub cy: Vec<f64>,
    /// World Z coordinates of spheres.
    pub cz: Vec<f64>,
    /// Sphere radii.
    pub radius: Vec<f64>,
    /// Range `link_ranges[link_id.index()]` into `cx, cy, cz, radius` for each link.
    pub link_ranges: Vec<Range<usize>>,
    /// Range `payload_ranges[payload_id.index()]` into `cx, cy, cz, radius` for each attached payload.
    pub payload_ranges: Vec<Range<usize>>,
}

impl WorldSphereSoA {
    /// Finds the closest signed-distance witness between sphere ranges `range_a` and `range_b`
    /// within `cutoff` meters.
    pub fn closest_spheres_witness(
        &self,
        range_a: &Range<usize>,
        range_b: &Range<usize>,
        cutoff: f64,
    ) -> Option<ShapeWitness> {
        if range_a.is_empty() || range_b.is_empty() {
            return None;
        }
        let mut best: Option<ShapeWitness> = None;
        for ia in range_a.clone() {
            let ca = Point3::new(self.cx[ia], self.cy[ia], self.cz[ia]);
            let ra = self.radius[ia];
            for ib in range_b.clone() {
                let cb = Point3::new(self.cx[ib], self.cy[ib], self.cz[ib]);
                let rb = self.radius[ib];
                let diff = ca - cb;
                let center_dist = diff.norm();
                let dist = center_dist - (ra + rb);
                if dist <= cutoff && best.as_ref().is_none_or(|b| dist < b.0) {
                    let normal = if center_dist > 1e-12 {
                        diff / center_dist
                    } else {
                        Vector3::z()
                    };
                    let p_a = ca - normal * ra;
                    let p_b = cb + normal * rb;
                    best = Some((dist, p_a, p_b, normal));
                }
            }
        }
        best
    }

    /// Checks using `f64x4` SIMD whether any sphere in `range_a` overlaps (within `margin`)
    /// any sphere in `range_b`.
    #[inline]
    pub fn ranges_intersect_simd(
        &self,
        range_a: &Range<usize>,
        range_b: &Range<usize>,
        margin: f64,
    ) -> bool {
        if range_a.is_empty() || range_b.is_empty() {
            return false;
        }
        let cx_b = &self.cx[range_b.clone()];
        let cy_b = &self.cy[range_b.clone()];
        let cz_b = &self.cz[range_b.clone()];
        let r_b = &self.radius[range_b.clone()];
        let len_b = cx_b.len();
        let chunks = len_b / 4;

        for idx_a in range_a.clone() {
            let ax = f64x4::splat(self.cx[idx_a]);
            let ay = f64x4::splat(self.cy[idx_a]);
            let az = f64x4::splat(self.cz[idx_a]);
            let ar = f64x4::splat(self.radius[idx_a] + margin);

            for c in 0..chunks {
                let off = c * 4;
                let bx = f64x4::from(&cx_b[off..off + 4]);
                let by = f64x4::from(&cy_b[off..off + 4]);
                let bz = f64x4::from(&cz_b[off..off + 4]);
                let br = f64x4::from(&r_b[off..off + 4]);

                let dx = ax - bx;
                let dy = ay - by;
                let dz = az - bz;
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let r_sum = ar + br;
                let r_sum_sq = r_sum * r_sum;

                let diff_arr: [f64; 4] = (dist_sq - r_sum_sq).into();
                if diff_arr.iter().any(|&d| d <= 0.0) {
                    return true;
                }
            }

            let ax_s = self.cx[idx_a];
            let ay_s = self.cy[idx_a];
            let az_s = self.cz[idx_a];
            let ar_s = self.radius[idx_a] + margin;
            for rem in (chunks * 4)..len_b {
                let dx = ax_s - cx_b[rem];
                let dy = ay_s - cy_b[rem];
                let dz = az_s - cz_b[rem];
                let r_sum = ar_s + r_b[rem];
                if dx * dx + dy * dy + dz * dz <= r_sum * r_sum {
                    return true;
                }
            }
        }
        false
    }
}

/// Rule governing whether a pair of links `(link_a, link_b)` is checked for collision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CollisionRule {
    /// Pair is checked with the default safety margin (`0.0`).
    Enabled,
    /// Pair is checked with a custom minimum allowed separation margin (meters).
    CustomMargin(f64),
    /// Disabled because the two links are directly connected by a joint.
    DisabledAdjacent,
    /// Disabled because the two links can never collide across the valid C-space.
    DisabledNeverColliding,
    /// Disabled by static configuration (e.g., SRDF default).
    DisabledDefault,
    /// Dynamically disabled at runtime (e.g., during grasping or dual-arm handover).
    DisabledDynamic,
}

/// Bitset-backed Allowed Collision Matrix (ACM) for self-collision and inter-arm collision filtering.
///
/// Uses an upper-triangular `u64` bitmask with hardware `trailing_zeros()` (`CTZ`) iteration
/// so disabled link pairs are skipped in $O(\text{active\_pairs})$ time.
#[derive(Debug, Clone)]
pub struct AllowedCollisionMatrix {
    num_links: usize,
    words_per_row: usize,
    active_upper_bitmask: Vec<u64>,
    rules: Vec<CollisionRule>,
}

impl AllowedCollisionMatrix {
    /// Creates an ACM for `num_links` with all distinct pairs `(i, j)` initially enabled.
    pub fn new(num_links: usize) -> Self {
        let words_per_row = num_links.div_ceil(64);
        let mut acm = Self {
            num_links,
            words_per_row,
            active_upper_bitmask: vec![0u64; num_links * words_per_row],
            rules: vec![CollisionRule::Enabled; num_links * num_links],
        };
        for i in 0..num_links {
            acm.rules[i * num_links + i] = CollisionRule::DisabledDefault;
            for j in (i + 1)..num_links {
                acm.set_bit(i, j, true);
            }
        }
        acm
    }

    /// Constructs an ACM from a [`MultiRobotModel`], automatically disabling self-pairs `(i, i)`
    /// and adjacent parent-child link pairs connected by a joint.
    pub fn from_model(model: &MultiRobotModel) -> Self {
        let mut acm = Self::new(model.num_links());
        for link in model.links() {
            if let Some(parent_id) = link.parent_link {
                acm.set_rule(parent_id, link.id, CollisionRule::DisabledAdjacent);
            }
        }
        acm
    }

    /// Number of links represented in this ACM.
    #[inline]
    pub fn num_links(&self) -> usize {
        self.num_links
    }

    #[inline(always)]
    fn set_bit(&mut self, i: usize, j: usize, active: bool) {
        let (r, c) = if i < j { (i, j) } else { (j, i) };
        let word_idx = r * self.words_per_row + (c / 64);
        let bit = 1u64 << (c % 64);
        if active {
            self.active_upper_bitmask[word_idx] |= bit;
        } else {
            self.active_upper_bitmask[word_idx] &= !bit;
        }
    }

    /// Sets the symmetric collision rule between `a` and `b`.
    pub fn set_rule(&mut self, a: LinkId, b: LinkId, rule: CollisionRule) {
        let (i, j) = (a.index(), b.index());
        if i == j || i >= self.num_links || j >= self.num_links {
            return;
        }
        self.rules[i * self.num_links + j] = rule;
        self.rules[j * self.num_links + i] = rule;
        let is_active = matches!(rule, CollisionRule::Enabled | CollisionRule::CustomMargin(_));
        self.set_bit(i, j, is_active);
    }

    /// Returns the configured [`CollisionRule`] for `(a, b)`.
    #[inline]
    pub fn rule(&self, a: LinkId, b: LinkId) -> CollisionRule {
        self.rules[a.index() * self.num_links + b.index()]
    }

    /// Returns the custom safety margin for `(a, b)` (`0.0` for `CollisionRule::Enabled`).
    #[inline]
    pub fn pair_margin(&self, a: LinkId, b: LinkId) -> f64 {
        self.custom_margin(a, b).unwrap_or(0.0)
    }

    /// Returns `Some(margin)` if `(a, b)` is configured with [`CollisionRule::CustomMargin`].
    #[inline]
    pub fn custom_margin(&self, a: LinkId, b: LinkId) -> Option<f64> {
        let (i, j) = (a.index(), b.index());
        if i == j || i >= self.num_links || j >= self.num_links {
            return None;
        }
        match self.rule(a, b) {
            CollisionRule::CustomMargin(m) => Some(m),
            _ => None,
        }
    }

    /// Assigns `CollisionRule::CustomMargin(margin)` to all currently active inter-arm link pairs
    /// (`model.link(a).robot != model.link(b).robot`), preserving all disabled pairs.
    pub fn set_inter_arm_margin(&mut self, model: &MultiRobotModel, margin: f64) {
        let mut pairs = Vec::new();
        self.for_each_active_pair(|a, b| {
            if model.link(a).robot != model.link(b).robot {
                pairs.push((a, b));
            }
        });
        for (a, b) in pairs {
            self.set_rule(a, b, CollisionRule::CustomMargin(margin));
        }
    }

    /// Assigns `CollisionRule::CustomMargin(margin)` to all currently active same-arm link pairs
    /// (`model.link(a).robot == model.link(b).robot`), preserving all disabled pairs.
    pub fn set_self_collision_margin(&mut self, model: &MultiRobotModel, margin: f64) {
        let mut pairs = Vec::new();
        self.for_each_active_pair(|a, b| {
            if model.link(a).robot == model.link(b).robot {
                pairs.push((a, b));
            }
        });
        for (a, b) in pairs {
            self.set_rule(a, b, CollisionRule::CustomMargin(margin));
        }
    }

    /// $O(1)$ bit-test returning `true` if pair `(a, b)` must be checked.
    #[inline(always)]
    pub fn is_pair_active(&self, a: LinkId, b: LinkId) -> bool {
        let (i, j) = (a.index(), b.index());
        if i == j || i >= self.num_links || j >= self.num_links {
            return false;
        }
        let (r, c) = if i < j { (i, j) } else { (j, i) };
        let word = self.active_upper_bitmask[r * self.words_per_row + (c / 64)];
        (word & (1u64 << (c % 64))) != 0
    }

    /// Iterates over all active link pairs `(a, b)` with `a < b` in $O(\text{active\_pairs})$ time
    /// using hardware count-trailing-zeros (`CTZ`).
    #[inline]
    pub fn for_each_active_pair(&self, mut f: impl FnMut(LinkId, LinkId)) {
        for i in 0..self.num_links {
            let row_offset = i * self.words_per_row;
            for w in 0..self.words_per_row {
                let mut word = self.active_upper_bitmask[row_offset + w];
                while word != 0 {
                    let tz = word.trailing_zeros() as usize;
                    let j = w * 64 + tz;
                    f(LinkId(i as u16), LinkId(j as u16));
                    word &= word - 1;
                }
            }
        }
    }

    /// Early-exit variant of `for_each_active_pair`: stops immediately and returns `true`
    /// if `predicate(a, b)` returns `true` for any active pair.
    #[inline]
    pub fn any_active_pair(&self, mut predicate: impl FnMut(LinkId, LinkId) -> bool) -> bool {
        for i in 0..self.num_links {
            let row_offset = i * self.words_per_row;
            for w in 0..self.words_per_row {
                let mut word = self.active_upper_bitmask[row_offset + w];
                while word != 0 {
                    let tz = word.trailing_zeros() as usize;
                    let j = w * 64 + tz;
                    if predicate(LinkId(i as u16), LinkId(j as u16)) {
                        return true;
                    }
                    word &= word - 1;
                }
            }
        }
        false
    }
}

/// Identifies the two geometric entities involved in a collision or signed-distance witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollidingPair {
    /// A robot link and a static environment obstacle.
    LinkToObstacle {
        /// Robot link ID.
        link: LinkId,
        /// Environment obstacle ID.
        obstacle: ObstacleId,
    },
    /// Two robot links (either self-collision within one arm or inter-arm collision between two arms).
    LinkToLink {
        /// First link ID.
        link_a: LinkId,
        /// Second link ID.
        link_b: LinkId,
    },
    /// An attached payload/tool and a static environment obstacle.
    PayloadToObstacle {
        /// Attached payload ID.
        payload: PayloadId,
        /// Robot link carrying the payload.
        parent_link: LinkId,
        /// Environment obstacle ID.
        obstacle: ObstacleId,
    },
    /// An attached payload/tool and another robot link.
    PayloadToLink {
        /// Attached payload ID.
        payload: PayloadId,
        /// Robot link carrying the payload.
        parent_link: LinkId,
        /// Other robot link ID.
        link: LinkId,
    },
    /// Two attached payloads/tools (e.g., tool-to-tool clearance between two arms).
    PayloadToPayload {
        /// First attached payload ID.
        payload_a: PayloadId,
        /// Robot link carrying payload A.
        parent_link_a: LinkId,
        /// Second attached payload ID.
        payload_b: PayloadId,
        /// Robot link carrying payload B.
        parent_link_b: LinkId,
    },
}

/// Differentiable signed-distance witness between two bodies for reactive QP/CBF control and TrajOpt.
#[derive(Debug, Clone, PartialEq)]
pub struct SignedDistanceWitness {
    /// Pair of entities involved.
    pub pair: CollidingPair,
    /// Signed distance in meters (`> 0.0` separated, `<= 0.0` touching or penetrating).
    pub signed_distance: f64,
    /// Closest witness point on body A in world coordinates.
    pub witness_a_world: Point3<f64>,
    /// Closest witness point on body B in world coordinates.
    pub witness_b_world: Point3<f64>,
    /// Unit normal in world coordinates pointing from body B toward body A ($\nabla_{\mathbf{p}_A} d$).
    pub normal_world: Vector3<f64>,
}

impl SignedDistanceWitness {
    /// Populates the $1 \times N_{\text{dof}}$ configuration-space distance Jacobian
    /// $J_{\text{dist}}(q) = \frac{\partial d}{\partial q}$ directly from joint screw axes
    /// with **zero heap allocations**.
    pub fn compute_distance_jacobian(
        &self,
        model: &MultiRobotModel,
        transforms: &TransformBuffer,
        out_j_dist: &mut [f64],
    ) {
        out_j_dist.fill(0.0);
        match self.pair {
            CollidingPair::LinkToObstacle { link, .. }
            | CollidingPair::PayloadToObstacle {
                parent_link: link, ..
            } => {
                model.accumulate_point_jacobian_projection(
                    transforms,
                    link,
                    &self.witness_a_world,
                    &self.normal_world,
                    1.0,
                    out_j_dist,
                );
            }
            CollidingPair::LinkToLink { link_a, link_b }
            | CollidingPair::PayloadToLink {
                parent_link: link_a,
                link: link_b,
                ..
            }
            | CollidingPair::PayloadToPayload {
                parent_link_a: link_a,
                parent_link_b: link_b,
                ..
            } => {
                model.accumulate_point_jacobian_projection(
                    transforms,
                    link_a,
                    &self.witness_a_world,
                    &self.normal_world,
                    1.0,
                    out_j_dist,
                );
                model.accumulate_point_jacobian_projection(
                    transforms,
                    link_b,
                    &self.witness_b_world,
                    &self.normal_world,
                    -1.0,
                    out_j_dist,
                );
            }
        }
    }
}

/// Pre-allocated per-thread scratchpad for zero-allocation collision and signed-distance evaluation.
#[derive(Debug, Clone)]
pub struct CollisionWorkspace {
    /// Evaluated link and frame poses.
    pub transforms: TransformBuffer,
    /// World-space AABBs for every link (`len = num_links`).
    pub link_aabbs: Vec<Aabb>,
    /// World-space poses for every attached payload (`len = num_payloads`).
    pub payload_poses: Vec<Isometry3<f64>>,
    /// World-space AABBs for every attached payload (`len = num_payloads`).
    pub payload_aabbs: Vec<Aabb>,
    /// World-space transformed Tier-1 spheres in SoA layout.
    pub world_spheres: WorldSphereSoA,
    /// Scratch buffer for continuous collision interpolation.
    pub scratch_q: Vec<f64>,
}

impl CollisionWorkspace {
    /// Creates a pre-allocated collision workspace sized for `model` and `env`.
    pub fn new(model: &MultiRobotModel, env: &CollisionEnvironment) -> Self {
        let num_links = model.num_links();
        let num_payloads = env.attached_payloads.len();
        let link_spheres: usize = (0..num_links)
            .map(|i| {
                env.link_geometries
                    .get(i)
                    .and_then(|g| g.as_ref())
                    .map_or(0, |g| g.local_spheres.len())
            })
            .sum();
        let payload_spheres: usize = env
            .attached_payloads
            .iter()
            .map(|p| p.geometry.local_spheres.len())
            .sum();
        let total_spheres = link_spheres + payload_spheres;

        Self {
            transforms: model.create_transform_buffer(),
            link_aabbs: vec![Aabb::new_invalid(); num_links],
            payload_poses: vec![Isometry3::identity(); num_payloads],
            payload_aabbs: vec![Aabb::new_invalid(); num_payloads],
            world_spheres: WorldSphereSoA {
                cx: vec![0.0; total_spheres],
                cy: vec![0.0; total_spheres],
                cz: vec![0.0; total_spheres],
                radius: vec![0.0; total_spheres],
                link_ranges: vec![0..0; num_links],
                payload_ranges: vec![0..0; num_payloads],
            },
            scratch_q: vec![0.0; model.dof()],
        }
    }
}

/// A static obstacle in the robot's world frame.
#[derive(Debug, Clone)]
pub struct Obstacle {
    /// Shape supplied by `parry3d` (`parry3d-f64`).
    pub shape: SharedShape,
    /// Pose of the shape in world coordinates.
    pub pose: Isometry3<f64>,
}

impl Obstacle {
    /// Computes the world-space AABB of this obstacle.
    #[inline]
    pub fn world_aabb(&self) -> Aabb {
        let pose = isometry_to_parry(&self.pose);
        self.shape.compute_aabb(&pose)
    }
}

/// Per-pair collision rule for attached payloads, tools, and environment obstacles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PayloadPairRule {
    /// Whether collision checks and CBF witnesses are active for this pair.
    pub enabled: bool,
    /// Optional custom minimum allowed separation margin in meters.
    pub margin: Option<f64>,
    /// Optional custom CBF activation distance in meters.
    pub activation_distance: Option<f64>,
}

impl Default for PayloadPairRule {
    fn default() -> Self {
        Self {
            enabled: true,
            margin: None,
            activation_distance: None,
        }
    }
}

#[inline]
fn canonical_link_pair(a: LinkId, b: LinkId) -> (LinkId, LinkId) {
    if a <= b { (a, b) } else { (b, a) }
}

#[inline]
fn canonical_payload_id_pair(a: PayloadId, b: PayloadId) -> (PayloadId, PayloadId) {
    if a <= b { (a, b) } else { (b, a) }
}

#[inline]
fn canonical_payload_pair(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

#[inline]
fn combine_optional_max(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    }
}

/// Multi-arm collision scene holding environment obstacles (indexed by a `Bvh`),
/// multi-tier robot link geometries, an [`AllowedCollisionMatrix`], and attached payloads.
#[derive(Debug, Default, Clone)]
pub struct CollisionEnvironment {
    /// Obstacles in the world frame. Inactive/removed slots remain in place as tombstones
    /// so [`ObstacleId`] indices never invalidate active siblings.
    pub obstacles: Vec<Obstacle>,
    /// Active flag per slot in `obstacles` (missing entries default to `true` for manual pushes).
    obstacle_active: Vec<bool>,
    /// Broad-phase Bounding Volume Hierarchy over active `obstacles`.
    obstacle_bvh: Option<Bvh>,
    /// Compact BVH leaf index -> [`ObstacleId`] lookup (since `parry3d::Bvh` requires dense `0..N` keys).
    bvh_leaf_obstacles: Vec<ObstacleId>,
    /// Per-link collision geometries indexed by `LinkId`.
    link_geometries: Vec<Option<LinkGeometry>>,
    /// Allowed Collision Matrix governing link-to-link self and inter-arm collision pairs.
    acm: Option<AllowedCollisionMatrix>,
    /// Dynamically grasped/attached payloads and tools.
    attached_payloads: Vec<AttachedPayload>,
    /// Custom CBF activation distances for active `(link_a, link_b)` pairs.
    link_pair_activations: HashMap<(LinkId, LinkId), f64>,
    /// Per-obstacle default safety margins indexed by `ObstacleId`.
    obstacle_margins: Vec<Option<f64>>,
    /// Per-obstacle default CBF activation distances indexed by `ObstacleId`.
    obstacle_activations: Vec<Option<f64>>,
    /// Per-pair overrides for `(link, obstacle)`.
    link_obstacle_rules: HashMap<(LinkId, ObstacleId), PayloadPairRule>,
    /// Per-pair overrides for `(payload_a, payload_b)`.
    payload_pair_rules: HashMap<(String, String), PayloadPairRule>,
    /// Per-pair overrides for `(payload_id, obstacle)`.
    payload_obstacle_rules: HashMap<(String, ObstacleId), PayloadPairRule>,
    /// Per-pair overrides for `(payload_id, link)`.
    payload_link_rules: HashMap<(String, LinkId), PayloadPairRule>,
    /// Zero-alloc integer-keyed cache for `payload_pair_rules`.
    compiled_payload_pair_rules: HashMap<(PayloadId, PayloadId), PayloadPairRule>,
    /// Zero-alloc integer-keyed cache for `payload_obstacle_rules`.
    compiled_payload_obstacle_rules: HashMap<(PayloadId, ObstacleId), PayloadPairRule>,
    /// Zero-alloc integer-keyed cache for `payload_link_rules`.
    compiled_payload_link_rules: HashMap<(PayloadId, LinkId), PayloadPairRule>,
}

impl CollisionEnvironment {
    /// Creates an empty collision environment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if `id` refers to an existing, active (non-removed) obstacle.
    #[inline]
    pub fn is_obstacle_active(&self, id: ObstacleId) -> bool {
        let idx = id.0 as usize;
        idx < self.obstacles.len() && self.obstacle_active.get(idx).copied().unwrap_or(true)
    }

    /// Returns the obstacle at `id` if it exists and is active.
    #[inline]
    pub fn obstacle(&self, id: ObstacleId) -> Option<&Obstacle> {
        self.is_obstacle_active(id)
            .then(|| &self.obstacles[id.0 as usize])
    }

    /// Iterates over all active `(ObstacleId, &Obstacle)` entries in index order.
    #[inline]
    pub fn active_obstacles(&self) -> impl Iterator<Item = (ObstacleId, &Obstacle)> + '_ {
        self.obstacles.iter().enumerate().filter_map(|(i, obs)| {
            let id = ObstacleId(i as u32);
            self.is_obstacle_active(id).then_some((id, obs))
        })
    }

    /// Number of currently active obstacles in the environment.
    #[inline]
    pub fn num_active_obstacles(&self) -> usize {
        (0..self.obstacles.len())
            .filter(|&i| self.obstacle_active.get(i).copied().unwrap_or(true))
            .count()
    }

    /// Adds a `parry3d` obstacle to the environment and updates the broad-phase BVH.
    pub fn add_obstacle(&mut self, shape: SharedShape, pose: Isometry3<f64>) -> ObstacleId {
        let id = self.add_obstacle_deferred(shape, pose);
        self.rebuild_bvh();
        id
    }

    pub(crate) fn add_obstacle_deferred(
        &mut self,
        shape: SharedShape,
        pose: Isometry3<f64>,
    ) -> ObstacleId {
        if self.obstacle_active.len() < self.obstacles.len() {
            self.obstacle_active.resize(self.obstacles.len(), true);
        }
        let id = ObstacleId(self.obstacles.len() as u32);
        self.obstacles.push(Obstacle { shape, pose });
        self.obstacle_active.push(true);
        self.obstacle_margins.push(None);
        self.obstacle_activations.push(None);
        id
    }

    /// Updates the world pose of an existing active obstacle and rebuilds the obstacle BVH
    /// without disturbing per-obstacle margin/activation or rule tables.
    ///
    /// Returns `false` if `id` is out of bounds or has been removed.
    pub fn set_obstacle_pose(&mut self, id: ObstacleId, pose: Isometry3<f64>) -> bool {
        if self.set_obstacle_pose_deferred(id, pose) {
            self.rebuild_bvh();
            true
        } else {
            false
        }
    }

    pub(crate) fn set_obstacle_pose_deferred(
        &mut self,
        id: ObstacleId,
        pose: Isometry3<f64>,
    ) -> bool {
        if !self.is_obstacle_active(id) {
            return false;
        }
        self.obstacles[id.0 as usize].pose = pose;
        true
    }

    /// Replaces the geometry shape of an existing active obstacle and rebuilds the obstacle BVH.
    ///
    /// Returns `false` if `id` is out of bounds or has been removed.
    pub fn set_obstacle_shape(&mut self, id: ObstacleId, shape: SharedShape) -> bool {
        if self.set_obstacle_shape_deferred(id, shape) {
            self.rebuild_bvh();
            true
        } else {
            false
        }
    }

    pub(crate) fn set_obstacle_shape_deferred(
        &mut self,
        id: ObstacleId,
        shape: SharedShape,
    ) -> bool {
        if !self.is_obstacle_active(id) {
            return false;
        }
        self.obstacles[id.0 as usize].shape = shape;
        true
    }

    /// Atomically updates both the geometry shape and world pose of an active obstacle and rebuilds the BVH.
    ///
    /// Returns `false` if `id` is out of bounds or has been removed.
    pub fn update_obstacle(
        &mut self,
        id: ObstacleId,
        shape: SharedShape,
        pose: Isometry3<f64>,
    ) -> bool {
        if self.update_obstacle_deferred(id, shape, pose) {
            self.rebuild_bvh();
            true
        } else {
            false
        }
    }

    pub(crate) fn update_obstacle_deferred(
        &mut self,
        id: ObstacleId,
        shape: SharedShape,
        pose: Isometry3<f64>,
    ) -> bool {
        if !self.is_obstacle_active(id) {
            return false;
        }
        let obs = &mut self.obstacles[id.0 as usize];
        obs.shape = shape;
        obs.pose = pose;
        true
    }

    /// Tombstones an obstacle, clears its margin/activation and pair rules, and rebuilds the BVH.
    ///
    /// Remaining [`ObstacleId`] handles keep their numeric indices. Returns `false` if `id`
    /// is invalid or already removed.
    pub fn remove_obstacle(&mut self, id: ObstacleId) -> bool {
        if self.remove_obstacle_deferred(id) {
            self.rebuild_bvh();
            true
        } else {
            false
        }
    }

    pub(crate) fn remove_obstacle_deferred(&mut self, id: ObstacleId) -> bool {
        if !self.is_obstacle_active(id) {
            return false;
        }
        let idx = id.0 as usize;
        if self.obstacle_active.len() < self.obstacles.len() {
            self.obstacle_active.resize(self.obstacles.len(), true);
        }
        self.obstacle_active[idx] = false;
        if idx < self.obstacle_margins.len() {
            self.obstacle_margins[idx] = None;
        }
        if idx < self.obstacle_activations.len() {
            self.obstacle_activations[idx] = None;
        }
        self.link_obstacle_rules.retain(|&(_, oid), _| oid != id);
        self.payload_obstacle_rules
            .retain(|&(_, oid), _| oid != id);
        self.rebuild_payload_rule_caches();
        true
    }

    /// Rebuilds the static obstacle `Bvh` broad-phase acceleration structure over active obstacles.
    pub fn rebuild_bvh(&mut self) {
        let mut leaf_ids = Vec::with_capacity(self.obstacles.len());
        let mut leaves = Vec::with_capacity(self.obstacles.len());
        for (i, obs) in self.obstacles.iter().enumerate() {
            let id = ObstacleId(i as u32);
            if self.is_obstacle_active(id) {
                let compact_idx = leaves.len();
                leaf_ids.push(id);
                leaves.push((compact_idx, obs.world_aabb()));
            }
        }
        self.bvh_leaf_obstacles = leaf_ids;
        if leaves.is_empty() {
            self.obstacle_bvh = None;
            return;
        }
        self.obstacle_bvh = Some(Bvh::from_iter(BvhBuildStrategy::Binned, leaves));
    }

    /// Registers multi-tier collision geometry for `link`.
    pub fn set_link_geometry(&mut self, link: LinkId, geometry: LinkGeometry) {
        if self.link_geometries.len() <= link.index() {
            self.link_geometries.resize(link.index() + 1, None);
        }
        self.link_geometries[link.index()] = Some(geometry);
    }

    /// Returns the collision geometry attached to `link`, if any.
    #[inline]
    pub fn link_geometry(&self, link: LinkId) -> Option<&LinkGeometry> {
        self.link_geometries.get(link.index()).and_then(|g| g.as_ref())
    }

    /// Sets the [`AllowedCollisionMatrix`] for self-collision and inter-arm collision checking.
    pub fn set_acm(&mut self, acm: AllowedCollisionMatrix) {
        self.acm = Some(acm);
    }

    /// Returns a reference to the [`AllowedCollisionMatrix`], if configured.
    #[inline]
    pub fn acm(&self) -> Option<&AllowedCollisionMatrix> {
        self.acm.as_ref()
    }

    /// Mutably borrows the [`AllowedCollisionMatrix`], if configured.
    #[inline]
    pub fn acm_mut(&mut self) -> Option<&mut AllowedCollisionMatrix> {
        self.acm.as_mut()
    }

    fn payload_id_by_name(&self, id: &str) -> Option<PayloadId> {
        self.attached_payloads
            .iter()
            .position(|p| p.id == id)
            .map(|idx| PayloadId(idx as u16))
    }

    fn rebuild_payload_rule_caches(&mut self) {
        self.compiled_payload_pair_rules.clear();
        for ((a, b), &rule) in &self.payload_pair_rules {
            if let (Some(pa), Some(pb)) = (self.payload_id_by_name(a), self.payload_id_by_name(b)) {
                self.compiled_payload_pair_rules
                    .insert(canonical_payload_id_pair(pa, pb), rule);
            }
        }

        self.compiled_payload_obstacle_rules.clear();
        for ((pid_str, obs), &rule) in &self.payload_obstacle_rules {
            if let Some(pid) = self.payload_id_by_name(pid_str) {
                self.compiled_payload_obstacle_rules
                    .insert((pid, *obs), rule);
            }
        }

        self.compiled_payload_link_rules.clear();
        for ((pid_str, link), &rule) in &self.payload_link_rules {
            if let Some(pid) = self.payload_id_by_name(pid_str) {
                self.compiled_payload_link_rules.insert((pid, *link), rule);
            }
        }
    }

    /// Attaches a grasped payload or tool to a robot link (replacing any existing payload with the same `id`).
    pub fn attach_payload(&mut self, payload: AttachedPayload) -> PayloadId {
        let id = if let Some(pos) = self.attached_payloads.iter().position(|p| p.id == payload.id) {
            self.attached_payloads[pos] = payload;
            PayloadId(pos as u16)
        } else {
            let id = PayloadId(self.attached_payloads.len() as u16);
            self.attached_payloads.push(payload);
            id
        };
        self.rebuild_payload_rule_caches();
        id
    }

    /// Detaches a grasped payload by its ID.
    pub fn detach_payload(&mut self, id: &str) -> Option<AttachedPayload> {
        let pos = self.attached_payloads.iter().position(|p| p.id == id)?;
        let removed = self.attached_payloads.swap_remove(pos);
        self.rebuild_payload_rule_caches();
        Some(removed)
    }

    /// Returns all currently attached payloads and tools.
    #[inline]
    pub fn attached_payloads(&self) -> &[AttachedPayload] {
        &self.attached_payloads
    }

    /// Returns an attached payload by [`PayloadId`].
    #[inline]
    pub fn attached_payload(&self, id: PayloadId) -> Option<&AttachedPayload> {
        self.attached_payloads.get(id.index())
    }

    /// Configures a custom safety margin on an active `(link_a, link_b)` pair in the ACM.
    pub fn set_link_pair_margin(&mut self, a: LinkId, b: LinkId, margin: f64) {
        if let Some(acm) = &mut self.acm {
            if acm.is_pair_active(a, b) {
                acm.set_rule(a, b, CollisionRule::CustomMargin(margin));
            }
        }
    }

    /// Configures a custom CBF/MPPI activation distance for `(link_a, link_b)`.
    pub fn set_link_pair_activation(&mut self, a: LinkId, b: LinkId, activation: f64) {
        self.link_pair_activations
            .insert(canonical_link_pair(a, b), activation);
    }

    /// Configures a default safety margin (meters) for an environment obstacle across all links/payloads.
    pub fn set_obstacle_margin(&mut self, obstacle: ObstacleId, margin: f64) {
        let idx = obstacle.0 as usize;
        if self.obstacle_margins.len() <= idx {
            self.obstacle_margins.resize(idx + 1, None);
        }
        self.obstacle_margins[idx] = Some(margin);
    }

    /// Configures a default CBF activation distance (meters) for an environment obstacle.
    pub fn set_obstacle_activation(&mut self, obstacle: ObstacleId, activation: f64) {
        let idx = obstacle.0 as usize;
        if self.obstacle_activations.len() <= idx {
            self.obstacle_activations.resize(idx + 1, None);
        }
        self.obstacle_activations[idx] = Some(activation);
    }

    /// Configures a custom safety margin for a specific `(link, obstacle)` pair.
    pub fn set_link_obstacle_margin(&mut self, link: LinkId, obstacle: ObstacleId, margin: f64) {
        self.link_obstacle_rules
            .entry((link, obstacle))
            .or_default()
            .margin = Some(margin);
    }

    /// Configures a custom CBF activation distance for a specific `(link, obstacle)` pair.
    pub fn set_link_obstacle_activation(
        &mut self,
        link: LinkId,
        obstacle: ObstacleId,
        activation: f64,
    ) {
        self.link_obstacle_rules
            .entry((link, obstacle))
            .or_default()
            .activation_distance = Some(activation);
    }

    /// Enables or disables collision checking between `link` and `obstacle`.
    pub fn set_link_obstacle_enabled(&mut self, link: LinkId, obstacle: ObstacleId, enabled: bool) {
        self.link_obstacle_rules
            .entry((link, obstacle))
            .or_default()
            .enabled = enabled;
    }

    /// Configures a custom safety margin between two attached payloads/tools `(id_a, id_b)`.
    pub fn set_payload_pair_margin(&mut self, id_a: &str, id_b: &str, margin: f64) {
        self.payload_pair_rules
            .entry(canonical_payload_pair(id_a, id_b))
            .or_default()
            .margin = Some(margin);
        self.rebuild_payload_rule_caches();
    }

    /// Configures a custom CBF/MPPI activation distance between two attached payloads/tools `(id_a, id_b)`.
    pub fn set_payload_pair_activation(&mut self, id_a: &str, id_b: &str, activation: f64) {
        self.payload_pair_rules
            .entry(canonical_payload_pair(id_a, id_b))
            .or_default()
            .activation_distance = Some(activation);
        self.rebuild_payload_rule_caches();
    }

    /// Enables or disables collision checking between two attached payloads `(id_a, id_b)`.
    pub fn set_payload_pair_enabled(&mut self, id_a: &str, id_b: &str, enabled: bool) {
        self.payload_pair_rules
            .entry(canonical_payload_pair(id_a, id_b))
            .or_default()
            .enabled = enabled;
        self.rebuild_payload_rule_caches();
    }

    /// Configures a custom safety margin between payload `payload_id` and `obstacle`.
    pub fn set_payload_obstacle_margin(
        &mut self,
        payload_id: &str,
        obstacle: ObstacleId,
        margin: f64,
    ) {
        self.payload_obstacle_rules
            .entry((payload_id.to_string(), obstacle))
            .or_default()
            .margin = Some(margin);
        self.rebuild_payload_rule_caches();
    }

    /// Configures a custom CBF activation distance between payload `payload_id` and `obstacle`.
    pub fn set_payload_obstacle_activation(
        &mut self,
        payload_id: &str,
        obstacle: ObstacleId,
        activation: f64,
    ) {
        self.payload_obstacle_rules
            .entry((payload_id.to_string(), obstacle))
            .or_default()
            .activation_distance = Some(activation);
        self.rebuild_payload_rule_caches();
    }

    /// Enables or disables collision checking between payload `payload_id` and `obstacle`.
    pub fn set_payload_obstacle_enabled(
        &mut self,
        payload_id: &str,
        obstacle: ObstacleId,
        enabled: bool,
    ) {
        self.payload_obstacle_rules
            .entry((payload_id.to_string(), obstacle))
            .or_default()
            .enabled = enabled;
        self.rebuild_payload_rule_caches();
    }

    /// Configures a custom safety margin between payload `payload_id` and `link`.
    pub fn set_payload_link_margin(&mut self, payload_id: &str, link: LinkId, margin: f64) {
        self.payload_link_rules
            .entry((payload_id.to_string(), link))
            .or_default()
            .margin = Some(margin);
        self.rebuild_payload_rule_caches();
    }

    /// Configures a custom CBF/MPPI activation distance between payload `payload_id` and `link`.
    pub fn set_payload_link_activation(
        &mut self,
        payload_id: &str,
        link: LinkId,
        activation: f64,
    ) {
        self.payload_link_rules
            .entry((payload_id.to_string(), link))
            .or_default()
            .activation_distance = Some(activation);
        self.rebuild_payload_rule_caches();
    }

    /// Enables or disables collision checking between payload `payload_id` and `link`.
    pub fn set_payload_link_enabled(&mut self, payload_id: &str, link: LinkId, enabled: bool) {
        self.payload_link_rules
            .entry((payload_id.to_string(), link))
            .or_default()
            .enabled = enabled;
        self.rebuild_payload_rule_caches();
    }

    /// Zero-allocation ID-based check for whether payload pair `(a, b)` is enabled.
    #[inline]
    pub fn is_payload_pair_id_enabled(&self, a: PayloadId, b: PayloadId) -> bool {
        if self.compiled_payload_pair_rules.is_empty() {
            return true;
        }
        self.compiled_payload_pair_rules
            .get(&canonical_payload_id_pair(a, b))
            .is_none_or(|r| r.enabled)
    }

    /// Zero-allocation ID-based check for whether `(payload, obstacle)` is enabled.
    #[inline]
    pub fn is_payload_obstacle_id_enabled(&self, payload: PayloadId, obstacle: ObstacleId) -> bool {
        if self.compiled_payload_obstacle_rules.is_empty() {
            return true;
        }
        self.compiled_payload_obstacle_rules
            .get(&(payload, obstacle))
            .is_none_or(|r| r.enabled)
    }

    /// Zero-allocation ID-based check for whether `(payload, link)` is enabled.
    #[inline]
    pub fn is_payload_link_id_enabled(&self, payload: PayloadId, link: LinkId) -> bool {
        if self.compiled_payload_link_rules.is_empty() {
            return true;
        }
        self.compiled_payload_link_rules
            .get(&(payload, link))
            .is_none_or(|r| r.enabled)
    }

    /// Returns whether collision checking between two payloads is enabled (allocation-free).
    #[inline]
    pub fn is_payload_pair_enabled(&self, id_a: &str, id_b: &str) -> bool {
        if self.payload_pair_rules.is_empty() {
            return true;
        }
        let (lo, hi) = if id_a <= id_b {
            (id_a, id_b)
        } else {
            (id_b, id_a)
        };
        self.payload_pair_rules
            .iter()
            .find(|((a, b), _)| a == lo && b == hi)
            .is_none_or(|(_, r)| r.enabled)
    }

    /// Returns whether collision checking between `payload_id` and `obstacle` is enabled.
    #[inline]
    pub fn is_payload_obstacle_enabled(&self, payload_id: &str, obstacle: ObstacleId) -> bool {
        if self.payload_obstacle_rules.is_empty() {
            return true;
        }
        self.payload_obstacle_rules
            .iter()
            .find(|((pid, obs), _)| *obs == obstacle && pid == payload_id)
            .is_none_or(|(_, r)| r.enabled)
    }

    /// Returns whether collision checking between `payload_id` and `link` is enabled.
    #[inline]
    pub fn is_payload_link_enabled(&self, payload_id: &str, link: LinkId) -> bool {
        if self.payload_link_rules.is_empty() {
            return true;
        }
        self.payload_link_rules
            .iter()
            .find(|((pid, l), _)| *l == link && pid == payload_id)
            .is_none_or(|(_, r)| r.enabled)
    }

    /// Returns whether collision checking between `link` and `obstacle` is enabled.
    #[inline]
    pub fn is_link_obstacle_enabled(&self, link: LinkId, obstacle: ObstacleId) -> bool {
        self.link_obstacle_rules
            .get(&(link, obstacle))
            .is_none_or(|r| r.enabled)
    }

    /// Returns the custom per-pair safety margin (meters) configured for `pair`, if any.
    pub fn custom_pair_margin(&self, pair: &CollidingPair) -> Option<f64> {
        match *pair {
            CollidingPair::LinkToLink { link_a, link_b } => {
                self.acm.as_ref().and_then(|acm| acm.custom_margin(link_a, link_b))
            }
            CollidingPair::LinkToObstacle { link, obstacle } => self
                .link_obstacle_rules
                .get(&(link, obstacle))
                .and_then(|r| r.margin)
                .or_else(|| {
                    self.obstacle_margins
                        .get(obstacle.0 as usize)
                        .copied()
                        .flatten()
                }),
            CollidingPair::PayloadToPayload {
                payload_a,
                payload_b,
                ..
            } => {
                let pa = self.attached_payload(payload_a);
                let pb = self.attached_payload(payload_b);
                let explicit = if self.compiled_payload_pair_rules.is_empty() {
                    None
                } else {
                    self.compiled_payload_pair_rules
                        .get(&canonical_payload_id_pair(payload_a, payload_b))
                        .and_then(|r| r.margin)
                };
                explicit.or_else(|| {
                    combine_optional_max(
                        pa.and_then(|p| p.safety_margin),
                        pb.and_then(|p| p.safety_margin),
                    )
                })
            }
            CollidingPair::PayloadToObstacle {
                payload, obstacle, ..
            } => {
                let p = self.attached_payload(payload);
                let explicit = if self.compiled_payload_obstacle_rules.is_empty() {
                    None
                } else {
                    self.compiled_payload_obstacle_rules
                        .get(&(payload, obstacle))
                        .and_then(|r| r.margin)
                };
                explicit
                    .or_else(|| p.and_then(|pl| pl.safety_margin))
                    .or_else(|| {
                        self.obstacle_margins
                            .get(obstacle.0 as usize)
                            .copied()
                            .flatten()
                    })
            }
            CollidingPair::PayloadToLink { payload, link, .. } => {
                let p = self.attached_payload(payload);
                let explicit = if self.compiled_payload_link_rules.is_empty() {
                    None
                } else {
                    self.compiled_payload_link_rules
                        .get(&(payload, link))
                        .and_then(|r| r.margin)
                };
                explicit.or_else(|| p.and_then(|pl| pl.safety_margin))
            }
        }
    }

    /// Returns the effective safety margin (meters) for `pair`, falling back to `default_margin`.
    #[inline]
    pub fn effective_pair_margin(&self, pair: &CollidingPair, default_margin: f64) -> f64 {
        self.custom_pair_margin(pair).unwrap_or(default_margin)
    }

    /// Returns the custom per-pair activation distance (meters) configured for `pair`, if any.
    pub fn custom_pair_activation(&self, pair: &CollidingPair) -> Option<f64> {
        match *pair {
            CollidingPair::LinkToLink { link_a, link_b } => {
                self.link_pair_activations
                    .get(&canonical_link_pair(link_a, link_b))
                    .copied()
            }
            CollidingPair::LinkToObstacle { link, obstacle } => self
                .link_obstacle_rules
                .get(&(link, obstacle))
                .and_then(|r| r.activation_distance)
                .or_else(|| {
                    self.obstacle_activations
                        .get(obstacle.0 as usize)
                        .copied()
                        .flatten()
                }),
            CollidingPair::PayloadToPayload {
                payload_a,
                payload_b,
                ..
            } => {
                let pa = self.attached_payload(payload_a);
                let pb = self.attached_payload(payload_b);
                let explicit = if self.compiled_payload_pair_rules.is_empty() {
                    None
                } else {
                    self.compiled_payload_pair_rules
                        .get(&canonical_payload_id_pair(payload_a, payload_b))
                        .and_then(|r| r.activation_distance)
                };
                explicit.or_else(|| {
                    combine_optional_max(
                        pa.and_then(|p| p.activation_distance),
                        pb.and_then(|p| p.activation_distance),
                    )
                })
            }
            CollidingPair::PayloadToObstacle {
                payload, obstacle, ..
            } => {
                let p = self.attached_payload(payload);
                let explicit = if self.compiled_payload_obstacle_rules.is_empty() {
                    None
                } else {
                    self.compiled_payload_obstacle_rules
                        .get(&(payload, obstacle))
                        .and_then(|r| r.activation_distance)
                };
                explicit
                    .or_else(|| p.and_then(|pl| pl.activation_distance))
                    .or_else(|| {
                        self.obstacle_activations
                            .get(obstacle.0 as usize)
                            .copied()
                            .flatten()
                    })
            }
            CollidingPair::PayloadToLink { payload, link, .. } => {
                let p = self.attached_payload(payload);
                let explicit = if self.compiled_payload_link_rules.is_empty() {
                    None
                } else {
                    self.compiled_payload_link_rules
                        .get(&(payload, link))
                        .and_then(|r| r.activation_distance)
                };
                explicit.or_else(|| p.and_then(|pl| pl.activation_distance))
            }
        }
    }

    /// Returns the effective CBF/MPPI activation distance (meters) for `pair`, falling back to `default_activation`.
    #[inline]
    pub fn effective_pair_activation(&self, pair: &CollidingPair, default_activation: f64) -> f64 {
        self.custom_pair_activation(pair).unwrap_or(default_activation)
    }

    /// Returns the maximum configured activation distance across `default_activation` and all custom overrides.
    pub fn max_activation_distance(&self, default_activation: f64) -> f64 {
        let mut max_act = default_activation;
        for &a in self.link_pair_activations.values() {
            max_act = max_act.max(a);
        }
        for &a in self.obstacle_activations.iter().flatten() {
            max_act = max_act.max(a);
        }
        for r in self.link_obstacle_rules.values() {
            if let Some(a) = r.activation_distance {
                max_act = max_act.max(a);
            }
        }
        for r in self.payload_pair_rules.values() {
            if let Some(a) = r.activation_distance {
                max_act = max_act.max(a);
            }
        }
        for r in self.payload_obstacle_rules.values() {
            if let Some(a) = r.activation_distance {
                max_act = max_act.max(a);
            }
        }
        for r in self.payload_link_rules.values() {
            if let Some(a) = r.activation_distance {
                max_act = max_act.max(a);
            }
        }
        for p in &self.attached_payloads {
            if let Some(a) = p.activation_distance {
                max_act = max_act.max(a);
            }
        }
        max_act
    }

    /// Allocates a [`CollisionWorkspace`] sized for `model` and this environment.
    pub fn create_workspace(&self, model: &MultiRobotModel) -> CollisionWorkspace {
        CollisionWorkspace::new(model, self)
    }

    /// Updates `workspace` with Forward Kinematics, world-space link & payload AABBs, and SoA spheres for `q`.
    fn update_workspace(
        &self,
        model: &MultiRobotModel,
        q: &[f64],
        tier: GeometryTier,
        workspace: &mut CollisionWorkspace,
    ) -> Result<(), CollisionError> {
        model.validate_configuration(q)?;
        model.forward_kinematics(q, &mut workspace.transforms)?;

        let num_links = model.num_links();
        let num_payloads = self.attached_payloads.len();
        workspace.link_aabbs.resize(num_links, Aabb::new_invalid());
        workspace.payload_poses.resize(num_payloads, Isometry3::identity());
        workspace.payload_aabbs.resize(num_payloads, Aabb::new_invalid());
        workspace.world_spheres.link_ranges.resize(num_links, 0..0);
        workspace.world_spheres.payload_ranges.resize(num_payloads, 0..0);

        if tier == GeometryTier::AnalyticSsv {
            workspace.world_spheres.cx.clear();
            workspace.world_spheres.cy.clear();
            workspace.world_spheres.cz.clear();
            workspace.world_spheres.radius.clear();
        }

        for link_idx in 0..num_links {
            let link_id = LinkId(link_idx as u16);
            let link_pose = workspace.transforms.link_pose(link_id);
            if let Some(geom) = self.link_geometry(link_id) {
                workspace.link_aabbs[link_idx] = geom.world_aabb(link_pose);
                if tier == GeometryTier::AnalyticSsv {
                    let start = workspace.world_spheres.cx.len();
                    for s in &geom.local_spheres {
                        let p_w = link_pose.transform_point(&s.center);
                        workspace.world_spheres.cx.push(p_w.x);
                        workspace.world_spheres.cy.push(p_w.y);
                        workspace.world_spheres.cz.push(p_w.z);
                        workspace.world_spheres.radius.push(s.radius);
                    }
                    let end = workspace.world_spheres.cx.len();
                    workspace.world_spheres.link_ranges[link_idx] = start..end;
                }
            } else {
                workspace.link_aabbs[link_idx] = Aabb::new_invalid();
                workspace.world_spheres.link_ranges[link_idx] = 0..0;
            }
        }

        for (p_idx, payload) in self.attached_payloads.iter().enumerate() {
            if payload.parent_link.index() >= num_links {
                workspace.payload_aabbs[p_idx] = Aabb::new_invalid();
                workspace.world_spheres.payload_ranges[p_idx] = 0..0;
                continue;
            }
            let world_pose = *workspace.transforms.link_pose(payload.parent_link) * payload.local_pose;
            workspace.payload_poses[p_idx] = world_pose;
            workspace.payload_aabbs[p_idx] = payload.geometry.world_aabb(&world_pose);
            if tier == GeometryTier::AnalyticSsv {
                let start = workspace.world_spheres.cx.len();
                for s in &payload.geometry.local_spheres {
                    let p_w = world_pose.transform_point(&s.center);
                    workspace.world_spheres.cx.push(p_w.x);
                    workspace.world_spheres.cy.push(p_w.y);
                    workspace.world_spheres.cz.push(p_w.z);
                    workspace.world_spheres.radius.push(s.radius);
                }
                let end = workspace.world_spheres.cx.len();
                workspace.world_spheres.payload_ranges[p_idx] = start..end;
            }
        }

        Ok(())
    }

    fn spheres_collide_with_obstacle(
        world_spheres: &WorldSphereSoA,
        range: &Range<usize>,
        obs: &Obstacle,
        margin: f64,
    ) -> bool {
        let obs_pose = isometry_to_parry(&obs.pose);
        for idx in range.clone() {
            let ball_pose = ParryPose::from_parts(
                ParryVector::new(
                    world_spheres.cx[idx],
                    world_spheres.cy[idx],
                    world_spheres.cz[idx],
                ),
                ParryRotation::IDENTITY,
            );
            let r = world_spheres.radius[idx];
            if margin == 0.0 {
                let ball = parry3d::shape::Ball::new(r);
                if parry3d::query::intersection_test(&ball_pose, &ball, &obs_pose, obs.shape.as_ref())
                    .unwrap_or(false)
                {
                    return true;
                }
            } else if margin > 0.0 {
                let ball = parry3d::shape::Ball::new(r + margin);
                if parry3d::query::intersection_test(&ball_pose, &ball, &obs_pose, obs.shape.as_ref())
                    .unwrap_or(false)
                {
                    return true;
                }
            } else {
                let ball = parry3d::shape::Ball::new(r);
                if let Ok(d) = parry3d::query::distance(&ball_pose, &ball, &obs_pose, obs.shape.as_ref())
                {
                    if d <= margin {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn closest_spheres_to_obstacle_witness(
        world_spheres: &WorldSphereSoA,
        range: &Range<usize>,
        obs: &Obstacle,
        cutoff: f64,
    ) -> Option<ShapeWitness> {
        let obs_pose = isometry_to_parry(&obs.pose);
        let mut best: Option<ShapeWitness> = None;
        for idx in range.clone() {
            let ball_pose = ParryPose::from_parts(
                ParryVector::new(
                    world_spheres.cx[idx],
                    world_spheres.cy[idx],
                    world_spheres.cz[idx],
                ),
                ParryRotation::IDENTITY,
            );
            let ball = parry3d::shape::Ball::new(world_spheres.radius[idx]);
            if let Ok(Some(contact)) = parry3d::query::contact(
                &ball_pose,
                &ball,
                &obs_pose,
                obs.shape.as_ref(),
                cutoff.max(0.0),
            ) {
                let dist = contact.dist;
                if dist <= cutoff && best.as_ref().is_none_or(|b| dist < b.0) {
                    let p_a = parry_to_point(contact.point1);
                    let p_b = parry_to_point(contact.point2);
                    let n1 = parry_to_vector(contact.normal1);
                    best = Some((dist, p_a, p_b, -n1));
                }
            }
        }
        best
    }

    /// Evaluates whether configuration `q` is free of self-collision, inter-arm collision,
    /// attached payload/tool collisions, and environment obstacle collisions using a pre-allocated [`CollisionWorkspace`].
    pub fn is_state_collision_free(
        &self,
        model: &MultiRobotModel,
        q: &[f64],
        tier: GeometryTier,
        workspace: &mut CollisionWorkspace,
    ) -> Result<bool, CollisionError> {
        self.update_workspace(model, q, tier, workspace)?;

        // 1. Self-collision & Inter-arm collision via bitset ACM + AABB + Narrow-phase
        if let Some(acm) = &self.acm {
            let self_collision = acm.any_active_pair(|link_a, link_b| {
                let (Some(geom_a), Some(geom_b)) =
                    (self.link_geometry(link_a), self.link_geometry(link_b))
                else {
                    return false;
                };
                let margin = acm.pair_margin(link_a, link_b);
                let aabb_a = &workspace.link_aabbs[link_a.index()];
                let aabb_b = &workspace.link_aabbs[link_b.index()];
                if !aabb_a.loosened(margin.max(0.0)).intersects(aabb_b) {
                    return false;
                }

                match tier {
                    GeometryTier::AnalyticSsv => {
                        let range_a = &workspace.world_spheres.link_ranges[link_a.index()];
                        let range_b = &workspace.world_spheres.link_ranges[link_b.index()];
                        if !range_a.is_empty() && !range_b.is_empty() {
                            return workspace
                                .world_spheres
                                .ranges_intersect_simd(range_a, range_b, margin);
                        }
                        if let (Some(cap_a), Some(cap_b)) =
                            (&geom_a.bounding_capsule, &geom_b.bounding_capsule)
                        {
                            let w_a = cap_a.transformed(workspace.transforms.link_pose(link_a));
                            let w_b = cap_b.transformed(workspace.transforms.link_pose(link_b));
                            let (dist, _, _, _) = w_a.signed_distance_to_capsule(&w_b);
                            return dist <= margin;
                        }
                        false
                    }
                    GeometryTier::Convex => {
                        let pose_a = workspace.transforms.link_pose(link_a);
                        let pose_b = workspace.transforms.link_pose(link_b);
                        Self::shapes_collide(
                            pose_a,
                            &geom_a.convex_shapes,
                            pose_b,
                            &geom_b.convex_shapes,
                            margin,
                        )
                    }
                }
            });

            if self_collision {
                return Ok(false);
            }
        }

        // 2. Robot links vs Static Environment Obstacles via BVH Broad-Phase
        if let Some(bvh) = &self.obstacle_bvh {
            for link_idx in 0..model.num_links() {
                let link_id = LinkId(link_idx as u16);
                let Some(geom) = self.link_geometry(link_id) else {
                    continue;
                };
                let link_aabb = &workspace.link_aabbs[link_idx];
                let link_pose = workspace.transforms.link_pose(link_id);
                let max_obs_margin = self
                    .obstacle_margins
                    .iter()
                    .flatten()
                    .copied()
                    .fold(0.0_f64, f64::max);
                let query_aabb = if max_obs_margin > 0.0 || !self.link_obstacle_rules.is_empty() {
                    let rule_max = self
                        .link_obstacle_rules
                        .iter()
                        .filter(|((l, _), _)| *l == link_id)
                        .filter_map(|(_, r)| r.margin)
                        .fold(max_obs_margin, f64::max);
                    link_aabb.loosened(rule_max.max(0.0))
                } else {
                    *link_aabb
                };

                for leaf_idx in bvh.intersect_aabb(&query_aabb) {
                    let obs_id = self.bvh_leaf_obstacles[leaf_idx as usize];
                    if !self.is_link_obstacle_enabled(link_id, obs_id) {
                        continue;
                    }
                    let margin = self
                        .custom_pair_margin(&CollidingPair::LinkToObstacle {
                            link: link_id,
                            obstacle: obs_id,
                        })
                        .unwrap_or(0.0);
                    let obs = &self.obstacles[obs_id.0 as usize];
                    if !link_aabb
                        .loosened(margin.max(0.0))
                        .intersects(&obs.world_aabb())
                    {
                        continue;
                    }
                    let collides = match tier {
                        GeometryTier::AnalyticSsv => {
                            let range = &workspace.world_spheres.link_ranges[link_idx];
                            Self::spheres_collide_with_obstacle(
                                &workspace.world_spheres,
                                range,
                                obs,
                                margin,
                            )
                        }
                        GeometryTier::Convex => Self::link_collides_with_obstacle(
                            link_pose,
                            &geom.convex_shapes,
                            obs,
                            margin,
                        ),
                    };
                    if collides {
                        return Ok(false);
                    }
                }
            }
        }

        // 3. Attached Payloads / Tools (evaluated even when there are no static obstacles)
        let num_payloads = self.attached_payloads.len();
        for p_idx in 0..num_payloads {
            let payload = &self.attached_payloads[p_idx];
            if payload.parent_link.index() >= model.num_links() {
                continue;
            }
            let payload_id = PayloadId(p_idx as u16);
            let payload_pose = &workspace.payload_poses[p_idx];
            let payload_aabb = &workspace.payload_aabbs[p_idx];

            // 3a. Payload vs Static Environment Obstacles
            if let Some(bvh) = &self.obstacle_bvh {
                let base_margin = payload.safety_margin.unwrap_or(0.0);
                let max_obs_margin = self
                    .obstacle_margins
                    .iter()
                    .flatten()
                    .copied()
                    .fold(base_margin, f64::max);
                let rule_max = self
                    .payload_obstacle_rules
                    .iter()
                    .filter(|((pid, _), _)| pid == &payload.id)
                    .filter_map(|(_, r)| r.margin)
                    .fold(max_obs_margin, f64::max);
                let query_aabb = payload_aabb.loosened(rule_max.max(0.0));

                for leaf_idx in bvh.intersect_aabb(&query_aabb) {
                    let obs_id = self.bvh_leaf_obstacles[leaf_idx as usize];
                    if !self.is_payload_obstacle_id_enabled(payload_id, obs_id) {
                        continue;
                    }
                    let pair = CollidingPair::PayloadToObstacle {
                        payload: payload_id,
                        parent_link: payload.parent_link,
                        obstacle: obs_id,
                    };
                    let margin = self.custom_pair_margin(&pair).unwrap_or(0.0);
                    let obs = &self.obstacles[obs_id.0 as usize];
                    if !payload_aabb
                        .loosened(margin.max(0.0))
                        .intersects(&obs.world_aabb())
                    {
                        continue;
                    }
                    let collides = match tier {
                        GeometryTier::AnalyticSsv => {
                            let range = &workspace.world_spheres.payload_ranges[p_idx];
                            if !range.is_empty() {
                                Self::spheres_collide_with_obstacle(
                                    &workspace.world_spheres,
                                    range,
                                    obs,
                                    margin,
                                )
                            } else {
                                Self::link_collides_with_obstacle(
                                    payload_pose,
                                    &payload.geometry.convex_shapes,
                                    obs,
                                    margin,
                                )
                            }
                        }
                        GeometryTier::Convex => Self::link_collides_with_obstacle(
                            payload_pose,
                            &payload.geometry.convex_shapes,
                            obs,
                            margin,
                        ),
                    };
                    if collides {
                        return Ok(false);
                    }
                }
            }

            // 3b. Payload vs Non-Touch Robot Links
            for link_idx in 0..model.num_links() {
                let link_id = LinkId(link_idx as u16);
                if link_id == payload.parent_link || payload.touch_links.contains(&link_id) {
                    continue;
                }
                if !self.is_payload_link_id_enabled(payload_id, link_id) {
                    continue;
                }
                let Some(geom) = self.link_geometry(link_id) else {
                    continue;
                };
                let pair = CollidingPair::PayloadToLink {
                    payload: payload_id,
                    parent_link: payload.parent_link,
                    link: link_id,
                };
                let margin = self.custom_pair_margin(&pair).unwrap_or(0.0);
                if !payload_aabb
                    .loosened(margin.max(0.0))
                    .intersects(&workspace.link_aabbs[link_idx])
                {
                    continue;
                }
                let collides = match tier {
                    GeometryTier::AnalyticSsv => {
                        let range_p = &workspace.world_spheres.payload_ranges[p_idx];
                        let range_l = &workspace.world_spheres.link_ranges[link_idx];
                        if !range_p.is_empty() && !range_l.is_empty() {
                            workspace
                                .world_spheres
                                .ranges_intersect_simd(range_p, range_l, margin)
                        } else {
                            let link_pose = workspace.transforms.link_pose(link_id);
                            Self::shapes_collide(
                                payload_pose,
                                &payload.geometry.convex_shapes,
                                link_pose,
                                &geom.convex_shapes,
                                margin,
                            )
                        }
                    }
                    GeometryTier::Convex => {
                        let link_pose = workspace.transforms.link_pose(link_id);
                        Self::shapes_collide(
                            payload_pose,
                            &payload.geometry.convex_shapes,
                            link_pose,
                            &geom.convex_shapes,
                            margin,
                        )
                    }
                };
                if collides {
                    return Ok(false);
                }
            }

            // 3c. Payload vs Other Attached Payloads (Tool-to-Tool)
            for p_b_idx in (p_idx + 1)..num_payloads {
                let payload_b = &self.attached_payloads[p_b_idx];
                if payload_b.parent_link.index() >= model.num_links()
                    || payload.parent_link == payload_b.parent_link
                {
                    continue;
                }
                let payload_b_id = PayloadId(p_b_idx as u16);
                if !self.is_payload_pair_id_enabled(payload_id, payload_b_id) {
                    continue;
                }
                let pair = CollidingPair::PayloadToPayload {
                    payload_a: payload_id,
                    parent_link_a: payload.parent_link,
                    payload_b: payload_b_id,
                    parent_link_b: payload_b.parent_link,
                };
                let margin = self.custom_pair_margin(&pair).unwrap_or(0.0);
                if !payload_aabb
                    .loosened(margin.max(0.0))
                    .intersects(&workspace.payload_aabbs[p_b_idx])
                {
                    continue;
                }
                let collides = match tier {
                    GeometryTier::AnalyticSsv => {
                        let range_a = &workspace.world_spheres.payload_ranges[p_idx];
                        let range_b = &workspace.world_spheres.payload_ranges[p_b_idx];
                        if !range_a.is_empty() && !range_b.is_empty() {
                            workspace
                                .world_spheres
                                .ranges_intersect_simd(range_a, range_b, margin)
                        } else {
                            Self::shapes_collide(
                                payload_pose,
                                &payload.geometry.convex_shapes,
                                &workspace.payload_poses[p_b_idx],
                                &payload_b.geometry.convex_shapes,
                                margin,
                            )
                        }
                    }
                    GeometryTier::Convex => Self::shapes_collide(
                        payload_pose,
                        &payload.geometry.convex_shapes,
                        &workspace.payload_poses[p_b_idx],
                        &payload_b.geometry.convex_shapes,
                        margin,
                    ),
                };
                if collides {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Continuous collision checking along the geodesic C-space segment `[q0, q1]`.
    ///
    /// Uses C-space Lipschitz lever-arm bounds (including attached tool extents) to determine
    /// a conservative subdivision count.
    pub fn is_segment_collision_free(
        &self,
        model: &MultiRobotModel,
        q0: &[f64],
        q1: &[f64],
        max_workspace_step: f64,
        tier: GeometryTier,
        workspace: &mut CollisionWorkspace,
    ) -> Result<bool, CollisionError> {
        if !self.is_state_collision_free(model, q0, tier, workspace)?
            || !self.is_state_collision_free(model, q1, tier, workspace)?
        {
            return Ok(false);
        }

        // Compute upper bound on workspace displacement across all links & attached tools using Lipschitz lever arms
        let mut max_disp = 0.0_f64;
        for link_idx in 0..model.num_links() {
            let disp = model.max_link_displacement(LinkId(link_idx as u16), q0, q1);
            if disp > max_disp {
                max_disp = disp;
            }
        }
        for payload in &self.attached_payloads {
            if payload.parent_link.index() >= model.num_links() {
                continue;
            }
            let tool_reach = payload.local_pose.translation.vector.norm()
                + payload.geometry.max_extent_from_origin();
            let mut disp = model.max_link_displacement(payload.parent_link, q0, q1);
            if tool_reach > 0.0 {
                for q_idx in 0..model.dof() {
                    if model.is_dof_ancestor_of_link(payload.parent_link, q_idx) {
                        let joint = model.joint(model.actuated_joint(q_idx));
                        if !matches!(joint.joint_type, JointType::Prismatic { .. }) {
                            disp += tool_reach * model.dof_distance(q_idx, q0[q_idx], q1[q_idx]);
                        }
                    }
                }
            }
            if disp > max_disp {
                max_disp = disp;
            }
        }
        if max_disp == 0.0 {
            max_disp = model.distance(q0, q1);
        }

        let step = max_workspace_step.max(1e-3);
        let num_segments = ((max_disp / step).ceil() as usize).clamp(1, 256);
        if num_segments <= 1 {
            return Ok(true);
        }

        // Check internal points along the segment
        let mut scratch_q = std::mem::take(&mut workspace.scratch_q);
        scratch_q.resize(model.dof(), 0.0);

        let result = (1..num_segments).try_for_each(|idx| {
            let alpha = idx as f64 / num_segments as f64;
            model.interpolate(q0, q1, alpha, &mut scratch_q);
            match self.is_state_collision_free(model, &scratch_q, tier, workspace) {
                Ok(true) => Ok(()),
                Ok(false) => Err(None),
                Err(e) => Err(Some(e)),
            }
        });

        workspace.scratch_q = scratch_q;
        match result {
            Ok(()) => Ok(true),
            Err(None) => Ok(false),
            Err(Some(e)) => Err(e),
        }
    }

    /// Queries all link-obstacle, active link-link, and attached payload signed distance witnesses
    /// within `cutoff_distance`.
    ///
    /// Populates `out_witnesses` with signed distances, world-space witness points, and unit normals
    /// suitable for [`SignedDistanceWitness::compute_distance_jacobian`].
    pub fn query_signed_distances(
        &self,
        model: &MultiRobotModel,
        q: &[f64],
        cutoff_distance: f64,
        tier: GeometryTier,
        workspace: &mut CollisionWorkspace,
        out_witnesses: &mut Vec<SignedDistanceWitness>,
    ) -> Result<(), CollisionError> {
        self.update_workspace(model, q, tier, workspace)?;
        out_witnesses.clear();
        let margin = cutoff_distance.max(0.0);

        // 1. Active Link-to-Link pairs
        if let Some(acm) = &self.acm {
            acm.for_each_active_pair(|link_a, link_b| {
                let (Some(geom_a), Some(geom_b)) =
                    (self.link_geometry(link_a), self.link_geometry(link_b))
                else {
                    return;
                };
                if !workspace.link_aabbs[link_a.index()]
                    .loosened(margin)
                    .intersects(&workspace.link_aabbs[link_b.index()])
                {
                    return;
                }

                let pose_a = workspace.transforms.link_pose(link_a);
                let pose_b = workspace.transforms.link_pose(link_b);

                if tier == GeometryTier::AnalyticSsv {
                    if let (Some(cap_a), Some(cap_b)) =
                        (&geom_a.bounding_capsule, &geom_b.bounding_capsule)
                    {
                        let w_a = cap_a.transformed(pose_a);
                        let w_b = cap_b.transformed(pose_b);
                        let (dist, p_a, p_b, normal) = w_a.signed_distance_to_capsule(&w_b);
                        if dist <= cutoff_distance {
                            out_witnesses.push(SignedDistanceWitness {
                                pair: CollidingPair::LinkToLink { link_a, link_b },
                                signed_distance: dist,
                                witness_a_world: p_a,
                                witness_b_world: p_b,
                                normal_world: normal,
                            });
                        }
                        return;
                    }
                }

                if let Some((dist, p_a, p_b, normal)) = Self::closest_shapes_witness(
                    pose_a,
                    &geom_a.convex_shapes,
                    pose_b,
                    &geom_b.convex_shapes,
                    cutoff_distance,
                ) {
                    out_witnesses.push(SignedDistanceWitness {
                        pair: CollidingPair::LinkToLink { link_a, link_b },
                        signed_distance: dist,
                        witness_a_world: p_a,
                        witness_b_world: p_b,
                        normal_world: normal,
                    });
                }
            });
        }

        // 2. Link-to-Obstacle pairs via BVH broad-phase with loosened AABB
        if let Some(bvh) = &self.obstacle_bvh {
            for link_idx in 0..model.num_links() {
                let link_id = LinkId(link_idx as u16);
                let Some(geom) = self.link_geometry(link_id) else {
                    continue;
                };
                let query_aabb = workspace.link_aabbs[link_idx].loosened(margin);
                let link_pose = workspace.transforms.link_pose(link_id);

                for leaf_idx in bvh.intersect_aabb(&query_aabb) {
                    let obs_id = self.bvh_leaf_obstacles[leaf_idx as usize];
                    if !self.is_link_obstacle_enabled(link_id, obs_id) {
                        continue;
                    }
                    let obs = &self.obstacles[obs_id.0 as usize];
                    let obs_slice = [(Isometry3::identity(), obs.shape.clone())];
                    if let Some((dist, p_a, p_b, normal)) = Self::closest_shapes_witness(
                        link_pose,
                        &geom.convex_shapes,
                        &obs.pose,
                        &obs_slice,
                        cutoff_distance,
                    ) {
                        out_witnesses.push(SignedDistanceWitness {
                            pair: CollidingPair::LinkToObstacle {
                                link: link_id,
                                obstacle: obs_id,
                            },
                            signed_distance: dist,
                            witness_a_world: p_a,
                            witness_b_world: p_b,
                            normal_world: normal,
                        });
                    }
                }
            }
        }

        // 3-5. Attached Payloads vs Obstacles, Non-Touch Links, and Other Payloads
        let num_payloads = self.attached_payloads.len();
        for p_idx in 0..num_payloads {
            let payload = &self.attached_payloads[p_idx];
            if payload.parent_link.index() >= model.num_links() {
                continue;
            }
            let payload_id = PayloadId(p_idx as u16);
            let payload_pose = &workspace.payload_poses[p_idx];
            let payload_aabb = &workspace.payload_aabbs[p_idx];
            let loosened_payload = payload_aabb.loosened(margin);

            // 3. Payload-to-Obstacle
            if let Some(bvh) = &self.obstacle_bvh {
                for leaf_idx in bvh.intersect_aabb(&loosened_payload) {
                    let obs_id = self.bvh_leaf_obstacles[leaf_idx as usize];
                    if !self.is_payload_obstacle_id_enabled(payload_id, obs_id) {
                        continue;
                    }
                    let obs = &self.obstacles[obs_id.0 as usize];
                    let witness = if tier == GeometryTier::AnalyticSsv
                        && !workspace.world_spheres.payload_ranges[p_idx].is_empty()
                    {
                        Self::closest_spheres_to_obstacle_witness(
                            &workspace.world_spheres,
                            &workspace.world_spheres.payload_ranges[p_idx],
                            obs,
                            cutoff_distance,
                        )
                    } else {
                        let obs_slice = [(Isometry3::identity(), obs.shape.clone())];
                        Self::closest_shapes_witness(
                            payload_pose,
                            &payload.geometry.convex_shapes,
                            &obs.pose,
                            &obs_slice,
                            cutoff_distance,
                        )
                    };
                    if let Some((dist, p_a, p_b, normal)) = witness {
                        out_witnesses.push(SignedDistanceWitness {
                            pair: CollidingPair::PayloadToObstacle {
                                payload: payload_id,
                                parent_link: payload.parent_link,
                                obstacle: obs_id,
                            },
                            signed_distance: dist,
                            witness_a_world: p_a,
                            witness_b_world: p_b,
                            normal_world: normal,
                        });
                    }
                }
            }

            // 4. Payload-to-Link
            for link_idx in 0..model.num_links() {
                let link_id = LinkId(link_idx as u16);
                if link_id == payload.parent_link || payload.touch_links.contains(&link_id) {
                    continue;
                }
                if !self.is_payload_link_id_enabled(payload_id, link_id) {
                    continue;
                }
                let Some(geom) = self.link_geometry(link_id) else {
                    continue;
                };
                if !loosened_payload.intersects(&workspace.link_aabbs[link_idx]) {
                    continue;
                }
                let witness = if tier == GeometryTier::AnalyticSsv
                    && !workspace.world_spheres.payload_ranges[p_idx].is_empty()
                    && !workspace.world_spheres.link_ranges[link_idx].is_empty()
                {
                    workspace.world_spheres.closest_spheres_witness(
                        &workspace.world_spheres.payload_ranges[p_idx],
                        &workspace.world_spheres.link_ranges[link_idx],
                        cutoff_distance,
                    )
                } else {
                    let link_pose = workspace.transforms.link_pose(link_id);
                    Self::closest_shapes_witness(
                        payload_pose,
                        &payload.geometry.convex_shapes,
                        link_pose,
                        &geom.convex_shapes,
                        cutoff_distance,
                    )
                };
                if let Some((dist, p_a, p_b, normal)) = witness {
                    out_witnesses.push(SignedDistanceWitness {
                        pair: CollidingPair::PayloadToLink {
                            payload: payload_id,
                            parent_link: payload.parent_link,
                            link: link_id,
                        },
                        signed_distance: dist,
                        witness_a_world: p_a,
                        witness_b_world: p_b,
                        normal_world: normal,
                    });
                }
            }

            // 5. Payload-to-Payload
            for p_b_idx in (p_idx + 1)..num_payloads {
                let payload_b = &self.attached_payloads[p_b_idx];
                if payload_b.parent_link.index() >= model.num_links()
                    || payload.parent_link == payload_b.parent_link
                {
                    continue;
                }
                let payload_b_id = PayloadId(p_b_idx as u16);
                if !self.is_payload_pair_id_enabled(payload_id, payload_b_id) {
                    continue;
                }
                if !loosened_payload.intersects(&workspace.payload_aabbs[p_b_idx]) {
                    continue;
                }
                let witness = if tier == GeometryTier::AnalyticSsv
                    && !workspace.world_spheres.payload_ranges[p_idx].is_empty()
                    && !workspace.world_spheres.payload_ranges[p_b_idx].is_empty()
                {
                    workspace.world_spheres.closest_spheres_witness(
                        &workspace.world_spheres.payload_ranges[p_idx],
                        &workspace.world_spheres.payload_ranges[p_b_idx],
                        cutoff_distance,
                    )
                } else {
                    Self::closest_shapes_witness(
                        payload_pose,
                        &payload.geometry.convex_shapes,
                        &workspace.payload_poses[p_b_idx],
                        &payload_b.geometry.convex_shapes,
                        cutoff_distance,
                    )
                };
                if let Some((dist, p_a, p_b, normal)) = witness {
                    out_witnesses.push(SignedDistanceWitness {
                        pair: CollidingPair::PayloadToPayload {
                            payload_a: payload_id,
                            parent_link_a: payload.parent_link,
                            payload_b: PayloadId(p_b_idx as u16),
                            parent_link_b: payload_b.parent_link,
                        },
                        signed_distance: dist,
                        witness_a_world: p_a,
                        witness_b_world: p_b,
                        normal_world: normal,
                    });
                }
            }
        }

        Ok(())
    }

    fn shapes_collide(
        pose_a: &Isometry3<f64>,
        shapes_a: &[(Isometry3<f64>, SharedShape)],
        pose_b: &Isometry3<f64>,
        shapes_b: &[(Isometry3<f64>, SharedShape)],
        margin: f64,
    ) -> bool {
        for (off_a, shape_a) in shapes_a {
            let pa = isometry_to_parry(&(pose_a * off_a));
            for (off_b, shape_b) in shapes_b {
                let pb = isometry_to_parry(&(pose_b * off_b));
                if margin <= 0.0 {
                    if parry3d::query::intersection_test(&pa, shape_a.as_ref(), &pb, shape_b.as_ref())
                        .unwrap_or(false)
                    {
                        return true;
                    }
                } else if let Ok(d) =
                    parry3d::query::distance(&pa, shape_a.as_ref(), &pb, shape_b.as_ref())
                {
                    if d <= margin {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn link_collides_with_obstacle(
        link_pose: &Isometry3<f64>,
        shapes_a: &[(Isometry3<f64>, SharedShape)],
        obs: &Obstacle,
        margin: f64,
    ) -> bool {
        let pb = isometry_to_parry(&obs.pose);
        for (off_a, shape_a) in shapes_a {
            let pa = isometry_to_parry(&(link_pose * off_a));
            if margin <= 0.0 {
                if parry3d::query::intersection_test(&pa, shape_a.as_ref(), &pb, obs.shape.as_ref())
                    .unwrap_or(false)
                {
                    return true;
                }
            } else if let Ok(d) =
                parry3d::query::distance(&pa, shape_a.as_ref(), &pb, obs.shape.as_ref())
            {
                if d <= margin {
                    return true;
                }
            }
        }
        false
    }

    fn closest_shapes_witness(
        pose_a: &Isometry3<f64>,
        shapes_a: &[(Isometry3<f64>, SharedShape)],
        pose_b: &Isometry3<f64>,
        shapes_b: &[(Isometry3<f64>, SharedShape)],
        cutoff: f64,
    ) -> Option<ShapeWitness> {
        let mut best: Option<ShapeWitness> = None;

        for (off_a, shape_a) in shapes_a {
            let pa = isometry_to_parry(&(pose_a * off_a));
            for (off_b, shape_b) in shapes_b {
                let pb = isometry_to_parry(&(pose_b * off_b));
                if let Ok(Some(contact)) = parry3d::query::contact(
                    &pa,
                    shape_a.as_ref(),
                    &pb,
                    shape_b.as_ref(),
                    cutoff.max(0.0),
                ) {
                    let dist = contact.dist;
                    if dist <= cutoff && best.as_ref().is_none_or(|b| dist < b.0) {
                        // In parry3d, `contact.normal1` points from shape 1 (A) toward shape 2 (B).
                        // Therefore the gradient of distance w.r.t. A (pointing from B toward A) is `-normal1`.
                        let p_a = parry_to_point(contact.point1);
                        let p_b = parry_to_point(contact.point2);
                        let n1 = parry_to_vector(contact.normal1);
                        let normal_b_to_a = -n1;
                        best = Some((dist, p_a, p_b, normal_b_to_a));
                    }
                }
            }
        }
        best
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
        robot.validate_configuration(configuration)?;
        if let Some(model) = &robot.model {
            let mut workspace = self.create_workspace(model);
            return self.is_state_collision_free(
                model,
                configuration.as_slice(),
                GeometryTier::Convex,
                &mut workspace,
            );
        }
        Ok(true)
    }
}
