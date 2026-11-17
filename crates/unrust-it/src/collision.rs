//! Multi-tier collision detection, bitset Allowed Collision Matrix (ACM),
//! BVH broad-phase, continuous collision checking, and differentiable signed-distance queries.

use crate::kinematics::{KinematicsError, TransformBuffer};
use crate::robot::{
    AttachedPayload, Configuration, LinkId, MultiRobotModel, ObstacleId, Robot, RobotError,
};
use nalgebra::{Isometry3, Point3, Vector3};
use parry3d::bounding_volume::{Aabb, BoundingVolume};
use parry3d::math::{Pose as ParryPose, Rotation as ParryRotation, Vector as ParryVector};
use parry3d::partitioning::{Bvh, BvhBuildStrategy};
use parry3d::shape::SharedShape;
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

/// Multi-tier collision geometry attached to a single robot link.
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
    /// Creates a link geometry from a single `parry3d` shape and automatically populates a Tier-1 sphere/capsule bound.
    pub fn from_shape(local_pose: Isometry3<f64>, shape: SharedShape) -> Self {
        let parry_pose = isometry_to_parry(&local_pose);
        let local_aabb = shape.compute_aabb(&parry_pose);
        let center = parry_to_point(local_aabb.center());
        let half_extents = parry_to_vector(local_aabb.half_extents());
        let radius = half_extents.norm();
        Self {
            local_aabb,
            bounding_capsule: Some(LinkCapsule::new(center, center, radius)),
            local_spheres: vec![BoundingSphere::new(center, radius)],
            convex_shapes: vec![(local_pose, shape)],
        }
    }

    /// Creates a link geometry from a capsule (generating both Tier-1 spheres/capsule and a Tier-2 `parry3d` capsule).
    pub fn from_capsule(endpoint_a: Point3<f64>, endpoint_b: Point3<f64>, radius: f64) -> Self {
        let seg_vec = endpoint_b - endpoint_a;
        let len = seg_vec.norm();
        let num_spheres = ((len / radius.max(1e-3)).ceil() as usize).clamp(1, 8) + 1;
        let mut local_spheres = Vec::with_capacity(num_spheres);
        for i in 0..num_spheres {
            let alpha = if num_spheres > 1 {
                i as f64 / (num_spheres - 1) as f64
            } else {
                0.5
            };
            local_spheres.push(BoundingSphere::new(endpoint_a + seg_vec * alpha, radius));
        }

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

/// Structure-of-Arrays (SoA) batch storage of world-space spheres across all links.
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
}

impl WorldSphereSoA {
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
        match self.rule(a, b) {
            CollisionRule::CustomMargin(m) => m,
            _ => 0.0,
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
            CollidingPair::LinkToObstacle { link, .. } => {
                model.accumulate_point_jacobian_projection(
                    transforms,
                    link,
                    &self.witness_a_world,
                    &self.normal_world,
                    1.0,
                    out_j_dist,
                );
            }
            CollidingPair::LinkToLink { link_a, link_b } => {
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
    /// World-space transformed Tier-1 spheres in SoA layout.
    pub world_spheres: WorldSphereSoA,
    /// Scratch buffer for continuous collision interpolation.
    pub scratch_q: Vec<f64>,
}

impl CollisionWorkspace {
    /// Creates a pre-allocated collision workspace sized for `model` and `env`.
    pub fn new(model: &MultiRobotModel, env: &CollisionEnvironment) -> Self {
        let num_links = model.num_links();
        let total_spheres: usize = (0..num_links)
            .map(|i| {
                env.link_geometries
                    .get(i)
                    .and_then(|g| g.as_ref())
                    .map_or(0, |g| g.local_spheres.len())
            })
            .sum();

        Self {
            transforms: model.create_transform_buffer(),
            link_aabbs: vec![Aabb::new_invalid(); num_links],
            world_spheres: WorldSphereSoA {
                cx: vec![0.0; total_spheres],
                cy: vec![0.0; total_spheres],
                cz: vec![0.0; total_spheres],
                radius: vec![0.0; total_spheres],
                link_ranges: vec![0..0; num_links],
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

/// Multi-arm collision scene holding environment obstacles (indexed by a `Bvh`),
/// multi-tier robot link geometries, an [`AllowedCollisionMatrix`], and attached payloads.
#[derive(Debug, Default, Clone)]
pub struct CollisionEnvironment {
    /// Obstacles in the world frame.
    pub obstacles: Vec<Obstacle>,
    /// Broad-phase Bounding Volume Hierarchy over `obstacles`.
    obstacle_bvh: Option<Bvh>,
    /// Per-link collision geometries indexed by `LinkId`.
    link_geometries: Vec<Option<LinkGeometry>>,
    /// Allowed Collision Matrix governing link-to-link self and inter-arm collision pairs.
    acm: Option<AllowedCollisionMatrix>,
    /// Dynamically grasped/attached payloads.
    attached_payloads: Vec<AttachedPayload>,
}

impl CollisionEnvironment {
    /// Creates an empty collision environment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a `parry3d` obstacle to the environment and updates the broad-phase BVH.
    pub fn add_obstacle(&mut self, shape: SharedShape, pose: Isometry3<f64>) -> ObstacleId {
        let id = ObstacleId(self.obstacles.len() as u32);
        self.obstacles.push(Obstacle { shape, pose });
        self.rebuild_bvh();
        id
    }

    /// Rebuilds the static obstacle `Bvh` broad-phase acceleration structure.
    pub fn rebuild_bvh(&mut self) {
        if self.obstacles.is_empty() {
            self.obstacle_bvh = None;
            return;
        }
        let leaves: Vec<(usize, Aabb)> = self
            .obstacles
            .iter()
            .enumerate()
            .map(|(i, obs)| (i, obs.world_aabb()))
            .collect();
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

    /// Attaches a grasped payload to a robot link.
    pub fn attach_payload(&mut self, payload: AttachedPayload) {
        self.attached_payloads.retain(|p| p.id != payload.id);
        self.attached_payloads.push(payload);
    }

    /// Detaches a grasped payload by its ID.
    pub fn detach_payload(&mut self, id: &str) -> Option<AttachedPayload> {
        let pos = self.attached_payloads.iter().position(|p| p.id == id)?;
        Some(self.attached_payloads.swap_remove(pos))
    }

    /// Allocates a [`CollisionWorkspace`] sized for `model` and this environment.
    pub fn create_workspace(&self, model: &MultiRobotModel) -> CollisionWorkspace {
        CollisionWorkspace::new(model, self)
    }

    /// Updates `workspace` with Forward Kinematics, world-space link AABBs, and SoA spheres for `q`.
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
        workspace.link_aabbs.resize(num_links, Aabb::new_invalid());
        workspace.world_spheres.link_ranges.resize(num_links, 0..0);

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

        Ok(())
    }

    /// Evaluates whether configuration `q` is free of self-collision, inter-arm collision,
    /// and environment obstacle collisions using a pre-allocated [`CollisionWorkspace`].
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

                for obs_idx in bvh.intersect_aabb(link_aabb) {
                    let obs = &self.obstacles[obs_idx as usize];
                    let collides = match tier {
                        GeometryTier::AnalyticSsv => {
                            let obs_pose = isometry_to_parry(&obs.pose);
                            geom.local_spheres.iter().any(|s| {
                                let center_w = link_pose.transform_point(&s.center);
                                let ball = parry3d::shape::Ball::new(s.radius);
                                let ball_pose = ParryPose::from_parts(
                                    ParryVector::new(center_w.x, center_w.y, center_w.z),
                                    ParryRotation::IDENTITY,
                                );
                                parry3d::query::intersection_test(
                                    &ball_pose,
                                    &ball,
                                    &obs_pose,
                                    obs.shape.as_ref(),
                                )
                                .unwrap_or(false)
                            })
                        }
                        GeometryTier::Convex => Self::link_collides_with_obstacle(
                            link_pose,
                            &geom.convex_shapes,
                            obs,
                            0.0,
                        ),
                    };
                    if collides {
                        return Ok(false);
                    }
                }
            }

            // 3. Attached Payloads vs Environment Obstacles & Non-Touch Links
            for payload in &self.attached_payloads {
                if payload.parent_link.index() >= model.num_links() {
                    continue;
                }
                let payload_world_pose =
                    workspace.transforms.link_pose(payload.parent_link) * payload.local_pose;
                let payload_parry_pose = isometry_to_parry(&payload_world_pose);
                let payload_aabb = payload.shape.compute_aabb(&payload_parry_pose);

                for obs_idx in bvh.intersect_aabb(&payload_aabb) {
                    let obs = &self.obstacles[obs_idx as usize];
                    let obs_pose = isometry_to_parry(&obs.pose);
                    if parry3d::query::intersection_test(
                        &payload_parry_pose,
                        payload.shape.as_ref(),
                        &obs_pose,
                        obs.shape.as_ref(),
                    )
                    .unwrap_or(false)
                    {
                        return Ok(false);
                    }
                }

                for link_idx in 0..model.num_links() {
                    let link_id = LinkId(link_idx as u16);
                    if link_id == payload.parent_link || payload.touch_links.contains(&link_id) {
                        continue;
                    }
                    let Some(geom) = self.link_geometry(link_id) else {
                        continue;
                    };
                    if !payload_aabb.intersects(&workspace.link_aabbs[link_idx]) {
                        continue;
                    }
                    let link_pose = workspace.transforms.link_pose(link_id);
                    for (local_offset, shape) in &geom.convex_shapes {
                        let shape_pose = isometry_to_parry(&(link_pose * local_offset));
                        if parry3d::query::intersection_test(
                            &payload_parry_pose,
                            payload.shape.as_ref(),
                            &shape_pose,
                            shape.as_ref(),
                        )
                        .unwrap_or(false)
                        {
                            return Ok(false);
                        }
                    }
                }
            }
        }

        Ok(true)
    }

    /// Continuous collision checking along the geodesic C-space segment `[q0, q1]`.
    ///
    /// Uses C-space Lipschitz lever-arm bounds to determine a conservative subdivision count
    /// paired with binary bisection ordering (checking midpoints first for fast collision rejection).
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

        // Compute upper bound on workspace displacement across all links using Lipschitz lever arms
        let mut max_disp = 0.0_f64;
        for link_idx in 0..model.num_links() {
            let disp = model.max_link_displacement(LinkId(link_idx as u16), q0, q1);
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

        // Check internal points in van der Corput / binary bisection order (1/2, 1/4, 3/4, ...)
        // so collisions near the middle of a segment are detected in O(1) evaluations.
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

    /// Queries all link-obstacle and active link-link signed distance witnesses within `cutoff_distance`.
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

                for obs_idx in bvh.intersect_aabb(&query_aabb) {
                    let obs = &self.obstacles[obs_idx as usize];
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
                                obstacle: ObstacleId(obs_idx),
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
