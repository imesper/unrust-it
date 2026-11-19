//! Robot descriptions, multi-arm kinematic forests, and strongly typed configuration spaces.

use nalgebra::{DVector, Isometry3, Point3, UnitVector3, Vector3};
use std::f64::consts::{PI, TAU};
use std::sync::Arc;
use thiserror::Error;

/// Strongly typed identifier for a robot or manipulator in a multi-robot system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RobotId(pub u16);

/// Strongly typed dense index for a link in a [`MultiRobotModel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LinkId(pub u16);

impl LinkId {
    /// Index as `usize` for slice lookups.
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Strongly typed dense index for a joint in a [`MultiRobotModel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JointId(pub u16);

impl JointId {
    /// Index as `usize` for slice lookups.
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Strongly typed dense index for a coordinate frame in a [`MultiRobotModel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FrameId(pub u16);

impl FrameId {
    /// The canonical inertial world root frame (always index 0).
    pub const WORLD: Self = Self(0);

    /// Index as `usize` for slice lookups.
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Strongly typed identifier for an environment obstacle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObstacleId(pub u32);

/// Strongly typed dense index for an attached payload or tool in a [`crate::collision::CollisionEnvironment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PayloadId(pub u16);

impl PayloadId {
    /// Index as `usize` for slice lookups.
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A joint's inclusive limits, expressed in radians (revolute) or meters (prismatic).
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

    /// Creates an unbounded joint range (`[-inf, +inf]`), e.g. for continuous joints.
    pub const fn unbounded() -> Self {
        Self {
            min: f64::NEG_INFINITY,
            max: f64::INFINITY,
        }
    }

    /// Returns whether a joint value is inside this limit.
    #[inline]
    pub fn contains(&self, value: f64) -> bool {
        (self.min..=self.max).contains(&value)
    }

    /// Clamps `value` into `[self.min, self.max]`.
    #[inline]
    pub fn clamp(&self, value: f64) -> f64 {
        value.clamp(self.min, self.max)
    }

    /// Returns the midpoint of the joint range (or `0.0` if unbounded).
    #[inline]
    pub fn midpoint(&self) -> f64 {
        if self.min.is_finite() && self.max.is_finite() {
            0.5 * (self.min + self.max)
        } else {
            0.0
        }
    }

    /// Returns the width `max - min` of the joint range.
    #[inline]
    pub fn span(&self) -> f64 {
        self.max - self.min
    }
}

/// First-, second-, and third-order kinodynamic bounds for an actuated degree of freedom.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KinodynamicLimits {
    /// Position limits (rad or m).
    pub position: JointLimit,
    /// Maximum absolute velocity (rad/s or m/s).
    pub max_velocity: f64,
    /// Maximum absolute acceleration (rad/s^2 or m/s^2).
    pub max_acceleration: f64,
    /// Maximum absolute jerk (rad/s^3 or m/s^3).
    pub max_jerk: f64,
    /// Maximum generalized torque or force (N*m or N).
    pub max_effort: f64,
}

impl KinodynamicLimits {
    /// Creates kinodynamic limits from position bounds with sensible default dynamic bounds.
    pub fn from_position(position: JointLimit) -> Self {
        Self {
            position,
            max_velocity: PI,
            max_acceleration: 10.0,
            max_jerk: 100.0,
            max_effort: 100.0,
        }
    }

    /// Creates fully specified kinodynamic bounds.
    pub fn new(
        position: JointLimit,
        max_velocity: f64,
        max_acceleration: f64,
        max_jerk: f64,
        max_effort: f64,
    ) -> Self {
        assert!(max_velocity >= 0.0, "max_velocity must be non-negative");
        assert!(
            max_acceleration >= 0.0,
            "max_acceleration must be non-negative"
        );
        assert!(max_jerk >= 0.0, "max_jerk must be non-negative");
        assert!(max_effort >= 0.0, "max_effort must be non-negative");
        Self {
            position,
            max_velocity,
            max_acceleration,
            max_jerk,
            max_effort,
        }
    }
}

/// A point in configuration space (C-space), one value per degree of freedom.
#[derive(Debug, Clone, PartialEq)]
pub struct Configuration(pub Vec<f64>);

impl Configuration {
    /// Creates a joint-space configuration.
    pub fn new(values: Vec<f64>) -> Self {
        Self(values)
    }

    /// Creates a zero-initialized configuration of dimension `dof`.
    pub fn zeros(dof: usize) -> Self {
        Self(vec![0.0; dof])
    }

    /// Creates a configuration from a borrowed slice.
    pub fn from_slice(slice: &[f64]) -> Self {
        Self(slice.to_vec())
    }

    /// Returns the number of degrees of freedom represented.
    #[inline]
    pub fn dof(&self) -> usize {
        self.0.len()
    }

    /// Borrows the underlying joint values as a contiguous slice.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }

    /// Mutably borrows the underlying joint values as a contiguous slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.0
    }
}

/// Full kinodynamic joint state ($q, \dot{q}, \ddot{q}, \tau$) for dynamic manipulation.
#[derive(Debug, Clone, PartialEq)]
pub struct JointState {
    /// Generalized joint positions $q$.
    pub q: DVector<f64>,
    /// Generalized joint velocities $\dot{q}$.
    pub dq: DVector<f64>,
    /// Generalized joint accelerations $\ddot{q}$.
    pub ddq: DVector<f64>,
    /// Optional generalized joint efforts/torques $\tau$.
    pub tau: Option<DVector<f64>>,
}

impl JointState {
    /// Creates a stationary joint state at position `q`.
    pub fn from_position(q: DVector<f64>) -> Self {
        let n = q.len();
        Self {
            q,
            dq: DVector::zeros(n),
            ddq: DVector::zeros(n),
            tau: None,
        }
    }

    /// Returns the number of degrees of freedom.
    #[inline]
    pub fn dof(&self) -> usize {
        self.q.len()
    }
}

/// A target point in 3D Cartesian task space, kept distinct from a joint configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskSpacePoint {
    /// Cartesian position in the robot's world frame.
    pub position: Point3<f64>,
}

impl TaskSpacePoint {
    /// Creates a Cartesian task-space point target.
    pub fn new(position: Point3<f64>) -> Self {
        Self { position }
    }
}

/// A full 6-DOF $SE(3)$ pose target for a specific coordinate frame (e.g., an arm's TCP).
#[derive(Debug, Clone, PartialEq)]
pub struct TaskPose {
    /// The frame whose world pose is targeted.
    pub frame: FrameId,
    /// Desired world-from-frame rigid transformation $T_{\text{world}}^{\text{frame}} \in SE(3)$.
    pub pose: Isometry3<f64>,
}

impl TaskPose {
    /// Creates a new $SE(3)$ task-space pose target for `frame`.
    pub fn new(frame: FrameId, pose: Isometry3<f64>) -> Self {
        Self { frame, pose }
    }
}

/// Errors raised while constructing or validating a robot or multi-robot model.
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
    /// Invalid parent link or frame index when building a kinematic tree.
    #[error("invalid topological reference: {0}")]
    InvalidTopology(String),
}

/// Articulation type of a joint connecting a parent link to a child link.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointType {
    /// Rigid weld (0 DOF).
    Fixed,
    /// Bounded 1-DOF rotation about `axis` in the joint frame.
    Revolute { axis: UnitVector3<f64> },
    /// Unbounded $SO(2)$ 1-DOF rotation about `axis` with $-\pi \equiv \pi$ wrap-around.
    Continuous { axis: UnitVector3<f64> },
    /// Bounded 1-DOF translation along `axis` in the joint frame.
    Prismatic { axis: UnitVector3<f64> },
}

impl JointType {
    /// Number of configuration degrees of freedom contributed by this joint.
    #[inline]
    pub const fn dof(&self) -> usize {
        match self {
            Self::Fixed => 0,
            Self::Revolute { .. } | Self::Continuous { .. } | Self::Prismatic { .. } => 1,
        }
    }

    /// Evaluates the local joint transform $T_{\text{joint}}(q_i)$.
    #[inline]
    pub fn local_transform(&self, q_val: f64) -> Isometry3<f64> {
        match self {
            Self::Fixed => Isometry3::identity(),
            Self::Revolute { axis } | Self::Continuous { axis } => {
                Isometry3::rotation(axis.into_inner() * q_val)
            }
            Self::Prismatic { axis } => Isometry3::translation(
                axis.x * q_val,
                axis.y * q_val,
                axis.z * q_val,
            ),
        }
    }

    /// Returns the local joint axis vector if actuated.
    #[inline]
    pub fn local_axis(&self) -> Option<UnitVector3<f64>> {
        match *self {
            Self::Fixed => None,
            Self::Revolute { axis } | Self::Continuous { axis } | Self::Prismatic { axis } => {
                Some(axis)
            }
        }
    }
}

/// Specification of a joint in a [`MultiRobotModel`].
#[derive(Debug, Clone)]
pub struct JointSpec {
    /// Human-readable joint name.
    pub name: String,
    /// Joint identifier.
    pub id: JointId,
    /// Owning robot identifier.
    pub robot: RobotId,
    /// Parent link (`None` if anchored directly to `FrameId::WORLD`).
    pub parent_link: Option<LinkId>,
    /// Child link driven by this joint.
    pub child_link: LinkId,
    /// Static origin transform from parent link (or world) to the joint frame.
    pub origin_transform: Isometry3<f64>,
    /// Joint articulation type.
    pub joint_type: JointType,
    /// Index into the global configuration vector `q` (`None` if `JointType::Fixed`).
    pub q_index: Option<usize>,
    /// Kinodynamic bounds (`None` if `JointType::Fixed`).
    pub limits: Option<KinodynamicLimits>,
}

/// Specification of a rigid link in a [`MultiRobotModel`].
#[derive(Debug, Clone)]
pub struct LinkSpec {
    /// Human-readable link name.
    pub name: String,
    /// Dense link identifier (`0..num_links`).
    pub id: LinkId,
    /// Owning robot identifier.
    pub robot: RobotId,
    /// Associated coordinate frame ID in the [`FrameGraph`].
    pub frame: FrameId,
    /// Driving joint that positions this link relative to its parent.
    pub parent_joint: JointId,
    /// Parent link (`None` if root link attached to world).
    pub parent_link: Option<LinkId>,
    /// Bounding radius of the link's geometry relative to its origin (used for Lipschitz CCD bounds).
    pub bounding_radius: f64,
}

/// How a coordinate frame in [`FrameGraph`] is attached to its parent.
#[derive(Debug, Clone, Copy)]
pub enum FrameAttachment {
    /// The canonical root world frame (`FrameId::WORLD`).
    WorldRoot,
    /// Attached directly to a robot link (`LinkId`).
    Link(LinkId),
    /// Fixed offset relative to a parent frame (e.g., tool center point / TCP or camera frame).
    FixedOffset {
        parent: FrameId,
        local_pose: Isometry3<f64>,
    },
}

/// A named coordinate frame in the topological [`FrameGraph`].
#[derive(Debug, Clone)]
pub struct FrameNode {
    /// Frame identifier.
    pub id: FrameId,
    /// Human-readable frame name.
    pub name: String,
    /// Attachment rule.
    pub attachment: FrameAttachment,
    /// Closest ancestor link (`None` if attached to world without an intervening link).
    pub associated_link: Option<LinkId>,
    /// Fixed offset from `associated_link` (`Isometry3::identity()` for link frames).
    pub link_to_frame: Isometry3<f64>,
}

/// Topologically sorted coordinate frame tree supporting single-pass forward kinematics.
#[derive(Debug, Clone)]
pub struct FrameGraph {
    nodes: Vec<FrameNode>,
}

impl Default for FrameGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameGraph {
    /// Creates a new frame graph initialized with `FrameId::WORLD` at index 0.
    pub fn new() -> Self {
        Self {
            nodes: vec![FrameNode {
                id: FrameId::WORLD,
                name: "world".to_string(),
                attachment: FrameAttachment::WorldRoot,
                associated_link: None,
                link_to_frame: Isometry3::identity(),
            }],
        }
    }

    /// Total number of frames in the graph (including `FrameId::WORLD`).
    #[inline]
    pub fn num_frames(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the slice of all frame nodes in topological order.
    #[inline]
    pub fn nodes(&self) -> &[FrameNode] {
        &self.nodes
    }

    /// Returns a specific frame node by ID.
    #[inline]
    pub fn node(&self, id: FrameId) -> Option<&FrameNode> {
        self.nodes.get(id.index())
    }

    /// Finds a frame ID by name.
    pub fn find_by_name(&self, name: &str) -> Option<FrameId> {
        self.nodes.iter().find(|n| n.name == name).map(|n| n.id)
    }
}

/// A named subset of joints, links, and end-effector frames (e.g., `"left_arm"`, `"right_arm"`, `"dual_arms"`).
#[derive(Debug, Clone)]
pub struct PlanningGroup {
    /// Group name.
    pub name: String,
    /// Actuated joints included in this group.
    pub joints: Vec<JointId>,
    /// Indices into the full system configuration vector `q`.
    pub q_indices: Vec<usize>,
    /// Links that move when this group's joints change.
    pub active_links: Vec<LinkId>,
    /// End-effector or tool frames controlled by this group.
    pub end_effectors: Vec<FrameId>,
}

impl PlanningGroup {
    /// Number of actuated degrees of freedom in this group.
    #[inline]
    pub fn dof(&self) -> usize {
        self.q_indices.len()
    }

    /// Extracts this group's joint values from a full-system configuration `full_q`.
    pub fn extract_group_q(&self, full_q: &[f64], out_group_q: &mut [f64]) {
        assert_eq!(out_group_q.len(), self.q_indices.len());
        for (dst, &q_idx) in out_group_q.iter_mut().zip(&self.q_indices) {
            *dst = full_q[q_idx];
        }
    }

    /// Writes this group's joint values into a full-system configuration `full_q`.
    pub fn inject_group_q(&self, group_q: &[f64], full_q: &mut [f64]) {
        assert_eq!(group_q.len(), self.q_indices.len());
        for (&src, &q_idx) in group_q.iter().zip(&self.q_indices) {
            full_q[q_idx] = src;
        }
    }

    /// Forms a composite multi-arm `PlanningGroup` by taking the ordered union of `self` and `other`.
    pub fn union(&self, name: impl Into<String>, other: &Self) -> Self {
        let mut joints = self.joints.clone();
        let mut q_indices = self.q_indices.clone();
        for (&j, &q_idx) in other.joints.iter().zip(&other.q_indices) {
            if !joints.contains(&j) {
                joints.push(j);
                q_indices.push(q_idx);
            }
        }
        let mut active_links = self.active_links.clone();
        for &link in &other.active_links {
            if !active_links.contains(&link) {
                active_links.push(link);
            }
        }
        let mut end_effectors = self.end_effectors.clone();
        for &ee in &other.end_effectors {
            if !end_effectors.contains(&ee) {
                end_effectors.push(ee);
            }
        }
        Self {
            name: name.into(),
            joints,
            q_indices,
            active_links,
            end_effectors,
        }
    }
}

pub use crate::collision::AttachedPayload;

/// Immutable topological and kinodynamic description of a single- or multi-arm robotic system.
///
/// `MultiRobotModel` is `Send + Sync` and separates immutable robot structure from
/// per-thread [`crate::kinematics::TransformBuffer`] and [`crate::collision::CollisionWorkspace`] scratchpads.
#[derive(Debug, Clone)]
pub struct MultiRobotModel {
    links: Vec<LinkSpec>,
    joints: Vec<JointSpec>,
    frame_graph: FrameGraph,
    /// Actuated joint IDs in order of `q_index` (`0..dof`).
    actuated_joints: Vec<JointId>,
    /// Per-DOF kinodynamic limits in order of `q_index` (`0..dof`).
    dof_limits: Vec<KinodynamicLimits>,
    /// Named planning groups (e.g. individual arms, dual-arm composites).
    groups: Vec<PlanningGroup>,
    /// Flattened `num_links * dof` boolean table: `is_ancestor_dof[link * dof + q_idx]`
    /// is `true` iff actuated DOF `q_idx` is on the kinematic path from World to `link`.
    ancestor_dof_mask: Vec<bool>,
    /// Flattened `num_links * dof` upper-bound lever arms $L_{i,k}$ (meters/rad or meters/m)
    /// for C-space Lipschitz continuous collision detection.
    lever_arm_bounds: Vec<f64>,
}

impl Default for MultiRobotModel {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiRobotModel {
    /// Creates an empty multi-robot model with `FrameId::WORLD`.
    pub fn new() -> Self {
        Self {
            links: Vec::new(),
            joints: Vec::new(),
            frame_graph: FrameGraph::new(),
            actuated_joints: Vec::new(),
            dof_limits: Vec::new(),
            groups: Vec::new(),
            ancestor_dof_mask: Vec::new(),
            lever_arm_bounds: Vec::new(),
        }
    }

    /// Total number of actuated degrees of freedom across all robots in the model.
    #[inline]
    pub fn dof(&self) -> usize {
        self.actuated_joints.len()
    }

    /// Total number of rigid links across all robots in the model.
    #[inline]
    pub fn num_links(&self) -> usize {
        self.links.len()
    }

    /// Total number of joints (including fixed joints).
    #[inline]
    pub fn num_joints(&self) -> usize {
        self.joints.len()
    }

    /// Returns all link specifications in topological order (`parent < child`).
    #[inline]
    pub fn links(&self) -> &[LinkSpec] {
        &self.links
    }

    /// Returns a specific link specification.
    #[inline]
    pub fn link(&self, id: LinkId) -> &LinkSpec {
        &self.links[id.index()]
    }

    /// Returns all joint specifications in topological order.
    #[inline]
    pub fn joints(&self) -> &[JointSpec] {
        &self.joints
    }

    /// Returns a specific joint specification.
    #[inline]
    pub fn joint(&self, id: JointId) -> &JointSpec {
        &self.joints[id.index()]
    }

    /// Returns the coordinate frame graph.
    #[inline]
    pub fn frame_graph(&self) -> &FrameGraph {
        &self.frame_graph
    }

    /// Returns the per-DOF kinodynamic limits slice (length = `self.dof()`).
    #[inline]
    pub fn dof_limits(&self) -> &[KinodynamicLimits] {
        &self.dof_limits
    }

    /// Returns the actuated joint ID for configuration index `q_idx`.
    #[inline]
    pub fn actuated_joint(&self, q_idx: usize) -> JointId {
        self.actuated_joints[q_idx]
    }

    /// Returns all registered planning groups.
    #[inline]
    pub fn groups(&self) -> &[PlanningGroup] {
        &self.groups
    }

    /// Finds a planning group by name.
    pub fn group_by_name(&self, name: &str) -> Option<&PlanningGroup> {
        self.groups.iter().find(|g| g.name == name)
    }

    /// Finds a link ID by name.
    pub fn link_by_name(&self, name: &str) -> Option<LinkId> {
        self.links.iter().find(|l| l.name == name).map(|l| l.id)
    }

    /// Returns `true` if actuated DOF `q_idx` is a kinematic ancestor of `link`.
    #[inline]
    pub fn is_dof_ancestor_of_link(&self, link: LinkId, q_idx: usize) -> bool {
        self.ancestor_dof_mask[link.index() * self.dof() + q_idx]
    }

    /// Returns the Lipschitz lever-arm bound $L_{i,k}$ of `link` with respect to actuated DOF `q_idx`.
    #[inline]
    pub fn lever_arm_bound(&self, link: LinkId, q_idx: usize) -> f64 {
        self.lever_arm_bounds[link.index() * self.dof() + q_idx]
    }

    /// Computes an upper bound on the maximum workspace displacement (meters) of any point
    /// on `link` when moving linearly in C-space from `q0` to `q1`.
    #[inline]
    pub fn max_link_displacement(&self, link: LinkId, q0: &[f64], q1: &[f64]) -> f64 {
        let dof = self.dof();
        let row = &self.lever_arm_bounds[link.index() * dof..(link.index() + 1) * dof];
        let mut disp = 0.0;
        for (k, &lever) in row.iter().enumerate() {
            if lever > 0.0 {
                disp += lever * self.dof_distance(k, q0[k], q1[k]);
            }
        }
        disp
    }

    /// Adds a new link driven by a joint attached to `parent_link` (`None` for world root).
    #[allow(clippy::too_many_arguments)]
    pub fn add_link_and_joint(
        &mut self,
        robot: RobotId,
        link_name: impl Into<String>,
        joint_name: impl Into<String>,
        parent_link: Option<LinkId>,
        origin_transform: Isometry3<f64>,
        joint_type: JointType,
        limits: Option<KinodynamicLimits>,
        bounding_radius: f64,
    ) -> Result<(LinkId, JointId, FrameId), RobotError> {
        if let Some(parent) = parent_link {
            if parent.index() >= self.links.len() {
                return Err(RobotError::InvalidTopology(format!(
                    "parent link index {} out of bounds (num_links = {})",
                    parent.0,
                    self.links.len()
                )));
            }
        }

        let link_id = LinkId(self.links.len() as u16);
        let joint_id = JointId(self.joints.len() as u16);
        let frame_id = FrameId(self.frame_graph.nodes.len() as u16);

        let q_index = if joint_type.dof() > 0 {
            let idx = self.actuated_joints.len();
            self.actuated_joints.push(joint_id);
            let lim = limits.unwrap_or_else(|| {
                KinodynamicLimits::from_position(match joint_type {
                    JointType::Continuous { .. } => JointLimit::unbounded(),
                    _ => JointLimit::new(-PI, PI),
                })
            });
            self.dof_limits.push(lim);
            Some(idx)
        } else {
            None
        };

        let link_name_str = link_name.into();
        self.frame_graph.nodes.push(FrameNode {
            id: frame_id,
            name: link_name_str.clone(),
            attachment: FrameAttachment::Link(link_id),
            associated_link: Some(link_id),
            link_to_frame: Isometry3::identity(),
        });

        self.joints.push(JointSpec {
            name: joint_name.into(),
            id: joint_id,
            robot,
            parent_link,
            child_link: link_id,
            origin_transform,
            joint_type,
            q_index,
            limits,
        });

        self.links.push(LinkSpec {
            name: link_name_str,
            id: link_id,
            robot,
            frame: frame_id,
            parent_joint: joint_id,
            parent_link,
            bounding_radius: bounding_radius.max(0.01),
        });

        self.rebuild_kinematic_tables();
        Ok((link_id, joint_id, frame_id))
    }

    /// Attaches a fixed coordinate frame (e.g., end-effector TCP, sensor frame) to an existing frame.
    pub fn add_fixed_frame(
        &mut self,
        name: impl Into<String>,
        parent: FrameId,
        local_pose: Isometry3<f64>,
    ) -> Result<FrameId, RobotError> {
        let Some(parent_node) = self.frame_graph.node(parent).cloned() else {
            return Err(RobotError::InvalidTopology(format!(
                "parent frame {} does not exist",
                parent.0
            )));
        };
        let frame_id = FrameId(self.frame_graph.nodes.len() as u16);
        self.frame_graph.nodes.push(FrameNode {
            id: frame_id,
            name: name.into(),
            attachment: FrameAttachment::FixedOffset { parent, local_pose },
            associated_link: parent_node.associated_link,
            link_to_frame: parent_node.link_to_frame * local_pose,
        });
        Ok(frame_id)
    }

    /// Registers a named [`PlanningGroup`] (e.g. a single arm or dual-arm subgroup).
    pub fn add_planning_group(&mut self, group: PlanningGroup) -> usize {
        let idx = self.groups.len();
        self.groups.push(group);
        idx
    }

    /// Convenience helper to add an $N$-DOF serial robotic arm to the model and register its [`PlanningGroup`].
    ///
    /// Each entry in `segments` specifies `(joint_axis, link_translation, joint_limit, link_radius)`.
    pub fn add_serial_arm(
        &mut self,
        robot: RobotId,
        arm_name: impl Into<String>,
        base_pose: Isometry3<f64>,
        segments: &[(UnitVector3<f64>, Vector3<f64>, JointLimit, f64)],
        tcp_offset: Isometry3<f64>,
    ) -> Result<PlanningGroup, RobotError> {
        let arm_name = arm_name.into();
        let mut parent_link = None;
        let mut joints = Vec::with_capacity(segments.len());
        let mut q_indices = Vec::with_capacity(segments.len());
        let mut active_links = Vec::with_capacity(segments.len());

        for (i, &(axis, translation, limit, radius)) in segments.iter().enumerate() {
            let origin = if i == 0 {
                base_pose
            } else {
                Isometry3::from_parts(translation.into(), nalgebra::UnitQuaternion::identity())
            };
            let (link_id, joint_id, _) = self.add_link_and_joint(
                robot,
                format!("{arm_name}_link_{i}"),
                format!("{arm_name}_joint_{i}"),
                parent_link,
                origin,
                JointType::Revolute { axis },
                Some(KinodynamicLimits::from_position(limit)),
                radius,
            )?;
            parent_link = Some(link_id);
            joints.push(joint_id);
            q_indices.push(self.joint(joint_id).q_index.unwrap());
            active_links.push(link_id);
        }

        let ee_frame = if let Some(last_link) = parent_link {
            let last_frame = self.link(last_link).frame;
            self.add_fixed_frame(format!("{arm_name}_tcp"), last_frame, tcp_offset)?
        } else {
            FrameId::WORLD
        };

        let group = PlanningGroup {
            name: arm_name,
            joints,
            q_indices,
            active_links,
            end_effectors: vec![ee_frame],
        };
        self.add_planning_group(group.clone());
        Ok(group)
    }

    /// Recomputes the ancestor mask and conservative Lipschitz lever-arm matrix $L_{i,k}$.
    fn rebuild_kinematic_tables(&mut self) {
        let num_links = self.links.len();
        let dof = self.dof();
        self.ancestor_dof_mask = vec![false; num_links * dof];
        self.lever_arm_bounds = vec![0.0; num_links * dof];

        // Accumulate maximum link-to-child offset distances along the kinematic tree
        for link_idx in 0..num_links {
            // Walk up from `link_idx` to the root, summing conservative link lengths
            let mut accumulated_length = self.links[link_idx].bounding_radius;
            let mut cursor = Some(LinkId(link_idx as u16));

            while let Some(curr_link_id) = cursor {
                let curr_link = &self.links[curr_link_id.index()];
                let joint = &self.joints[curr_link.parent_joint.index()];

                if let Some(q_idx) = joint.q_index {
                    self.ancestor_dof_mask[link_idx * dof + q_idx] = true;
                    let lever = match joint.joint_type {
                        JointType::Fixed => 0.0,
                        JointType::Prismatic { .. } => 1.0,
                        JointType::Revolute { .. } | JointType::Continuous { .. } => {
                            accumulated_length.max(1e-3)
                        }
                    };
                    self.lever_arm_bounds[link_idx * dof + q_idx] = lever;
                }

                accumulated_length += joint.origin_transform.translation.vector.norm()
                    + curr_link.bounding_radius;
                cursor = curr_link.parent_link;
            }
        }
    }

    /// Returns the neutral (midpoint) configuration of the multi-robot system.
    pub fn neutral_configuration(&self) -> Configuration {
        Configuration::new(
            self.dof_limits
                .iter()
                .map(|lim| lim.position.midpoint())
                .collect(),
        )
    }

    /// Validates that `q` has length `self.dof()` and respects all joint position limits.
    pub fn validate_configuration(&self, q: &[f64]) -> Result<(), RobotError> {
        if q.len() != self.dof() {
            return Err(RobotError::WrongDof {
                expected: self.dof(),
                actual: q.len(),
            });
        }
        for (joint_idx, (&val, lim)) in q.iter().zip(&self.dof_limits).enumerate() {
            if !lim.position.contains(val) {
                return Err(RobotError::JointOutOfBounds {
                    joint: joint_idx,
                    value: val,
                    min: lim.position.min,
                    max: lim.position.max,
                });
            }
        }
        Ok(())
    }

    /// Geodesic distance for a single actuated DOF `q_idx` (handles $SO(2)$ wrap-around for `Continuous` joints).
    #[inline]
    pub fn dof_distance(&self, q_idx: usize, a: f64, b: f64) -> f64 {
        let joint = &self.joints[self.actuated_joints[q_idx].index()];
        match joint.joint_type {
            JointType::Continuous { .. } => {
                let diff = (b - a).rem_euclid(TAU);
                if diff > PI {
                    TAU - diff
                } else {
                    diff
                }
            }
            _ => (b - a).abs(),
        }
    }

    /// Manifold-aware L2 distance in C-space between `q0` and `q1`.
    pub fn distance(&self, q0: &[f64], q1: &[f64]) -> f64 {
        debug_assert_eq!(q0.len(), self.dof());
        debug_assert_eq!(q1.len(), self.dof());
        let mut sum_sq = 0.0;
        for k in 0..self.dof() {
            let d = self.dof_distance(k, q0[k], q1[k]);
            sum_sq += d * d;
        }
        sum_sq.sqrt()
    }

    /// Geodesic interpolation in C-space between `q0` and `q1` at `alpha` $\in [0, 1]$ into `out_q`.
    pub fn interpolate(&self, q0: &[f64], q1: &[f64], alpha: f64, out_q: &mut [f64]) {
        debug_assert_eq!(q0.len(), self.dof());
        debug_assert_eq!(q1.len(), self.dof());
        debug_assert_eq!(out_q.len(), self.dof());
        for k in 0..self.dof() {
            let joint = &self.joints[self.actuated_joints[k].index()];
            out_q[k] = match joint.joint_type {
                JointType::Continuous { .. } => {
                    let mut diff = (q1[k] - q0[k]).rem_euclid(TAU);
                    if diff > PI {
                        diff -= TAU;
                    }
                    (q0[k] + alpha * diff + PI).rem_euclid(TAU) - PI
                }
                _ => q0[k] + alpha * (q1[k] - q0[k]),
            };
        }
    }
}

/// The single- or multi-arm robot facade used by kinematics, collision checking, and planning.
///
/// Can be constructed directly from joint limits (`Robot::new`) or backed by a full
/// topological [`MultiRobotModel`] (`Robot::from_model`).
#[derive(Debug, Clone)]
pub struct Robot {
    /// Per-joint limits defining the valid portion of C-space.
    pub joint_limits: Vec<JointLimit>,
    /// The robot's current joint-space configuration.
    pub configuration: Configuration,
    /// Optional articulated multi-robot kinematic and geometric model.
    pub model: Option<Arc<MultiRobotModel>>,
}

impl Robot {
    /// Creates a robot at the midpoint of every joint limit.
    pub fn new(joint_limits: Vec<JointLimit>) -> Result<Self, RobotError> {
        let configuration =
            Configuration::new(joint_limits.iter().map(JointLimit::midpoint).collect());
        Ok(Self {
            joint_limits,
            configuration,
            model: None,
        })
    }

    /// Creates a `Robot` backed by an articulated [`MultiRobotModel`].
    pub fn from_model(model: Arc<MultiRobotModel>) -> Self {
        let joint_limits: Vec<JointLimit> =
            model.dof_limits().iter().map(|l| l.position).collect();
        let configuration = model.neutral_configuration();
        Self {
            joint_limits,
            configuration,
            model: Some(model),
        }
    }

    /// Returns the number of degrees of freedom of this robot.
    #[inline]
    pub fn dof(&self) -> usize {
        self.joint_limits.len()
    }

    /// Updates the current configuration after checking DOF and joint limits.
    pub fn set_configuration(&mut self, configuration: Configuration) -> Result<(), RobotError> {
        self.validate_configuration(&configuration)?;
        self.configuration = configuration;
        Ok(())
    }

    /// Checks whether a configuration is valid for this robot.
    pub fn validate_configuration(&self, configuration: &Configuration) -> Result<(), RobotError> {
        self.validate_slice(configuration.as_slice())
    }

    /// Checks whether a borrowed joint slice is valid for this robot.
    pub fn validate_slice(&self, q: &[f64]) -> Result<(), RobotError> {
        if q.len() != self.joint_limits.len() {
            return Err(RobotError::WrongDof {
                expected: self.joint_limits.len(),
                actual: q.len(),
            });
        }
        for (joint, (&value, limit)) in q.iter().zip(&self.joint_limits).enumerate() {
            if !limit.contains(value) {
                return Err(RobotError::JointOutOfBounds {
                    joint,
                    value,
                    min: limit.min,
                    max: limit.max,
                });
            }
        }
        Ok(())
    }
}
