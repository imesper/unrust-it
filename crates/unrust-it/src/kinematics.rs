//! Forward kinematics, analytical geometric Jacobians, distance Jacobian projections,
//! and multi-target damped least-squares inverse kinematics.

use crate::robot::{
    Configuration, FrameAttachment, FrameId, JointType, LinkId, MultiRobotModel, Robot, RobotError,
    TaskPose, TaskSpacePoint,
};
use nalgebra::{DMatrix, DMatrixViewMut, DVector, Isometry3, Point3, Vector3};
use thiserror::Error;

/// Errors a kinematics or inverse-kinematics operation may report.
#[derive(Debug, Error, PartialEq)]
pub enum KinematicsError {
    /// Invalid configuration dimension or out-of-bounds joint values.
    #[error(transparent)]
    InvalidConfiguration(#[from] RobotError),
    /// Referenced frame or link ID does not exist in the model.
    #[error("invalid frame or link id")]
    InvalidFrameOrLink,
    /// Jacobian output matrix dimensions do not match `(6, dof)`.
    #[error("jacobian matrix has shape ({actual_rows}, {actual_cols}), expected (6, {expected_cols})")]
    InvalidJacobianShape {
        /// Provided row count.
        actual_rows: usize,
        /// Provided column count.
        actual_cols: usize,
        /// Expected column count (`model.dof()`).
        expected_cols: usize,
    },
    /// No valid configuration converged to the requested pose(s) within tolerance.
    #[error("target pose is unreachable")]
    TargetUnreachable,
    /// The requested operation is not implemented by a model.
    #[error("kinematics operation is not implemented")]
    NotImplemented,
}

/// Pre-allocated, zero-allocation workspace holding evaluated world poses and joint screw axes.
#[derive(Debug, Clone)]
pub struct TransformBuffer {
    /// World-from-link poses `T_world_link[link_id]`.
    link_poses: Vec<Isometry3<f64>>,
    /// World-from-frame poses `T_world_frame[frame_id]`.
    frame_poses: Vec<Isometry3<f64>>,
    /// World-space joint origins `o_joint[joint_id]`.
    joint_origins_world: Vec<Point3<f64>>,
    /// World-space joint axes `axis_joint[joint_id]`.
    joint_axes_world: Vec<Vector3<f64>>,
}

impl TransformBuffer {
    /// Allocates a workspace sized for `model`.
    pub fn from_model(model: &MultiRobotModel) -> Self {
        Self {
            link_poses: vec![Isometry3::identity(); model.num_links()],
            frame_poses: vec![Isometry3::identity(); model.frame_graph().num_frames()],
            joint_origins_world: vec![Point3::origin(); model.num_joints()],
            joint_axes_world: vec![Vector3::zeros(); model.num_joints()],
        }
    }

    /// Ensures internal buffers are sized for `model` without reallocating if capacity suffices.
    pub fn resize_for(&mut self, model: &MultiRobotModel) {
        self.link_poses
            .resize(model.num_links(), Isometry3::identity());
        self.frame_poses
            .resize(model.frame_graph().num_frames(), Isometry3::identity());
        self.joint_origins_world
            .resize(model.num_joints(), Point3::origin());
        self.joint_axes_world
            .resize(model.num_joints(), Vector3::zeros());
    }

    /// $O(1)$ lookup of a link's world pose $T_{\text{world}}^{\text{link}}$.
    #[inline]
    pub fn link_pose(&self, link: LinkId) -> &Isometry3<f64> {
        &self.link_poses[link.index()]
    }

    /// Returns the slice of all link world poses.
    #[inline]
    pub fn link_poses(&self) -> &[Isometry3<f64>] {
        &self.link_poses
    }

    /// $O(1)$ lookup of a coordinate frame's world pose $T_{\text{world}}^{\text{frame}}$.
    #[inline]
    pub fn frame_pose(&self, frame: FrameId) -> &Isometry3<f64> {
        &self.frame_poses[frame.index()]
    }

    /// $O(1)$ relative transform $T_{\text{target}}^{\text{source}} = (T_{\text{world}}^{\text{target}})^{-1} T_{\text{world}}^{\text{source}}$.
    #[inline]
    pub fn relative_pose(&self, target: FrameId, source: FrameId) -> Isometry3<f64> {
        self.frame_pose(target).inverse() * self.frame_pose(source)
    }
}

impl MultiRobotModel {
    /// Allocates a new [`TransformBuffer`] sized for this model.
    pub fn create_transform_buffer(&self) -> TransformBuffer {
        TransformBuffer::from_model(self)
    }

    /// Evaluates forward kinematics for all links, joints, and frames in a single linear pass.
    ///
    /// Performs **zero heap allocations** when `buf` has been initialized for this model.
    pub fn forward_kinematics(
        &self,
        q: &[f64],
        buf: &mut TransformBuffer,
    ) -> Result<(), KinematicsError> {
        if q.len() != self.dof() {
            return Err(KinematicsError::InvalidConfiguration(
                RobotError::WrongDof {
                    expected: self.dof(),
                    actual: q.len(),
                },
            ));
        }
        buf.resize_for(self);

        // 1. Single linear pass over topologically ordered joints/links
        for joint in self.joints() {
            let parent_pose = match joint.parent_link {
                Some(parent_id) => buf.link_poses[parent_id.index()],
                None => Isometry3::identity(),
            };
            let joint_base_world = parent_pose * joint.origin_transform;
            buf.joint_origins_world[joint.id.index()] =
                Point3::from(joint_base_world.translation.vector);

            if let Some(local_axis) = joint.joint_type.local_axis() {
                buf.joint_axes_world[joint.id.index()] =
                    joint_base_world.rotation * local_axis.into_inner();
            } else {
                buf.joint_axes_world[joint.id.index()] = Vector3::zeros();
            }

            let q_val = joint.q_index.map(|idx| q[idx]).unwrap_or(0.0);
            let child_pose = joint_base_world * joint.joint_type.local_transform(q_val);
            buf.link_poses[joint.child_link.index()] = child_pose;
        }

        // 2. Single linear pass over topologically ordered coordinate frames
        for node in self.frame_graph().nodes() {
            buf.frame_poses[node.id.index()] = match node.attachment {
                FrameAttachment::WorldRoot => Isometry3::identity(),
                FrameAttachment::Link(link_id) => buf.link_poses[link_id.index()],
                FrameAttachment::FixedOffset { parent, local_pose } => {
                    buf.frame_poses[parent.index()] * local_pose
                }
            };
        }

        Ok(())
    }

    /// Computes the $6 \times N_{\text{dof}}$ world-aligned geometric Jacobian $J(q)$ of `target_frame`
    /// after `forward_kinematics` has been evaluated into `buf`.
    ///
    /// - Rows `0..3`: linear velocity Jacobian $J_v(q)$ in world coordinates ($\dot{p} = J_v \dot{q}$).
    /// - Rows `3..6`: angular velocity Jacobian $J_\omega(q)$ in world coordinates ($\omega = J_\omega \dot{q}$).
    pub fn geometric_jacobian(
        &self,
        buf: &TransformBuffer,
        target_frame: FrameId,
        mut out_jacobian: DMatrixViewMut<'_, f64>,
    ) -> Result<(), KinematicsError> {
        let dof = self.dof();
        if out_jacobian.nrows() != 6 || out_jacobian.ncols() != dof {
            return Err(KinematicsError::InvalidJacobianShape {
                actual_rows: out_jacobian.nrows(),
                actual_cols: out_jacobian.ncols(),
                expected_cols: dof,
            });
        }
        let Some(frame_node) = self.frame_graph().node(target_frame) else {
            return Err(KinematicsError::InvalidFrameOrLink);
        };
        out_jacobian.fill(0.0);

        let Some(assoc_link) = frame_node.associated_link else {
            // Frame is fixed to the world root; Jacobian is identically zero.
            return Ok(());
        };

        let p_target = Point3::from(buf.frame_pose(target_frame).translation.vector);
        self.compute_point_and_angular_jacobian_on_link(
            buf,
            assoc_link,
            &p_target,
            &mut out_jacobian,
        );
        Ok(())
    }

    /// Populates the $6 \times N_{\text{dof}}$ world-aligned geometric Jacobian for an arbitrary
    /// world point `point_world` rigidly attached to `link`.
    pub fn compute_point_and_angular_jacobian_on_link(
        &self,
        buf: &TransformBuffer,
        link: LinkId,
        point_world: &Point3<f64>,
        out_jacobian: &mut DMatrixViewMut<'_, f64>,
    ) {
        let dof = self.dof();
        for q_idx in 0..dof {
            if !self.is_dof_ancestor_of_link(link, q_idx) {
                continue;
            }
            let joint_id = self.actuated_joint(q_idx);
            let joint = self.joint(joint_id);
            let axis_w = &buf.joint_axes_world[joint_id.index()];

            match joint.joint_type {
                JointType::Fixed => {}
                JointType::Revolute { .. } | JointType::Continuous { .. } => {
                    let origin_w = &buf.joint_origins_world[joint_id.index()];
                    let r = point_world - origin_w;
                    let v = axis_w.cross(&r);
                    out_jacobian[(0, q_idx)] = v.x;
                    out_jacobian[(1, q_idx)] = v.y;
                    out_jacobian[(2, q_idx)] = v.z;
                    out_jacobian[(3, q_idx)] = axis_w.x;
                    out_jacobian[(4, q_idx)] = axis_w.y;
                    out_jacobian[(5, q_idx)] = axis_w.z;
                }
                JointType::Prismatic { .. } => {
                    out_jacobian[(0, q_idx)] = axis_w.x;
                    out_jacobian[(1, q_idx)] = axis_w.y;
                    out_jacobian[(2, q_idx)] = axis_w.z;
                }
            }
        }
    }

    /// Allocation-free projection of a link point's linear Jacobian onto a world-space direction:
    /// accumulates `scale * (normal_world^T * J_p(q))` into `out_grad_q` ($1 \times N_{\text{dof}}$).
    ///
    /// Uses the cross-product screw identity:
    /// $\hat{\mathbf{n}} \cdot (\boldsymbol{\omega}_k \times (\mathbf{p} - \mathbf{o}_k)) = ((\mathbf{p} - \mathbf{o}_k) \times \hat{\mathbf{n}}) \cdot \boldsymbol{\omega}_k$.
    #[inline]
    pub fn accumulate_point_jacobian_projection(
        &self,
        buf: &TransformBuffer,
        link: LinkId,
        point_world: &Point3<f64>,
        normal_world: &Vector3<f64>,
        scale: f64,
        out_grad_q: &mut [f64],
    ) {
        let dof = self.dof();
        debug_assert_eq!(out_grad_q.len(), dof);
        let scaled_n = normal_world * scale;

        for (q_idx, grad_slot) in out_grad_q.iter_mut().enumerate().take(dof) {
            if !self.is_dof_ancestor_of_link(link, q_idx) {
                continue;
            }
            let joint_id = self.actuated_joint(q_idx);
            let joint = self.joint(joint_id);
            let axis_w = &buf.joint_axes_world[joint_id.index()];

            match joint.joint_type {
                JointType::Fixed => {}
                JointType::Revolute { .. } | JointType::Continuous { .. } => {
                    let origin_w = &buf.joint_origins_world[joint_id.index()];
                    let r = point_world - origin_w;
                    *grad_slot += r.cross(&scaled_n).dot(axis_w);
                }
                JointType::Prismatic { .. } => {
                    *grad_slot += scaled_n.dot(axis_w);
                }
            }
        }
    }

    /// Solves multi-target (single-arm or simultaneous multi-arm) 6-DOF Inverse Kinematics
    /// using Damped Least-Squares (Levenberg-Marquardt) starting from `q_seed`.
    pub fn solve_ik_dls(
        &self,
        targets: &[TaskPose],
        q_seed: &[f64],
        max_iterations: usize,
        tolerance: f64,
        damping: f64,
        buf: &mut TransformBuffer,
    ) -> Result<Configuration, KinematicsError> {
        let dof = self.dof();
        if q_seed.len() != dof {
            return Err(KinematicsError::InvalidConfiguration(
                RobotError::WrongDof {
                    expected: dof,
                    actual: q_seed.len(),
                },
            ));
        }
        if targets.is_empty() {
            return Ok(Configuration::from_slice(q_seed));
        }

        let m = 6 * targets.len();
        let mut q = DVector::from_column_slice(q_seed);
        // Clamp initial seed into bounds
        for (k, lim) in self.dof_limits().iter().enumerate() {
            q[k] = lim.position.clamp(q[k]);
        }

        let mut err = DVector::<f64>::zeros(m);
        let mut j_stacked = DMatrix::<f64>::zeros(m, dof);

        for _ in 0..max_iterations {
            self.forward_kinematics(q.as_slice(), buf)?;

            for (i, target) in targets.iter().enumerate() {
                if target.frame.index() >= self.frame_graph().num_frames() {
                    return Err(KinematicsError::InvalidFrameOrLink);
                }
                let current_pose = buf.frame_pose(target.frame);
                let pos_err = target.pose.translation.vector - current_pose.translation.vector;
                let rot_err = (target.pose.rotation * current_pose.rotation.inverse()).scaled_axis();

                let row_offset = 6 * i;
                err[row_offset] = pos_err.x;
                err[row_offset + 1] = pos_err.y;
                err[row_offset + 2] = pos_err.z;
                err[row_offset + 3] = rot_err.x;
                err[row_offset + 4] = rot_err.y;
                err[row_offset + 5] = rot_err.z;

                let sub_j = j_stacked.view_mut((row_offset, 0), (6, dof));
                self.geometric_jacobian(buf, target.frame, sub_j)?;
            }

            if err.norm() <= tolerance {
                return Ok(Configuration::new(q.as_slice().to_vec()));
            }

            // Solve (J * J^T + lambda^2 * I) * y = err, dq = J^T * y
            let mut gram = &j_stacked * j_stacked.transpose();
            let lambda_sq = damping * damping;
            for r in 0..m {
                gram[(r, r)] += lambda_sq;
            }
            let Some(chol) = gram.cholesky() else {
                return Err(KinematicsError::TargetUnreachable);
            };
            let y = chol.solve(&err);
            let mut dq = j_stacked.transpose() * y;

            // Step size trust-region limiting to prevent wild jumps near singularities
            let max_step = 0.35;
            let dq_inf = dq.amax();
            if dq_inf > max_step {
                dq *= max_step / dq_inf;
            }

            q += dq;
            for (k, lim) in self.dof_limits().iter().enumerate() {
                let joint = self.joint(self.actuated_joint(k));
                if matches!(joint.joint_type, JointType::Continuous { .. }) {
                    q[k] = (q[k] + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
                        - std::f64::consts::PI;
                } else {
                    q[k] = lim.position.clamp(q[k]);
                }
            }
        }

        self.forward_kinematics(q.as_slice(), buf)?;
        for (i, target) in targets.iter().enumerate() {
            let current_pose = buf.frame_pose(target.frame);
            let pos_err = target.pose.translation.vector - current_pose.translation.vector;
            let rot_err = (target.pose.rotation * current_pose.rotation.inverse()).scaled_axis();
            let row_offset = 6 * i;
            err[row_offset] = pos_err.x;
            err[row_offset + 1] = pos_err.y;
            err[row_offset + 2] = pos_err.z;
            err[row_offset + 3] = rot_err.x;
            err[row_offset + 4] = rot_err.y;
            err[row_offset + 5] = rot_err.z;
        }

        if err.norm() <= tolerance {
            Ok(Configuration::new(q.as_slice().to_vec()))
        } else {
            Err(KinematicsError::TargetUnreachable)
        }
    }
}

/// Computes link or end-effector poses from a joint-space configuration.
pub trait ForwardKinematics {
    /// Computes the primary end-effector pose in the robot's world frame.
    fn forward(&self, configuration: &Configuration) -> Result<Isometry3<f64>, KinematicsError>;
}

/// Solves for a joint-space configuration that reaches a task-space target.
pub trait InverseKinematics {
    /// Returns a valid configuration for `target`, if one exists.
    fn inverse(&self, target: &TaskSpacePoint) -> Result<Configuration, KinematicsError>;
}

impl ForwardKinematics for Robot {
    fn forward(&self, configuration: &Configuration) -> Result<Isometry3<f64>, KinematicsError> {
        self.validate_configuration(configuration)?;
        let Some(model) = &self.model else {
            return Err(KinematicsError::NotImplemented);
        };
        let mut buf = model.create_transform_buffer();
        model.forward_kinematics(configuration.as_slice(), &mut buf)?;
        let target_frame = model
            .groups()
            .first()
            .and_then(|g| g.end_effectors.first().copied())
            .unwrap_or_else(|| FrameId((model.frame_graph().num_frames() - 1) as u16));
        Ok(*buf.frame_pose(target_frame))
    }
}
