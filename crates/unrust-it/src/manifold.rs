//! Holonomic constraint manifolds for closed-chain bimanual manipulation
//! and task-space orientation/pose constraints.

use crate::kinematics::{KinematicsError, TransformBuffer};
use crate::robot::{FrameId, JointType, MultiRobotModel, RobotError};
use nalgebra::{DMatrix, DVector, Isometry3, Point3};

/// A differentiable holonomic constraint manifold $\mathcal{M} = \{ q \in \mathcal{C} \mid F(q) = \mathbf{0} \}$.
pub trait ConstraintManifold: Send + Sync {
    /// Dimension $m$ of the constraint residual vector $F(q) \in \mathbb{R}^m$.
    fn constraint_dim(&self) -> usize;

    /// Evaluates the constraint residual $F(q) \in \mathbb{R}^m$ given an updated [`TransformBuffer`].
    fn evaluate(
        &self,
        model: &MultiRobotModel,
        buf: &TransformBuffer,
        out_residual: &mut DVector<f64>,
    ) -> Result<(), KinematicsError>;

    /// Evaluates the $m \times N_{\text{dof}}$ constraint Jacobian $J_F(q) = \frac{\partial F}{\partial q}$.
    fn jacobian(
        &self,
        model: &MultiRobotModel,
        buf: &TransformBuffer,
        out_jacobian: &mut DMatrix<f64>,
    ) -> Result<(), KinematicsError>;

    /// Projects an ambient configuration `q_inout` onto the constraint manifold $\mathcal{M}$
    /// using Damped Least-Squares Newton-Raphson retraction.
    fn project(
        &self,
        model: &MultiRobotModel,
        q_inout: &mut [f64],
        buf: &mut TransformBuffer,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<(), KinematicsError> {
        let dof = model.dof();
        if q_inout.len() != dof {
            return Err(KinematicsError::InvalidConfiguration(
                RobotError::WrongDof {
                    expected: dof,
                    actual: q_inout.len(),
                },
            ));
        }
        let m = self.constraint_dim();
        if m == 0 {
            return Ok(());
        }

        let mut residual = DVector::<f64>::zeros(m);
        let mut j_mat = DMatrix::<f64>::zeros(m, dof);
        let damping_sq = 1e-4;

        for _ in 0..max_iterations {
            model.forward_kinematics(q_inout, buf)?;
            self.evaluate(model, buf, &mut residual)?;
            if residual.norm() <= tolerance {
                return Ok(());
            }

            self.jacobian(model, buf, &mut j_mat)?;

            // Solve (J_F * J_F^T + lambda^2 * I) * y = -F(q), dq = J_F^T * y
            let mut gram = &j_mat * j_mat.transpose();
            for r in 0..m {
                gram[(r, r)] += damping_sq;
            }
            let Some(chol) = gram.cholesky() else {
                return Err(KinematicsError::TargetUnreachable);
            };
            let y = chol.solve(&(-&residual));
            let mut dq = j_mat.transpose() * y;

            let max_step = 0.35;
            let dq_inf = dq.amax();
            if dq_inf > max_step {
                dq *= max_step / dq_inf;
            }

            for (k, lim) in model.dof_limits().iter().enumerate() {
                let next_val = q_inout[k] + dq[k];
                let joint = model.joint(model.actuated_joint(k));
                q_inout[k] = if matches!(joint.joint_type, JointType::Continuous { .. }) {
                    (next_val + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
                        - std::f64::consts::PI
                } else {
                    lim.position.clamp(next_val)
                };
            }
        }

        model.forward_kinematics(q_inout, buf)?;
        self.evaluate(model, buf, &mut residual)?;
        if residual.norm() <= tolerance {
            Ok(())
        } else {
            Err(KinematicsError::TargetUnreachable)
        }
    }
}

/// Closed-chain relative pose constraint between two frames `frame_a` and `frame_b`
/// (e.g., two robotic arms grasping a shared rigid object or performing a synchronized handover).
///
/// Enforces $(T_{\text{world}}^A(q))^{-1} T_{\text{world}}^B(q) = T_{\text{rel}}^*$ along the active twist dimensions.
#[derive(Debug, Clone)]
pub struct ClosedChainGraspConstraint {
    /// First end-effector or grasp frame $A$.
    pub frame_a: FrameId,
    /// Second end-effector or grasp frame $B$.
    pub frame_b: FrameId,
    /// Desired relative transform ${}^A T_B^* \in SE(3)$ from frame $A$ to frame $B$.
    pub desired_relative_pose: Isometry3<f64>,
    /// Active constraint dimensions `[vx, vy, vz, wx, wy, wz]` in world-aligned coordinates.
    pub active_dims: [bool; 6],
}

impl ClosedChainGraspConstraint {
    /// Creates a full 6-DOF rigid closed-chain grasp constraint between `frame_a` and `frame_b`.
    pub fn new_rigid(
        frame_a: FrameId,
        frame_b: FrameId,
        desired_relative_pose: Isometry3<f64>,
    ) -> Self {
        Self {
            frame_a,
            frame_b,
            desired_relative_pose,
            active_dims: [true; 6],
        }
    }

    /// Creates a closed-chain constraint with custom active twist dimensions (e.g., `[true, true, true, false, false, false]` for a ball-joint grasp).
    pub fn with_active_dims(
        frame_a: FrameId,
        frame_b: FrameId,
        desired_relative_pose: Isometry3<f64>,
        active_dims: [bool; 6],
    ) -> Self {
        Self {
            frame_a,
            frame_b,
            desired_relative_pose,
            active_dims,
        }
    }
}

impl ConstraintManifold for ClosedChainGraspConstraint {
    fn constraint_dim(&self) -> usize {
        self.active_dims.iter().filter(|&&b| b).count()
    }

    fn evaluate(
        &self,
        model: &MultiRobotModel,
        buf: &TransformBuffer,
        out_residual: &mut DVector<f64>,
    ) -> Result<(), KinematicsError> {
        let num_frames = model.frame_graph().num_frames();
        if self.frame_a.index() >= num_frames || self.frame_b.index() >= num_frames {
            return Err(KinematicsError::InvalidFrameOrLink);
        }

        let t_w_a = buf.frame_pose(self.frame_a);
        let t_w_b = buf.frame_pose(self.frame_b);
        let t_w_b_desired = t_w_a * self.desired_relative_pose;

        // World-aligned position residual: p_B(q) - p_B_desired(q)
        let pos_res = t_w_b.translation.vector - t_w_b_desired.translation.vector;
        // World-aligned orientation residual: Log(R_B(q) * R_B_desired(q)^T)
        let rot_res = (t_w_b.rotation * t_w_b_desired.rotation.inverse()).scaled_axis();
        let full_res = [
            pos_res.x, pos_res.y, pos_res.z, rot_res.x, rot_res.y, rot_res.z,
        ];

        let mut row = 0;
        for (dim, &active) in self.active_dims.iter().enumerate() {
            if active {
                out_residual[row] = full_res[dim];
                row += 1;
            }
        }
        Ok(())
    }

    fn jacobian(
        &self,
        model: &MultiRobotModel,
        buf: &TransformBuffer,
        out_jacobian: &mut DMatrix<f64>,
    ) -> Result<(), KinematicsError> {
        let dof = model.dof();
        let Some(node_a) = model.frame_graph().node(self.frame_a) else {
            return Err(KinematicsError::InvalidFrameOrLink);
        };
        let Some(node_b) = model.frame_graph().node(self.frame_b) else {
            return Err(KinematicsError::InvalidFrameOrLink);
        };

        let p_b = Point3::from(buf.frame_pose(self.frame_b).translation.vector);
        let t_w_a = buf.frame_pose(self.frame_a);
        let t_w_b_desired = t_w_a * self.desired_relative_pose;
        let p_b_des = Point3::from(t_w_b_desired.translation.vector);

        out_jacobian.fill(0.0);
        for q_idx in 0..dof {
            let col_b = match node_b.associated_link {
                Some(link_b) => model.point_screw_column(buf, link_b, &p_b, q_idx),
                None => [0.0; 6],
            };
            let col_a = match node_a.associated_link {
                Some(link_a) => model.point_screw_column(buf, link_a, &p_b_des, q_idx),
                None => [0.0; 6],
            };

            let mut row = 0;
            for (dim, &active) in self.active_dims.iter().enumerate() {
                if active {
                    out_jacobian[(row, q_idx)] = col_b[dim] - col_a[dim];
                    row += 1;
                }
            }
        }
        Ok(())
    }
}
