//! Real-time 1 kHz reactive velocity controller combining multi-arm differential IK,
//! closed-chain manifold stabilization, velocity-damper joint bounds, and Control Barrier Functions (CBFs).

use crate::collision::{
    CollisionEnvironment, CollisionError, CollisionWorkspace, GeometryTier, SignedDistanceWitness,
};
use crate::kinematics::KinematicsError;
use crate::manifold::ConstraintManifold;
use crate::robot::{FrameId, MultiRobotModel, RobotError};
use nalgebra::{DMatrix, DVector, Isometry3, SVector};
use thiserror::Error;

/// Errors raised by the real-time reactive servo controller.
#[derive(Debug, Error, PartialEq)]
pub enum ControlError {
    /// Invalid configuration or DOF dimension.
    #[error(transparent)]
    Robot(#[from] RobotError),
    /// Kinematics evaluation failed.
    #[error(transparent)]
    Kinematics(#[from] KinematicsError),
    /// Collision / signed-distance evaluation failed.
    #[error(transparent)]
    Collision(#[from] CollisionError),
}

/// Real-time Cartesian tracking command for a coordinate frame (e.g., an arm's TCP).
#[derive(Debug, Clone, PartialEq)]
pub struct ServoTarget {
    /// Target coordinate frame.
    pub frame: FrameId,
    /// Feedforward world-aligned spatial twist $[v_x, v_y, v_z, \omega_x, \omega_y, \omega_z]^\top$.
    pub feedforward_twist: SVector<f64, 6>,
    /// Optional desired world pose $T_{\text{world}}^{\text{frame}}$ for proportional pose feedback.
    pub desired_pose: Option<Isometry3<f64>>,
    /// Proportional feedback gain $K_p$ on pose error (`1/s`).
    pub pose_gain: f64,
}

impl ServoTarget {
    /// Creates a pose-tracking servo target with proportional gain `pose_gain`.
    pub fn pose(frame: FrameId, desired_pose: Isometry3<f64>, pose_gain: f64) -> Self {
        Self {
            frame,
            feedforward_twist: SVector::zeros(),
            desired_pose: Some(desired_pose),
            pose_gain,
        }
    }

    /// Creates a pure velocity-mode twist servo target.
    pub fn twist(frame: FrameId, feedforward_twist: SVector<f64, 6>) -> Self {
        Self {
            frame,
            feedforward_twist,
            desired_pose: None,
            pose_gain: 0.0,
        }
    }
}

/// Configuration parameters for [`ReactiveServoController`].
#[derive(Debug, Clone)]
pub struct ReactiveServoConfig {
    /// Control loop timestep $\Delta t$ in seconds (e.g., `0.001` for 1 kHz).
    pub dt: f64,
    /// Minimum allowable safety clearance $d_{\min}$ in meters.
    pub safety_margin: f64,
    /// Distance threshold $d_{\text{act}}$ below which CBF collision avoidance inequalities activate.
    pub cbf_activation_distance: f64,
    /// Control Barrier Function decay rate $\gamma > 0$ ($\dot{d} + \gamma (d - d_{\min}) \ge 0$).
    pub cbf_gamma: f64,
    /// Damped Least-Squares regularization $\lambda$.
    pub damping: f64,
    /// Null-space posture attraction gain toward neutral configuration.
    pub posture_gain: f64,
}

impl Default for ReactiveServoConfig {
    fn default() -> Self {
        Self {
            dt: 0.002,
            safety_margin: 0.05,
            cbf_activation_distance: 0.25,
            cbf_gamma: 5.0,
            damping: 0.03,
            posture_gain: 0.5,
        }
    }
}

/// Pre-allocated, real-time reactive velocity controller with Control Barrier Function (CBF)
/// self-collision, inter-arm collision, and environment obstacle avoidance.
pub struct ReactiveServoController {
    /// Controller tuning parameters.
    pub config: ReactiveServoConfig,
    workspace: CollisionWorkspace,
    witnesses: Vec<SignedDistanceWitness>,
    j_dist_row: Vec<f64>,
}

impl ReactiveServoController {
    /// Allocates a reactive controller and all scratch buffers for `model` and `env`.
    pub fn new(
        model: &MultiRobotModel,
        env: &CollisionEnvironment,
        config: ReactiveServoConfig,
    ) -> Self {
        Self {
            config,
            workspace: env.create_workspace(model),
            witnesses: Vec::with_capacity(32),
            j_dist_row: vec![0.0; model.dof()],
        }
    }

    /// Computes a collision-avoiding, limit-respecting joint velocity command `out_dq_cmd`
    /// for a single control tick $\Delta t$.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        model: &MultiRobotModel,
        env: &CollisionEnvironment,
        q_current: &[f64],
        dq_current: &[f64],
        targets: &[ServoTarget],
        manifold: Option<&dyn ConstraintManifold>,
        out_dq_cmd: &mut [f64],
    ) -> Result<(), ControlError> {
        let dof = model.dof();
        if q_current.len() != dof || dq_current.len() != dof || out_dq_cmd.len() != dof {
            return Err(ControlError::Robot(RobotError::WrongDof {
                expected: dof,
                actual: q_current.len(),
            }));
        }

        // 1. Evaluate FK and query all signed distance witnesses within cbf_activation_distance
        env.query_signed_distances(
            model,
            q_current,
            self.config.cbf_activation_distance,
            GeometryTier::Convex,
            &mut self.workspace,
            &mut self.witnesses,
        )?;

        // 2. Primary Damped Least-Squares task-space velocity + null-space posture attraction
        let mut dq = DVector::<f64>::zeros(dof);
        for k in 0..dof {
            let q_mid = model.dof_limits()[k].position.midpoint();
            dq[k] = self.config.posture_gain * (q_mid - q_current[k]);
        }

        if !targets.is_empty() {
            let m = 6 * targets.len();
            let mut v_des = DVector::<f64>::zeros(m);
            let mut j_task = DMatrix::<f64>::zeros(m, dof);

            for (i, target) in targets.iter().enumerate() {
                let row_off = 6 * i;
                let mut twist = target.feedforward_twist;
                if let Some(des_pose) = &target.desired_pose {
                    let curr_pose = self.workspace.transforms.frame_pose(target.frame);
                    let pos_err = des_pose.translation.vector - curr_pose.translation.vector;
                    let rot_err =
                        (des_pose.rotation * curr_pose.rotation.inverse()).scaled_axis();
                    twist[0] += target.pose_gain * pos_err.x;
                    twist[1] += target.pose_gain * pos_err.y;
                    twist[2] += target.pose_gain * pos_err.z;
                    twist[3] += target.pose_gain * rot_err.x;
                    twist[4] += target.pose_gain * rot_err.y;
                    twist[5] += target.pose_gain * rot_err.z;
                }
                for r in 0..6 {
                    v_des[row_off + r] = twist[r];
                }
                let sub_j = j_task.view_mut((row_off, 0), (6, dof));
                model.geometric_jacobian(&self.workspace.transforms, target.frame, sub_j)?;
            }

            let mut gram = &j_task * j_task.transpose();
            let lam_sq = self.config.damping * self.config.damping;
            for r in 0..m {
                gram[(r, r)] += lam_sq;
            }
            if let Some(chol) = gram.cholesky() {
                let residual = v_des - &j_task * &dq;
                let y = chol.solve(&residual);
                dq += j_task.transpose() * y;
            }
        }

        // 3. Enforce closed-chain bimanual manifold velocity constraint: J_F * dq = -K_F * F(q)
        if let Some(mf) = manifold {
            let m_c = mf.constraint_dim();
            if m_c > 0 {
                let mut f_res = DVector::<f64>::zeros(m_c);
                let mut j_f = DMatrix::<f64>::zeros(m_c, dof);
                mf.evaluate(model, &self.workspace.transforms, &mut f_res)?;
                mf.jacobian(model, &self.workspace.transforms, &mut j_f)?;

                let mut gram_f = &j_f * j_f.transpose();
                for r in 0..m_c {
                    gram_f[(r, r)] += 1e-6;
                }
                if let Some(chol) = gram_f.cholesky() {
                    let rhs = -10.0 * f_res - &j_f * &dq;
                    let y = chol.solve(&rhs);
                    dq += j_f.transpose() * y;
                }
            }
        }

        // 4. Compute velocity-damper joint position, velocity, and acceleration box bounds
        let dt = self.config.dt.max(1e-4);
        let mut lb = vec![0.0; dof];
        let mut ub = vec![0.0; dof];
        for (k, lim) in model.dof_limits().iter().enumerate() {
            let v_max = lim.max_velocity;
            let a_max = lim.max_acceleration;
            let pos_lb = if lim.position.min.is_finite() {
                (lim.position.min - q_current[k]) / dt
            } else {
                -v_max
            };
            let pos_ub = if lim.position.max.is_finite() {
                (lim.position.max - q_current[k]) / dt
            } else {
                v_max
            };
            lb[k] = (-v_max).max(dq_current[k] - a_max * dt).max(pos_lb);
            ub[k] = v_max.min(dq_current[k] + a_max * dt).min(pos_ub);
            if lb[k] > ub[k] {
                let mid = 0.5 * (lb[k] + ub[k]);
                lb[k] = mid;
                ub[k] = mid;
            }
        }

        // 5. Iterative Control Barrier Function (CBF) half-space & box-bound projection:
        //    For each witness i: J_dist_i * dq >= -gamma * (d_i - safety_margin)
        self.j_dist_row.resize(dof, 0.0);
        for _iter in 0..10 {
            for w in &self.witnesses {
                w.compute_distance_jacobian(
                    model,
                    &self.workspace.transforms,
                    &mut self.j_dist_row,
                );
                let norm_sq: f64 = self.j_dist_row.iter().map(|&v| v * v).sum();
                if norm_sq <= 1e-12 {
                    continue;
                }
                let j_dot_dq: f64 = self
                    .j_dist_row
                    .iter()
                    .zip(dq.iter())
                    .map(|(&j, &v)| j * v)
                    .sum();
                let h = w.signed_distance - self.config.safety_margin;
                let min_rate = -self.config.cbf_gamma * h;

                if j_dot_dq < min_rate {
                    let alpha = (min_rate - j_dot_dq) / norm_sq;
                    for (k, &j_val) in self.j_dist_row.iter().enumerate() {
                        dq[k] += alpha * j_val;
                    }
                }
            }

            // Clamp onto velocity-damper box bounds
            for k in 0..dof {
                dq[k] = dq[k].clamp(lb[k], ub[k]);
            }
        }

        out_dq_cmd.copy_from_slice(dq.as_slice());
        Ok(())
    }
}
