//! Real-time 1 kHz reactive velocity controller combining multi-arm differential IK,
//! closed-chain manifold stabilization, velocity-damper joint bounds, and Control Barrier Functions (CBFs).

use crate::collision::{
    CollidingPair, CollisionEnvironment, CollisionError, CollisionWorkspace, GeometryTier,
    SignedDistanceWitness,
};
use crate::kinematics::KinematicsError;
use crate::manifold::ConstraintManifold;
use crate::robot::{FrameId, LinkId, MultiRobotModel, PlanningGroup, RobotError, RobotId};
use nalgebra::{DMatrix, DVector, Isometry3, Point3, SVector, Vector3};
use std::collections::BTreeMap;
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

/// Real-time tracking command for [`ReactiveServoController`].
#[derive(Debug, Clone, PartialEq)]
pub enum ServoTarget {
    /// Cartesian task-space tracking command for a coordinate frame (e.g., an arm's TCP).
    Cartesian {
        /// Target coordinate frame.
        frame: FrameId,
        /// Feedforward world-aligned spatial twist $[v_x, v_y, v_z, \omega_x, \omega_y, \omega_z]^\top$.
        feedforward_twist: SVector<f64, 6>,
        /// Optional desired world pose $T_{\text{world}}^{\text{frame}}$ for proportional pose feedback.
        desired_pose: Option<Isometry3<f64>>,
        /// Proportional feedback gain $K_p$ on pose error (`1/s`).
        pose_gain: f64,
        /// Active twist dimensions `[vx, vy, vz, wx, wy, wz]` tracked by DLS IK.
        active_dims: [bool; 6],
    },
    /// Joint-space trajectory reference tracked through the 1 kHz CBF safety filter:
    /// $\dot{q}_{\text{nom}} = \dot{q}_{\text{ref}} + k_p (q_{\text{ref}} - q)$.
    JointReference {
        /// Desired joint positions (full model DOF when `q_indices` is `None`, else matches `q_indices`).
        q_ref: Vec<f64>,
        /// Feedforward joint velocities (matching length of `q_ref`).
        dq_ref: Vec<f64>,
        /// Proportional position feedback gain `kp` (`1/s`).
        kp: f64,
        /// Optional subset of global DOF indices (e.g. from a [`PlanningGroup`]).
        q_indices: Option<Vec<usize>>,
    },
}

impl ServoTarget {
    /// Creates a full 6-DOF pose-tracking servo target with proportional gain `pose_gain`.
    pub fn pose(frame: FrameId, desired_pose: Isometry3<f64>, pose_gain: f64) -> Self {
        Self::Cartesian {
            frame,
            feedforward_twist: SVector::zeros(),
            desired_pose: Some(desired_pose),
            pose_gain,
            active_dims: [true; 6],
        }
    }

    /// Creates a 3-DOF Cartesian position-only servo target (`[vx, vy, vz]` active).
    pub fn position(frame: FrameId, desired_position: Point3<f64>, pose_gain: f64) -> Self {
        Self::Cartesian {
            frame,
            feedforward_twist: SVector::zeros(),
            desired_pose: Some(Isometry3::translation(
                desired_position.x,
                desired_position.y,
                desired_position.z,
            )),
            pose_gain,
            active_dims: [true, true, true, false, false, false],
        }
    }

    /// Creates a pure velocity-mode twist servo target.
    pub fn twist(frame: FrameId, feedforward_twist: SVector<f64, 6>) -> Self {
        Self::Cartesian {
            frame,
            feedforward_twist,
            desired_pose: None,
            pose_gain: 0.0,
            active_dims: [true; 6],
        }
    }

    /// Creates a full-model joint-space reference target (`dq_nom = dq_ref + kp * (q_ref - q)`).
    pub fn joint_reference(q_ref: Vec<f64>, dq_ref: Vec<f64>, kp: f64) -> Self {
        Self::JointReference {
            q_ref,
            dq_ref,
            kp,
            q_indices: None,
        }
    }

    /// Creates a subgroup joint-space reference target scoped to `group`'s actuated DOFs.
    pub fn group_joint_reference(
        group: &PlanningGroup,
        q_ref: Vec<f64>,
        dq_ref: Vec<f64>,
        kp: f64,
    ) -> Self {
        Self::JointReference {
            q_ref,
            dq_ref,
            kp,
            q_indices: Some(group.q_indices.clone()),
        }
    }
}

/// Configuration parameters for [`ReactiveServoController`].
#[derive(Debug, Clone)]
pub struct ReactiveServoConfig {
    /// Control loop timestep $\Delta t$ in seconds (e.g., `0.001` for 1 kHz).
    pub dt: f64,
    /// Default minimum allowable safety clearance $d_{\min}$ in meters when no pair-specific margin is configured.
    pub safety_margin: f64,
    /// Default distance threshold $d_{\text{act}}$ below which CBF collision avoidance inequalities activate.
    pub cbf_activation_distance: f64,
    /// Control Barrier Function decay rate $\gamma > 0$ ($\dot{d} + \gamma (d - d_{\min}) \ge 0$).
    pub cbf_gamma: f64,
    /// Damped Least-Squares regularization $\lambda$.
    pub damping: f64,
    /// Null-space posture attraction gain toward neutral configuration.
    pub posture_gain: f64,
    /// Per-arm metric weights for priority-weighted QP-CBF (higher weight resists yielding; default `1.0`).
    pub arm_priorities: BTreeMap<RobotId, f64>,
    /// Gain for tangential symmetry-breaking curl injected when CBF constraints oppose nominal motion.
    pub tangential_curl_gain: f64,
    /// Maximum Dykstra active-set iterations per control tick for the priority-weighted QP-CBF.
    pub qp_max_iterations: usize,
    /// Soft-slack quadratic penalty regularization $\rho \ge 0$ preventing QP infeasibility when constraints conflict.
    pub qp_slack_reg: f64,
    /// Discrete-time CBF curvature safety buffer (meters) added to `d_min` so forward Euler steps stay strictly clear.
    pub cbf_margin_padding: f64,
    /// Optional per-DOF mask (`false` freezes that joint completely at zero velocity).
    pub active_dof_mask: Option<Vec<bool>>,
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
            arm_priorities: BTreeMap::new(),
            tangential_curl_gain: 0.35,
            qp_max_iterations: 30,
            qp_slack_reg: 1e-5,
            cbf_margin_padding: 0.001,
            active_dof_mask: None,
        }
    }
}

impl ReactiveServoConfig {
    /// Sets the priority weight for `robot` in the QP-CBF objective (`priority > 1.0` holds course while lower-priority arms yield).
    pub fn with_arm_priority(mut self, robot: RobotId, priority: f64) -> Self {
        self.arm_priorities.insert(robot, priority.max(1e-3));
        self
    }

    /// Restricts reactive velocity commands and CBF yielding to DOFs where `mask[k]` is `true`.
    pub fn with_active_dof_mask(mut self, mask: Vec<bool>) -> Self {
        self.active_dof_mask = Some(mask);
        self
    }

    /// Restricts reactive velocity commands and CBF yielding to `group`'s DOFs on `model`.
    pub fn with_planning_group(mut self, model: &MultiRobotModel, group: &PlanningGroup) -> Self {
        let mut mask = vec![false; model.dof()];
        for &idx in &group.q_indices {
            if idx < mask.len() {
                mask[idx] = true;
            }
        }
        self.active_dof_mask = Some(mask);
        self
    }
}

/// Pre-allocated, zero-allocation real-time reactive velocity controller with priority-weighted QP
/// Control Barrier Function (CBF) self-collision, inter-arm collision, attached tool/payload avoidance, and symmetry breaking.
pub struct ReactiveServoController {
    /// Controller tuning parameters.
    pub config: ReactiveServoConfig,
    workspace: CollisionWorkspace,
    witnesses: Vec<SignedDistanceWitness>,
    j_dist_row: Vec<f64>,
    j_tan_row: Vec<f64>,
    curl_dq: Vec<f64>,
    dq_scratch: Vec<f64>,
    j_full_buf: DMatrix<f64>,
    v_des_buf: Vec<f64>,
    j_task_buf: Vec<f64>,
    gram_buf: Vec<f64>,
    rhs_buf: Vec<f64>,
    mf_res_buf: DVector<f64>,
    mf_jac_buf: DMatrix<f64>,
    inv_dof_weights: Vec<f64>,
    cbf_a_rows: Vec<f64>,
    cbf_b_rhs: Vec<f64>,
    cbf_a_winv_a: Vec<f64>,
    cbf_lambdas: Vec<f64>,
    qp_p_box: Vec<f64>,
    lb: Vec<f64>,
    ub: Vec<f64>,
}

impl ReactiveServoController {
    /// Allocates a reactive controller and all scratch buffers for `model` and `env`.
    pub fn new(
        model: &MultiRobotModel,
        env: &CollisionEnvironment,
        config: ReactiveServoConfig,
    ) -> Self {
        let dof = model.dof();
        Self {
            config,
            workspace: env.create_workspace(model),
            witnesses: Vec::with_capacity(64),
            j_dist_row: vec![0.0; dof],
            j_tan_row: vec![0.0; dof],
            curl_dq: vec![0.0; dof],
            dq_scratch: vec![0.0; dof],
            j_full_buf: DMatrix::zeros(6, dof),
            v_des_buf: Vec::with_capacity(24),
            j_task_buf: Vec::with_capacity(24 * dof),
            gram_buf: Vec::with_capacity(24 * 24),
            rhs_buf: Vec::with_capacity(24),
            mf_res_buf: DVector::zeros(6),
            mf_jac_buf: DMatrix::zeros(6, dof),
            inv_dof_weights: vec![1.0; dof],
            cbf_a_rows: Vec::with_capacity(64 * dof),
            cbf_b_rhs: Vec::with_capacity(64),
            cbf_a_winv_a: Vec::with_capacity(64),
            cbf_lambdas: Vec::with_capacity(64),
            qp_p_box: vec![0.0; dof],
            lb: vec![0.0; dof],
            ub: vec![0.0; dof],
        }
    }

    /// Computes a collision-avoiding, limit-respecting joint velocity command `out_dq_cmd`
    /// for a single control tick $\Delta t$ with zero heap allocations on warm scratch buffers.
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
        self.step_internal(
            model,
            env,
            q_current,
            dq_current,
            targets,
            None,
            manifold,
            out_dq_cmd,
        )
    }

    /// Zero-allocation 1 kHz CBF safety filter tracking a full-DOF joint-space reference
    /// `(q_ref, dq_ref)` without allocating a [`ServoTarget`] wrapper.
    #[allow(clippy::too_many_arguments)]
    pub fn step_joint_reference(
        &mut self,
        model: &MultiRobotModel,
        env: &CollisionEnvironment,
        q_current: &[f64],
        dq_current: &[f64],
        q_ref: &[f64],
        dq_ref: &[f64],
        kp: f64,
        manifold: Option<&dyn ConstraintManifold>,
        out_dq_cmd: &mut [f64],
    ) -> Result<(), ControlError> {
        self.step_internal(
            model,
            env,
            q_current,
            dq_current,
            &[],
            Some((q_ref, dq_ref, kp)),
            manifold,
            out_dq_cmd,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn step_internal(
        &mut self,
        model: &MultiRobotModel,
        env: &CollisionEnvironment,
        q_current: &[f64],
        dq_current: &[f64],
        targets: &[ServoTarget],
        direct_joint_ref: Option<(&[f64], &[f64], f64)>,
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
        if let Some(mask) = &self.config.active_dof_mask {
            if mask.len() != dof {
                return Err(ControlError::Robot(RobotError::WrongDof {
                    expected: dof,
                    actual: mask.len(),
                }));
            }
        }
        let is_active = |k: usize| self.config.active_dof_mask.as_ref().is_none_or(|m| m[k]);

        // Refresh inverse priority weights per DOF
        self.inv_dof_weights.resize(dof, 1.0);
        for k in 0..dof {
            if !is_active(k) {
                self.inv_dof_weights[k] = 0.0;
                continue;
            }
            let robot_id = model.joint(model.actuated_joint(k)).robot;
            let w = self
                .config
                .arm_priorities
                .get(&robot_id)
                .copied()
                .unwrap_or(1.0)
                .max(1e-3);
            self.inv_dof_weights[k] = 1.0 / w;
        }

        // 1. Evaluate FK and query all signed distance witnesses within max configured activation distance
        let max_act = env.max_activation_distance(self.config.cbf_activation_distance);
        env.query_signed_distances(
            model,
            q_current,
            max_act,
            GeometryTier::Convex,
            &mut self.workspace,
            &mut self.witnesses,
        )?;

        // 2a. Initialize nominal velocity (posture attraction + JointReference overrides)
        self.dq_scratch.resize(dof, 0.0);
        for k in 0..dof {
            let q_mid = model.dof_limits()[k].position.midpoint();
            self.dq_scratch[k] = if is_active(k) {
                self.config.posture_gain * (q_mid - q_current[k])
            } else {
                0.0
            };
        }

        if let Some((q_ref, dq_ref, kp)) = direct_joint_ref {
            if q_ref.len() != dof || dq_ref.len() != dof {
                return Err(ControlError::Robot(RobotError::WrongDof {
                    expected: dof,
                    actual: q_ref.len().min(dq_ref.len()),
                }));
            }
            for k in 0..dof {
                if is_active(k) {
                    let err = model.dof_signed_diff(k, q_current[k], q_ref[k]);
                    self.dq_scratch[k] = dq_ref[k] + kp * err;
                }
            }
        }

        for target in targets {
            if let ServoTarget::JointReference {
                q_ref,
                dq_ref,
                kp,
                q_indices,
            } = target
            {
                match q_indices {
                    None => {
                        if q_ref.len() != dof || dq_ref.len() != dof {
                            return Err(ControlError::Robot(RobotError::WrongDof {
                                expected: dof,
                                actual: q_ref.len().min(dq_ref.len()),
                            }));
                        }
                        for k in 0..dof {
                            if is_active(k) {
                                let err = model.dof_signed_diff(k, q_current[k], q_ref[k]);
                                self.dq_scratch[k] = dq_ref[k] + kp * err;
                            }
                        }
                    }
                    Some(indices) => {
                        if q_ref.len() != indices.len() || dq_ref.len() != indices.len() {
                            return Err(ControlError::Robot(RobotError::WrongDof {
                                expected: indices.len(),
                                actual: q_ref.len().min(dq_ref.len()),
                            }));
                        }
                        for (local_i, &k) in indices.iter().enumerate() {
                            if k >= dof {
                                return Err(ControlError::Robot(RobotError::WrongDof {
                                    expected: dof,
                                    actual: k + 1,
                                }));
                            }
                            if is_active(k) {
                                let err = model.dof_signed_diff(k, q_current[k], q_ref[local_i]);
                                self.dq_scratch[k] = dq_ref[local_i] + kp * err;
                            }
                        }
                    }
                }
            }
        }

        // 2b. Zero-allocation Cartesian DLS task-space velocity solve
        let m: usize = targets
            .iter()
            .map(|t| match t {
                ServoTarget::Cartesian { active_dims, .. } => {
                    active_dims.iter().filter(|&&b| b).count()
                }
                ServoTarget::JointReference { .. } => 0,
            })
            .sum();
        if m > 0 {
            if self.j_full_buf.nrows() != 6 || self.j_full_buf.ncols() != dof {
                self.j_full_buf = DMatrix::zeros(6, dof);
            }
            self.v_des_buf.clear();
            self.j_task_buf.clear();
            self.v_des_buf.reserve(m.saturating_sub(self.v_des_buf.capacity()));
            self.j_task_buf
                .reserve((m * dof).saturating_sub(self.j_task_buf.capacity()));

            for target in targets {
                let ServoTarget::Cartesian {
                    frame,
                    feedforward_twist,
                    desired_pose,
                    pose_gain,
                    active_dims,
                } = target
                else {
                    continue;
                };

                let mut twist = *feedforward_twist;
                if let Some(des_pose) = desired_pose {
                    let curr_pose = self.workspace.transforms.frame_pose(*frame);
                    let pos_err = des_pose.translation.vector - curr_pose.translation.vector;
                    let rot_err =
                        (des_pose.rotation * curr_pose.rotation.inverse()).scaled_axis();
                    twist[0] += pose_gain * pos_err.x;
                    twist[1] += pose_gain * pos_err.y;
                    twist[2] += pose_gain * pos_err.z;
                    twist[3] += pose_gain * rot_err.x;
                    twist[4] += pose_gain * rot_err.y;
                    twist[5] += pose_gain * rot_err.z;
                }
                model.geometric_jacobian(
                    &self.workspace.transforms,
                    *frame,
                    self.j_full_buf.as_view_mut(),
                )?;
                for (r, &active) in active_dims.iter().enumerate() {
                    if active {
                        self.v_des_buf.push(twist[r]);
                        for c in 0..dof {
                            self.j_task_buf.push(if is_active(c) {
                                self.j_full_buf[(r, c)]
                            } else {
                                0.0
                            });
                        }
                    }
                }
            }

            let lam_sq = self.config.damping * self.config.damping;
            self.gram_buf.clear();
            self.gram_buf.resize(m * m, 0.0);
            self.rhs_buf.clear();
            self.rhs_buf.resize(m, 0.0);

            for r in 0..m {
                let row_r = &self.j_task_buf[r * dof..(r + 1) * dof];
                let j_dq: f64 = row_r
                    .iter()
                    .zip(self.dq_scratch.iter())
                    .map(|(&j, &qdot)| j * qdot)
                    .sum();
                self.rhs_buf[r] = self.v_des_buf[r] - j_dq;
                for c in 0..=r {
                    let row_c = &self.j_task_buf[c * dof..(c + 1) * dof];
                    let dot: f64 = row_r.iter().zip(row_c.iter()).map(|(&a, &b)| a * b).sum();
                    let val = if r == c { dot + lam_sq } else { dot };
                    self.gram_buf[r * m + c] = val;
                    self.gram_buf[c * m + r] = val;
                }
            }

            if cholesky_solve_in_place(m, &mut self.gram_buf[..m * m], &mut self.rhs_buf[..m]) {
                for r in 0..m {
                    let y_r = self.rhs_buf[r];
                    let row_r = &self.j_task_buf[r * dof..(r + 1) * dof];
                    for (dst, &j) in self.dq_scratch.iter_mut().zip(row_r.iter()) {
                        *dst += j * y_r;
                    }
                }
            }
        }

        // 3. Enforce closed-chain bimanual manifold velocity constraint: J_F * dq = -K_F * F(q)
        if let Some(mf) = manifold {
            let m_c = mf.constraint_dim();
            if m_c > 0 {
                if self.mf_res_buf.len() != m_c {
                    self.mf_res_buf = DVector::zeros(m_c);
                }
                if self.mf_jac_buf.nrows() != m_c || self.mf_jac_buf.ncols() != dof {
                    self.mf_jac_buf = DMatrix::zeros(m_c, dof);
                }
                mf.evaluate(model, &self.workspace.transforms, &mut self.mf_res_buf)?;
                mf.jacobian(model, &self.workspace.transforms, &mut self.mf_jac_buf)?;

                for c in 0..dof {
                    if !is_active(c) {
                        for r in 0..m_c {
                            self.mf_jac_buf[(r, c)] = 0.0;
                        }
                    }
                }

                self.gram_buf.clear();
                self.gram_buf.resize(m_c * m_c, 0.0);
                self.rhs_buf.clear();
                self.rhs_buf.resize(m_c, 0.0);

                for r in 0..m_c {
                    let mut j_dq = 0.0;
                    for (c, &qdot) in self.dq_scratch.iter().enumerate() {
                        j_dq += self.mf_jac_buf[(r, c)] * qdot;
                    }
                    self.rhs_buf[r] = -10.0 * self.mf_res_buf[r] - j_dq;
                    for c in 0..=r {
                        let mut dot = 0.0;
                        for k in 0..dof {
                            dot += self.mf_jac_buf[(r, k)] * self.mf_jac_buf[(c, k)];
                        }
                        let val = if r == c { dot + 1e-6 } else { dot };
                        self.gram_buf[r * m_c + c] = val;
                        self.gram_buf[c * m_c + r] = val;
                    }
                }

                if cholesky_solve_in_place(
                    m_c,
                    &mut self.gram_buf[..m_c * m_c],
                    &mut self.rhs_buf[..m_c],
                ) {
                    for r in 0..m_c {
                        let y_r = self.rhs_buf[r];
                        for (c, dst) in self.dq_scratch.iter_mut().enumerate() {
                            *dst += self.mf_jac_buf[(r, c)] * y_r;
                        }
                    }
                }
            }
        }

        // 4. Compute velocity-damper joint position, velocity, and acceleration box bounds
        let dt = self.config.dt.max(1e-4);
        self.lb.resize(dof, 0.0);
        self.ub.resize(dof, 0.0);
        for (k, lim) in model.dof_limits().iter().enumerate() {
            if !is_active(k) {
                self.lb[k] = 0.0;
                self.ub[k] = 0.0;
                continue;
            }
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
            self.lb[k] = (-v_max).max(dq_current[k] - a_max * dt).max(pos_lb);
            self.ub[k] = v_max.min(dq_current[k] + a_max * dt).min(pos_ub);
            if self.lb[k] > self.ub[k] {
                let mid = 0.5 * (self.lb[k] + self.ub[k]);
                self.lb[k] = mid;
                self.ub[k] = mid;
            }
        }

        // 5. Build active per-pair CBF constraints and compute tangential symmetry-breaking curl
        self.j_dist_row.resize(dof, 0.0);
        self.j_tan_row.resize(dof, 0.0);
        self.curl_dq.resize(dof, 0.0);
        self.curl_dq.fill(0.0);
        self.cbf_a_rows.clear();
        self.cbf_b_rhs.clear();
        self.cbf_a_winv_a.clear();

        for w in &self.witnesses {
            let d_min = env.effective_pair_margin(&w.pair, self.config.safety_margin)
                + self.config.cbf_margin_padding.max(0.0);
            let d_act = env
                .effective_pair_activation(&w.pair, self.config.cbf_activation_distance)
                .max(d_min + 1e-3);
            if w.signed_distance > d_act {
                continue;
            }

            w.compute_distance_jacobian(
                model,
                &self.workspace.transforms,
                &mut self.j_dist_row,
            );
            let a_winv_a: f64 = self
                .j_dist_row
                .iter()
                .zip(self.inv_dof_weights.iter())
                .map(|(&j, &iw)| j * j * iw)
                .sum();
            if a_winv_a <= 1e-12 {
                continue;
            }

            let b_i = -self.config.cbf_gamma * (w.signed_distance - d_min);
            self.cbf_a_rows.extend_from_slice(&self.j_dist_row);
            self.cbf_b_rhs.push(b_i);
            self.cbf_a_winv_a.push(a_winv_a);

            if self.config.tangential_curl_gain > 0.0 {
                let j_dot_dq_nom: f64 = self
                    .j_dist_row
                    .iter()
                    .zip(self.dq_scratch.iter())
                    .map(|(&j, &v)| j * v)
                    .sum();
                let violation = (b_i - j_dot_dq_nom).max(0.0);
                if violation > 1e-9 {
                    let up = if w.normal_world.z.abs() < 0.9 {
                        Vector3::z()
                    } else {
                        Vector3::x()
                    };
                    let tangent_raw = up.cross(&w.normal_world);
                    let t_norm = tangent_raw.norm();
                    if t_norm > 1e-9 {
                        let mut tangent = tangent_raw / t_norm;
                        self.j_tan_row.fill(0.0);
                        match w.pair {
                            CollidingPair::LinkToObstacle { link, .. }
                            | CollidingPair::PayloadToObstacle {
                                parent_link: link, ..
                            } => {
                                model.accumulate_point_jacobian_projection(
                                    &self.workspace.transforms,
                                    link,
                                    &w.witness_a_world,
                                    &tangent,
                                    1.0,
                                    &mut self.j_tan_row,
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
                                let root_a = root_link_of(model, link_a);
                                let root_b = root_link_of(model, link_b);
                                if root_a != root_b {
                                    let base_dir = self
                                        .workspace
                                        .transforms
                                        .link_pose(root_a)
                                        .translation
                                        .vector
                                        - self
                                            .workspace
                                            .transforms
                                            .link_pose(root_b)
                                            .translation
                                            .vector;
                                    if base_dir.dot(&tangent) < -0.1 * base_dir.norm() {
                                        tangent = -tangent;
                                    }
                                }
                                model.accumulate_point_jacobian_projection(
                                    &self.workspace.transforms,
                                    link_a,
                                    &w.witness_a_world,
                                    &tangent,
                                    1.0,
                                    &mut self.j_tan_row,
                                );
                                model.accumulate_point_jacobian_projection(
                                    &self.workspace.transforms,
                                    link_b,
                                    &w.witness_b_world,
                                    &tangent,
                                    -1.0,
                                    &mut self.j_tan_row,
                                );
                            }
                        }

                        let t_winv_t: f64 = self
                            .j_tan_row
                            .iter()
                            .zip(self.inv_dof_weights.iter())
                            .map(|(&jt, &iw)| jt * jt * iw)
                            .sum();
                        if t_winv_t > 1e-12 {
                            let tan_dot_dq: f64 = self
                                .j_tan_row
                                .iter()
                                .zip(self.dq_scratch.iter())
                                .map(|(&jt, &v)| jt * v)
                                .sum();
                            let dir_sign = if tan_dot_dq < -1e-6 { -1.0 } else { 1.0 };
                            let prox = ((d_act - w.signed_distance)
                                / (d_act - d_min).max(1e-4))
                            .clamp(0.0, 1.0);
                            let scale = dir_sign
                                * self.config.tangential_curl_gain
                                * prox
                                * violation
                                / (t_winv_t + 1e-4);
                            for k in 0..dof {
                                self.curl_dq[k] +=
                                    scale * self.inv_dof_weights[k] * self.j_tan_row[k];
                            }
                        }
                    }
                }
            }
        }

        // Inject tangential curl into the nominal command BEFORE QP safety projection
        for (dst, &curl) in self.dq_scratch.iter_mut().zip(self.curl_dq.iter()) {
            *dst += curl;
        }

        // 6. Priority-weighted soft-slack QP-CBF with box bounds via Dykstra's projection in W-metric
        let num_cbf = self.cbf_b_rhs.len();
        let rho = self.config.qp_slack_reg.max(0.0);
        self.cbf_lambdas.clear();
        self.cbf_lambdas.resize(num_cbf, 0.0);
        self.qp_p_box.resize(dof, 0.0);
        self.qp_p_box.fill(0.0);

        let max_iters = self.config.qp_max_iterations.max(1);
        for _iter in 0..max_iters {
            let mut max_delta = 0.0_f64;

            for i in 0..num_cbf {
                let a_i = &self.cbf_a_rows[i * dof..(i + 1) * dof];
                let a_winv_a_raw = self.cbf_a_winv_a[i];
                let dot_curr: f64 = a_i
                    .iter()
                    .zip(self.dq_scratch.iter())
                    .map(|(&a, &v)| a * v)
                    .sum();
                let y_dot_a = dot_curr - self.cbf_lambdas[i] * a_winv_a_raw;
                let new_lambda =
                    ((self.cbf_b_rhs[i] - y_dot_a) / (a_winv_a_raw + rho)).max(0.0);
                let d_lam = new_lambda - self.cbf_lambdas[i];
                if d_lam.abs() > 1e-12 {
                    for k in 0..dof {
                        let step = d_lam * self.inv_dof_weights[k] * a_i[k];
                        self.dq_scratch[k] += step;
                        max_delta = max_delta.max(step.abs());
                    }
                    self.cbf_lambdas[i] = new_lambda;
                }
            }

            for k in 0..dof {
                let y_k = self.dq_scratch[k] - self.qp_p_box[k];
                let clamped = y_k.clamp(self.lb[k], self.ub[k]);
                max_delta = max_delta.max((clamped - self.dq_scratch[k]).abs());
                self.qp_p_box[k] = clamped - y_k;
                self.dq_scratch[k] = clamped;
            }

            if max_delta < 1e-7 {
                break;
            }
        }

        for k in 0..dof {
            self.dq_scratch[k] = self.dq_scratch[k].clamp(self.lb[k], self.ub[k]);
        }

        out_dq_cmd.copy_from_slice(&self.dq_scratch[..dof]);
        Ok(())
    }
}

fn cholesky_solve_in_place(n: usize, gram: &mut [f64], rhs: &mut [f64]) -> bool {
    for i in 0..n {
        for j in 0..=i {
            let mut sum = gram[i * n + j];
            for k in 0..j {
                sum -= gram[i * n + k] * gram[j * n + k];
            }
            if i == j {
                if sum <= 1e-18 || !sum.is_finite() {
                    return false;
                }
                gram[i * n + i] = sum.sqrt();
            } else {
                gram[i * n + j] = sum / gram[j * n + j];
            }
        }
    }
    for i in 0..n {
        let mut sum = rhs[i];
        for k in 0..i {
            sum -= gram[i * n + k] * rhs[k];
        }
        rhs[i] = sum / gram[i * n + i];
    }
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for k in (i + 1)..n {
            sum -= gram[k * n + i] * rhs[k];
        }
        rhs[i] = sum / gram[i * n + i];
    }
    true
}

#[inline]
fn root_link_of(model: &MultiRobotModel, mut link: LinkId) -> LinkId {
    while let Some(parent) = model.link(link).parent_link {
        link = parent;
    }
    link
}
