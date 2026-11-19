//! Real-time 1 kHz reactive velocity controller combining multi-arm differential IK,
//! closed-chain manifold stabilization, velocity-damper joint bounds, and Control Barrier Functions (CBFs).

use crate::collision::{
    CollidingPair, CollisionEnvironment, CollisionError, CollisionWorkspace, GeometryTier,
    SignedDistanceWitness,
};
use crate::kinematics::KinematicsError;
use crate::manifold::ConstraintManifold;
use crate::robot::{FrameId, LinkId, MultiRobotModel, RobotError, RobotId};
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
    /// Active twist dimensions `[vx, vy, vz, wx, wy, wz]` tracked by DLS IK.
    pub active_dims: [bool; 6],
}

impl ServoTarget {
    /// Creates a full 6-DOF pose-tracking servo target with proportional gain `pose_gain`.
    pub fn pose(frame: FrameId, desired_pose: Isometry3<f64>, pose_gain: f64) -> Self {
        Self {
            frame,
            feedforward_twist: SVector::zeros(),
            desired_pose: Some(desired_pose),
            pose_gain,
            active_dims: [true; 6],
        }
    }

    /// Creates a 3-DOF Cartesian position-only servo target (`[vx, vy, vz]` active).
    pub fn position(frame: FrameId, desired_position: Point3<f64>, pose_gain: f64) -> Self {
        Self {
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
        Self {
            frame,
            feedforward_twist,
            desired_pose: None,
            pose_gain: 0.0,
            active_dims: [true; 6],
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
        }
    }
}

impl ReactiveServoConfig {
    /// Sets the priority weight for `robot` in the QP-CBF objective (`priority > 1.0` holds course while lower-priority arms yield).
    pub fn with_arm_priority(mut self, robot: RobotId, priority: f64) -> Self {
        self.arm_priorities.insert(robot, priority.max(1e-3));
        self
    }
}

/// Pre-allocated, real-time reactive velocity controller with priority-weighted QP Control Barrier
/// Function (CBF) self-collision, inter-arm collision, attached tool/payload avoidance, and symmetry breaking.
pub struct ReactiveServoController {
    /// Controller tuning parameters.
    pub config: ReactiveServoConfig,
    workspace: CollisionWorkspace,
    witnesses: Vec<SignedDistanceWitness>,
    j_dist_row: Vec<f64>,
    j_tan_row: Vec<f64>,
    curl_dq: Vec<f64>,
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

        // Refresh inverse priority weights per DOF
        self.inv_dof_weights.resize(dof, 1.0);
        for k in 0..dof {
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

        // 2. Primary Damped Least-Squares task-space velocity + null-space posture attraction
        let mut dq = DVector::<f64>::zeros(dof);
        for k in 0..dof {
            let q_mid = model.dof_limits()[k].position.midpoint();
            dq[k] = self.config.posture_gain * (q_mid - q_current[k]);
        }

        let m: usize = targets
            .iter()
            .map(|t| t.active_dims.iter().filter(|&&b| b).count())
            .sum();
        if m > 0 {
            let mut v_des = DVector::<f64>::zeros(m);
            let mut j_task = DMatrix::<f64>::zeros(m, dof);
            let mut j_full = DMatrix::<f64>::zeros(6, dof);
            let mut row_cursor = 0;

            for target in targets {
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
                model.geometric_jacobian(
                    &self.workspace.transforms,
                    target.frame,
                    j_full.as_view_mut(),
                )?;
                for r in 0..6 {
                    if target.active_dims[r] {
                        v_des[row_cursor] = twist[r];
                        for c in 0..dof {
                            j_task[(row_cursor, c)] = j_full[(r, c)];
                        }
                        row_cursor += 1;
                    }
                }
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
        self.lb.resize(dof, 0.0);
        self.ub.resize(dof, 0.0);
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
                    .zip(dq.iter())
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
                                .zip(dq.iter())
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
        for k in 0..dof {
            dq[k] += self.curl_dq[k];
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
                let dot_curr: f64 = a_i.iter().zip(dq.iter()).map(|(&a, &v)| a * v).sum();
                let y_dot_a = dot_curr - self.cbf_lambdas[i] * a_winv_a_raw;
                let new_lambda =
                    ((self.cbf_b_rhs[i] - y_dot_a) / (a_winv_a_raw + rho)).max(0.0);
                let d_lam = new_lambda - self.cbf_lambdas[i];
                if d_lam.abs() > 1e-12 {
                    for k in 0..dof {
                        let step = d_lam * self.inv_dof_weights[k] * a_i[k];
                        dq[k] += step;
                        max_delta = max_delta.max(step.abs());
                    }
                    self.cbf_lambdas[i] = new_lambda;
                }
            }

            for k in 0..dof {
                let y_k = dq[k] - self.qp_p_box[k];
                let clamped = y_k.clamp(self.lb[k], self.ub[k]);
                max_delta = max_delta.max((clamped - dq[k]).abs());
                self.qp_p_box[k] = clamped - y_k;
                dq[k] = clamped;
            }

            if max_delta < 1e-7 {
                break;
            }
        }

        for k in 0..dof {
            dq[k] = dq[k].clamp(self.lb[k], self.ub[k]);
        }

        out_dq_cmd.copy_from_slice(dq.as_slice());
        Ok(())
    }
}

#[inline]
fn root_link_of(model: &MultiRobotModel, mut link: LinkId) -> LinkId {
    while let Some(parent) = model.link(link).parent_link {
        link = parent;
    }
    link
}
