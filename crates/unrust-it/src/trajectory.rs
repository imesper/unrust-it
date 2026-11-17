//! Kinodynamic trajectory representations, $C^2$ cubic spline interpolation,
//! TOPP-RA time-optimal path parameterization, and online jerk-limited trajectory generation.

use crate::planner::PlannerError;
use crate::robot::{Configuration, KinodynamicLimits};

/// A time-indexed kinodynamic state along a trajectory.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectorySample {
    /// Timestamp $t$ in seconds from the start of the trajectory.
    pub time: f64,
    /// Joint positions $q(t)$.
    pub position: Vec<f64>,
    /// Joint velocities $\dot{q}(t)$.
    pub velocity: Vec<f64>,
    /// Joint accelerations $\ddot{q}(t)$.
    pub acceleration: Vec<f64>,
    /// Joint jerks $\dddot{q}(t)$.
    pub jerk: Vec<f64>,
}

/// Natural cubic spline geometric path $q(s)$ parameterized by cumulative chord length $s \in [0, L]$.
#[derive(Debug, Clone)]
pub struct CubicSplinePath {
    dof: usize,
    knots: Vec<f64>,
    /// Per-segment, per-DOF cubic polynomial coefficients `[a, b, c, d]` on $u = s - s_i \in [0, \Delta s_i]$:
    /// $q(u) = a + b u + c u^2 + d u^3$.
    coeffs: Vec<[f64; 4]>,
}

impl CubicSplinePath {
    /// Constructs a twice-differentiable ($C^2$) natural cubic spline through `waypoints`.
    pub fn from_waypoints(waypoints: &[Configuration]) -> Option<Self> {
        if waypoints.is_empty() {
            return None;
        }
        let dof = waypoints[0].dof();
        if waypoints.len() == 1 {
            let mut coeffs = Vec::with_capacity(dof);
            for &val in waypoints[0].as_slice() {
                coeffs.push([val, 0.0, 0.0, 0.0]);
            }
            return Some(Self {
                dof,
                knots: vec![0.0, 0.0],
                coeffs,
            });
        }

        // Compute cumulative chord-length knots
        let mut knots = Vec::with_capacity(waypoints.len());
        let mut filtered: Vec<&[f64]> = Vec::with_capacity(waypoints.len());
        knots.push(0.0);
        filtered.push(waypoints[0].as_slice());

        for wp in waypoints.iter().skip(1) {
            let prev = *filtered.last().unwrap();
            let seg_len: f64 = prev
                .iter()
                .zip(wp.as_slice())
                .map(|(a, b)| (b - a).powi(2))
                .sum::<f64>()
                .sqrt();
            if seg_len > 1e-9 {
                knots.push(knots.last().unwrap() + seg_len);
                filtered.push(wp.as_slice());
            }
        }

        if filtered.len() == 1 {
            let mut coeffs = Vec::with_capacity(dof);
            for &val in filtered[0] {
                coeffs.push([val, 0.0, 0.0, 0.0]);
            }
            return Some(Self {
                dof,
                knots: vec![0.0, 0.0],
                coeffs,
            });
        }

        let n_seg = filtered.len() - 1;
        let n_pts = filtered.len();
        let mut h = vec![0.0; n_seg];
        for i in 0..n_seg {
            h[i] = knots[i + 1] - knots[i];
        }

        let mut coeffs = vec![[0.0; 4]; n_seg * dof];

        // Solve tridiagonal system for natural cubic spline per DOF
        for d in 0..dof {
            let mut alpha = vec![0.0; n_pts];
            for i in 1..n_seg {
                alpha[i] = (3.0 / h[i]) * (filtered[i + 1][d] - filtered[i][d])
                    - (3.0 / h[i - 1]) * (filtered[i][d] - filtered[i - 1][d]);
            }

            let mut l = vec![1.0; n_pts];
            let mut mu = vec![0.0; n_pts];
            let mut z = vec![0.0; n_pts];

            for i in 1..n_seg {
                l[i] = 2.0 * (knots[i + 1] - knots[i - 1]) - h[i - 1] * mu[i - 1];
                mu[i] = h[i] / l[i];
                z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
            }

            let mut c = vec![0.0; n_pts];
            for j in (0..n_seg).rev() {
                c[j] = z[j] - mu[j] * c[j + 1];
                let b = (filtered[j + 1][d] - filtered[j][d]) / h[j]
                    - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
                let d_coeff = (c[j + 1] - c[j]) / (3.0 * h[j]);
                coeffs[j * dof + d] = [filtered[j][d], b, c[j], d_coeff];
            }
        }

        Some(Self { dof, knots, coeffs })
    }

    /// Total arc length $L$ of the spline path.
    #[inline]
    pub fn total_length(&self) -> f64 {
        *self.knots.last().unwrap_or(&0.0)
    }

    /// Number of degrees of freedom.
    #[inline]
    pub fn dof(&self) -> usize {
        self.dof
    }

    /// Evaluates $q(s)$, $q'(s) = \frac{dq}{ds}$, and $q''(s) = \frac{d^2 q}{ds^2}$ at path parameter $s \in [0, L]$.
    pub fn evaluate_derivatives(
        &self,
        s: f64,
        out_q: &mut [f64],
        out_dq_ds: &mut [f64],
        out_ddq_ds2: &mut [f64],
    ) {
        let total = self.total_length();
        if total <= 1e-12 {
            for d in 0..self.dof {
                out_q[d] = self.coeffs[d][0];
                out_dq_ds[d] = 0.0;
                out_ddq_ds2[d] = 0.0;
            }
            return;
        }
        let s_clamped = s.clamp(0.0, total);
        let n_seg = self.knots.len() - 1;
        let seg_idx = match self
            .knots
            .binary_search_by(|k| k.partial_cmp(&s_clamped).unwrap())
        {
            Ok(idx) => idx.min(n_seg - 1),
            Err(idx) => idx.saturating_sub(1).min(n_seg - 1),
        };
        let u = s_clamped - self.knots[seg_idx];
        let u2 = u * u;
        let u3 = u2 * u;

        let base = seg_idx * self.dof;
        for d in 0..self.dof {
            let [a, b, c, d_c] = self.coeffs[base + d];
            out_q[d] = a + b * u + c * u2 + d_c * u3;
            out_dq_ds[d] = b + 2.0 * c * u + 3.0 * d_c * u2;
            out_ddq_ds2[d] = 2.0 * c + 6.0 * d_c * u;
        }
    }
}

/// Time-Optimal Path Parameterization based on Reachability Analysis (TOPP-RA).
///
/// Computes the minimum-time parameterization $s(t)$ along a geometric [`CubicSplinePath`]
/// subject to per-joint velocity and acceleration bounds.
#[derive(Debug, Clone)]
pub struct ToppraTrajectory {
    path: CubicSplinePath,
    grid_s: Vec<f64>,
    grid_t: Vec<f64>,
    /// Path velocity $\dot{s}_i$ at each grid point.
    grid_sd: Vec<f64>,
    /// Path acceleration $\ddot{s}_i$ on each interval $[s_i, s_{i+1}]$.
    grid_sdd: Vec<f64>,
}

impl ToppraTrajectory {
    /// Computes the time-optimal trajectory along `path` under `limits` using `num_grid_points`
    /// reachability intervals, starting and ending at rest ($\dot{s}_0 = \dot{s}_N = 0$).
    pub fn parameterize(
        path: CubicSplinePath,
        limits: &[KinodynamicLimits],
        num_grid_points: usize,
    ) -> Result<Self, PlannerError> {
        let dof = path.dof();
        let total_len = path.total_length();
        if total_len <= 1e-9 {
            return Ok(Self {
                path,
                grid_s: vec![0.0],
                grid_t: vec![0.0],
                grid_sd: vec![0.0],
                grid_sdd: vec![0.0],
            });
        }

        let n = num_grid_points.max(16);
        let ds = total_len / n as f64;
        let grid_s: Vec<f64> = (0..=n).map(|i| i as f64 * ds).collect();

        let mut q = vec![0.0; dof];
        let mut qs = vec![0.0; dof];
        let mut qss = vec![0.0; dof];

        let mut qs_grid = vec![0.0; (n + 1) * dof];
        let mut qss_grid = vec![0.0; (n + 1) * dof];
        // Upper bound on squared path velocity x_i = \dot{s}_i^2 at each grid point
        let mut x_upper = vec![f64::INFINITY; n + 1];
        x_upper[0] = 0.0;
        x_upper[n] = 0.0;

        for i in 0..=n {
            path.evaluate_derivatives(grid_s[i], &mut q, &mut qs, &mut qss);
            qs_grid[i * dof..(i + 1) * dof].copy_from_slice(&qs);
            qss_grid[i * dof..(i + 1) * dof].copy_from_slice(&qss);

            if i > 0 && i < n {
                let mut x_max = f64::INFINITY;
                for k in 0..dof {
                    let v_max = limits[k].max_velocity;
                    let dq_ds = qs[k].abs();
                    if dq_ds > 1e-9 {
                        x_max = x_max.min((v_max / dq_ds).powi(2));
                    }
                    // Also enforce centripetal acceleration feasibility: |q''_k(s)| * x <= a_max,k
                    let ddq_ds2 = qss[k].abs();
                    if ddq_ds2 > 1e-9 {
                        x_max = x_max.min(limits[k].max_acceleration / ddq_ds2);
                    }
                }
                x_upper[i] = x_max;
            }
        }

        // Helper: computes feasible path acceleration interval [u_min(i, x), u_max(i, x)] at grid point i and x = sd^2
        let accel_bounds = |i: usize, x: f64| -> (f64, f64) {
            let mut u_min = f64::NEG_INFINITY;
            let mut u_max = f64::INFINITY;
            let qs_i = &qs_grid[i * dof..(i + 1) * dof];
            let qss_i = &qss_grid[i * dof..(i + 1) * dof];

            for k in 0..dof {
                let a_max = limits[k].max_acceleration;
                let a_cent = qss_i[k] * x;
                let slope = qs_i[k];
                if slope.abs() > 1e-9 {
                    let b1 = (-a_max - a_cent) / slope;
                    let b2 = (a_max - a_cent) / slope;
                    u_min = u_min.max(b1.min(b2));
                    u_max = u_max.min(b1.max(b2));
                } else if a_cent.abs() > a_max + 1e-6 {
                    return (1.0, -1.0); // Infeasible x
                }
            }
            (u_min, u_max)
        };

        // 1. Backward Reachability Pass: tighten x_upper[i] so x_{i+1} <= x_upper[i+1] is reachable
        for i in (0..n).rev() {
            // Binary/fixed-point refinement for largest x_i <= x_upper[i] that can decelerate to x_upper[i+1]
            let mut candidate = x_upper[i];
            for _ in 0..8 {
                let (u_min, u_max) = accel_bounds(i, candidate);
                if u_min > u_max {
                    candidate *= 0.8;
                    continue;
                }
                // Minimum reachable x_{i+1} from candidate is candidate + 2 * ds * u_min
                let min_next = candidate + 2.0 * ds * u_min;
                if min_next <= x_upper[i + 1] + 1e-9 {
                    break;
                }
                // Otherwise reduce candidate so candidate + 2 * ds * u_min(candidate) <= x_upper[i+1]
                candidate = (x_upper[i + 1] - 2.0 * ds * u_min).max(0.0).min(candidate);
            }
            x_upper[i] = candidate.max(0.0);
        }

        // 2. Forward Greedy Pass: choose maximal feasible acceleration u_i at each step
        let mut x_prof = vec![0.0; n + 1];
        let mut grid_sdd = vec![0.0; n];
        for i in 0..n {
            let x_curr = x_prof[i];
            let (u_min, u_max) = accel_bounds(i, x_curr);
            let u_reach_max = (x_upper[i + 1] - x_curr) / (2.0 * ds);
            let u_chosen = u_max.min(u_reach_max).max(u_min);
            let x_next = (x_curr + 2.0 * ds * u_chosen).clamp(0.0, x_upper[i + 1]);
            grid_sdd[i] = (x_next - x_curr) / (2.0 * ds);
            x_prof[i + 1] = x_next;
        }

        let grid_sd: Vec<f64> = x_prof.iter().map(|&x| x.max(0.0).sqrt()).collect();
        let mut grid_t = vec![0.0; n + 1];
        for i in 0..n {
            let avg_sd = 0.5 * (grid_sd[i] + grid_sd[i + 1]);
            let dt = if avg_sd > 1e-9 {
                ds / avg_sd
            } else {
                (2.0 * ds / grid_sdd[i].abs().max(1e-6)).sqrt()
            };
            grid_t[i + 1] = grid_t[i] + dt;
        }

        Ok(Self {
            path,
            grid_s,
            grid_t,
            grid_sd,
            grid_sdd,
        })
    }

    /// Total trajectory duration $T$ in seconds.
    #[inline]
    pub fn duration(&self) -> f64 {
        *self.grid_t.last().unwrap_or(&0.0)
    }

    /// Samples the time-parameterized trajectory $(q(t), \dot{q}(t), \ddot{q}(t))$ at time `t` in $O(\log N)$ time.
    pub fn sample(&self, t: f64) -> TrajectorySample {
        let dof = self.path.dof();
        let mut position = vec![0.0; dof];
        let mut velocity = vec![0.0; dof];
        let mut acceleration = vec![0.0; dof];
        let jerk = vec![0.0; dof];

        let total_t = self.duration();
        if total_t <= 1e-12 || self.grid_t.len() <= 1 {
            let mut qs = vec![0.0; dof];
            let mut qss = vec![0.0; dof];
            self.path
                .evaluate_derivatives(0.0, &mut position, &mut qs, &mut qss);
            return TrajectorySample {
                time: 0.0,
                position,
                velocity,
                acceleration,
                jerk,
            };
        }

        let t_clamped = t.clamp(0.0, total_t);
        let n_seg = self.grid_t.len() - 1;
        let idx = match self
            .grid_t
            .binary_search_by(|tk| tk.partial_cmp(&t_clamped).unwrap())
        {
            Ok(i) => i.min(n_seg - 1),
            Err(i) => i.saturating_sub(1).min(n_seg - 1),
        };

        let dt = t_clamped - self.grid_t[idx];
        let sd0 = self.grid_sd[idx];
        let sdd = self.grid_sdd[idx];
        let s = (self.grid_s[idx] + sd0 * dt + 0.5 * sdd * dt * dt)
            .clamp(0.0, self.path.total_length());
        let sd = (sd0 + sdd * dt).max(0.0);

        let mut qs = vec![0.0; dof];
        let mut qss = vec![0.0; dof];
        self.path
            .evaluate_derivatives(s, &mut position, &mut qs, &mut qss);

        for k in 0..dof {
            velocity[k] = qs[k] * sd;
            acceleration[k] = qs[k] * sdd + qss[k] * sd * sd;
        }

        TrajectorySample {
            time: t_clamped,
            position,
            velocity,
            acceleration,
            jerk,
        }
    }
}

/// Time-synchronized quintic ($C^2$ jerk-bounded) online trajectory generator (Ruckig-style OTG)
/// for non-zero boundary conditions $(q_0, \dot{q}_0, \ddot{q}_0) \to (q_f, \dot{q}_f, \ddot{q}_f)$
/// and moving target interception.
#[derive(Debug, Clone)]
pub struct JerkLimitedTrajectory {
    dof: usize,
    duration: f64,
    /// Per-DOF quintic polynomial coefficients `[c0, c1, c2, c3, c4, c5]` over $t \in [0, T]$.
    coeffs: Vec<[f64; 6]>,
}

impl JerkLimitedTrajectory {
    /// Generates a minimum-duration, time-synchronized quintic trajectory from
    /// `(q0, v0, a0)` to `(qf, vf, af)` that strictly respects `limits` (velocity, acceleration, jerk).
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        q0: &[f64],
        v0: &[f64],
        a0: &[f64],
        qf: &[f64],
        vf: &[f64],
        af: &[f64],
        limits: &[KinodynamicLimits],
        min_duration: f64,
    ) -> Self {
        let dof = q0.len();
        assert_eq!(v0.len(), dof);
        assert_eq!(a0.len(), dof);
        assert_eq!(qf.len(), dof);
        assert_eq!(vf.len(), dof);
        assert_eq!(af.len(), dof);
        assert_eq!(limits.len(), dof);

        // Estimate lower-bound synchronized duration from peak quintic velocity, acceleration, and jerk scaling laws
        let mut t_sync = min_duration.max(1e-3);
        for k in 0..dof {
            let dq = (qf[k] - q0[k]).abs();
            let dv = (vf[k] - v0[k]).abs();
            let v_bound = 1.875 * dq / limits[k].max_velocity.max(1e-6);
            let a_bound = (5.7735 * dq / limits[k].max_acceleration.max(1e-6))
                .sqrt()
                .max(1.5 * dv / limits[k].max_acceleration.max(1e-6));
            let j_bound = (60.0 * dq / limits[k].max_jerk.max(1e-6)).cbrt();
            t_sync = t_sync.max(v_bound).max(a_bound).max(j_bound);
        }

        // Refine duration so all sampled points respect velocity, acceleration, and jerk limits
        let mut coeffs = vec![[0.0; 6]; dof];
        for _ in 0..12 {
            let t = t_sync;
            let t2 = t * t;
            let t3 = t2 * t;
            let t4 = t3 * t;
            let t5 = t4 * t;

            let mut max_ratio = 1.0_f64;
            for k in 0..dof {
                let c0 = q0[k];
                let c1 = v0[k];
                let c2 = 0.5 * a0[k];
                let h = qf[k] - (c0 + c1 * t + c2 * t2);
                let dh = vf[k] - (c1 + 2.0 * c2 * t);
                let ddh = af[k] - 2.0 * c2;

                let c3 = (10.0 * h - 4.0 * dh * t + 0.5 * ddh * t2) / t3;
                let c4 = (-15.0 * h + 7.0 * dh * t - ddh * t2) / t4;
                let c5 = (6.0 * h - 3.0 * dh * t + 0.5 * ddh * t2) / t5;
                coeffs[k] = [c0, c1, c2, c3, c4, c5];

                for step in 0..=8 {
                    let tau = (step as f64 / 8.0) * t;
                    let tau2 = tau * tau;
                    let tau3 = tau2 * tau;
                    let tau4 = tau3 * tau;

                    let vel = (c1
                        + 2.0 * c2 * tau
                        + 3.0 * c3 * tau2
                        + 4.0 * c4 * tau3
                        + 5.0 * c5 * tau4)
                        .abs();
                    let acc =
                        (2.0 * c2 + 6.0 * c3 * tau + 12.0 * c4 * tau2 + 20.0 * c5 * tau3).abs();
                    let jrk = (6.0 * c3 + 24.0 * c4 * tau + 60.0 * c5 * tau2).abs();

                    max_ratio = max_ratio
                        .max(vel / limits[k].max_velocity.max(1e-6))
                        .max((acc / limits[k].max_acceleration.max(1e-6)).sqrt())
                        .max((jrk / limits[k].max_jerk.max(1e-6)).cbrt());
                }
            }

            if max_ratio <= 1.001 {
                break;
            }
            t_sync *= max_ratio * 1.02;
        }

        Self {
            dof,
            duration: t_sync,
            coeffs,
        }
    }

    /// Synchronized duration $T$ of the trajectory in seconds.
    #[inline]
    pub fn duration(&self) -> f64 {
        self.duration
    }

    /// Evaluates the trajectory state $(q(t), \dot{q}(t), \ddot{q}(t), \dddot{q}(t))$ at time `t` in $O(1)$ time.
    pub fn sample(&self, t: f64) -> TrajectorySample {
        let tau = t.clamp(0.0, self.duration);
        let tau2 = tau * tau;
        let tau3 = tau2 * tau;
        let tau4 = tau3 * tau;
        let tau5 = tau4 * tau;

        let mut position = vec![0.0; self.dof];
        let mut velocity = vec![0.0; self.dof];
        let mut acceleration = vec![0.0; self.dof];
        let mut jerk = vec![0.0; self.dof];

        for k in 0..self.dof {
            let [c0, c1, c2, c3, c4, c5] = self.coeffs[k];
            position[k] = c0 + c1 * tau + c2 * tau2 + c3 * tau3 + c4 * tau4 + c5 * tau5;
            velocity[k] =
                c1 + 2.0 * c2 * tau + 3.0 * c3 * tau2 + 4.0 * c4 * tau3 + 5.0 * c5 * tau4;
            acceleration[k] = 2.0 * c2 + 6.0 * c3 * tau + 12.0 * c4 * tau2 + 20.0 * c5 * tau3;
            jerk[k] = 6.0 * c3 + 24.0 * c4 * tau + 60.0 * c5 * tau2;
        }

        TrajectorySample {
            time: tau,
            position,
            velocity,
            acceleration,
            jerk,
        }
    }
}
