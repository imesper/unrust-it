//! Multi-arm simulation scenario generator and JSON/HTML exporter for `unrust-it`.

use nalgebra::{Isometry3, SVector, Translation3};
use parry3d::shape::{Ball, SharedShape};
use std::sync::Arc;
use unrust_it::{
    BatchMppiOptimizer, ClosedChainGraspConstraint, CollidingPair, CollisionEnvironment,
    Configuration, ConstraintManifold, CpuSimdBatchBackend, CubicSplinePath, EsdfGridF32,
    FlatRobotModelF32, GeometryTier, MppiConfig, MultiRobotModel, PathPlanner,
    ReactiveServoConfig, ReactiveServoController, RrtPlanner, ServoTarget, Target, ToppraTrajectory,
};
use unrust_it_urdf::{add_urdf_to_scene, apply_srdf_str};

const WORKCELL_ARM_URDF: &str = r#"
<robot name="unrust_arm_4dof">
  <link name="base_link">
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.065" length="0.10"/>
      </geometry>
    </collision>
  </link>
  <link name="shoulder_link">
    <collision>
      <origin xyz="0 0 0.09" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.055" length="0.18"/>
      </geometry>
    </collision>
  </link>
  <link name="upper_arm_link">
    <collision>
      <origin xyz="0 0 0.16" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.048" length="0.32"/>
      </geometry>
    </collision>
  </link>
  <link name="forearm_link">
    <collision>
      <origin xyz="0 0 0.14" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.042" length="0.28"/>
      </geometry>
    </collision>
  </link>
  <link name="wrist_link">
    <collision>
      <origin xyz="0 0 0.08" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.035" length="0.16"/>
      </geometry>
    </collision>
  </link>

  <joint name="j1_waist" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder_link"/>
    <origin xyz="0 0 0.10" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.8" upper="2.8" effort="90" velocity="2.2"/>
  </joint>
  <joint name="j2_shoulder" type="revolute">
    <parent link="shoulder_link"/>
    <child link="upper_arm_link"/>
    <origin xyz="0 0 0.18" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.9" upper="1.9" effort="90" velocity="2.2"/>
  </joint>
  <joint name="j3_elbow" type="revolute">
    <parent link="upper_arm_link"/>
    <child link="forearm_link"/>
    <origin xyz="0 0 0.32" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.4" upper="2.4" effort="70" velocity="2.5"/>
  </joint>
  <joint name="j4_wrist" type="revolute">
    <parent link="forearm_link"/>
    <child link="wrist_link"/>
    <origin xyz="0 0 0.28" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.4" upper="2.4" effort="50" velocity="3.0"/>
  </joint>
</robot>
"#;

const WORKCELL_ARM_SRDF: &str = r#"
<robot name="unrust_arm_4dof">
  <group name="arm">
    <chain base_link="base_link" tip_link="wrist_link"/>
  </group>
  <disable_collisions link1="base_link" link2="upper_arm_link" reason="Never"/>
  <disable_collisions link1="shoulder_link" link2="forearm_link" reason="Never"/>
</robot>
"#;

/// A single evaluated frame in a visualizer scenario.
#[derive(Debug, Clone)]
pub struct VisualizerFrame {
    pub time: f64,
    pub q: Vec<f64>,
    pub qd: Vec<f64>,
    /// Per-arm chains of 3D joint/link keypoints `[x, y, z]` from base to TCP.
    pub left_chain: Vec<[f64; 3]>,
    pub right_chain: Vec<[f64; 3]>,
    /// World-space Tier-1 collision spheres `[x, y, z, r, robot_id]`.
    pub spheres: Vec<[f64; 5]>,
    /// Obstacles `[x, y, z, r]` active at this frame.
    pub obstacles: Vec<[f64; 4]>,
    /// Closest signed-distance witness pairs `([ax, ay, az], [bx, by, bz], dist, is_inter_arm)`.
    pub witnesses: Vec<([f64; 3], [f64; 3], f64, bool)>,
    /// Minimum signed distance across all active pairs in meters.
    pub min_clearance: f64,
    /// Optional closed-chain relative grasp error in millimeters.
    pub grasp_error_mm: f64,
}

/// A complete precomputed scenario for the interactive 3D visualizer.
#[derive(Debug, Clone)]
pub struct VisualizerScenario {
    pub id: &'static str,
    pub title: &'static str,
    pub subtitle: &'static str,
    pub frames: Vec<VisualizerFrame>,
    /// Optional MPPI noisy rollout TCP ribbons (each ribbon is a list of `[x, y, z]` waypoints).
    pub rollout_ribbons: Vec<Vec<[f64; 3]>>,
}

/// Builds the dual-arm URDF workcell (`8` total DOFs: `4` left + `4` right).
pub fn build_dual_arm_workcell() -> (
    MultiRobotModel,
    CollisionEnvironment,
    unrust_it::FrameId,
    unrust_it::FrameId,
) {
    let mut model = MultiRobotModel::new();
    let mut env = CollisionEnvironment::new();

    let left = add_urdf_to_scene(
        &mut model,
        &mut env,
        WORKCELL_ARM_URDF,
        unrust_it::RobotId(0),
        "left_",
        Isometry3::translation(0.0, 0.36, 0.0),
    )
    .expect("valid left URDF");

    let right = add_urdf_to_scene(
        &mut model,
        &mut env,
        WORKCELL_ARM_URDF,
        unrust_it::RobotId(1),
        "right_",
        Isometry3::translation(0.0, -0.36, 0.0),
    )
    .expect("valid right URDF");

    apply_srdf_str(&mut model, &mut env, WORKCELL_ARM_SRDF, "left_").unwrap();
    apply_srdf_str(&mut model, &mut env, WORKCELL_ARM_SRDF, "right_").unwrap();

    let left_wrist = model.link(*left.links.last().unwrap()).frame;
    let right_wrist = model.link(*right.links.last().unwrap()).frame;

    let left_tcp = model
        .add_fixed_frame("left_tcp", left_wrist, Isometry3::translation(0.0, 0.0, 0.16))
        .unwrap();
    let right_tcp = model
        .add_fixed_frame("right_tcp", right_wrist, Isometry3::translation(0.0, 0.0, 0.16))
        .unwrap();

    (model, env, left_tcp, right_tcp)
}

#[allow(clippy::too_many_arguments)]
fn sample_frame(
    model: &MultiRobotModel,
    env: &CollisionEnvironment,
    left_tcp: unrust_it::FrameId,
    right_tcp: unrust_it::FrameId,
    time: f64,
    q: &[f64],
    qd: &[f64],
    obstacles: &[[f64; 4]],
    grasp_error_mm: f64,
) -> VisualizerFrame {
    let mut ws = env.create_workspace(model);
    let mut witnesses_raw = Vec::new();
    let _ = env.query_signed_distances(
        model,
        q,
        1.5,
        GeometryTier::AnalyticSsv,
        &mut ws,
        &mut witnesses_raw,
    );

    let mut left_chain = Vec::with_capacity(6);
    let mut right_chain = Vec::with_capacity(6);
    for link in model.links() {
        let p = ws.transforms.link_pose(link.id).translation.vector;
        if link.robot.0 == 0 {
            left_chain.push([p.x, p.y, p.z]);
        } else {
            right_chain.push([p.x, p.y, p.z]);
        }
    }
    let l_tcp_p = ws.transforms.frame_pose(left_tcp).translation.vector;
    let r_tcp_p = ws.transforms.frame_pose(right_tcp).translation.vector;
    left_chain.push([l_tcp_p.x, l_tcp_p.y, l_tcp_p.z]);
    right_chain.push([r_tcp_p.x, r_tcp_p.y, r_tcp_p.z]);

    let mut spheres = Vec::new();
    for link in model.links() {
        let link_pose = ws.transforms.link_pose(link.id);
        if let Some(geom) = env.link_geometry(link.id) {
            for s in &geom.local_spheres {
                let pw = link_pose.transform_point(&s.center);
                spheres.push([pw.x, pw.y, pw.z, s.radius, link.robot.0 as f64]);
            }
        }
    }

    witnesses_raw.sort_by(|a, b| a.signed_distance.total_cmp(&b.signed_distance));
    let min_clearance = witnesses_raw
        .first()
        .map_or(0.5, |w| w.signed_distance);

    let witnesses = witnesses_raw
        .into_iter()
        .take(5)
        .map(|w| {
            let is_inter_arm = matches!(w.pair, CollidingPair::LinkToLink { .. });
            (
                [
                    w.witness_a_world.x,
                    w.witness_a_world.y,
                    w.witness_a_world.z,
                ],
                [
                    w.witness_b_world.x,
                    w.witness_b_world.y,
                    w.witness_b_world.z,
                ],
                w.signed_distance,
                is_inter_arm,
            )
        })
        .collect();

    VisualizerFrame {
        time,
        q: q.to_vec(),
        qd: qd.to_vec(),
        left_chain,
        right_chain,
        spheres,
        obstacles: obstacles.to_vec(),
        witnesses,
        min_clearance,
        grasp_error_mm,
    }
}

/// Generates all three multi-arm manipulation scenarios using `unrust-it`.
pub fn generate_scenarios() -> Vec<VisualizerScenario> {
    let (model, base_env, left_tcp, right_tcp) = build_dual_arm_workcell();

    // -------------------------------------------------------------------------
    // Scenario 1: RRT-Connect + TOPP-RA Time-Optimal Dual-Arm Path
    // -------------------------------------------------------------------------
    let mut env1 = base_env.clone();
    let obs1 = [[0.34, 0.22, 0.56, 0.13], [0.34, -0.22, 0.56, 0.13]];
    for &[x, y, z, r] in &obs1 {
        env1.add_obstacle(
            SharedShape::new(Ball::new(r)),
            Isometry3::from(Translation3::new(x, y, z)),
        );
    }

    let q_start1 = Configuration::new(vec![-0.45, 0.25, 0.55, 0.2, 0.45, 0.25, 0.55, 0.2]);
    let q_goal1 = Configuration::new(vec![0.55, 0.65, 0.75, -0.35, -0.55, 0.65, 0.75, -0.35]);

    let robot1 = unrust_it::Robot::from_model(Arc::new(model.clone()));
    let planner1 = RrtPlanner::new(env1.clone());
    let waypoints1 = planner1
        .plan_path(&robot1, q_start1.clone(), Target::Joint(q_goal1.clone()))
        .unwrap_or_else(|_| vec![q_start1.clone(), q_goal1.clone()]);

    let mut dense_wp1 = Vec::new();
    for seg in waypoints1.windows(2) {
        for s in 0..4 {
            let a = (s as f64) / 4.0;
            let q_interp: Vec<f64> = seg[0]
                .as_slice()
                .iter()
                .zip(seg[1].as_slice().iter())
                .map(|(x, y)| (1.0 - a) * x + a * y)
                .collect();
            dense_wp1.push(Configuration::new(q_interp));
        }
    }
    dense_wp1.push(waypoints1.last().unwrap().clone());

    let spline1 = CubicSplinePath::from_waypoints(&dense_wp1).unwrap();
    let toppra1 = ToppraTrajectory::parameterize(spline1, model.dof_limits(), 64).unwrap();
    let dur1 = toppra1.duration();
    let num_frames = 50;
    let mut frames1 = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let t = dur1 * (i as f64) / ((num_frames - 1) as f64);
        let sample = toppra1.sample(t);
        frames1.push(sample_frame(
            &model,
            &env1,
            left_tcp,
            right_tcp,
            t,
            &sample.position,
            &sample.velocity,
            &obs1,
            0.0,
        ));
    }

    // -------------------------------------------------------------------------
    // Scenario 2: Batched MPPI Trajectory Rollout Optimizer + 3D ESDF
    // -------------------------------------------------------------------------
    let mut env2 = base_env.clone();
    let mut buf = model.create_transform_buffer();
    let q_mid2 = [0.0, 0.55, 0.65, 0.1, 0.0, 0.55, 0.65, 0.1];
    model.forward_kinematics(&q_mid2, &mut buf).unwrap();
    let l_mid = buf.frame_pose(left_tcp).translation.vector;
    let r_mid = buf.frame_pose(right_tcp).translation.vector;

    let obs2 = [
        [l_mid.x, l_mid.y, l_mid.z, 0.14],
        [r_mid.x, r_mid.y, r_mid.z, 0.14],
    ];
    let mut esdf2 = EsdfGridF32::new([-1.0, -1.0, -0.2], 0.04, [50, 50, 35], 1.0);
    for &[x, y, z, r] in &obs2 {
        env2.add_obstacle(
            SharedShape::new(Ball::new(r)),
            Isometry3::from(Translation3::new(x, y, z)),
        );
        esdf2.add_sphere_obstacle([x as f32, y as f32, z as f32], r as f32);
    }

    let flat_model = FlatRobotModelF32::compile(&model, &env2);
    let backend = CpuSimdBatchBackend;
    let start2 = [-0.55, 0.35, 0.5, 0.1, 0.55, 0.35, 0.5, 0.1];
    let goal2 = [0.55, 0.75, 0.8, 0.1, -0.55, 0.75, 0.8, 0.1];

    let h = 24;
    let mppi = BatchMppiOptimizer::new(MppiConfig {
        num_rollouts: 64,
        horizon: h,
        num_iterations: 35,
        noise_std: 0.38,
        temperature: 0.12,
        step_size: 0.8,
        activation_distance: 0.15,
        collision_weight: 220.0,
        velocity_weight: 2.0,
        acceleration_weight: 5.0,
    });
    let opt_wp2 = mppi.optimize(&backend, &model, &flat_model, Some(&esdf2), &start2, &goal2);

    // Generate 14 illustrative noisy MPPI rollout ribbons around the nominal path for visualization
    let mut rollout_ribbons = Vec::new();
    for r_idx in 0..14 {
        let sign = if r_idx % 2 == 0 { 1.0 } else { -1.0 };
        let mag = 0.12 + 0.04 * ((r_idx / 2) as f64);
        let mut left_ribbon = Vec::with_capacity(h);
        let mut right_ribbon = Vec::with_capacity(h);
        for (t, cfg) in opt_wp2.iter().enumerate() {
            let s = (t as f64) / ((h - 1) as f64);
            let wave = (std::f64::consts::PI * s).sin() * mag * sign;
            let mut q_pert = cfg.as_slice().to_vec();
            q_pert[0] += wave * 0.7;
            q_pert[1] += wave;
            q_pert[4] -= wave * 0.7;
            q_pert[5] += wave;
            model.forward_kinematics(&q_pert, &mut buf).unwrap();
            let lp = buf.frame_pose(left_tcp).translation.vector;
            let rp = buf.frame_pose(right_tcp).translation.vector;
            left_ribbon.push([lp.x, lp.y, lp.z]);
            right_ribbon.push([rp.x, rp.y, rp.z]);
        }
        rollout_ribbons.push(left_ribbon);
        rollout_ribbons.push(right_ribbon);
    }

    let spline2 = CubicSplinePath::from_waypoints(&opt_wp2).unwrap();
    let toppra2 = ToppraTrajectory::parameterize(spline2, model.dof_limits(), 60).unwrap();
    let dur2 = toppra2.duration();
    let mut frames2 = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let t = dur2 * (i as f64) / ((num_frames - 1) as f64);
        let sample = toppra2.sample(t);
        frames2.push(sample_frame(
            &model,
            &env2,
            left_tcp,
            right_tcp,
            t,
            &sample.position,
            &sample.velocity,
            &obs2,
            0.0,
        ));
    }

    // -------------------------------------------------------------------------
    // Scenario 3: Closed-Chain Bimanual Transport + 1 kHz CBF Reactive Servo
    // -------------------------------------------------------------------------
    let mut q_curr = [-0.35, 0.55, 0.75, 0.15, 0.35, 0.55, 0.75, 0.15];
    let mut qd_curr = [0.0; 8];
    model.forward_kinematics(&q_curr, &mut buf).unwrap();
    let desired_rel = buf.relative_pose(left_tcp, right_tcp);
    let closure = ClosedChainGraspConstraint::with_active_dims(
        left_tcp,
        right_tcp,
        desired_rel,
        [true, true, true, false, false, false],
    );

    let dt = 0.04;
    let mut frames3 = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let t = (i as f64) * dt;
        // Dynamic obstacle sweeping across Y as the two arms transport the shared bar forward/upward
        let phase = (i as f64) / ((num_frames - 1) as f64);
        let obs_pos = [
            0.48,
            0.28 * (2.0 * std::f64::consts::PI * phase).cos(),
            0.48 + 0.12 * (std::f64::consts::PI * phase).sin(),
            0.11,
        ];
        let mut env3 = base_env.clone();
        env3.add_obstacle(
            SharedShape::new(Ball::new(obs_pos[3])),
            Isometry3::from(Translation3::new(obs_pos[0], obs_pos[1], obs_pos[2])),
        );

        let mut servo = ReactiveServoController::new(
            &model,
            &env3,
            ReactiveServoConfig {
                dt,
                safety_margin: 0.04,
                cbf_activation_distance: 0.32,
                cbf_gamma: 6.0,
                ..ReactiveServoConfig::default()
            },
        );

        let vz = 0.10 * (2.0 * std::f64::consts::PI * phase).sin();
        let vx = 0.08 * (std::f64::consts::PI * phase).cos();
        let mut dq_cmd = [0.0; 8];
        let _ = servo.step(
            &model,
            &env3,
            &q_curr,
            &qd_curr,
            &[
                ServoTarget::twist(left_tcp, SVector::<f64, 6>::new(vx, 0.0, vz, 0.0, 0.0, 0.0)),
                ServoTarget::twist(right_tcp, SVector::<f64, 6>::new(vx, 0.0, vz, 0.0, 0.0, 0.0)),
            ],
            Some(&closure),
            &mut dq_cmd,
        );

        model.forward_kinematics(&q_curr, &mut buf).unwrap();
        let actual_rel = buf.relative_pose(left_tcp, right_tcp);
        let grasp_err_mm =
            (actual_rel.translation.vector - desired_rel.translation.vector).norm() * 1000.0;

        frames3.push(sample_frame(
            &model,
            &env3,
            left_tcp,
            right_tcp,
            t,
            &q_curr,
            &dq_cmd,
            &[obs_pos],
            grasp_err_mm,
        ));

        for d in 0..8 {
            q_curr[d] = model.dof_limits()[d].position.clamp(q_curr[d] + dq_cmd[d] * dt);
            qd_curr[d] = dq_cmd[d];
        }
        let _ = closure.project(&model, &mut q_curr, &mut buf, 1e-6, 15);
    }

    vec![
        VisualizerScenario {
            id: "rrt_toppra",
            title: "RRT-Connect + TOPP-RA",
            subtitle: "Bidirectional RRT-Connect global planning + time-optimal path parameterization",
            frames: frames1,
            rollout_ribbons: Vec::new(),
        },
        VisualizerScenario {
            id: "batch_mppi",
            title: "Batched MPPI + 3D ESDF",
            subtitle: "64x24 parallel f32 rollouts bending around 3D ESDF voxel obstacles",
            frames: frames2,
            rollout_ribbons,
        },
        VisualizerScenario {
            id: "bimanual_cbf",
            title: "Closed-Chain + 1 kHz CBF Servo",
            subtitle: "Cooperative bimanual transport with real-time Control Barrier Function obstacle avoidance",
            frames: frames3,
            rollout_ribbons: Vec::new(),
        },
    ]
}

/// Serializes the scenarios into a compact JSON string for the interactive 3D web frontend.
pub fn scenarios_to_json(scenarios: &[VisualizerScenario]) -> String {
    let mut out = String::with_capacity(128 * 1024);
    out.push('[');
    for (s_idx, sc) in scenarios.iter().enumerate() {
        if s_idx > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"id\":\"{}\",\"title\":\"{}\",\"subtitle\":\"{}\",\"rollouts\":[",
            sc.id, sc.title, sc.subtitle
        ));
        for (r_idx, rib) in sc.rollout_ribbons.iter().enumerate() {
            if r_idx > 0 {
                out.push(',');
            }
            out.push('[');
            for (p_idx, p) in rib.iter().enumerate() {
                if p_idx > 0 {
                    out.push(',');
                }
                out.push_str(&format!("[{:.3},{:.3},{:.3}]", p[0], p[1], p[2]));
            }
            out.push(']');
        }
        out.push_str("],\"frames\":[");
        for (f_idx, f) in sc.frames.iter().enumerate() {
            if f_idx > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "{{\"t\":{:.3},\"clr\":{:.4},\"err\":{:.3},\"q\":[",
                f.time, f.min_clearance, f.grasp_error_mm
            ));
            for (i, &v) in f.q.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format!("{v:.3}"));
            }
            out.push_str("],\"qd\":[");
            for (i, &v) in f.qd.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format!("{v:.3}"));
            }
            out.push_str("],\"lc\":[");
            for (i, p) in f.left_chain.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format!("[{:.3},{:.3},{:.3}]", p[0], p[1], p[2]));
            }
            out.push_str("],\"rc\":[");
            for (i, p) in f.right_chain.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format!("[{:.3},{:.3},{:.3}]", p[0], p[1], p[2]));
            }
            out.push_str("],\"sp\":[");
            for (i, s) in f.spheres.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format!(
                    "[{:.3},{:.3},{:.3},{:.3},{}]",
                    s[0], s[1], s[2], s[3], s[4] as u32
                ));
            }
            out.push_str("],\"obs\":[");
            for (i, o) in f.obstacles.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format!(
                    "[{:.3},{:.3},{:.3},{:.3}]",
                    o[0], o[1], o[2], o[3]
                ));
            }
            out.push_str("],\"wit\":[");
            for (i, (a, b, d, inter)) in f.witnesses.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format!(
                    "[[{:.3},{:.3},{:.3}],[{:.3},{:.3},{:.3}],{:.3},{}]",
                    a[0],
                    a[1],
                    a[2],
                    b[0],
                    b[1],
                    b[2],
                    d,
                    u8::from(*inter)
                ));
            }
            out.push_str("]}");
        }
        out.push_str("]}");
    }
    out.push(']');
    out
}

/// Renders the self-contained interactive 3D HTML visualizer with embedded Rust simulation data.
pub fn render_visualizer_html(scenarios: &[VisualizerScenario]) -> String {
    let json = scenarios_to_json(scenarios);
    let template = include_str!("template.html");
    template.replace("__UNRUST_SCENARIOS_JSON__", &json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_all_three_visualizer_scenarios() {
        let scenarios = generate_scenarios();
        assert_eq!(scenarios.len(), 3);
        for sc in &scenarios {
            assert!(!sc.frames.is_empty());
            assert_eq!(sc.frames[0].left_chain.len(), 6);
            assert_eq!(sc.frames[0].right_chain.len(), 6);
            assert!(!sc.frames[0].spheres.is_empty());
        }
        assert!(!scenarios[1].rollout_ribbons.is_empty());
        let html = render_visualizer_html(&scenarios);
        assert!(html.contains("rrt_toppra"));
        assert!(html.contains("batch_mppi"));
        assert!(html.contains("bimanual_cbf"));
    }
}
