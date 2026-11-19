//! Built-in declarative workcell presets covering Phase 2A–2C robotics capabilities.

use crate::scene::spec::{
    arm_palette_color, ArmSpec, ArmTargetSpec, CollisionRulesSpec, ComputeBackendChoice, EsdfSpec,
    ObstacleRole, ObstacleSpec, PlannerSpec, PoseSpec, PresetId, ShapePrimitiveSpec,
    Stage1Strategy, ToolPreset, UrdfSource, WorkcellSpec,
};
use nalgebra::Vector3;
use std::f64::consts::PI;
use unrust_it::{MppiConfig, ReactiveServoConfig, RobotId};

/// Builds a declarative [`WorkcellSpec`] for one of the built-in multi-arm presets.
pub fn preset_workcell(id: PresetId) -> WorkcellSpec {
    match id {
        PresetId::ThreeArmCloseProximity => three_arm_close_proximity_spec(),
        PresetId::ThreeArmCrossing => three_arm_crossing_spec(),
        PresetId::DualArmMppiEsdf => dual_arm_mppi_esdf_spec(),
        PresetId::DualArmRrtToppra => dual_arm_rrt_toppra_spec(),
        PresetId::DualArmClosedChainCbf => dual_arm_closed_chain_cbf_spec(),
    }
}

/// Returns all built-in preset identifiers.
pub fn all_preset_ids() -> &'static [PresetId] {
    PresetId::ALL
}

fn three_arm_base_arms(targets: [ArmTargetSpec; 3], q_start: Vec<f64>) -> Vec<ArmSpec> {
    let r_base = 0.56;
    let angles = [0.0_f64, 2.0 * PI / 3.0, 4.0 * PI / 3.0];
    let tools = [
        ToolPreset::ParallelGripper,
        ToolPreset::Screwdriver,
        ToolPreset::InspectionProbe,
    ];
    let labels = [
        "Arm 0 (Gripper)",
        "Arm 1 (Screwdriver)",
        "Arm 2 (Probe)",
    ];
    let priorities = [3.0, 2.0, 1.0];

    (0..3)
        .map(|i| {
            let th = angles[i];
            ArmSpec {
                uid: format!("arm{i}"),
                label: labels[i].to_string(),
                urdf_source: UrdfSource::Builtin6Dof,
                base_pose: PoseSpec::from_xyz_yaw(
                    r_base * th.cos(),
                    r_base * th.sin(),
                    0.0,
                    th + PI,
                ),
                flange_link_raw: "flange_link".to_string(),
                default_tcp_z: 0.12,
                tool: tools[i].clone(),
                color_rgba: arm_palette_color(i),
                q_start: q_start.clone(),
                priority: priorities[i],
                target: targets[i].clone(),
            }
        })
        .collect()
}

fn three_arm_close_proximity_spec() -> WorkcellSpec {
    let ready = vec![0.0, -0.35, 1.35, 0.0, 0.70, 0.0];
    let targets = [
        ArmTargetSpec::ApproachPose {
            final_pose: PoseSpec::from_z_approach(
                Vector3::new(0.052, 0.0, 0.438),
                Vector3::new(-0.85, 0.0, -0.52),
            ),
            standoff_local: [0.0, 0.0, -0.06],
        },
        ArmTargetSpec::ApproachPose {
            final_pose: PoseSpec::from_z_approach(
                Vector3::new(0.0, 0.0095, 0.457),
                Vector3::new(0.22, -0.38, -0.90),
            ),
            standoff_local: [0.0, 0.0, -0.06],
        },
        ArmTargetSpec::ApproachPose {
            final_pose: PoseSpec::from_z_approach(
                Vector3::new(0.0, -0.0095, 0.457),
                Vector3::new(0.22, 0.38, -0.90),
            ),
            standoff_local: [0.0, 0.0, -0.06],
        },
    ];

    let servo = ReactiveServoConfig {
        dt: 0.01,
        safety_margin: 0.04,
        cbf_activation_distance: 0.16,
        cbf_gamma: 8.0,
        damping: 0.02,
        posture_gain: 0.05,
        tangential_curl_gain: 0.35,
        qp_max_iterations: 35,
        ..ReactiveServoConfig::default()
    }
    .with_arm_priority(RobotId(0), 3.0)
    .with_arm_priority(RobotId(1), 2.0)
    .with_arm_priority(RobotId(2), 1.0);

    WorkcellSpec {
        preset_id: Some(PresetId::ThreeArmCloseProximity),
        title: PresetId::ThreeArmCloseProximity.title().to_string(),
        subtitle: PresetId::ThreeArmCloseProximity.subtitle().to_string(),
        arms: three_arm_base_arms(targets, ready),
        obstacles: vec![
            ObstacleSpec {
                uid: "pedestal".to_string(),
                label: "Workpiece Pedestal".to_string(),
                shape: ShapePrimitiveSpec::CylinderZ {
                    half_height: 0.195,
                    radius: 0.032,
                },
                pose: PoseSpec::translation(0.0, 0.0, 0.195),
                role: ObstacleRole::Pedestal,
                color_rgba: [0.35, 0.42, 0.52, 1.0],
                safety_margin: None,
                activation_distance: None,
                carve_esdf: true,
            },
            ObstacleSpec {
                uid: "workpiece".to_string(),
                label: "Target Workpiece".to_string(),
                shape: ShapePrimitiveSpec::Cuboid {
                    half_extents: [0.04, 0.04, 0.025],
                },
                pose: PoseSpec::translation(0.0, 0.0, 0.42),
                role: ObstacleRole::Workpiece,
                color_rgba: [0.96, 0.62, 0.14, 1.0],
                safety_margin: Some(0.003),
                activation_distance: Some(0.02),
                carve_esdf: true,
            },
        ],
        rules: CollisionRulesSpec::default(),
        esdf: EsdfSpec::default(),
        planner: PlannerSpec::TwoStageProximity {
            strategy: Stage1Strategy::MppiWithRrtFallback,
            mppi: MppiConfig {
                num_rollouts: 48,
                horizon: 16,
                num_iterations: 20,
                noise_std: 0.20,
                temperature: 0.12,
                step_size: 0.75,
                activation_distance: 0.08,
                collision_weight: 160.0,
                velocity_weight: 2.0,
                acceleration_weight: 5.0,
            },
            rrt_step_size: 0.35,
            rrt_max_workspace_step: 0.025,
            rrt_max_iterations: 2500,
            ik_num_seeds: 48,
            ik_max_iterations: 120,
            ik_tolerance: 5e-4,
            servo,
            approach_pose_gain: 6.0,
            max_refinement_steps: 450,
            final_pos_tol: 1.8e-3,
            final_rot_tol: 1.8e-2,
            waypoint_stride: 4,
        },
        compute_backend: ComputeBackendChoice::CpuSimd,
    }
}

fn three_arm_crossing_spec() -> WorkcellSpec {
    let ready = vec![0.0, -0.35, 1.35, 0.0, 0.70, 0.0];
    let start_tips = [
        Vector3::new(0.07, -0.04, 0.46),
        Vector3::new(-0.02, 0.08, 0.46),
        Vector3::new(-0.06, -0.06, 0.46),
    ];
    let goal_tips = [
        Vector3::new(0.01, 0.06, 0.40),
        Vector3::new(0.04, -0.03, 0.48),
        Vector3::new(-0.06, 0.02, 0.44),
    ];
    let sector_angles = [0.0_f64, 2.0 * PI / 3.0, 4.0 * PI / 3.0];

    let mut start_poses = Vec::with_capacity(3);
    let mut goal_targets = Vec::with_capacity(3);
    for i in 0..3 {
        let s = sector_angles[i];
        let z_in = Vector3::new(-0.42 * s.cos(), -0.42 * s.sin(), -0.91);
        start_poses.push(PoseSpec::from_z_approach(start_tips[i], z_in));
        goal_targets.push(ArmTargetSpec::DirectPose(PoseSpec::from_z_approach(
            goal_tips[i],
            z_in,
        )));
    }
    let targets: [ArmTargetSpec; 3] = goal_targets.try_into().unwrap();

    let servo = ReactiveServoConfig {
        dt: 0.01,
        safety_margin: 0.04,
        cbf_activation_distance: 0.16,
        cbf_gamma: 7.0,
        damping: 0.025,
        posture_gain: 0.05,
        tangential_curl_gain: 0.55,
        qp_max_iterations: 35,
        cbf_margin_padding: 0.002,
        ..ReactiveServoConfig::default()
    }
    .with_arm_priority(RobotId(0), 3.0)
    .with_arm_priority(RobotId(1), 2.0)
    .with_arm_priority(RobotId(2), 1.0);

    WorkcellSpec {
        preset_id: Some(PresetId::ThreeArmCrossing),
        title: PresetId::ThreeArmCrossing.title().to_string(),
        subtitle: PresetId::ThreeArmCrossing.subtitle().to_string(),
        arms: three_arm_base_arms(targets, ready),
        obstacles: Vec::new(),
        rules: CollisionRulesSpec::default(),
        esdf: EsdfSpec::default(),
        planner: PlannerSpec::ThreeArmCrossing {
            start_poses,
            ik_num_seeds: 48,
            servo,
            num_steps: 500,
            record_stride: 5,
        },
        compute_backend: ComputeBackendChoice::CpuSimd,
    }
}

fn dual_arm_4dof_arms(
    q_left_start: Vec<f64>,
    q_right_start: Vec<f64>,
    left_target: ArmTargetSpec,
    right_target: ArmTargetSpec,
) -> Vec<ArmSpec> {
    vec![
        ArmSpec {
            uid: "left".to_string(),
            label: "Left 4-DOF Arm".to_string(),
            urdf_source: UrdfSource::Builtin4Dof,
            base_pose: PoseSpec::translation(0.0, 0.36, 0.0),
            flange_link_raw: "wrist_link".to_string(),
            default_tcp_z: 0.16,
            tool: ToolPreset::None,
            color_rgba: arm_palette_color(0),
            q_start: q_left_start,
            priority: 1.0,
            target: left_target,
        },
        ArmSpec {
            uid: "right".to_string(),
            label: "Right 4-DOF Arm".to_string(),
            urdf_source: UrdfSource::Builtin4Dof,
            base_pose: PoseSpec::translation(0.0, -0.36, 0.0),
            flange_link_raw: "wrist_link".to_string(),
            default_tcp_z: 0.16,
            tool: ToolPreset::None,
            color_rgba: arm_palette_color(1),
            q_start: q_right_start,
            priority: 1.0,
            target: right_target,
        },
    ]
}

fn dual_arm_mppi_esdf_spec() -> WorkcellSpec {
    WorkcellSpec {
        preset_id: Some(PresetId::DualArmMppiEsdf),
        title: PresetId::DualArmMppiEsdf.title().to_string(),
        subtitle: PresetId::DualArmMppiEsdf.subtitle().to_string(),
        arms: dual_arm_4dof_arms(
            vec![-0.55, 0.35, 0.5, 0.1],
            vec![0.55, 0.35, 0.5, 0.1],
            ArmTargetSpec::JointGoal(vec![0.55, 0.75, 0.8, 0.1]),
            ArmTargetSpec::JointGoal(vec![-0.55, 0.75, 0.8, 0.1]),
        ),
        obstacles: vec![
            ObstacleSpec {
                uid: "obs_left".to_string(),
                label: "Left Mid-Air Obstacle".to_string(),
                shape: ShapePrimitiveSpec::Sphere { radius: 0.14 },
                pose: PoseSpec::translation(0.5824, 0.36, 0.6971),
                role: ObstacleRole::StaticObstacle,
                color_rgba: [0.94, 0.32, 0.42, 0.85],
                safety_margin: None,
                activation_distance: None,
                carve_esdf: true,
            },
            ObstacleSpec {
                uid: "obs_right".to_string(),
                label: "Right Mid-Air Obstacle".to_string(),
                shape: ShapePrimitiveSpec::Sphere { radius: 0.14 },
                pose: PoseSpec::translation(0.5824, -0.36, 0.6971),
                role: ObstacleRole::StaticObstacle,
                color_rgba: [0.94, 0.32, 0.42, 0.85],
                safety_margin: None,
                activation_distance: None,
                carve_esdf: true,
            },
        ],
        rules: CollisionRulesSpec::default(),
        esdf: EsdfSpec {
            origin: [-1.0, -1.0, -0.2],
            voxel_size: 0.04,
            dims: [50, 50, 35],
            truncation: 1.0,
        },
        planner: PlannerSpec::BatchMppiEsdf {
            mppi: MppiConfig {
                num_rollouts: 64,
                horizon: 24,
                num_iterations: 35,
                noise_std: 0.38,
                temperature: 0.12,
                step_size: 0.8,
                activation_distance: 0.15,
                collision_weight: 220.0,
                velocity_weight: 2.0,
                acceleration_weight: 5.0,
            },
            num_visual_ribbons: 16,
            toppra_grid_points: 60,
            num_playback_frames: 60,
        },
        compute_backend: ComputeBackendChoice::CpuSimd,
    }
}

fn dual_arm_rrt_toppra_spec() -> WorkcellSpec {
    WorkcellSpec {
        preset_id: Some(PresetId::DualArmRrtToppra),
        title: PresetId::DualArmRrtToppra.title().to_string(),
        subtitle: PresetId::DualArmRrtToppra.subtitle().to_string(),
        arms: dual_arm_4dof_arms(
            vec![-0.45, 0.25, 0.55, 0.2],
            vec![0.45, 0.25, 0.55, 0.2],
            ArmTargetSpec::JointGoal(vec![0.55, 0.65, 0.75, -0.35]),
            ArmTargetSpec::JointGoal(vec![-0.55, 0.65, 0.75, -0.35]),
        ),
        obstacles: vec![
            ObstacleSpec {
                uid: "obs_left".to_string(),
                label: "Left Sphere".to_string(),
                shape: ShapePrimitiveSpec::Sphere { radius: 0.13 },
                pose: PoseSpec::translation(0.34, 0.22, 0.56),
                role: ObstacleRole::StaticObstacle,
                color_rgba: [0.94, 0.32, 0.42, 0.85],
                safety_margin: None,
                activation_distance: None,
                carve_esdf: true,
            },
            ObstacleSpec {
                uid: "obs_right".to_string(),
                label: "Right Sphere".to_string(),
                shape: ShapePrimitiveSpec::Sphere { radius: 0.13 },
                pose: PoseSpec::translation(0.34, -0.22, 0.56),
                role: ObstacleRole::StaticObstacle,
                color_rgba: [0.94, 0.32, 0.42, 0.85],
                safety_margin: None,
                activation_distance: None,
                carve_esdf: true,
            },
        ],
        rules: CollisionRulesSpec::default(),
        esdf: EsdfSpec {
            origin: [-0.9, -0.9, -0.1],
            voxel_size: 0.04,
            dims: [45, 45, 35],
            truncation: 1.0,
        },
        planner: PlannerSpec::RrtConnectToppra {
            step_size: 0.35,
            max_workspace_step: 0.03,
            max_iterations: 3000,
            seed: 42,
            toppra_grid_points: 64,
            num_playback_frames: 60,
        },
        compute_backend: ComputeBackendChoice::CpuSimd,
    }
}

fn dual_arm_closed_chain_cbf_spec() -> WorkcellSpec {
    WorkcellSpec {
        preset_id: Some(PresetId::DualArmClosedChainCbf),
        title: PresetId::DualArmClosedChainCbf.title().to_string(),
        subtitle: PresetId::DualArmClosedChainCbf.subtitle().to_string(),
        arms: dual_arm_4dof_arms(
            vec![-0.35, 0.55, 0.75, 0.15],
            vec![0.35, 0.55, 0.75, 0.15],
            ArmTargetSpec::Twist {
                linear: [0.08, 0.0, 0.10],
                angular: [0.0, 0.0, 0.0],
            },
            ArmTargetSpec::Twist {
                linear: [0.08, 0.0, 0.10],
                angular: [0.0, 0.0, 0.0],
            },
        ),
        obstacles: vec![ObstacleSpec {
            uid: "sweeping_ball".to_string(),
            label: "Dynamic Sweeping Obstacle".to_string(),
            shape: ShapePrimitiveSpec::Sphere { radius: 0.11 },
            pose: PoseSpec::translation(0.48, 0.28, 0.48),
            role: ObstacleRole::DynamicSweep {
                center: [0.48, 0.0, 0.48],
                amp_xyz: [0.0, 0.28, 0.12],
                freq_hz: [0.0, 1.0, 0.5],
                phase_rad: [0.0, 0.0, 0.0],
            },
            color_rgba: [0.96, 0.38, 0.26, 0.90],
            safety_margin: None,
            activation_distance: None,
            carve_esdf: false,
        }],
        rules: CollisionRulesSpec::default(),
        esdf: EsdfSpec::default(),
        planner: PlannerSpec::ClosedChainCbfServo {
            arm_a_uid: "left".to_string(),
            arm_b_uid: "right".to_string(),
            active_grasp_dims: [true, true, true, false, false, false],
            servo: ReactiveServoConfig {
                dt: 0.04,
                safety_margin: 0.04,
                cbf_activation_distance: 0.32,
                cbf_gamma: 6.0,
                ..ReactiveServoConfig::default()
            },
            num_steps: 55,
            linear_twist_amp: [0.08, 0.0, 0.10],
        },
        compute_backend: ComputeBackendChoice::CpuSimd,
    }
}
