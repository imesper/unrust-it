use approx::assert_relative_eq;
use nalgebra::Vector3;
use std::path::PathBuf;
use unrust_it_io::{
    resolve_trajectory_set, AcmOverrideSpec, AcmRuleValueSpec, EntityRefSpec, IoError,
    ObstacleRoleSpec, ObstacleSpec, PairOverrideSpec, PlannerConfigSpec, PoseSpec, ReferenceFrame,
    RobotSourceSpec, RobotSpec, RobotTrajectoriesSpec, ShapeSpec, SubframeRole, SubframeSpec,
    ToolPayloadSpec, TrajectoryFrameSpec, TrajectorySetSpec, TrajectorySpec, WorldSpec,
};
use unrust_it_urdf::presets::{ARM_4DOF_URDF, ARM_6DOF_SRDF, ARM_6DOF_URDF};

fn temp_test_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("unrust_it_io_test_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp test dir");
    dir
}

fn sample_dual_world() -> WorldSpec {
    let mut world = WorldSpec::new("dual_cell");
    world.description = "Dual arm welding & inspection demo cell".into();
    world.planner = Some(PlannerConfigSpec::default());

    world.robots.push(RobotSpec {
        uid: "arm_a".into(),
        label: "Arm A (6-DOF)".into(),
        prefix: "a_".into(),
        priority: 0,
        active_group: None,
        tcp_frame_name: Some("a_torch_tcp".into()),
        tcp_parent_frame: None,
        color_rgba: None,
        q_start: vec![0.1, -0.3, 0.5, 0.0, 0.4, 0.0],
        source: RobotSourceSpec::Builtin6Dof,
        base_pose: PoseSpec::from_xyz([0.0, -0.45, 0.0]),
        tcp_offset: Some(PoseSpec::from_xyz([0.0, 0.0, 0.12])),
        tool_payload: Some(ToolPayloadSpec {
            name: "a_torch".into(),
            parent_link: None,
            mass: 0.75,
            safety_margin: Some(0.004),
            activation_distance: Some(0.06),
            touch_links: Vec::new(),
            local_pose: PoseSpec::from_xyz([0.0, 0.0, 0.06]),
            shape: ShapeSpec::Capsule {
                radius: 0.018,
                half_height: 0.05,
            },
        }),
    });

    world.robots.push(RobotSpec {
        uid: "arm_b".into(),
        label: "Arm B (4-DOF)".into(),
        prefix: "b_".into(),
        priority: 1,
        active_group: None,
        tcp_frame_name: Some("b_sensor_tcp".into()),
        tcp_parent_frame: None,
        color_rgba: Some([0.9, 0.5, 0.2, 1.0]),
        q_start: vec![-0.2, 0.4, -0.5, 0.2],
        source: RobotSourceSpec::Builtin4Dof,
        base_pose: PoseSpec::from_xyz([0.0, 0.45, 0.0]),
        tcp_offset: Some(PoseSpec::from_xyz([0.0, 0.0, 0.09])),
        tool_payload: Some(ToolPayloadSpec {
            name: "b_scanner".into(),
            parent_link: None,
            mass: 0.45,
            safety_margin: None,
            activation_distance: None,
            touch_links: Vec::new(),
            local_pose: PoseSpec::IDENTITY,
            shape: ShapeSpec::Cylinder {
                radius: 0.025,
                half_height: 0.04,
            },
        }),
    });

    world.obstacles.push(ObstacleSpec {
        uid: "workpiece".into(),
        label: "Main Manifold".into(),
        role: ObstacleRoleSpec::Workpiece,
        carve_esdf: true,
        color_rgba: None,
        safety_margin: Some(0.006),
        activation_distance: Some(0.08),
        pose: PoseSpec::from_xyz([0.45, 0.0, 0.25]),
        shape: ShapeSpec::Box {
            half_extents: [0.12, 0.10, 0.08],
        },
    });

    world.obstacles.push(ObstacleSpec {
        uid: "table".into(),
        label: "Optical Table".into(),
        role: ObstacleRoleSpec::Static,
        carve_esdf: true,
        color_rgba: None,
        safety_margin: None,
        activation_distance: None,
        pose: PoseSpec::from_xyz([0.0, 0.0, -0.05]),
        shape: ShapeSpec::Box {
            half_extents: [1.0, 1.0, 0.05],
        },
    });

    world.collision_rules.acm_overrides.push(AcmOverrideSpec {
        link_a: "a_base_link".into(),
        link_b: "b_base_link".into(),
        rule: AcmRuleValueSpec::Disabled,
    });
    world
        .collision_rules
        .pair_overrides
        .push(PairOverrideSpec {
            enabled: true,
            margin: Some(0.003),
            activation_distance: Some(0.05),
            a: EntityRefSpec::Payload("a_torch".into()),
            b: EntityRefSpec::Obstacle("workpiece".into()),
        });

    world
}

#[test]
fn world_spec_toml_and_json_roundtrip_and_compile() {
    let world = sample_dual_world();

    let toml_text = world.to_toml_string().expect("serialize world TOML");
    let from_toml = WorldSpec::from_toml_str(&toml_text).expect("parse world TOML");
    assert_eq!(from_toml, world);

    let json_text = world
        .to_json_string_pretty()
        .expect("serialize world JSON");
    let from_json = WorldSpec::from_json_str(&json_text).expect("parse world JSON");
    assert_eq!(from_json, world);

    let compiled = from_toml.compile().expect("compile dual world");
    assert_eq!(compiled.robots.len(), 2);
    assert_eq!(compiled.scene.model().dof(), 10);
    assert_eq!(compiled.obstacles.len(), 2);
    assert!(compiled.scene.esdf().is_some());
    assert!(compiled.robot("arm_a").is_some());
    assert!(compiled.robot("arm_b").is_some());
    assert!(compiled.obstacle_by_uid.contains_key("workpiece"));

    let arm_a = compiled.robot("arm_a").expect("arm_a present");
    assert_eq!(arm_a.tcp_frame_name, "a_torch_tcp");
    assert_eq!(arm_a.payload_name.as_deref(), Some("a_torch"));
    assert_eq!(arm_a.q_range, 0..6);

    let mut tf = compiled.scene.create_transform_buffer();
    compiled
        .scene
        .model()
        .forward_kinematics(compiled.q_start.as_slice(), &mut tf)
        .expect("fk on compiled world");
    let torch_pos = tf.frame_pose(arm_a.tcp_frame_id).translation.vector;
    assert!(torch_pos.norm() > 0.1);
}

#[test]
fn world_spec_loads_relative_urdf_srdf_files_from_disk() {
    let dir = temp_test_dir("rel_paths");
    let urdf_dir = dir.join("robots");
    std::fs::create_dir_all(&urdf_dir).expect("create robots subdir");
    std::fs::write(urdf_dir.join("arm6.urdf"), ARM_6DOF_URDF).expect("write arm6.urdf");
    std::fs::write(urdf_dir.join("arm6.srdf"), ARM_6DOF_SRDF).expect("write arm6.srdf");
    std::fs::write(urdf_dir.join("arm4.urdf"), ARM_4DOF_URDF).expect("write arm4.urdf");

    let mut world = WorldSpec::new("file_backed");
    world.robots.push(RobotSpec {
        uid: "r6".into(),
        label: String::new(),
        prefix: "r6_".into(),
        priority: 0,
        active_group: None,
        tcp_frame_name: None,
        tcp_parent_frame: None,
        color_rgba: None,
        q_start: Vec::new(),
        source: RobotSourceSpec::Files {
            urdf_path: PathBuf::from("robots/arm6.urdf"),
            srdf_path: Some(PathBuf::from("robots/arm6.srdf")),
        },
        base_pose: PoseSpec::IDENTITY,
        tcp_offset: None,
        tool_payload: None,
    });
    world.robots.push(RobotSpec {
        uid: "r4".into(),
        label: String::new(),
        prefix: "r4_".into(),
        priority: 1,
        active_group: None,
        tcp_frame_name: None,
        tcp_parent_frame: None,
        color_rgba: None,
        q_start: Vec::new(),
        source: RobotSourceSpec::Files {
            urdf_path: PathBuf::from("robots/arm4.urdf"),
            srdf_path: None,
        },
        base_pose: PoseSpec::from_xyz([0.0, 0.7, 0.0]),
        tcp_offset: None,
        tool_payload: None,
    });

    let world_file = dir.join("workcell.toml");
    world.save_file(&world_file).expect("save workcell.toml");
    let loaded = WorldSpec::load_file(&world_file).expect("load workcell.toml");
    assert_eq!(loaded.base_dir(), Some(dir.as_path()));
    let compiled = loaded.compile().expect("compile file-backed world");
    assert_eq!(compiled.scene.model().dof(), 10);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn trajectory_set_roundtrip_and_hierarchical_subframe_resolution() {
    let world = sample_dual_world();
    let compiled = world.compile().expect("compile world");

    let set = TrajectorySetSpec {
        schema_version: 1,
        name: "weld_and_scan_batch".into(),
        description: "Multi-robot trajectory batch with frames & nested subframes".into(),
        world_name: Some("dual_cell".into()),
        default_reference_frame: Some(ReferenceFrame::Workpiece {
            obstacle_uid: "workpiece".into(),
        }),
        robots: vec![
            RobotTrajectoriesSpec {
                robot_uid: "arm_a".into(),
                group_name: None,
                tcp_frame: Some("a_torch_tcp".into()),
                reference_frame: None,
                trajectories: vec![TrajectorySpec {
                    id: "seam_1".into(),
                    label: "Front Seam".into(),
                    enabled: true,
                    speed_scale: 0.8,
                    tcp_frame: None,
                    reference_frame: None,
                    frames: vec![TrajectoryFrameSpec {
                        id: "spot_a".into(),
                        label: "Spot A".into(),
                        role: SubframeRole::Contact,
                        time_s: Some(1.0),
                        execute_ik: None,
                        tcp_frame: None,
                        joint_seed: Vec::new(),
                        color_rgba: None,
                        reference_frame: None,
                        pose: PoseSpec::from_xyz([0.0, -0.05, 0.08]),
                        subframes: vec![
                            SubframeSpec {
                                id: "lead_out".into(),
                                label: "Retract".into(),
                                role: SubframeRole::Retract,
                                time_offset_s: 0.4,
                                execute_ik: None,
                                tcp_frame: None,
                                color_rgba: None,
                                pose: PoseSpec::from_xyz([0.0, 0.0, 0.08]),
                                subframes: Vec::new(),
                            },
                            SubframeSpec {
                                id: "lead_in".into(),
                                label: "Approach".into(),
                                role: SubframeRole::Approach,
                                time_offset_s: -0.35,
                                execute_ik: None,
                                tcp_frame: None,
                                color_rgba: None,
                                pose: PoseSpec::from_xyz([0.0, 0.0, 0.06]),
                                subframes: vec![SubframeSpec {
                                    id: "standoff_note".into(),
                                    label: "Sensor Ref".into(),
                                    role: SubframeRole::ReferenceOnly,
                                    time_offset_s: 0.0,
                                    execute_ik: None,
                                    tcp_frame: None,
                                    color_rgba: None,
                                    pose: PoseSpec::from_xyz([0.02, 0.0, 0.0]),
                                    subframes: Vec::new(),
                                }],
                            },
                        ],
                    }],
                }],
            },
            RobotTrajectoriesSpec {
                robot_uid: "arm_b".into(),
                group_name: None,
                tcp_frame: None,
                reference_frame: Some(ReferenceFrame::RobotBase),
                trajectories: vec![TrajectorySpec {
                    id: "inspect".into(),
                    label: "Camera Sweep".into(),
                    enabled: true,
                    speed_scale: 1.0,
                    tcp_frame: None,
                    reference_frame: None,
                    frames: vec![TrajectoryFrameSpec {
                        id: "look_1".into(),
                        label: "Look 1".into(),
                        role: SubframeRole::Waypoint,
                        time_s: Some(0.5),
                        execute_ik: None,
                        tcp_frame: None,
                        joint_seed: vec![0.0, 0.2, -0.2, 0.1],
                        color_rgba: None,
                        reference_frame: None,
                        pose: PoseSpec::from_xyz([0.35, -0.20, 0.45]),
                        subframes: Vec::new(),
                    }],
                }],
            },
        ],
    };

    // Round-trip both TOML and JSON with nested subframes
    let toml_text = set.to_toml_string().expect("serialize traj TOML");
    let parsed_toml = TrajectorySetSpec::from_toml_str(&toml_text).expect("parse traj TOML");
    assert_eq!(parsed_toml, set);

    let json_text = set.to_json_string_pretty().expect("serialize traj JSON");
    let parsed_json = TrajectorySetSpec::from_json_str(&json_text).expect("parse traj JSON");
    assert_eq!(parsed_json, set);

    let resolved = resolve_trajectory_set(&parsed_toml, &compiled, compiled.q_start.as_slice())
        .expect("resolve trajectories");
    assert_eq!(resolved.robots.len(), 2);

    let arm_a_wps = &resolved.robots[0].trajectories[0].waypoints;
    // Order: pre (`lead_in`), child of pre (`standoff_note`), parent (`spot_a`), post (`lead_out`)
    let ids: Vec<&str> = arm_a_wps.iter().map(|w| w.node_id.as_str()).collect();
    assert_eq!(
        ids,
        vec![
            "spot_a/lead_in",
            "spot_a/lead_in/standoff_note",
            "spot_a",
            "spot_a/lead_out",
        ]
    );

    // Workpiece at [0.45, 0.0, 0.25], spot_a at [0.0, -0.05, 0.08] => [0.45, -0.05, 0.33]
    let spot = &arm_a_wps[2];
    assert_relative_eq!(
        spot.world_pose.translation.vector,
        Vector3::new(0.45, -0.05, 0.33),
        epsilon = 1e-12
    );
    assert!(spot.execute_ik);

    // Depth-2 reference_only subframe composed accurately and excluded from executable stream
    let note = &arm_a_wps[1];
    assert_eq!(note.depth, 2);
    assert!(!note.execute_ik);
    assert_relative_eq!(
        note.world_pose.translation.vector,
        Vector3::new(0.47, -0.05, 0.39),
        epsilon = 1e-12
    );

    let exec_a: Vec<&str> = resolved
        .executable_waypoints_for_robot("arm_a")
        .map(|w| w.node_id.as_str())
        .collect();
    assert_eq!(exec_a, vec!["spot_a/lead_in", "spot_a", "spot_a/lead_out"]);

    // Robot B uses RobotBase ([0.0, 0.45, 0.0]) + [0.35, -0.20, 0.45] => [0.35, 0.25, 0.45]
    let look_b = &resolved.robots[1].trajectories[0].waypoints[0];
    assert_relative_eq!(
        look_b.world_pose.translation.vector,
        Vector3::new(0.35, 0.25, 0.45),
        epsilon = 1e-12
    );
    assert!(look_b.joint_seed.is_some());
}

#[test]
fn trajectory_resolution_rejects_unknown_robot_or_wrong_seed() {
    let compiled = sample_dual_world().compile().expect("compile world");

    let mut bad_robot = TrajectorySetSpec::new("bad_robot");
    bad_robot.robots.push(RobotTrajectoriesSpec {
        robot_uid: "nonexistent_arm".into(),
        group_name: None,
        tcp_frame: None,
        reference_frame: None,
        trajectories: Vec::new(),
    });
    assert!(matches!(
        resolve_trajectory_set(&bad_robot, &compiled, compiled.q_start.as_slice()),
        Err(IoError::UnknownRobot(_))
    ));

    let mut bad_seed = TrajectorySetSpec::new("bad_seed");
    bad_seed.robots.push(RobotTrajectoriesSpec {
        robot_uid: "arm_a".into(),
        group_name: None,
        tcp_frame: None,
        reference_frame: None,
        trajectories: vec![TrajectorySpec {
            id: "t0".into(),
            label: String::new(),
            enabled: true,
            speed_scale: 1.0,
            tcp_frame: None,
            reference_frame: None,
            frames: vec![TrajectoryFrameSpec {
                id: "f0".into(),
                label: String::new(),
                role: SubframeRole::Waypoint,
                time_s: None,
                execute_ik: None,
                tcp_frame: None,
                joint_seed: vec![0.0, 0.0], // wrong DOF (arm_a is 6-DOF)
                color_rgba: None,
                reference_frame: None,
                pose: PoseSpec::IDENTITY,
                subframes: Vec::new(),
            }],
        }],
    });
    assert!(matches!(
        resolve_trajectory_set(&bad_seed, &compiled, compiled.q_start.as_slice()),
        Err(IoError::JointCountMismatch { .. })
    ));
}
