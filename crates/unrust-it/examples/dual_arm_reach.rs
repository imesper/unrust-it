//! Demonstrates multi-arm kinematic trees, bitset Allowed Collision Matrix (ACM),
//! signed-distance Jacobian evaluation, and simultaneous dual-arm pose planning.

use nalgebra::{Isometry3, Point3, Translation3, UnitVector3, Vector3};
use parry3d::shape::{Ball, SharedShape};
use std::f64::consts::PI;
use std::sync::Arc;
use unrust_it::{
    AllowedCollisionMatrix, CollisionEnvironment, Configuration, GeometryTier, JointLimit,
    LinkGeometry, MultiRobotModel, PathPlanner, Robot, RobotId, RrtPlanner, Target, TaskPose,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Construct a dual-arm robotic workcell (two 3-DOF manipulators sharing a workspace).
    let mut model = MultiRobotModel::new();
    let limit = JointLimit::new(-PI, PI);
    let segments = [
        (
            UnitVector3::new_normalize(Vector3::z()),
            Vector3::new(0.0, 0.0, 0.2),
            limit,
            0.05,
        ),
        (
            UnitVector3::new_normalize(Vector3::y()),
            Vector3::new(0.0, 0.0, 0.4),
            limit,
            0.05,
        ),
        (
            UnitVector3::new_normalize(Vector3::y()),
            Vector3::new(0.0, 0.0, 0.35),
            limit,
            0.04,
        ),
    ];

    let left_arm = model.add_serial_arm(
        RobotId(0),
        "left_arm",
        Isometry3::translation(0.0, 0.35, 0.0),
        &segments,
        Isometry3::translation(0.0, 0.0, 0.3),
    )?;
    let right_arm = model.add_serial_arm(
        RobotId(1),
        "right_arm",
        Isometry3::translation(0.0, -0.35, 0.0),
        &segments,
        Isometry3::translation(0.0, 0.0, 0.3),
    )?;
    let dual_arms = left_arm.union("dual_arms", &right_arm);
    model.add_planning_group(dual_arms);

    // 2. Populate multi-tier link geometries and bitset Allowed Collision Matrix (ACM).
    let mut env = CollisionEnvironment::new();
    env.set_acm(AllowedCollisionMatrix::from_model(&model));
    for link in model.links() {
        env.set_link_geometry(
            link.id,
            LinkGeometry::from_capsule(
                Point3::origin(),
                Point3::new(0.0, 0.0, 0.3),
                link.bounding_radius,
            ),
        );
    }

    // Add a central spherical obstacle in the shared workspace.
    env.add_obstacle(
        SharedShape::new(Ball::new(0.12)),
        Isometry3::from(Translation3::new(0.45, 0.35, 0.55)),
    );

    // 3. Evaluate Forward Kinematics and Signed Distance Jacobians at the start state.
    let start = Configuration::new(vec![0.0, -0.2, 0.0, 0.0, -0.2, 0.0]);
    let mut workspace = env.create_workspace(&model);
    let mut witnesses = Vec::new();
    env.query_signed_distances(
        &model,
        start.as_slice(),
        1.0,
        GeometryTier::Convex,
        &mut workspace,
        &mut witnesses,
    )?;
    println!(
        "Evaluated {} proximity witness pairs within 1.0 m at start configuration",
        witnesses.len()
    );

    // 4. Plan a simultaneous dual-arm motion to desired 6-DOF end-effector poses.
    let mut buf = model.create_transform_buffer();
    let goal_q_ref = [0.25, 0.9, -0.3, -0.25, 0.7, -0.2];
    model.forward_kinematics(&goal_q_ref, &mut buf)?;
    let left_target_pose = *buf.frame_pose(left_arm.end_effectors[0]);
    let right_target_pose = *buf.frame_pose(right_arm.end_effectors[0]);

    let robot = Robot::from_model(Arc::new(model));
    let planner = RrtPlanner::new(env);
    let path = planner.plan_path(
        &robot,
        start,
        Target::MultiPose(vec![
            TaskPose::new(left_arm.end_effectors[0], left_target_pose),
            TaskPose::new(right_arm.end_effectors[0], right_target_pose),
        ]),
    )?;

    println!(
        "Planned collision-free dual-arm trajectory with {} waypoints",
        path.len()
    );
    Ok(())
}
