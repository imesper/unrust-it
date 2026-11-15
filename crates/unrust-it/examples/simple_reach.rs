//! A small end-to-end example using the initial unrust-it API.

use nalgebra::{Isometry3, Translation3};
use parry3d::shape::{Ball, SharedShape};
use unrust_it::{
    CollisionEnvironment, Configuration, JointLimit, PathPlanner, Robot, RrtPlanner, Target,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Joint limits describe the robot's configuration space (C-space).
    let robot = Robot::new(vec![
        JointLimit::new(-std::f64::consts::PI, std::f64::consts::PI),
        JointLimit::new(-1.5, 1.5),
        JointLimit::new(-2.0, 2.0),
    ])?;

    // Obstacles use parry3d shapes and poses in the world frame.
    let mut environment = CollisionEnvironment::new();
    environment.add_obstacle(
        SharedShape::new(Ball::new(0.25)),
        Isometry3::from(Translation3::new(0.5, 0.0, 0.2)),
    );

    // A real application would add link geometry and a robot-specific
    // kinematics model; the MVP RRT skeleton handles joint-space endpoints.
    let planner = RrtPlanner::new(environment);
    let path = planner.plan_path(
        &robot,
        Configuration::new(vec![0.0, 0.0, 0.0]),
        Target::Joint(Configuration::new(vec![1.0, 0.5, -0.5])),
    )?;

    println!("planned {} configuration(s)", path.len());
    Ok(())
}
