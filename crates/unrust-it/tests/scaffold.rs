use nalgebra::Isometry3;
use parry3d::shape::{Ball, SharedShape};
use unrust_it::{CollisionEnvironment, JointLimit, Robot};

#[test]
fn environment_accepts_parry_obstacles() {
    let mut environment = CollisionEnvironment::new();
    environment.add_obstacle(SharedShape::new(Ball::new(1.0)), Isometry3::identity());
    assert_eq!(environment.obstacles.len(), 1);
}

#[test]
fn robot_dof_matches_joint_limits() {
    let robot = Robot::new(vec![JointLimit::new(-1.0, 1.0), JointLimit::new(-2.0, 2.0)]).unwrap();
    assert_eq!(robot.configuration.dof(), 2);
}
