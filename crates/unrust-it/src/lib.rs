//! Building blocks for planning collision-free paths for robotic manipulators.
//!
//! The crate deliberately keeps the mathematical model small: joint-space
//! values are distinct from task-space points, while users can provide their
//! own kinematics and collision geometry for a particular robot.

pub mod collision;
pub mod kinematics;
pub mod planner;
pub mod robot;

pub use collision::{CollisionChecker, CollisionEnvironment, CollisionError};
pub use kinematics::{ForwardKinematics, InverseKinematics};
pub use planner::{PathPlanner, PlannerError, RrtPlanner, Target};
pub use robot::{Configuration, JointLimit, Robot, RobotError, TaskSpacePoint};

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    #[test]
    fn robot_rejects_configuration_with_wrong_dof() {
        let mut robot = Robot::new(vec![JointLimit::new(-1.0, 1.0)]).unwrap();
        assert!(robot
            .set_configuration(Configuration::new(vec![0.0, 0.0]))
            .is_err());
    }

    #[test]
    fn rrt_returns_a_joint_space_path() {
        let robot = Robot::new(vec![JointLimit::new(-1.0, 1.0)]).unwrap();
        let planner = RrtPlanner::new(CollisionEnvironment::new());
        let path = planner
            .plan_path(
                &robot,
                Configuration::new(vec![0.0]),
                Target::Joint(Configuration::new(vec![0.5])),
            )
            .unwrap();
        assert_eq!(
            path,
            vec![Configuration::new(vec![0.0]), Configuration::new(vec![0.5])]
        );
    }

    #[test]
    fn task_space_target_is_typed() {
        let target = TaskSpacePoint::new(Point3::new(1.0, 2.0, 3.0));
        assert_eq!(target.position, Point3::new(1.0, 2.0, 3.0));
    }
}
