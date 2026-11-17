//! Building blocks for planning collision-free paths and dynamic manipulation
//! for single- and multi-arm robotic systems in 3D space.
//!
//! `unrust-it` separates immutable robot topology ([`MultiRobotModel`]) from
//! pre-allocated per-thread workspaces ([`TransformBuffer`], [`CollisionWorkspace`])
//! to support both multi-threaded global motion planning and zero-allocation
//! real-time reactive collision avoidance.

pub mod collision;
pub mod kinematics;
pub mod planner;
pub mod robot;

pub use collision::{
    AllowedCollisionMatrix, BoundingSphere, CollidingPair, CollisionChecker, CollisionEnvironment,
    CollisionError, CollisionRule, CollisionWorkspace, GeometryTier, LinkCapsule, LinkGeometry,
    Obstacle, SignedDistanceWitness, WorldSphereSoA,
};
pub use kinematics::{ForwardKinematics, InverseKinematics, KinematicsError, TransformBuffer};
pub use planner::{PathPlanner, PlannerError, RrtPlanner, Target};
pub use robot::{
    AttachedPayload, Configuration, FrameAttachment, FrameGraph, FrameId, FrameNode, JointId,
    JointLimit, JointSpec, JointState, JointType, KinodynamicLimits, LinkId, LinkSpec,
    MultiRobotModel, ObstacleId, PlanningGroup, Robot, RobotError, RobotId, TaskPose,
    TaskSpacePoint,
};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, Isometry3, Point3, Translation3, UnitVector3, Vector3};
    use parry3d::shape::{Ball, SharedShape};
    use std::f64::consts::PI;
    use std::sync::Arc;

    /// Helper to build a dual-arm workcell (two 3-DOF arms mounted side by side).
    fn build_dual_arm_model() -> (MultiRobotModel, PlanningGroup, PlanningGroup, PlanningGroup) {
        let mut model = MultiRobotModel::new();
        let limit = JointLimit::new(-PI, PI);
        let segments = [
            (UnitVector3::new_normalize(Vector3::z()), Vector3::new(0.0, 0.0, 0.2), limit, 0.05),
            (UnitVector3::new_normalize(Vector3::y()), Vector3::new(0.0, 0.0, 0.4), limit, 0.05),
            (UnitVector3::new_normalize(Vector3::y()), Vector3::new(0.0, 0.0, 0.35), limit, 0.04),
        ];

        let left_arm = model
            .add_serial_arm(
                RobotId(0),
                "left_arm",
                Isometry3::translation(0.0, 0.35, 0.0),
                &segments,
                Isometry3::translation(0.0, 0.0, 0.3),
            )
            .unwrap();

        let right_arm = model
            .add_serial_arm(
                RobotId(1),
                "right_arm",
                Isometry3::translation(0.0, -0.35, 0.0),
                &segments,
                Isometry3::translation(0.0, 0.0, 0.3),
            )
            .unwrap();

        let dual_arms = left_arm.union("dual_arms", &right_arm);
        model.add_planning_group(dual_arms.clone());

        (model, left_arm, right_arm, dual_arms)
    }

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

    #[test]
    fn analytical_jacobian_matches_finite_differences_on_dual_arm() {
        let (model, left_arm, right_arm, _) = build_dual_arm_model();
        assert_eq!(model.dof(), 6);

        let q = [0.4, -0.6, 0.8, -0.3, 0.5, -0.7];
        let mut buf = model.create_transform_buffer();
        model.forward_kinematics(&q, &mut buf).unwrap();

        let eps = 1e-7;
        for &target_frame in &[left_arm.end_effectors[0], right_arm.end_effectors[0]] {
            let mut j_analytical = DMatrix::<f64>::zeros(6, model.dof());
            model
                .geometric_jacobian(&buf, target_frame, j_analytical.as_view_mut())
                .unwrap();

            let mut j_fd = DMatrix::<f64>::zeros(6, model.dof());
            let mut buf_plus = model.create_transform_buffer();
            let mut buf_minus = model.create_transform_buffer();

            for k in 0..model.dof() {
                let mut q_plus = q;
                let mut q_minus = q;
                q_plus[k] += eps;
                q_minus[k] -= eps;

                model.forward_kinematics(&q_plus, &mut buf_plus).unwrap();
                model.forward_kinematics(&q_minus, &mut buf_minus).unwrap();

                let pose_plus = buf_plus.frame_pose(target_frame);
                let pose_minus = buf_minus.frame_pose(target_frame);

                let v_fd = (pose_plus.translation.vector - pose_minus.translation.vector)
                    / (2.0 * eps);
                let w_fd = (pose_plus.rotation * pose_minus.rotation.inverse()).scaled_axis()
                    / (2.0 * eps);

                j_fd[(0, k)] = v_fd.x;
                j_fd[(1, k)] = v_fd.y;
                j_fd[(2, k)] = v_fd.z;
                j_fd[(3, k)] = w_fd.x;
                j_fd[(4, k)] = w_fd.y;
                j_fd[(5, k)] = w_fd.z;
            }

            assert_relative_eq!(j_analytical, j_fd, epsilon = 1e-5);
        }
    }

    #[test]
    fn acm_and_signed_distance_jacobian_work_for_inter_arm_and_obstacles() {
        let (model, _, _, _) = build_dual_arm_model();
        let acm = AllowedCollisionMatrix::from_model(&model);

        // Adjacent parent-child links on left arm (0, 1) and (1, 2) are disabled automatically
        assert!(!acm.is_pair_active(LinkId(0), LinkId(1)));
        assert!(!acm.is_pair_active(LinkId(1), LinkId(2)));
        // Non-adjacent and inter-arm pairs are active
        assert!(acm.is_pair_active(LinkId(0), LinkId(2)));
        assert!(acm.is_pair_active(LinkId(2), LinkId(5)));

        let mut env = CollisionEnvironment::new();
        env.set_acm(acm);
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
        env.add_obstacle(
            SharedShape::new(Ball::new(0.15)),
            Isometry3::from(Translation3::new(0.35, 0.35, 0.6)),
        );

        let q = [0.2, 0.4, 0.3, -0.2, 0.3, -0.1];
        let mut workspace = env.create_workspace(&model);
        let mut witnesses = Vec::new();
        env.query_signed_distances(
            &model,
            &q,
            2.0,
            GeometryTier::Convex,
            &mut workspace,
            &mut witnesses,
        )
        .unwrap();
        assert!(!witnesses.is_empty());

        // Verify that the distance Jacobian matches finite differences of the witness distance
        let obs_witness = witnesses
            .iter()
            .find(|w| matches!(w.pair, CollidingPair::LinkToObstacle { link: LinkId(2), .. }))
            .expect("expected witness between left forearm link and obstacle");

        let mut j_dist = vec![0.0; model.dof()];
        obs_witness.compute_distance_jacobian(&model, &workspace.transforms, &mut j_dist);

        let eps = 1e-6;
        for k in 0..model.dof() {
            let mut q_plus = q;
            let mut q_minus = q;
            q_plus[k] += eps;
            q_minus[k] -= eps;

            let mut w_plus = Vec::new();
            let mut w_minus = Vec::new();
            env.query_signed_distances(
                &model,
                &q_plus,
                2.0,
                GeometryTier::Convex,
                &mut workspace,
                &mut w_plus,
            )
            .unwrap();
            env.query_signed_distances(
                &model,
                &q_minus,
                2.0,
                GeometryTier::Convex,
                &mut workspace,
                &mut w_minus,
            )
            .unwrap();

            let d_plus = w_plus
                .iter()
                .find(|w| w.pair == obs_witness.pair)
                .unwrap()
                .signed_distance;
            let d_minus = w_minus
                .iter()
                .find(|w| w.pair == obs_witness.pair)
                .unwrap()
                .signed_distance;
            let fd_grad = (d_plus - d_minus) / (2.0 * eps);

            assert_relative_eq!(j_dist[k], fd_grad, epsilon = 1e-4);
        }
    }

    #[test]
    fn rrt_connect_routes_around_obstacle_and_solves_multi_pose_ik() {
        let (model, left_arm, right_arm, _) = build_dual_arm_model();
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

        // Place an obstacle that blocks the straight-line configuration path at q = [0, 0.5, 0, ...]
        let mut buf = model.create_transform_buffer();
        let q_mid = [0.0, 0.6, 0.0, 0.0, 0.0, 0.0];
        model.forward_kinematics(&q_mid, &mut buf).unwrap();
        let blocked_pos = buf.frame_pose(left_arm.end_effectors[0]).translation.vector;
        env.add_obstacle(
            SharedShape::new(Ball::new(0.12)),
            Isometry3::from(Translation3::from(blocked_pos)),
        );

        let robot = Robot::from_model(Arc::new(model));
        let start = Configuration::new(vec![0.0, -0.2, 0.0, 0.0, 0.0, 0.0]);
        let goal = Configuration::new(vec![0.0, 1.2, 0.0, 0.0, 0.0, 0.0]);

        // Verify midpoint is actually in collision while endpoints are collision-free
        assert!(env.is_collision_free(&robot, &start).unwrap());
        assert!(env.is_collision_free(&robot, &goal).unwrap());
        assert!(!env
            .is_collision_free(&robot, &Configuration::from_slice(&q_mid))
            .unwrap());

        let planner = RrtPlanner::new(env.clone());
        let path = planner
            .plan_path(&robot, start.clone(), Target::Joint(goal.clone()))
            .expect("RRT-Connect should route around the blocking obstacle");
        assert!(path.len() >= 3);
        assert_eq!(path.first().unwrap(), &start);
        assert_eq!(path.last().unwrap(), &goal);

        // Also test multi-arm simultaneous TaskPose IK target resolution
        let model_ref = robot.model.as_ref().unwrap();
        let q_desired = [0.3, -0.4, 0.5, -0.3, 0.4, -0.5];
        model_ref.forward_kinematics(&q_desired, &mut buf).unwrap();
        let left_pose = *buf.frame_pose(left_arm.end_effectors[0]);
        let right_pose = *buf.frame_pose(right_arm.end_effectors[0]);

        let ik_path = planner
            .plan_path(
                &robot,
                start,
                Target::MultiPose(vec![
                    TaskPose::new(left_arm.end_effectors[0], left_pose),
                    TaskPose::new(right_arm.end_effectors[0], right_pose),
                ]),
            )
            .expect("RRT-Connect should solve dual-arm IK and plan a valid path");
        let final_q = ik_path.last().unwrap();
        model_ref
            .forward_kinematics(final_q.as_slice(), &mut buf)
            .unwrap();
        assert_relative_eq!(
            buf.frame_pose(left_arm.end_effectors[0]).translation.vector,
            left_pose.translation.vector,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            buf.frame_pose(right_arm.end_effectors[0]).translation.vector,
            right_pose.translation.vector,
            epsilon = 1e-3
        );
    }
}
