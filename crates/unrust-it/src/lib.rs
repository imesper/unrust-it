//! High-performance path planning, dynamic manipulation, and collision avoidance
//! for single- and multi-arm ($K \ge 3$) robotic systems in 3D space.
//!
//! # Architecture Overview
//!
//! `unrust-it` is organized around two core design invariants:
//!
//! 1. **Dual-precision compute**:
//!    - **Exact `f64` layer** ([`robot`], [`kinematics`], [`collision`], [`manifold`],
//!      [`trajectory`], [`control`], [`planner`]) powered by `nalgebra` and `parry3d-f64` for
//!      analytical geometric Jacobians, GJK/EPA signed distances, closed-chain constraints,
//!      TOPP-RA time parameterization, and 1 kHz CBF reactive control.
//!    - **Batched `f32` rollout layer** ([`batch`]) evaluating $B \times H$ states in one
//!      dispatch via `wide::f32x8` + `rayon` ([`CpuSimdBatchBackend`]) and `wgpu` WGSL compute
//!      shaders (`WgpuBatchBackend` / resident `GpuRobotContext` when the `gpu` feature is enabled).
//! 2. **Zero-allocation hot paths**:
//!    Immutable `Send + Sync` topology ([`MultiRobotModel`]) and scenes ([`CollisionEnvironment`])
//!    are strictly separated from pre-allocated per-thread scratchpads ([`TransformBuffer`],
//!    [`CollisionWorkspace`], [`ReactiveServoController`]).
//!
//! # Module Guide
//!
//! - [`robot`]: Kinematic trees, joint types/limits, `PlanningGroup`, `AttachedPayload`, and `MultiRobotModel`.
//! - [`kinematics`]: Single-pass FK, analytical $6 \times N_{\text{dof}}$ Jacobians, screw-axis distance projection, DLS IK.
//! - [`collision`]: Tier-1 `f64x4` SIMD SSV spheres/capsules, Tier-2 `parry3d` convex shapes, hardware-`CTZ` bitset ACM, BVH broad-phase, Lipschitz segment checking.
//! - [`manifold`]: Holonomic `ConstraintManifold` and closed-chain grasp projection.
//! - [`trajectory`]: $C^2$ `CubicSplinePath`, `ToppraTrajectory`, and `JerkLimitedTrajectory`.
//! - [`control`]: 1 kHz `ReactiveServoController` with priority-weighted QP-CBF, tangential curl, and velocity dampers.
//! - [`planner`]: Bidirectional `RrtPlanner` and two-stage close-proximity `TwoStageApproachPlanner`.
//! - [`scene`]: Unified [`WorldScene`] owner with generational [`ObstacleHandle`]s and dirty-region ESDF sync.
//! - [`batch`]: `FlatRobotModelF32`, `EsdfGridF32`, `BatchIkSolver`, `BatchMppiOptimizer`, SIMD/GPU backends.
//!
//! # Feature Flags
//!
//! - `gpu` *(enabled by default)*: Enables `wgpu` WGSL compute shaders (`WgpuBatchBackend`, `GpuRobotContext`).
//!   Disable default features for pure CPU SIMD (`f32x8` + `rayon`) headless deployments.
//!
//! # Quick Start
//!
//! ```rust
//! use nalgebra::{Isometry3, Point3, UnitVector3, Vector3};
//! use parry3d::shape::{Ball, SharedShape};
//! use std::f64::consts::PI;
//! use std::sync::Arc;
//! use unrust_it::{
//!     AllowedCollisionMatrix, CollisionEnvironment, Configuration, GeometryTier, JointLimit,
//!     LinkGeometry, MultiRobotModel, PathPlanner, Robot, RobotId, RrtPlanner, Target, TaskPose,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut model = MultiRobotModel::new();
//! let limit = JointLimit::new(-PI, PI);
//! let segs = [
//!     (UnitVector3::new_normalize(Vector3::z()), Vector3::new(0.0, 0.0, 0.2), limit, 0.05),
//!     (UnitVector3::new_normalize(Vector3::y()), Vector3::new(0.0, 0.0, 0.4), limit, 0.04),
//! ];
//! let left = model.add_serial_arm(RobotId(0), "left", Isometry3::translation(0.0, 0.35, 0.0), &segs, Isometry3::translation(0.0, 0.0, 0.2))?;
//! let right = model.add_serial_arm(RobotId(1), "right", Isometry3::translation(0.0, -0.35, 0.0), &segs, Isometry3::translation(0.0, 0.0, 0.2))?;
//!
//! let mut env = CollisionEnvironment::new();
//! env.set_acm(AllowedCollisionMatrix::from_model(&model));
//! for l in model.links() {
//!     env.set_link_geometry(l.id, LinkGeometry::from_capsule(Point3::origin(), Point3::new(0.0, 0.0, 0.2), l.bounding_radius));
//! }
//! env.add_obstacle(SharedShape::new(Ball::new(0.08)), Isometry3::translation(0.4, 0.0, 0.4));
//!
//! let mut ws = env.create_workspace(&model);
//! let mut witnesses = Vec::new();
//! let start = Configuration::new(vec![0.0, 0.2, 0.0, -0.2]);
//! env.query_signed_distances(&model, start.as_slice(), 1.0, GeometryTier::Convex, &mut ws, &mut witnesses)?;
//!
//! let mut buf = model.create_transform_buffer();
//! model.forward_kinematics(&[0.2, 0.5, -0.2, -0.5], &mut buf)?;
//! let target = Target::MultiPose(vec![
//!     TaskPose::new(left.end_effectors[0], *buf.frame_pose(left.end_effectors[0])),
//!     TaskPose::new(right.end_effectors[0], *buf.frame_pose(right.end_effectors[0])),
//! ]);
//!
//! let path = RrtPlanner::new(env).plan_path(&Robot::from_model(Arc::new(model)), start, target)?;
//! assert!(path.len() >= 2);
//! # Ok(())
//! # }
//! ```

pub mod batch;
pub mod collision;
pub mod control;
pub mod kinematics;
pub mod manifold;
pub mod planner;
pub mod robot;
pub mod scene;
pub mod trajectory;

pub use batch::{
    BatchIkSolver, BatchMppiOptimizer, BatchRolloutBackend, CpuSimdBatchBackend, EsdfGridF32,
    FlatEndEffectorF32, FlatJointF32, FlatLinkSphereF32, FlatRobotModelF32, FlatSpherePairF32,
    MppiConfig, MppiOptimizationTrace, RolloutUniformsF32, SPHERE_FLAG_PAYLOAD,
};
#[cfg(feature = "gpu")]
pub use batch::{GpuRobotContext, ResidentGpuBatchBackend, WgpuBatchBackend};
pub use collision::{
    decompose_shape_to_spheres, parry_to_isometry, AllowedCollisionMatrix, BoundingSphere,
    CollidingPair, CollisionChecker, CollisionEnvironment, CollisionError, CollisionRule,
    CollisionWorkspace, GeometryTier, LinkCapsule, LinkGeometry, Obstacle, PayloadPairRule,
    SignedDistanceWitness, WorldSphereSoA,
};
pub use control::{ControlError, ReactiveServoConfig, ReactiveServoController, ServoTarget};
pub use kinematics::{ForwardKinematics, InverseKinematics, KinematicsError, TransformBuffer};
pub use manifold::{ClosedChainGraspConstraint, ConstraintManifold};
pub use planner::{
    ApproachTarget, CoarsePlannerMode, PathPlanner, PlannerError, RrtPlanner, Target,
    TwoStageApproachPlanner, TwoStagePlanConfig, TwoStagePlanResult,
};
pub use scene::{EsdfConfig, ObstacleHandle, ObstacleOptions, SceneError, WorldScene};
pub use robot::{
    AttachedPayload, Configuration, FrameAttachment, FrameGraph, FrameId, FrameNode, JointId,
    JointLimit, JointSpec, JointState, JointType, KinodynamicLimits, LinkId, LinkSpec,
    MultiRobotModel, ObstacleId, PayloadId, PlanningGroup, Robot, RobotError, RobotId, TaskPose,
    TaskSpacePoint,
};
pub use trajectory::{
    CubicSplinePath, JerkLimitedTrajectory, ToppraTrajectory, TrajectorySample,
};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, Isometry3, Point3, SVector, Translation3, UnitVector3, Vector3};
    use parry3d::shape::{Ball, SharedShape};
    use std::f64::consts::PI;
    use std::sync::Arc;

    /// Helper to build a dual-arm workcell (two 3-DOF arms mounted side by side).
    fn build_dual_arm_model() -> (MultiRobotModel, PlanningGroup, PlanningGroup, PlanningGroup) {
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

        // Place an obstacle that blocks the straight-line configuration path at q = [0, 0.6, 0, ...]
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

    #[test]
    fn closed_chain_bimanual_manifold_and_kinodynamic_trajectories_work() {
        let (model, left_arm, right_arm, _) = build_dual_arm_model();
        let mut buf = model.create_transform_buffer();

        // 1. Test closed-chain bimanual relative position constraint between left_arm and right_arm TCPs
        let q_ref = [0.1, 0.4, 0.2, -0.1, 0.4, 0.2];
        model.forward_kinematics(&q_ref, &mut buf).unwrap();
        let desired_rel = buf.relative_pose(left_arm.end_effectors[0], right_arm.end_effectors[0]);

        let closure = ClosedChainGraspConstraint::with_active_dims(
            left_arm.end_effectors[0],
            right_arm.end_effectors[0],
            desired_rel,
            [true, true, true, false, false, false],
        );

        let mut q_perturbed = [0.25, 0.55, 0.1, -0.3, 0.2, 0.4];
        closure
            .project(&model, &mut q_perturbed, &mut buf, 1e-6, 80)
            .expect("closed-chain projection should converge");

        model.forward_kinematics(&q_perturbed, &mut buf).unwrap();
        let projected_rel =
            buf.relative_pose(left_arm.end_effectors[0], right_arm.end_effectors[0]);
        assert_relative_eq!(
            projected_rel.translation.vector,
            desired_rel.translation.vector,
            epsilon = 1e-5
        );

        // 2. Test TOPP-RA time-optimal parameterization and JerkLimitedTrajectory
        let waypoints = vec![
            Configuration::new(vec![0.0; 6]),
            Configuration::from_slice(&q_ref),
            Configuration::from_slice(&q_perturbed),
        ];
        let spline = CubicSplinePath::from_waypoints(&waypoints).unwrap();
        let toppra = ToppraTrajectory::parameterize(spline, model.dof_limits(), 64).unwrap();
        assert!(toppra.duration() > 0.0);

        let mid_sample = toppra.sample(0.5 * toppra.duration());
        for (k, lim) in model.dof_limits().iter().enumerate() {
            assert!(mid_sample.velocity[k].abs() <= lim.max_velocity + 1e-2);
            assert!(mid_sample.acceleration[k].abs() <= lim.max_acceleration + 1e-1);
        }

        let otg = JerkLimitedTrajectory::generate(
            &[0.0; 6],
            &[0.1; 6],
            &[0.0; 6],
            &q_ref,
            &[0.0; 6],
            &[0.0; 6],
            model.dof_limits(),
            0.1,
        );
        let end_sample = otg.sample(otg.duration());
        for (k, &expected) in q_ref.iter().enumerate() {
            assert_relative_eq!(end_sample.position[k], expected, epsilon = 1e-6);
            assert_relative_eq!(end_sample.velocity[k], 0.0, epsilon = 1e-6);
        }

        // 3. Test 1 kHz ReactiveServoController with CBF obstacle avoidance
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
        let mut servo =
            ReactiveServoController::new(&model, &env, ReactiveServoConfig::default());
        let mut dq_cmd = [0.0; 6];
        servo
            .step(
                &model,
                &env,
                &q_perturbed,
                &[0.0; 6],
                &[ServoTarget::twist(
                    left_arm.end_effectors[0],
                    SVector::<f64, 6>::new(0.05, 0.0, 0.0, 0.0, 0.0, 0.0),
                )],
                Some(&closure),
                &mut dq_cmd,
            )
            .unwrap();
        assert!(dq_cmd.iter().any(|&v| v.abs() > 1e-4));
    }

    #[test]
    fn batch_f32_fk_and_esdf_rollout_match_f64_reference() {
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

        let flat_model = FlatRobotModelF32::compile(&model, &env);
        assert_eq!(flat_model.dof, 6);
        assert_eq!(flat_model.num_links, 6);
        assert!(!flat_model.spheres.is_empty());
        assert!(!flat_model.active_self_pairs.is_empty());

        // Evaluate a B x H = 16 x 16 = 256 rollout batch
        let batch_size = 256;
        let dof = flat_model.dof;
        let mut q_batch_f32 = vec![0.0_f32; batch_size * dof];
        for b in 0..batch_size {
            let phase = b as f32 * 0.03;
            for d in 0..dof {
                q_batch_f32[b * dof + d] = (phase + d as f32 * 0.4).sin() * 0.8;
            }
        }

        let backend = CpuSimdBatchBackend;
        let num_spheres = flat_model.spheres.len();
        let num_ees = flat_model.end_effectors.len();
        let mut out_spheres = vec![[0.0_f32; 4]; batch_size * num_spheres];
        let mut out_ees = vec![[0.0_f32; 7]; batch_size * num_ees];

        backend.forward_kinematics_batch_f32(
            &flat_model,
            &q_batch_f32,
            &mut out_spheres,
            &mut out_ees,
        );

        // Verify batched f32x8 SIMD poses match f64 MultiRobotModel::forward_kinematics
        let mut buf = model.create_transform_buffer();
        for b in [0, 1, 7, 8, 63, 128, 255] {
            let q_f64: Vec<f64> = q_batch_f32[b * dof..(b + 1) * dof]
                .iter()
                .map(|&v| v as f64)
                .collect();
            model.forward_kinematics(&q_f64, &mut buf).unwrap();

            let left_f64 = buf.frame_pose(left_arm.end_effectors[0]);
            let right_f64 = buf.frame_pose(right_arm.end_effectors[0]);
            let left_f32 = batch::pose_f32_to_isometry(&out_ees[b * num_ees]);
            let right_f32 = batch::pose_f32_to_isometry(&out_ees[b * num_ees + 1]);

            assert_relative_eq!(
                left_f64.translation.vector,
                left_f32.translation.vector,
                epsilon = 1e-5
            );
            assert_relative_eq!(
                right_f64.translation.vector,
                right_f32.translation.vector,
                epsilon = 1e-5
            );
        }

        // Carve an obstacle into a 3D ESDF voxel grid and evaluate batched rollout costs
        let mut esdf = EsdfGridF32::new([-1.0, -1.0, -0.5], 0.04, [50, 50, 50], 1.0);
        esdf.add_sphere_obstacle([0.3, 0.35, 0.5], 0.25);

        let mut costs = vec![0.0_f32; batch_size];
        backend.collision_cost_batch_f32(&flat_model, Some(&esdf), &q_batch_f32, 0.15, &mut costs);
        assert!(costs.iter().any(|&c| c > 0.0));

        #[cfg(feature = "gpu")]
        {
            let gpu_backend = WgpuBatchBackend::new().expect("wgpu backend should initialize and compile rollout.wgsl");
            let mut gpu_spheres = vec![[0.0_f32; 4]; batch_size * num_spheres];
            let mut gpu_ees = vec![[0.0_f32; 7]; batch_size * num_ees];
            let mut gpu_costs = vec![0.0_f32; batch_size];

            gpu_backend.forward_kinematics_batch_f32(
                &flat_model,
                &q_batch_f32,
                &mut gpu_spheres,
                &mut gpu_ees,
            );
            gpu_backend.collision_cost_batch_f32(
                &flat_model,
                Some(&esdf),
                &q_batch_f32,
                0.15,
                &mut gpu_costs,
            );

            for (cpu_ee, gpu_ee) in out_ees.iter().zip(gpu_ees.iter()) {
                for i in 0..7 {
                    assert_relative_eq!(cpu_ee[i], gpu_ee[i], epsilon = 1e-4);
                }
            }
            for (&cpu_c, &gpu_c) in costs.iter().zip(gpu_costs.iter()) {
                assert_relative_eq!(cpu_c, gpu_c, epsilon = 1e-4);
            }

            // Verify persistent GPU resident buffers (zero per-step allocations) + update_esdf
            let mut gpu_ctx = GpuRobotContext::new(
                &gpu_backend,
                flat_model.clone(),
                Some(esdf.clone()),
                batch_size,
            );
            let mut res_spheres = vec![[0.0_f32; 4]; batch_size * num_spheres];
            let mut res_ees = vec![[0.0_f32; 7]; batch_size * num_ees];
            let mut res_costs = vec![0.0_f32; batch_size];

            gpu_ctx.forward_kinematics_resident(
                &gpu_backend,
                &q_batch_f32,
                &mut res_spheres,
                &mut res_ees,
            );
            gpu_ctx.collision_cost_resident(&gpu_backend, &q_batch_f32, 0.15, &mut res_costs);

            for (&cpu_c, &res_c) in costs.iter().zip(res_costs.iter()) {
                assert_relative_eq!(cpu_c, res_c, epsilon = 1e-4);
            }

            let mut updated_esdf = esdf.clone();
            updated_esdf.add_sphere_obstacle([0.0, -0.35, 0.5], 0.2);
            gpu_ctx.update_esdf(&gpu_backend, updated_esdf);
            gpu_ctx.collision_cost_resident(&gpu_backend, &q_batch_f32, 0.15, &mut res_costs);
            assert!(res_costs.iter().any(|&c| c > 0.0));
        }
    }

    #[test]
    fn batch_ik_and_mppi_optimizer_avoid_obstacles() {
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

        // Place an obstacle along the straight-line trajectory at q_mid = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
        let mut buf = model.create_transform_buffer();
        let q_mid = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0];
        model.forward_kinematics(&q_mid, &mut buf).unwrap();
        let obs_pos = buf.frame_pose(left_arm.end_effectors[0]).translation.vector;
        env.add_obstacle(
            SharedShape::new(Ball::new(0.14)),
            Isometry3::from(Translation3::from(obs_pos)),
        );

        let mut esdf = EsdfGridF32::new([-1.2, -1.2, -0.5], 0.04, [60, 60, 50], 1.0);
        esdf.add_sphere_obstacle(
            [obs_pos.x as f32, obs_pos.y as f32, obs_pos.z as f32],
            0.14,
        );

        let flat_model = FlatRobotModelF32::compile(&model, &env);
        let backend = CpuSimdBatchBackend;

        // 1. Multi-seed collision-free IK
        let q_goal_ref = [0.25, -0.35, 0.45, -0.25, 0.35, -0.45];
        model.forward_kinematics(&q_goal_ref, &mut buf).unwrap();
        let left_target = TaskPose::new(
            left_arm.end_effectors[0],
            *buf.frame_pose(left_arm.end_effectors[0]),
        );
        let right_target = TaskPose::new(
            right_arm.end_effectors[0],
            *buf.frame_pose(right_arm.end_effectors[0]),
        );

        let ik_solver = BatchIkSolver::new(24);
        let q_sol = ik_solver
            .solve_collision_free(
                &backend,
                &model,
                &flat_model,
                &env,
                Some(&esdf),
                &[0.0; 6],
                &[left_target, right_target],
            )
            .expect("BatchIkSolver should find a collision-free solution");

        let mut ws = env.create_workspace(&model);
        assert!(env
            .is_state_collision_free(&model, q_sol.as_slice(), GeometryTier::Convex, &mut ws)
            .unwrap());

        // 2. Batched MPPI trajectory optimization around the obstacle
        let start = [0.0, -0.2, 0.0, 0.0, 0.0, 0.0];
        let goal = [0.0, 1.1, 0.0, 0.0, 0.0, 0.0];

        // Evaluate straight-line initial collision cost vs MPPI-optimized collision cost
        let h = 16;
        let mppi = BatchMppiOptimizer::new(MppiConfig {
            num_rollouts: 64,
            horizon: h,
            num_iterations: 30,
            noise_std: 0.35,
            temperature: 0.12,
            step_size: 0.8,
            activation_distance: 0.14,
            collision_weight: 200.0,
            velocity_weight: 2.0,
            acceleration_weight: 4.0,
        });

        let mut init_batch = Vec::with_capacity(h * 6);
        for t in 0..h {
            let a = (t as f64) / ((h - 1) as f64);
            for d in 0..6 {
                init_batch.push(((1.0 - a) * start[d] + a * goal[d]) as f32);
            }
        }
        let mut init_costs = vec![0.0_f32; h];
        backend.collision_cost_batch_f32(&flat_model, Some(&esdf), &init_batch, 0.14, &mut init_costs);
        let init_total_coll: f32 = init_costs.iter().sum();
        assert!(init_total_coll > 0.0, "straight line should intersect obstacle activation zone");

        let opt_traj = mppi.optimize(&backend, &model, &flat_model, Some(&esdf), &start, &goal);
        assert_eq!(opt_traj.len(), h);
        assert_eq!(opt_traj.first().unwrap().as_slice(), &start);
        assert_eq!(opt_traj.last().unwrap().as_slice(), &goal);

        let mut opt_batch = Vec::with_capacity(h * 6);
        for cfg in &opt_traj {
            for &v in cfg.as_slice() {
                opt_batch.push(v as f32);
            }
        }
        let mut opt_costs = vec![0.0_f32; h];
        backend.collision_cost_batch_f32(&flat_model, Some(&esdf), &opt_batch, 0.14, &mut opt_costs);
        let opt_total_coll: f32 = opt_costs.iter().sum();
        assert!(
            opt_total_coll < init_total_coll * 0.25,
            "MPPI should substantially reduce collision cost (init={init_total_coll}, opt={opt_total_coll})"
        );
    }

    #[test]
    fn attached_payload_collision_witnesses_and_jacobians_match_fd() {
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

        // Attach a multi-part compound tool to left wrist (LinkId(2)) with ZERO obstacles first.
        let left_wrist = *left_arm.active_links.last().unwrap();
        let left_forearm = left_arm.active_links[1];
        let right_wrist = *right_arm.active_links.last().unwrap();
        let left_tool = AttachedPayload::from_compound(
            "left_tool",
            left_wrist,
            Isometry3::translation(0.0, 0.0, 0.3),
            vec![
                (
                    Isometry3::translation(0.0, 0.0, 0.05),
                    SharedShape::new(parry3d::shape::Cuboid::new(parry3d::math::Vector::new(
                        0.03, 0.03, 0.05,
                    ))),
                ),
                (
                    Isometry3::translation(0.0, 0.0, 0.16),
                    SharedShape::new(parry3d::shape::Capsule::new_z(0.05, 0.015)),
                ),
            ],
        )
        .with_touch_links(vec![left_forearm, left_wrist]);
        let left_pid = env.attach_payload(left_tool);

        let mut ws = env.create_workspace(&model);

        // 1. Verify Payload <-> Link collision detection works even with ZERO environment obstacles
        // Bend both arms directly toward each other along Y so left_tool penetrates right_wrist.
        let q_collide = [
            -std::f64::consts::FRAC_PI_2,
            0.7,
            0.55,
            std::f64::consts::FRAC_PI_2,
            0.7,
            0.55,
        ];
        assert!(
            !env.is_state_collision_free(&model, &q_collide, GeometryTier::Convex, &mut ws)
                .unwrap(),
            "tool colliding with other arm must be caught even with zero static obstacles"
        );
        assert!(
            !env.is_state_collision_free(&model, &q_collide, GeometryTier::AnalyticSsv, &mut ws)
                .unwrap(),
            "AnalyticSsv tool-vs-link check must also detect collision"
        );

        // 2. Verify touch_links exemption: fold left arm tightly without hitting right arm
        let q_touch_safe = [-0.2, 0.2, 2.2, 0.0, -0.8, 0.0];
        assert!(
            env.is_state_collision_free(&model, &q_touch_safe, GeometryTier::Convex, &mut ws)
                .unwrap(),
            "touch_links should exempt parent & adjacent forearm links from payload collision"
        );

        // 3. Attach second tool to right wrist and add an obstacle near left tool
        let right_tool = AttachedPayload::from_shape(
            "right_tool",
            right_wrist,
            Isometry3::translation(0.0, 0.0, 0.3),
            SharedShape::new(parry3d::shape::Capsule::new_z(0.08, 0.015)),
        )
        .with_touch_links(vec![right_arm.active_links[1], right_wrist]);
        let right_pid = env.attach_payload(right_tool);

        let q_near = [
            -std::f64::consts::FRAC_PI_2,
            0.22,
            0.18,
            std::f64::consts::FRAC_PI_2,
            0.22,
            0.18,
        ];
        let mut buf = model.create_transform_buffer();
        model.forward_kinematics(&q_near, &mut buf).unwrap();
        let left_wrist_pose = *buf.link_pose(left_wrist);
        let obs_center = left_wrist_pose.transform_point(&Point3::new(0.15, 0.0, 0.45));
        let obs_id = env.add_obstacle(
            SharedShape::new(Ball::new(0.05)),
            Isometry3::from(Translation3::from(obs_center.coords)),
        );

        assert!(env
            .is_state_collision_free(&model, &q_near, GeometryTier::Convex, &mut ws)
            .unwrap());

        let mut witnesses = Vec::new();
        env.query_signed_distances(
            &model,
            &q_near,
            1.5,
            GeometryTier::Convex,
            &mut ws,
            &mut witnesses,
        )
        .unwrap();

        let w_obs = witnesses
            .iter()
            .find(|w| {
                matches!(
                    w.pair,
                    CollidingPair::PayloadToObstacle {
                        payload,
                        obstacle,
                        ..
                    } if payload == left_pid && obstacle == obs_id
                )
            })
            .expect("expected PayloadToObstacle witness")
            .clone();
        let w_link = witnesses
            .iter()
            .find(|w| {
                matches!(
                    w.pair,
                    CollidingPair::PayloadToLink {
                        payload,
                        link,
                        ..
                    } if payload == left_pid && link == right_wrist
                )
            })
            .expect("expected PayloadToLink witness")
            .clone();
        let w_pay = witnesses
            .iter()
            .find(|w| {
                matches!(
                    w.pair,
                    CollidingPair::PayloadToPayload {
                        payload_a,
                        payload_b,
                        ..
                    } if payload_a == left_pid && payload_b == right_pid
                )
            })
            .expect("expected PayloadToPayload witness")
            .clone();

        // Verify analytical distance Jacobians for all 3 payload witness types against FD
        let eps = 1e-6;
        let mut j_dist = vec![0.0; model.dof()];
        for witness in [&w_obs, &w_link, &w_pay] {
            witness.compute_distance_jacobian(&model, &ws.transforms, &mut j_dist);
            for k in 0..model.dof() {
                let mut q_plus = q_near;
                let mut q_minus = q_near;
                q_plus[k] += eps;
                q_minus[k] -= eps;
                let mut w_p = Vec::new();
                let mut w_m = Vec::new();
                env.query_signed_distances(
                    &model,
                    &q_plus,
                    1.5,
                    GeometryTier::Convex,
                    &mut ws,
                    &mut w_p,
                )
                .unwrap();
                env.query_signed_distances(
                    &model,
                    &q_minus,
                    1.5,
                    GeometryTier::Convex,
                    &mut ws,
                    &mut w_m,
                )
                .unwrap();
                let d_p = w_p
                    .iter()
                    .find(|w| w.pair == witness.pair)
                    .unwrap()
                    .signed_distance;
                let d_m = w_m
                    .iter()
                    .find(|w| w.pair == witness.pair)
                    .unwrap()
                    .signed_distance;
                let fd = (d_p - d_m) / (2.0 * eps);
                assert_relative_eq!(j_dist[k], fd, epsilon = 1e-4);
            }
        }

        // 4. Verify is_segment_collision_free catches an attached tool sweeping through an obstacle
        let q_sweep_a = [0.0, 0.3, 0.2, 0.0, -0.8, 0.0];
        let mut q_sweep_mid = q_sweep_a;
        q_sweep_mid[0] = -0.60;
        let mut q_sweep_end = q_sweep_a;
        q_sweep_end[0] = -1.20;
        model.forward_kinematics(&q_sweep_mid, &mut buf).unwrap();
        let sweep_tip = buf
            .link_pose(left_wrist)
            .transform_point(&Point3::new(0.0, 0.0, 0.45));
        let mut env_sweep = CollisionEnvironment::new();
        env_sweep.set_acm(AllowedCollisionMatrix::from_model(&model));
        for link in model.links() {
            env_sweep.set_link_geometry(
                link.id,
                LinkGeometry::from_capsule(
                    Point3::origin(),
                    Point3::new(0.0, 0.0, 0.3),
                    link.bounding_radius,
                ),
            );
        }
        env_sweep.attach_payload(
            AttachedPayload::from_shape(
                "long_probe",
                left_wrist,
                Isometry3::translation(0.0, 0.0, 0.45),
                SharedShape::new(Ball::new(0.04)),
            )
            .with_touch_links(vec![left_forearm, left_wrist]),
        );
        env_sweep.add_obstacle(
            SharedShape::new(Ball::new(0.05)),
            Isometry3::from(Translation3::from(sweep_tip.coords)),
        );
        let mut ws_sweep = env_sweep.create_workspace(&model);
        assert!(env_sweep
            .is_state_collision_free(&model, &q_sweep_a, GeometryTier::Convex, &mut ws_sweep)
            .unwrap());
        assert!(env_sweep
            .is_state_collision_free(&model, &q_sweep_end, GeometryTier::Convex, &mut ws_sweep)
            .unwrap());
        assert!(
            !env_sweep
                .is_segment_collision_free(
                    &model,
                    &q_sweep_a,
                    &q_sweep_end,
                    0.03,
                    GeometryTier::Convex,
                    &mut ws_sweep
                )
                .unwrap(),
            "Lipschitz CCD must catch a long attached tool sweeping through an obstacle"
        );
    }

    #[test]
    fn batch_f32_includes_attached_payloads_and_pair_margins_cpu_and_gpu() {
        let (model, left_arm, right_arm, _) = build_dual_arm_model();
        let mut env = CollisionEnvironment::new();
        let mut acm = AllowedCollisionMatrix::from_model(&model);
        acm.set_inter_arm_margin(&model, 0.06);
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

        let left_wrist = *left_arm.active_links.last().unwrap();
        let right_wrist = *right_arm.active_links.last().unwrap();
        env.attach_payload(
            AttachedPayload::from_shape(
                "tool_l",
                left_wrist,
                Isometry3::translation(0.0, 0.0, 0.35),
                SharedShape::new(Ball::new(0.02)),
            )
            .with_touch_links(vec![left_arm.active_links[1], left_wrist])
            .with_safety_margin(0.003)
            .with_activation_distance(0.02),
        );
        env.attach_payload(
            AttachedPayload::from_shape(
                "tool_r",
                right_wrist,
                Isometry3::translation(0.0, 0.0, 0.35),
                SharedShape::new(Ball::new(0.02)),
            )
            .with_touch_links(vec![right_arm.active_links[1], right_wrist])
            .with_safety_margin(0.003)
            .with_activation_distance(0.02),
        );
        env.set_payload_pair_margin("tool_l", "tool_r", 0.004);
        env.set_payload_pair_activation("tool_l", "tool_r", 0.02);

        let flat = FlatRobotModelF32::compile_with_default_margin(&model, &env, 0.04);
        assert!(flat.spheres.iter().any(|s| s.flags == SPHERE_FLAG_PAYLOAD));
        assert!(flat
            .active_self_pairs
            .iter()
            .any(|p| (p.activation_distance - 0.02).abs() < 1e-5));

        let mut esdf = EsdfGridF32::new([-1.2, -1.2, -0.4], 0.04, [60, 60, 45], 1.0);
        esdf.add_sphere_obstacle([0.45, 0.0, 0.65], 0.12);

        // Verify pair-specific activation override: construct a batch of varied configs
        let batch_size = 64;
        let dof = flat.dof;
        let mut q_batch = vec![0.0_f32; batch_size * dof];
        for b in 0..batch_size {
            let phase = b as f32 * 0.08;
            q_batch[b * dof] = -1.57 + 0.3 * phase.sin();
            q_batch[b * dof + 1] = 0.5 + 0.3 * phase.cos();
            q_batch[b * dof + 2] = 0.4 + 0.2 * phase.sin();
            q_batch[b * dof + 3] = 1.57 - 0.3 * phase.sin();
            q_batch[b * dof + 4] = 0.5 + 0.3 * phase.cos();
            q_batch[b * dof + 5] = 0.4 + 0.2 * phase.cos();
        }

        let cpu = CpuSimdBatchBackend;
        let mut cpu_costs = vec![0.0_f32; batch_size];
        cpu.collision_cost_batch_f32(&flat, Some(&esdf), &q_batch, 0.12, &mut cpu_costs);
        assert!(cpu_costs.iter().any(|&c| c > 0.0));

        // Specifically verify that tool-to-tool separation of ~35mm (> 4mm margin + 20mm pair activation)
        // yields ZERO cost even when global activation_distance is large (0.15m), proving custom activation works!
        let flat_tool_only = {
            let mut m_only = FlatRobotModelF32::compile(&model, &env);
            m_only.active_self_pairs.retain(|p| {
                flat.spheres[p.sphere_a as usize].flags == SPHERE_FLAG_PAYLOAD
                    && flat.spheres[p.sphere_b as usize].flags == SPHERE_FLAG_PAYLOAD
            });
            m_only
        };
        // Find distance when tool tips are separated (> 4mm margin + 20mm pair activation)
        let q_gap = [
            -std::f64::consts::FRAC_PI_2,
            0.20,
            0.12,
            std::f64::consts::FRAC_PI_2,
            0.20,
            0.12,
        ];
        let mut buf = model.create_transform_buffer();
        model.forward_kinematics(&q_gap, &mut buf).unwrap();
        let p_tl = buf
            .link_pose(left_wrist)
            .transform_point(&Point3::new(0.0, 0.0, 0.35));
        let p_tr = buf
            .link_pose(right_wrist)
            .transform_point(&Point3::new(0.0, 0.0, 0.35));
        let gap = (p_tl - p_tr).norm() - 0.04;
        assert!(gap > 0.024, "expected tool gap > 0.024, got {gap}");
        let q_gap_f32 = batch::config_to_f32(&q_gap);
        let mut gap_cost = [0.0_f32; 1];
        cpu.collision_cost_batch_f32(&flat_tool_only, None, &q_gap_f32, 0.50, &mut gap_cost);
        assert_relative_eq!(gap_cost[0], 0.0, epsilon = 1e-6);

        #[cfg(feature = "gpu")]
        {
            let gpu = WgpuBatchBackend::new().expect("wgpu backend should initialize");
            let mut gpu_costs = vec![0.0_f32; batch_size];
            gpu.collision_cost_batch_f32(&flat, Some(&esdf), &q_batch, 0.12, &mut gpu_costs);
            for (&c, &g) in cpu_costs.iter().zip(gpu_costs.iter()) {
                assert_relative_eq!(c, g, epsilon = 1e-4);
            }

            let gpu_ctx = GpuRobotContext::new(&gpu, flat.clone(), Some(esdf.clone()), batch_size);
            let mut res_costs = vec![0.0_f32; batch_size];
            gpu_ctx.collision_cost_resident(&gpu, &q_batch, 0.12, &mut res_costs);
            for (&c, &r) in cpu_costs.iter().zip(res_costs.iter()) {
                assert_relative_eq!(c, r, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn priority_weighted_qp_cbf_and_curl_break_symmetric_deadlock() {
        let (model, left_arm, right_arm, _) = build_dual_arm_model();
        let mut env = CollisionEnvironment::new();
        env.set_acm(AllowedCollisionMatrix::from_model(&model));
        for link in model.links() {
            let cap_len = if link.id == LinkId(2) || link.id == LinkId(5) {
                0.65
            } else {
                0.30
            };
            env.set_link_geometry(
                link.id,
                LinkGeometry::from_capsule(
                    Point3::origin(),
                    Point3::new(0.0, 0.0, cap_len),
                    link.bounding_radius,
                ),
            );
        }

        // 1. Priority weighting test: place left & right arms facing each other symmetrically along Y.
        let q_sym = [
            -std::f64::consts::FRAC_PI_2,
            -0.50,
            1.30,
            std::f64::consts::FRAC_PI_2,
            -0.50,
            1.30,
        ];
        let mut ws = env.create_workspace(&model);
        assert!(env
            .is_state_collision_free(&model, &q_sym, GeometryTier::Convex, &mut ws)
            .unwrap());

        let left_tcp = left_arm.end_effectors[0];
        let right_tcp = right_arm.end_effectors[0];

        let cfg_prio = ReactiveServoConfig {
            dt: 0.1,
            safety_margin: 0.06,
            cbf_activation_distance: 0.35,
            cbf_gamma: 8.0,
            damping: 0.02,
            posture_gain: 0.0,
            tangential_curl_gain: 0.0,
            ..ReactiveServoConfig::default()
        }
        .with_arm_priority(RobotId(0), 10.0)
        .with_arm_priority(RobotId(1), 1.0);

        let mut servo_prio = ReactiveServoController::new(&model, &env, cfg_prio);
        let mut dq_cmd = [0.0; 6];
        let targets_approach = [
            ServoTarget::twist(left_tcp, SVector::<f64, 6>::new(0.0, -0.35, 0.0, 0.0, 0.0, 0.0)),
            ServoTarget::twist(right_tcp, SVector::<f64, 6>::new(0.0, 0.35, 0.0, 0.0, 0.0, 0.0)),
        ];
        servo_prio
            .step(
                &model,
                &env,
                &q_sym,
                &[0.0; 6],
                &targets_approach,
                None,
                &mut dq_cmd,
            )
            .unwrap();

        let mut buf = model.create_transform_buffer();
        model.forward_kinematics(&q_sym, &mut buf).unwrap();
        let mut j_l = DMatrix::<f64>::zeros(6, 6);
        let mut j_r = DMatrix::<f64>::zeros(6, 6);
        model.geometric_jacobian(&buf, left_tcp, j_l.as_view_mut()).unwrap();
        model.geometric_jacobian(&buf, right_tcp, j_r.as_view_mut()).unwrap();
        let dq_vec = nalgebra::DVector::from_column_slice(&dq_cmd);
        let v_l = (&j_l * &dq_vec)[1]; // Y velocity of left TCP (desires -0.35)
        let v_r = (&j_r * &dq_vec)[1]; // Y velocity of right TCP (desires +0.35)
        assert!(
            v_l < -0.20 && v_r < -0.10,
            "higher-priority left arm should keep advancing -Y while lower-priority right arm backs away (-Y) (v_l={v_l:.4}, v_r={v_r:.4})"
        );

        // 2. Deadlock breaking via tangential curl:
        // 2a. Exact radial obstacle deadlock: without curl, motion stays trapped on y = +0.35 plane;
        // with curl, arm swings around the obstacle and converges cleanly.
        let mut env_obs_deadlock = CollisionEnvironment::new();
        env_obs_deadlock.set_acm(AllowedCollisionMatrix::from_model(&model));
        for link in model.links() {
            env_obs_deadlock.set_link_geometry(
                link.id,
                LinkGeometry::from_capsule(
                    Point3::origin(),
                    Point3::new(0.0, 0.0, 0.30),
                    link.bounding_radius,
                ),
            );
        }
        let left_wrist = *left_arm.active_links.last().unwrap();
        let right_wrist = *right_arm.active_links.last().unwrap();
        env_obs_deadlock.attach_payload(
            AttachedPayload::from_shape(
                "probe_l",
                left_wrist,
                Isometry3::translation(0.0, 0.0, 0.30),
                SharedShape::new(Ball::new(0.035)),
            )
            .with_touch_links(vec![left_arm.active_links[1], left_wrist]),
        );

        let q_rad_start = [0.0, -0.35, 1.45, 0.0, -0.8, 0.0];
        let q_rad_goal = [0.0, 0.55, 0.70, 0.0, -0.8, 0.0];
        model.forward_kinematics(&q_rad_start, &mut buf).unwrap();
        let p_rad_start = buf.frame_pose(left_tcp).translation.vector;
        model.forward_kinematics(&q_rad_goal, &mut buf).unwrap();
        let p_rad_goal = buf.frame_pose(left_tcp).translation.vector;
        let obs_mid = Vector3::new(0.31, p_rad_start.y, 0.5 * (p_rad_start.z + p_rad_goal.z));
        // Vertical cylinder along Z at y = +0.35 blocks straight radial X-motion while keeping exact symmetry.
        let vertical_rot = nalgebra::UnitQuaternion::from_axis_angle(
            &Vector3::x_axis(),
            std::f64::consts::FRAC_PI_2,
        );
        env_obs_deadlock.add_obstacle(
            SharedShape::new(parry3d::shape::Cylinder::new(0.40, 0.04)),
            Isometry3::from_parts(Translation3::from(obs_mid), vertical_rot),
        );
        assert!(
            env_obs_deadlock
                .is_state_collision_free(&model, &q_rad_start, GeometryTier::Convex, &mut ws)
                .unwrap(),
            "start state for radial deadlock test must be collision-free"
        );

        let rad_target = [ServoTarget::position(
            left_tcp,
            Point3::from(p_rad_goal),
            1.8,
        )];
        let mut q_nocurl = q_rad_start;
        let mut qd_nocurl = [0.0; 6];
        let mut servo_nocurl = ReactiveServoController::new(
            &model,
            &env_obs_deadlock,
            ReactiveServoConfig {
                dt: 0.02,
                safety_margin: 0.03,
                cbf_activation_distance: 0.25,
                cbf_gamma: 5.0,
                damping: 0.02,
                posture_gain: 0.0,
                tangential_curl_gain: 0.0,
                ..ReactiveServoConfig::default()
            },
        );
        for _ in 0..150 {
            servo_nocurl
                .step(
                    &model,
                    &env_obs_deadlock,
                    &q_nocurl,
                    &qd_nocurl,
                    &rad_target,
                    None,
                    &mut dq_cmd,
                )
                .unwrap();
            for k in 0..6 {
                q_nocurl[k] = model.dof_limits()[k]
                    .position
                    .clamp(q_nocurl[k] + dq_cmd[k] * 0.02);
                qd_nocurl[k] = dq_cmd[k];
            }
            assert!(
                env_obs_deadlock
                    .is_state_collision_free(&model, &q_nocurl, GeometryTier::Convex, &mut ws)
                    .unwrap(),
                "no-curl trajectory must still remain collision-free while stalled"
            );
        }
        model.forward_kinematics(&q_nocurl, &mut buf).unwrap();
        let stalled_err = (buf.frame_pose(left_tcp).translation.vector - p_rad_goal).norm();
        assert!(
            stalled_err > 0.12 && q_nocurl[0].abs() < 1e-3,
            "without tangential curl, collinear obstacle must stall arm in symmetry plane (stalled_err={stalled_err:.4}, q0={:.4})",
            q_nocurl[0]
        );

        // Verify on a single tick that curl immediately breaks the y = +0.35 symmetry when blocked
        let mut servo_curl_impulse = ReactiveServoController::new(
            &model,
            &env_obs_deadlock,
            ReactiveServoConfig {
                dt: 0.02,
                safety_margin: 0.03,
                cbf_activation_distance: 0.25,
                cbf_gamma: 5.0,
                damping: 0.02,
                posture_gain: 0.0,
                tangential_curl_gain: 0.9,
                ..ReactiveServoConfig::default()
            },
        );
        servo_curl_impulse
            .step(
                &model,
                &env_obs_deadlock,
                &q_nocurl,
                &qd_nocurl,
                &rad_target,
                None,
                &mut dq_cmd,
            )
            .unwrap();
        assert!(
            dq_cmd[0].abs() > 0.05,
            "tangential curl should immediately inject lateral yaw escape velocity (dq0={:.4})",
            dq_cmd[0]
        );

        // With a bypassable sphere obstacle along the straight path, tangential curl routes around it and converges
        let mut env_obs_bypass = CollisionEnvironment::new();
        env_obs_bypass.set_acm(AllowedCollisionMatrix::from_model(&model));
        for link in model.links() {
            env_obs_bypass.set_link_geometry(
                link.id,
                LinkGeometry::from_capsule(
                    Point3::origin(),
                    Point3::new(0.0, 0.0, 0.30),
                    link.bounding_radius,
                ),
            );
        }
        env_obs_bypass.attach_payload(
            AttachedPayload::from_shape(
                "probe_l",
                left_wrist,
                Isometry3::translation(0.0, 0.0, 0.30),
                SharedShape::new(Ball::new(0.035)),
            )
            .with_touch_links(vec![left_arm.active_links[1], left_wrist]),
        );
        env_obs_bypass.add_obstacle(
            SharedShape::new(Ball::new(0.045)),
            Isometry3::from(Translation3::from(Vector3::new(0.31, p_rad_start.y, 0.85))),
        );

        let mut q_rad_curl = q_rad_start;
        let mut qd_rad_curl = [0.0; 6];
        let mut servo_rad_curl = ReactiveServoController::new(
            &model,
            &env_obs_bypass,
            ReactiveServoConfig {
                dt: 0.02,
                safety_margin: 0.02,
                cbf_activation_distance: 0.25,
                cbf_gamma: 5.0,
                damping: 0.02,
                posture_gain: 0.0,
                tangential_curl_gain: 0.9,
                ..ReactiveServoConfig::default()
            },
        );
        for _ in 0..240 {
            servo_rad_curl
                .step(
                    &model,
                    &env_obs_bypass,
                    &q_rad_curl,
                    &qd_rad_curl,
                    &rad_target,
                    None,
                    &mut dq_cmd,
                )
                .unwrap();
            for k in 0..6 {
                q_rad_curl[k] = model.dof_limits()[k]
                    .position
                    .clamp(q_rad_curl[k] + dq_cmd[k] * 0.02);
                qd_rad_curl[k] = dq_cmd[k];
            }
            assert!(
                env_obs_bypass
                    .is_state_collision_free(&model, &q_rad_curl, GeometryTier::Convex, &mut ws)
                    .unwrap(),
                "radial curl trajectory must stay strictly collision-free"
            );
        }
        model.forward_kinematics(&q_rad_curl, &mut buf).unwrap();
        let unblocked_err = (buf.frame_pose(left_tcp).translation.vector - p_rad_goal).norm();
        assert!(
            unblocked_err < 0.02,
            "tangential curl should route around collinear obstacle (unblocked_err={unblocked_err:.4})"
        );

        // 2b. Dual-arm head-on crossing with tool payloads converges cleanly and collision-free
        let mut env_cross = CollisionEnvironment::new();
        env_cross.set_acm(AllowedCollisionMatrix::from_model(&model));
        for link in model.links() {
            env_cross.set_link_geometry(
                link.id,
                LinkGeometry::from_capsule(
                    Point3::origin(),
                    Point3::new(0.0, 0.0, 0.30),
                    link.bounding_radius,
                ),
            );
        }
        env_cross.attach_payload(
            AttachedPayload::from_shape(
                "cross_tool_l",
                left_wrist,
                Isometry3::translation(0.0, 0.0, 0.30),
                SharedShape::new(Ball::new(0.045)),
            )
            .with_touch_links(vec![left_arm.active_links[1], left_wrist]),
        );
        env_cross.attach_payload(
            AttachedPayload::from_shape(
                "cross_tool_r",
                right_wrist,
                Isometry3::translation(0.0, 0.0, 0.30),
                SharedShape::new(Ball::new(0.045)),
            )
            .with_touch_links(vec![right_arm.active_links[1], right_wrist]),
        );

        model
            .forward_kinematics(&[0.0, 0.35, 0.70, 0.0, 0.35, 0.70], &mut buf)
            .unwrap();
        let r_actual = buf.frame_pose(left_tcp).translation.x;
        let x_off = (r_actual * r_actual - 0.35 * 0.35).sqrt();
        let q0_left = (-0.35_f64).atan2(-x_off);
        let q0_right = (0.35_f64).atan2(x_off);
        let mut q_curr = [q0_left, 0.35, 0.70, q0_right, 0.35, 0.70];
        let mut qd_curr = [0.0; 6];
        model.forward_kinematics(&q_curr, &mut buf).unwrap();
        let p_left_start = buf.frame_pose(left_tcp).translation.vector;
        let p_right_start = buf.frame_pose(right_tcp).translation.vector;
        let left_goal = Point3::from(p_right_start);
        let right_goal = Point3::from(p_left_start);

        let swap_targets = [
            ServoTarget::position(left_tcp, left_goal, 4.0),
            ServoTarget::position(right_tcp, right_goal, 4.0),
        ];

        let cfg_curl = ReactiveServoConfig {
            dt: 0.02,
            safety_margin: 0.025,
            cbf_activation_distance: 0.25,
            cbf_gamma: 8.0,
            damping: 0.02,
            posture_gain: 0.0,
            tangential_curl_gain: 0.8,
            ..ReactiveServoConfig::default()
        };
        let mut servo_curl = ReactiveServoController::new(&model, &env_cross, cfg_curl);

        for _ in 0..220 {
            servo_curl
                .step(
                    &model,
                    &env_cross,
                    &q_curr,
                    &qd_curr,
                    &swap_targets,
                    None,
                    &mut dq_cmd,
                )
                .unwrap();
            for k in 0..6 {
                q_curr[k] = model.dof_limits()[k]
                    .position
                    .clamp(q_curr[k] + dq_cmd[k] * 0.02);
                qd_curr[k] = dq_cmd[k];
            }
            assert!(
                env_cross
                    .is_state_collision_free(&model, &q_curr, GeometryTier::Convex, &mut ws)
                    .unwrap(),
                "reactive curl trajectory must remain strictly collision-free"
            );
        }

        model.forward_kinematics(&q_curr, &mut buf).unwrap();
        let p_l_end = buf.frame_pose(left_tcp).translation.vector;
        let p_r_end = buf.frame_pose(right_tcp).translation.vector;
        let err_l = (p_l_end - left_goal.coords).norm();
        let err_r = (p_r_end - right_goal.coords).norm();
        assert!(
            err_l < 0.02 && err_r < 0.02,
            "tangential curl should resolve symmetric head-on crossing without deadlock (err_l={err_l:.4}, err_r={err_r:.4}, p_l={p_l_end:?}, p_r={p_r_end:?}, q={q_curr:?})"
        );
    }

    #[test]
    fn esdf_carve_collision_environment_produces_signed_distances() {
        use parry3d::math::Vector as ParryVector;
        use parry3d::shape::{Cuboid, Cylinder};

        let mut env = CollisionEnvironment::new();
        let box_id = env.add_obstacle(
            SharedShape::new(Cuboid::new(ParryVector::new(0.08, 0.06, 0.05))),
            Isometry3::from_parts(
                Translation3::new(0.0, 0.0, 0.20),
                nalgebra::UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.45),
            ),
        );
        let cyl_rot =
            nalgebra::UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f64::consts::FRAC_PI_2);
        env.add_obstacle(
            SharedShape::new(Cylinder::new(0.10, 0.04)),
            Isometry3::from_parts(Translation3::new(0.25, 0.0, 0.20), cyl_rot),
        );

        let mut esdf = EsdfGridF32::new([-0.3, -0.3, 0.0], 0.02, [40, 30, 20], 1.0);
        esdf.carve_collision_environment(&env);

        // Interior of box and cylinder must be negative
        let (d_box, _) = esdf.sample_distance_and_gradient([0.0, 0.0, 0.20]).unwrap();
        let (d_cyl, _) = esdf.sample_distance_and_gradient([0.25, 0.0, 0.20]).unwrap();
        assert!(d_box < -0.02, "expected negative distance inside box, got {d_box}");
        assert!(d_cyl < -0.01, "expected negative distance inside cylinder, got {d_cyl}");

        // Move box via set_obstacle_pose and re-carve
        assert!(env.set_obstacle_pose(box_id, Isometry3::translation(-0.15, 0.0, 0.20)));
        esdf.clear(1.0);
        assert!(esdf.distances.iter().all(|&d| (d - 1.0).abs() < 1e-6));
        esdf.carve_collision_environment(&env);
        let (d_old, _) = esdf.sample_distance_and_gradient([0.0, 0.0, 0.20]).unwrap();
        let (d_new, _) = esdf.sample_distance_and_gradient([-0.15, 0.0, 0.20]).unwrap();
        assert!(d_old > 0.04, "old box location should be free, got {d_old}");
        assert!(d_new < -0.02, "moved box location should be interior, got {d_new}");
    }

    #[test]
    fn batch_mppi_optimize_with_trace_matches_optimize_and_exposes_rollouts() {
        let (model, left_arm, right_arm, _) = build_dual_arm_model();
        let mut env = CollisionEnvironment::new();
        env.set_acm(AllowedCollisionMatrix::from_model(&model));
        for link in model.links() {
            env.set_link_geometry(
                link.id,
                LinkGeometry::from_capsule(
                    Point3::origin(),
                    Point3::new(0.0, 0.0, 0.30),
                    link.bounding_radius,
                ),
            );
        }
        let mut esdf = EsdfGridF32::new([-1.0, -1.0, -0.2], 0.05, [40, 40, 28], 1.0);
        esdf.add_sphere_obstacle([0.3, 0.0, 0.65], 0.12);
        let flat = FlatRobotModelF32::compile(&model, &env);

        let _ = (left_arm, right_arm);
        let start = [-0.4, 0.3, 0.4, 0.4, 0.3, 0.4];
        let goal = [0.4, 0.6, 0.6, -0.4, 0.6, 0.6];

        let opt = BatchMppiOptimizer::new(MppiConfig {
            num_rollouts: 16,
            horizon: 10,
            num_iterations: 8,
            ..MppiConfig::default()
        });
        let trace = opt.optimize_with_trace(&CpuSimdBatchBackend, &model, &flat, Some(&esdf), &start, &goal);
        let nominal = opt.optimize(&CpuSimdBatchBackend, &model, &flat, Some(&esdf), &start, &goal);

        assert_eq!(trace.nominal, nominal);
        assert_eq!(trace.final_rollouts.len(), 16);
        assert_eq!(trace.rollout_costs.len(), 16);
        assert_eq!(trace.rollout_weights.len(), 16);
        assert_eq!(trace.iteration_min_costs.len(), 8);
        let weight_sum: f64 = trace.rollout_weights.iter().sum();
        assert_relative_eq!(weight_sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn joint_reference_cbf_filters_colliding_toppra_trajectory_safely() {
        let (model, left_arm, _right_arm, _dual_arms) = build_dual_arm_model();
        let mut env = CollisionEnvironment::new();
        env.set_acm(AllowedCollisionMatrix::from_model(&model));
        for link in model.links() {
            env.set_link_geometry(
                link.id,
                LinkGeometry::from_capsule(
                    Point3::origin(),
                    Point3::new(0.0, 0.0, 0.28),
                    link.bounding_radius,
                ),
            );
        }

        let start = [0.0, -0.35, -0.35, 0.0, -0.35, -0.35];
        let mid = [0.0, 0.0, 0.0, 0.0, -0.35, -0.35];
        let goal = [0.0, 0.55, 0.45, 0.0, -0.35, -0.35];
        let spline = CubicSplinePath::from_waypoints(&[
            Configuration::from_slice(&start),
            Configuration::from_slice(&mid),
            Configuration::from_slice(&goal),
        ])
        .unwrap();
        let traj = ToppraTrajectory::parameterize(spline, model.dof_limits(), 25).unwrap();

        // 1. Track unobstructed TOPP-RA trajectory using ServoTarget::JointReference
        let mut q = start;
        let mut dq = [0.0_f64; 6];
        let mut dq_cmd = [0.0_f64; 6];
        let mut q_ref = vec![0.0_f64; 6];
        let mut dq_ref = vec![0.0_f64; 6];
        let mut ddq_ref = vec![0.0_f64; 6];

        let cfg = ReactiveServoConfig {
            dt: 0.005,
            safety_margin: 0.04,
            cbf_activation_distance: 0.25,
            cbf_gamma: 6.0,
            ..ReactiveServoConfig::default()
        };
        let mut servo = ReactiveServoController::new(&model, &env, cfg.clone());
        let steps = ((traj.duration() / cfg.dt).ceil() as usize) + 80;
        for step in 0..steps {
            let t = (step as f64) * cfg.dt;
            traj.sample_into(t, &mut q_ref, &mut dq_ref, &mut ddq_ref);
            servo
                .step(
                    &model,
                    &env,
                    &q,
                    &dq,
                    &[ServoTarget::joint_reference(
                        q_ref.clone(),
                        dq_ref.clone(),
                        10.0,
                    )],
                    None,
                    &mut dq_cmd,
                )
                .unwrap();
            dq.copy_from_slice(&dq_cmd);
            for (q_k, &dq_k) in q.iter_mut().zip(dq_cmd.iter()) {
                *q_k += dq_k * cfg.dt;
            }
        }
        for k in 0..6 {
            assert!(
                (q[k] - goal[k]).abs() < 0.03,
                "unobstructed JointReference should converge closely on dof {k}: q={}, goal={}",
                q[k],
                goal[k]
            );
        }

        // 2. Place an obstacle directly across the swept path and re-run through CBF safety filter
        let mut buf = model.create_transform_buffer();
        let mid_pose = {
            model.forward_kinematics(&mid, &mut buf).unwrap();
            *buf.frame_pose(left_arm.end_effectors[0])
        };
        env.add_obstacle(
            SharedShape::new(Ball::new(0.12)),
            Isometry3::from(Translation3::from(mid_pose.translation.vector)),
        );

        let mut servo_obs = ReactiveServoController::new(&model, &env, cfg);
        let mut ws = env.create_workspace(&model);
        let mut witnesses = Vec::with_capacity(32);
        q = start;
        dq = [0.0; 6];
        let mut min_dist = f64::INFINITY;
        for step in 0..steps {
            let t = (step as f64) * 0.005;
            traj.sample_into(t, &mut q_ref, &mut dq_ref, &mut ddq_ref);
            servo_obs
                .step_joint_reference(&model, &env, &q, &dq, &q_ref, &dq_ref, 8.0, None, &mut dq_cmd)
                .unwrap();
            dq.copy_from_slice(&dq_cmd);
            for (q_k, &dq_k) in q.iter_mut().zip(dq_cmd.iter()) {
                *q_k += dq_k * 0.005;
            }
            env.query_signed_distances(
                &model,
                &q,
                0.5,
                GeometryTier::Convex,
                &mut ws,
                &mut witnesses,
            )
            .unwrap();
            for w in &witnesses {
                min_dist = min_dist.min(w.signed_distance);
            }
        }
        assert!(
            min_dist >= 0.035,
            "JointReference CBF filter must keep clearance near margin 0.04 m, got {min_dist:.4}"
        );
    }

    #[test]
    fn subgroup_planning_freezes_stationary_arms_in_rrt_and_two_stage() {
        let (model, left_arm, _right_arm, _dual_arms) = build_dual_arm_model();
        let mut env = CollisionEnvironment::new();
        env.set_acm(AllowedCollisionMatrix::from_model(&model));
        for link in model.links() {
            env.set_link_geometry(
                link.id,
                LinkGeometry::from_capsule(
                    Point3::origin(),
                    Point3::new(0.0, 0.0, 0.25),
                    link.bounding_radius,
                ),
            );
        }
        env.add_obstacle(
            SharedShape::new(Ball::new(0.10)),
            Isometry3::translation(0.28, 0.35, 0.52),
        );

        let start_vec = vec![0.1, -0.3, -0.3, -0.45, 0.6, 0.35];
        let start = Configuration::new(start_vec.clone());

        // Compute reachable target for left arm only
        let mut goal_ref = start_vec.clone();
        goal_ref[0] = -0.35;
        goal_ref[1] = 0.55;
        goal_ref[2] = 0.35;
        let mut buf = model.create_transform_buffer();
        model.forward_kinematics(&goal_ref, &mut buf).unwrap();
        let left_pose = *buf.frame_pose(left_arm.end_effectors[0]);

        // 1. Subgroup RRT-Connect planning for left_arm only
        let robot = Robot::from_model(Arc::new(model.clone()));
        let rrt = RrtPlanner::new(env.clone());
        let rrt_path = rrt
            .plan_path_for_group(
                &robot,
                &left_arm,
                start.clone(),
                Target::MultiPose(vec![TaskPose::new(left_arm.end_effectors[0], left_pose)]),
            )
            .unwrap();
        assert!(rrt_path.len() >= 2);
        for wp in &rrt_path {
            for k in 3..6 {
                assert_relative_eq!(wp.0[k], start_vec[k], epsilon = 1e-12);
            }
        }

        // 2. Subgroup Two-Stage close-proximity planning for left_arm only
        let pre_ref = [-0.30, 0.42, 0.28, -0.45, 0.6, 0.35];
        model.forward_kinematics(&pre_ref, &mut buf).unwrap();
        let pre_pose = *buf.frame_pose(left_arm.end_effectors[0]);

        let two_stage_cfg = TwoStagePlanConfig {
            coarse_mode: CoarsePlannerMode::RrtConnect,
            servo_config: ReactiveServoConfig {
                safety_margin: 0.025,
                ..TwoStagePlanConfig::default().servo_config
            },
            ..TwoStagePlanConfig::default()
        };
        let planner = TwoStageApproachPlanner::new(two_stage_cfg);
        let res = planner
            .plan_for_group(
                &CpuSimdBatchBackend,
                &model,
                &env,
                None,
                &left_arm,
                start,
                &[ApproachTarget::with_pre_approach(
                    TaskPose::new(left_arm.end_effectors[0], left_pose),
                    pre_pose,
                )],
            )
            .unwrap();
        assert!(res.max_position_error < 0.01);
        for wp in &res.full_path {
            for k in 3..6 {
                assert_relative_eq!(wp.0[k], start_vec[k], epsilon = 1e-12);
            }
        }

        // 3. Subgroup masked MPPI freezes stationary right_arm DOFs
        let flat = FlatRobotModelF32::compile(&model, &env);
        let mppi = BatchMppiOptimizer::new(MppiConfig {
            num_rollouts: 16,
            horizon: 8,
            num_iterations: 4,
            ..MppiConfig::default()
        });
        let left_mask = [true, true, true, false, false, false];
        let mppi_traj = mppi.optimize_masked(
            &CpuSimdBatchBackend,
            &model,
            &flat,
            None,
            &start_vec,
            &goal_ref,
            Some(&left_mask),
        );
        assert_eq!(mppi_traj.len(), 8);
        for wp in &mppi_traj {
            for k in 3..6 {
                assert_relative_eq!(wp.0[k], start_vec[k], epsilon = 1e-12);
            }
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn resident_gpu_batch_backend_matches_simd_and_runs_ik_mppi() {
        let Ok(wgpu_backend) = WgpuBatchBackend::new() else {
            return;
        };
        let (model, left_arm, right_arm, _dual_arms) = build_dual_arm_model();
        let mut env = CollisionEnvironment::new();
        env.set_acm(AllowedCollisionMatrix::from_model(&model));
        for link in model.links() {
            env.set_link_geometry(
                link.id,
                LinkGeometry::from_capsule(
                    Point3::origin(),
                    Point3::new(0.0, 0.0, 0.28),
                    link.bounding_radius,
                ),
            );
        }
        let flat = FlatRobotModelF32::compile(&model, &env);
        let mut esdf = EsdfGridF32::new([-0.8, -0.8, 0.0], 0.05, [32, 32, 24], 1.0);
        esdf.add_sphere_obstacle([0.35, 0.0, 0.6], 0.10);

        let wgpu_arc = Arc::new(wgpu_backend);
        // Intentionally small max_batch_size (16) to exercise multi-chunk dispatch in MPPI
        let resident = ResidentGpuBatchBackend::new(wgpu_arc, flat.clone(), Some(esdf.clone()), 16);

        let q_batch: Vec<f32> = (0..24)
            .flat_map(|b| {
                let a = (b as f32) * 0.05 - 0.5;
                [a, 0.3, -0.2, -a, 0.3, -0.2]
            })
            .collect();
        let mut sph_cpu = vec![[0.0_f32; 4]; 24 * flat.spheres.len()];
        let mut ee_cpu = vec![[0.0_f32; 7]; 24 * flat.end_effectors.len()];
        let mut sph_res = vec![[0.0_f32; 4]; 24 * flat.spheres.len()];
        let mut ee_res = vec![[0.0_f32; 7]; 24 * flat.end_effectors.len()];

        CpuSimdBatchBackend.forward_kinematics_batch_f32(&flat, &q_batch, &mut sph_cpu, &mut ee_cpu);
        resident.forward_kinematics_batch_f32(&flat, &q_batch, &mut sph_res, &mut ee_res);
        for (a, b) in ee_cpu.iter().zip(ee_res.iter()) {
            for k in 0..7 {
                assert_relative_eq!(a[k], b[k], epsilon = 1e-4);
            }
        }

        let mut costs_cpu = vec![0.0_f32; 24];
        let mut costs_res = vec![0.0_f32; 24];
        CpuSimdBatchBackend.collision_cost_batch_f32(&flat, Some(&esdf), &q_batch, 0.08, &mut costs_cpu);
        resident.collision_cost_batch_f32(&flat, Some(&esdf), &q_batch, 0.08, &mut costs_res);
        for (c0, c1) in costs_cpu.iter().zip(costs_res.iter()) {
            assert_relative_eq!(*c0, *c1, epsilon = 1e-4);
        }

        let mut buf = model.create_transform_buffer();
        let goal_q = [0.25, 0.5, -0.2, -0.25, 0.5, -0.2];
        model.forward_kinematics(&goal_q, &mut buf).unwrap();
        let targets = [
            TaskPose::new(left_arm.end_effectors[0], *buf.frame_pose(left_arm.end_effectors[0])),
            TaskPose::new(right_arm.end_effectors[0], *buf.frame_pose(right_arm.end_effectors[0])),
        ];
        let seed = [0.0, 0.1, 0.1, 0.0, 0.1, 0.1];
        let q_ik = BatchIkSolver::new(20)
            .solve_collision_free(&resident, &model, &flat, &env, Some(&esdf), &seed, &targets)
            .unwrap();

        let mppi = BatchMppiOptimizer::new(MppiConfig {
            num_rollouts: 8,
            horizon: 8,
            num_iterations: 4,
            ..MppiConfig::default()
        });
        let traj = mppi.optimize(&resident, &model, &flat, Some(&esdf), &seed, q_ik.as_slice());
        assert_eq!(traj.len(), 8);
    }
}
