//! Real-time allocation-freedom verification for 1 kHz control, collision queries,
//! closed-chain constraint Jacobians, and trajectory sampling.

use nalgebra::{Isometry3, Point3, UnitVector3, Vector3};
use parry3d::math::Vector as ParryVec;
use parry3d::shape::{Ball, Cuboid, SharedShape};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::f64::consts::PI;
use unrust_it::{
    AllowedCollisionMatrix, AttachedPayload, ClosedChainGraspConstraint, CollisionEnvironment,
    Configuration, CubicSplinePath, GeometryTier, JerkLimitedTrajectory, JointLimit, LinkGeometry,
    MultiRobotModel, ReactiveServoConfig, ReactiveServoController, RobotId, ServoTarget,
    ToppraTrajectory,
};

struct CountingAlloc;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = COUNTING.try_with(|active| {
            if active.get() {
                let _ = ALLOC_COUNT.try_with(|n| n.set(n.get() + 1));
            }
        });
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = COUNTING.try_with(|active| {
            if active.get() {
                let _ = ALLOC_COUNT.try_with(|n| n.set(n.get() + 1));
            }
        });
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOC: CountingAlloc = CountingAlloc;

fn assert_zero_allocs<F: FnMut()>(label: &str, mut f: F) {
    ALLOC_COUNT.with(|n| n.set(0));
    COUNTING.with(|active| active.set(true));
    f();
    COUNTING.with(|active| active.set(false));
    let count = ALLOC_COUNT.with(|n| n.get());
    assert_eq!(count, 0, "{label} performed {count} unexpected heap allocations");
}

fn build_workcell() -> (
    MultiRobotModel,
    CollisionEnvironment,
    unrust_it::PlanningGroup,
    unrust_it::PlanningGroup,
) {
    let mut model = MultiRobotModel::new();
    let limit = JointLimit::new(-PI, PI);
    let segs = [
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
            0.045,
        ),
        (
            UnitVector3::new_normalize(Vector3::y()),
            Vector3::new(0.0, 0.0, 0.35),
            limit,
            0.04,
        ),
    ];

    let left = model
        .add_serial_arm(
            RobotId(0),
            "left_arm",
            Isometry3::translation(0.0, 0.36, 0.0),
            &segs,
            Isometry3::translation(0.0, 0.0, 0.28),
        )
        .expect("left arm valid");
    let right = model
        .add_serial_arm(
            RobotId(1),
            "right_arm",
            Isometry3::translation(0.0, -0.36, 0.0),
            &segs,
            Isometry3::translation(0.0, 0.0, 0.28),
        )
        .expect("right arm valid");

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

    let obs_box = env.add_obstacle(
        SharedShape::new(Cuboid::new(ParryVec::new(0.06, 0.06, 0.06))),
        Isometry3::translation(0.32, 0.0, 0.45),
    );
    env.add_obstacle(
        SharedShape::new(Ball::new(0.07)),
        Isometry3::translation(0.42, 0.14, 0.62),
    );

    let left_flange = left.active_links.last().copied().unwrap();
    let right_flange = right.active_links.last().copied().unwrap();
    let tool_shape = SharedShape::new(Cuboid::new(ParryVec::new(0.02, 0.03, 0.02)));
    env.attach_payload(
        AttachedPayload::from_shape(
            "left_tool",
            left_flange,
            Isometry3::translation(0.0, 0.0, 0.15),
            tool_shape.clone(),
        )
        .with_touch_links(vec![left_flange])
        .with_safety_margin(0.005)
        .with_activation_distance(0.25),
    );
    env.attach_payload(
        AttachedPayload::from_shape(
            "right_tool",
            right_flange,
            Isometry3::translation(0.0, 0.0, 0.15),
            tool_shape,
        )
        .with_touch_links(vec![right_flange])
        .with_safety_margin(0.005)
        .with_activation_distance(0.25),
    );

    env.set_payload_pair_margin("left_tool", "right_tool", 0.004);
    env.set_payload_pair_activation("left_tool", "right_tool", 0.25);
    env.set_payload_obstacle_margin("left_tool", obs_box, 0.003);
    env.set_payload_obstacle_activation("left_tool", obs_box, 0.25);

    (model, env, left, right)
}

#[test]
fn hot_paths_execute_with_zero_heap_allocations() {
    let (model, env, left, right) = build_workcell();
    let dof = model.dof();

    let q_cur = [0.15, 0.45, 0.4, -0.15, 0.45, 0.4];
    let dq_cur = [0.0; 6];
    let mut dq_cmd = [0.0; 6];

    let mut tb = model.create_transform_buffer();
    model.forward_kinematics(&q_cur, &mut tb).unwrap();
    let p_l = *tb.frame_pose(left.end_effectors[0]);
    let p_r = *tb.frame_pose(right.end_effectors[0]);
    let rel_ab = p_l.inverse() * p_r;
    let closed_chain =
        ClosedChainGraspConstraint::new_rigid(left.end_effectors[0], right.end_effectors[0], rel_ab);

    let cart_targets = [
        ServoTarget::pose(
            left.end_effectors[0],
            Isometry3::translation(0.32, 0.12, 0.55) * p_l.rotation,
            3.0,
        ),
        ServoTarget::pose(
            right.end_effectors[0],
            Isometry3::translation(0.32, -0.12, 0.55) * p_r.rotation,
            3.0,
        ),
    ];
    let jref_targets = [ServoTarget::joint_reference(
        vec![0.25, 0.5, 0.3, -0.25, 0.5, 0.3],
        vec![0.05; 6],
        4.0,
    )];
    let q_ref = [0.25, 0.5, 0.3, -0.25, 0.5, 0.3];
    let dq_ref = [0.05; 6];

    let mut servo = ReactiveServoController::new(
        &model,
        &env,
        ReactiveServoConfig {
            dt: 0.001,
            safety_margin: 0.04,
            cbf_activation_distance: 0.30,
            cbf_gamma: 6.0,
            tangential_curl_gain: 0.35,
            ..ReactiveServoConfig::default()
        }
        .with_arm_priority(RobotId(0), 2.5),
    );

    // Warmup passes
    for _ in 0..2 {
        servo
            .step(&model, &env, &q_cur, &dq_cur, &cart_targets, None, &mut dq_cmd)
            .unwrap();
        servo
            .step(
                &model,
                &env,
                &q_cur,
                &dq_cur,
                &jref_targets,
                Some(&closed_chain),
                &mut dq_cmd,
            )
            .unwrap();
        servo
            .step_joint_reference(
                &model,
                &env,
                &q_cur,
                &dq_cur,
                &q_ref,
                &dq_ref,
                4.0,
                None,
                &mut dq_cmd,
            )
            .unwrap();
    }

    assert_zero_allocs("ReactiveServoController::step (Cartesian)", || {
        for _ in 0..100 {
            servo
                .step(&model, &env, &q_cur, &dq_cur, &cart_targets, None, &mut dq_cmd)
                .unwrap();
        }
    });

    assert_zero_allocs("ReactiveServoController::step (JointReference + ClosedChain)", || {
        for _ in 0..50 {
            servo
                .step(
                    &model,
                    &env,
                    &q_cur,
                    &dq_cur,
                    &jref_targets,
                    Some(&closed_chain),
                    &mut dq_cmd,
                )
                .unwrap();
        }
    });

    assert_zero_allocs("ReactiveServoController::step_joint_reference", || {
        for _ in 0..100 {
            servo
                .step_joint_reference(
                    &model,
                    &env,
                    &q_cur,
                    &dq_cur,
                    &q_ref,
                    &dq_ref,
                    4.0,
                    None,
                    &mut dq_cmd,
                )
                .unwrap();
        }
    });

    // Collision query + rule lookups
    let mut ws = env.create_workspace(&model);
    let mut witnesses = Vec::with_capacity(64);
    env.query_signed_distances(&model, &q_cur, 0.5, GeometryTier::Convex, &mut ws, &mut witnesses)
        .unwrap();
    assert!(!witnesses.is_empty());

    assert_zero_allocs("CollisionEnvironment::query_signed_distances + rules", || {
        for _ in 0..100 {
            env.query_signed_distances(
                &model,
                &q_cur,
                0.5,
                GeometryTier::Convex,
                &mut ws,
                &mut witnesses,
            )
            .unwrap();
            let mut sum = 0.0;
            for w in &witnesses {
                sum += env.effective_pair_margin(&w.pair, 0.04);
                sum += env.effective_pair_activation(&w.pair, 0.25);
            }
            assert!(sum >= 0.0);
        }
    });

    // Trajectory zero-alloc sampling
    let spline = CubicSplinePath::from_waypoints(&[
        Configuration::from_slice(&q_cur),
        Configuration::from_slice(&[0.2, 0.6, 0.5, -0.2, 0.6, 0.5]),
        Configuration::from_slice(&q_ref),
    ])
    .unwrap();
    let toppra = ToppraTrajectory::parameterize(spline, model.dof_limits(), 25).unwrap();
    let otg = JerkLimitedTrajectory::generate(
        &q_cur,
        &dq_cur,
        &[0.0; 6],
        &q_ref,
        &[0.0; 6],
        &[0.0; 6],
        model.dof_limits(),
        0.5,
    );

    let mut pos = vec![0.0; dof];
    let mut vel = vec![0.0; dof];
    let mut acc = vec![0.0; dof];
    let mut jerk = vec![0.0; dof];

    toppra.sample_into(0.1, &mut pos, &mut vel, &mut acc);
    otg.sample_into(0.1, &mut pos, &mut vel, &mut acc, &mut jerk);

    assert_zero_allocs("ToppraTrajectory::sample_into", || {
        for i in 0..100 {
            let t = (i as f64) * 0.005;
            toppra.sample_into(t, &mut pos, &mut vel, &mut acc);
        }
    });

    assert_zero_allocs("JerkLimitedTrajectory::sample_into", || {
        for i in 0..100 {
            let t = (i as f64) * 0.004;
            otg.sample_into(t, &mut pos, &mut vel, &mut acc, &mut jerk);
        }
    });
}
