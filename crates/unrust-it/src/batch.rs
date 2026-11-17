//! CuRobo-style batched `f32` kinematics, 3D Euclidean Signed Distance Fields (`EsdfGridF32`),
//! and GPU (`wgpu` Metal / Vulkan / DX12 / WebGPU) + CPU SIMD (`f32x8`) `B x H` rollout evaluation.
//!
//! While [`crate::kinematics`] and [`crate::collision`] provide exact `f64` single-query APIs
//! backed by `parry3d-f64`, GPU kernels and wide CPU SIMD require:
//! 1. **Branchless, pointer-free `#[repr(C)]` `bytemuck::Pod` `f32` data layouts** (`FlatRobotModelF32`)
//!    that upload directly to GPU storage buffers (`wgpu`, `cudarc`, `cubecl`, or Metal).
//! 2. **Batched tensor interfaces** evaluating $N = B \times H$ configurations in a single call.
//! 3. **Branchless geometric primitives** (link sphere trees + 3D ESDF voxel grids)
//!    with zero GJK/EPA while-loop warp divergence.

use crate::collision::{AllowedCollisionMatrix, CollisionEnvironment};
use crate::robot::{FrameAttachment, FrameId, JointType, LinkId, MultiRobotModel};
use bytemuck::{Pod, Zeroable};
use nalgebra::{Isometry3, Point3, Vector3};
use rayon::prelude::*;
use wide::f32x8;

/// Joint kind tag encoded as a `u32` for GPU shader / kernel compatibility.
pub const JOINT_KIND_FIXED: u32 = 0;
/// Revolute or continuous rotational joint (`u32` tag).
pub const JOINT_KIND_REVOLUTE: u32 = 1;
/// Prismatic linear joint (`u32` tag).
pub const JOINT_KIND_PRISMATIC: u32 = 2;

/// GPU-ready `#[repr(C)]` flat joint descriptor (`f32`, 64 bytes, 16-byte aligned).
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct FlatJointF32 {
    /// Static translation `[tx, ty, tz]` from parent link to joint frame.
    pub origin_xyz: [f32; 3],
    /// Index into configuration vector `q` (`u32::MAX` if fixed).
    pub q_index: u32,
    /// Static unit quaternion `[qx, qy, qz, qw]` from parent link to joint frame.
    pub origin_quat_xyzw: [f32; 4],
    /// Local joint axis `[ax, ay, az]`.
    pub axis_xyz: [f32; 3],
    /// Joint kind (`0 = Fixed`, `1 = Revolute/Continuous`, `2 = Prismatic`).
    pub kind: u32,
    /// Parent link index (`u32::MAX` if attached to world root).
    pub parent_link: u32,
    /// Child link index driven by this joint.
    pub child_link: u32,
    /// Position lower bound (`f32`).
    pub min_pos: f32,
    /// Position upper bound (`f32`).
    pub max_pos: f32,
}

/// GPU-ready `#[repr(C)]` sphere primitive attached to a robot link (`f32`, 32 bytes, 16-byte aligned).
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct FlatLinkSphereF32 {
    /// Local sphere center `[x, y, z]` in link coordinates.
    pub local_center: [f32; 3],
    /// Sphere radius in meters.
    pub radius: f32,
    /// Owning link index (`0..num_links`).
    pub link_index: u32,
    /// Padding for 16-byte alignment in GPU storage buffers.
    pub _pad: [u32; 3],
}

/// GPU-ready `#[repr(C)]` active self/inter-arm sphere collision pair (`f32`, 16 bytes, 16-byte aligned).
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct FlatSpherePairF32 {
    /// First sphere index in `FlatRobotModelF32::spheres`.
    pub sphere_a: u32,
    /// Second sphere index in `FlatRobotModelF32::spheres`.
    pub sphere_b: u32,
    /// Combined radius plus safety margin ($r_a + r_b + d_{\text{margin}}$).
    pub combined_radius: f32,
    /// Padding for 16-byte alignment.
    pub _pad: u32,
}

/// GPU-ready `#[repr(C)]` end-effector frame descriptor (`f32`, 32 bytes, 16-byte aligned).
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct FlatEndEffectorF32 {
    /// Local translation `[x, y, z]` relative to `link_index`.
    pub local_xyz: [f32; 3],
    /// Associated link index (`0..num_links`).
    pub link_index: u32,
    /// Local rotation quaternion `[qx, qy, qz, qw]` relative to `link_index`.
    pub local_quat_xyzw: [f32; 4],
}

/// GPU uniform buffer parameters for batched rollout shaders (`64 bytes`, 16-byte aligned).
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct RolloutUniformsF32 {
    /// Number of states $N = B \times H$ in the batch.
    pub batch_size: u32,
    /// Number of actuated degrees of freedom $D$.
    pub dof: u32,
    /// Number of links $L$.
    pub num_links: u32,
    /// Number of link spheres $S$.
    pub num_spheres: u32,
    /// Number of end-effector frames $E$.
    pub num_ees: u32,
    /// Number of active self/inter-arm sphere pairs.
    pub num_self_pairs: u32,
    /// `1` if an `EsdfGridF32` is bound, `0` otherwise.
    pub has_esdf: u32,
    /// Smooth collision hinge activation distance $d_{\text{act}}$.
    pub activation_distance: f32,
    /// World-space minimum corner of the ESDF grid.
    pub esdf_origin: [f32; 3],
    /// Reciprocal voxel size `1.0 / voxel_size`.
    pub esdf_inv_voxel_size: f32,
    /// ESDF grid dimensions `[nx, ny, nz]`.
    pub esdf_dims: [u32; 3],
    /// 16-byte alignment padding.
    pub _pad: u32,
}

/// Flattened, pointer-free `f32` representation of a [`MultiRobotModel`] and its Tier-1 sphere geometry.
///
/// Designed for zero-copy upload to GPU compute buffers (CUDA / Metal MPS / `wgpu`)
/// and 8-wide SIMD (`f32x8`) execution on CPU.
#[derive(Debug, Clone)]
pub struct FlatRobotModelF32 {
    /// Number of actuated degrees of freedom $D$.
    pub dof: usize,
    /// Number of links $L$.
    pub num_links: usize,
    /// Topologically sorted joints (`len = num_links`).
    pub joints: Vec<FlatJointF32>,
    /// Link spheres across all arms (`len = num_spheres`).
    pub spheres: Vec<FlatLinkSphereF32>,
    /// Pre-compiled active self/inter-arm sphere pairs from the [`AllowedCollisionMatrix`].
    pub active_self_pairs: Vec<FlatSpherePairF32>,
    /// End-effector descriptors for task-space rollouts.
    pub end_effectors: Vec<FlatEndEffectorF32>,
}

impl FlatRobotModelF32 {
    /// Compiles a [`MultiRobotModel`] and [`CollisionEnvironment`]
    /// into a contiguous GPU/SIMD-friendly `f32` model.
    pub fn compile(model: &MultiRobotModel, env: &CollisionEnvironment) -> Self {
        let dof = model.dof();
        let num_links = model.num_links();
        let mut joints = Vec::with_capacity(num_links);

        for joint in model.joints() {
            let t = joint.origin_transform.translation.vector;
            let q = joint.origin_transform.rotation.coords;
            let (axis, kind) = match joint.joint_type {
                JointType::Fixed => ([0.0, 0.0, 0.0], JOINT_KIND_FIXED),
                JointType::Revolute { axis } | JointType::Continuous { axis } => (
                    [axis.x as f32, axis.y as f32, axis.z as f32],
                    JOINT_KIND_REVOLUTE,
                ),
                JointType::Prismatic { axis } => (
                    [axis.x as f32, axis.y as f32, axis.z as f32],
                    JOINT_KIND_PRISMATIC,
                ),
            };
            let (min_pos, max_pos) = joint
                .limits
                .map(|l| (l.position.min as f32, l.position.max as f32))
                .unwrap_or((-std::f32::consts::PI, std::f32::consts::PI));

            joints.push(FlatJointF32 {
                origin_xyz: [t.x as f32, t.y as f32, t.z as f32],
                q_index: joint.q_index.map_or(u32::MAX, |idx| idx as u32),
                origin_quat_xyzw: [q.x as f32, q.y as f32, q.z as f32, q.w as f32],
                axis_xyz: axis,
                kind,
                parent_link: joint.parent_link.map_or(u32::MAX, |l| l.0 as u32),
                child_link: joint.child_link.0 as u32,
                min_pos,
                max_pos,
            });
        }

        let mut spheres = Vec::new();
        let mut link_sphere_ranges = vec![0..0; num_links];
        for (link_idx, range_slot) in link_sphere_ranges.iter_mut().enumerate() {
            let link_id = LinkId(link_idx as u16);
            let start = spheres.len();
            if let Some(geom) = env.link_geometry(link_id) {
                for s in &geom.local_spheres {
                    spheres.push(FlatLinkSphereF32 {
                        local_center: [s.center.x as f32, s.center.y as f32, s.center.z as f32],
                        radius: s.radius as f32,
                        link_index: link_idx as u32,
                        _pad: [0; 3],
                    });
                }
            }
            let end = spheres.len();
            *range_slot = start..end;
        }

        let default_acm;
        let acm = match env.acm() {
            Some(a) => a,
            None => {
                default_acm = AllowedCollisionMatrix::from_model(model);
                &default_acm
            }
        };

        let mut active_self_pairs = Vec::new();
        acm.for_each_active_pair(|link_a, link_b| {
            let margin = acm.pair_margin(link_a, link_b) as f32;
            let range_a = &link_sphere_ranges[link_a.index()];
            let range_b = &link_sphere_ranges[link_b.index()];
            for sa in range_a.clone() {
                for sb in range_b.clone() {
                    active_self_pairs.push(FlatSpherePairF32 {
                        sphere_a: sa as u32,
                        sphere_b: sb as u32,
                        combined_radius: spheres[sa].radius + spheres[sb].radius + margin,
                        _pad: 0,
                    });
                }
            }
        });

        let mut end_effectors = Vec::new();
        for group in model.groups() {
            for &ee_frame in &group.end_effectors {
                if let Some((link_id, local_iso)) = Self::resolve_frame_to_link(model, ee_frame) {
                    let t = local_iso.translation.vector;
                    let q = local_iso.rotation.coords;
                    end_effectors.push(FlatEndEffectorF32 {
                        local_xyz: [t.x as f32, t.y as f32, t.z as f32],
                        link_index: link_id.0 as u32,
                        local_quat_xyzw: [q.x as f32, q.y as f32, q.z as f32, q.w as f32],
                    });
                }
            }
        }

        Self {
            dof,
            num_links,
            joints,
            spheres,
            active_self_pairs,
            end_effectors,
        }
    }

    fn resolve_frame_to_link(
        model: &MultiRobotModel,
        frame: FrameId,
    ) -> Option<(LinkId, Isometry3<f64>)> {
        let node = model.frame_graph().node(frame)?;
        match node.attachment {
            FrameAttachment::WorldRoot => None,
            FrameAttachment::Link(link_id) => Some((link_id, Isometry3::identity())),
            FrameAttachment::FixedOffset { .. } => {
                Some((node.associated_link?, node.link_to_frame))
            }
        }
    }
}

/// 3D Euclidean Signed Distance Field (`f32`) for branchless $O(1)$ GPU/SIMD trilinear distance & gradient queries.
#[derive(Debug, Clone)]
pub struct EsdfGridF32 {
    /// World-space minimum corner `[ox, oy, oz]` in meters.
    pub origin: [f32; 3],
    /// Voxel size in meters.
    pub voxel_size: f32,
    /// Precomputed `1.0 / voxel_size`.
    pub inv_voxel_size: f32,
    /// Grid dimensions `[nx, ny, nz]`.
    pub dims: [usize; 3],
    /// Flattened `Z -> Y -> X` contiguous buffer of signed distances in meters (`len = nx * ny * nz`).
    pub distances: Vec<f32>,
}

impl EsdfGridF32 {
    /// Allocates a new ESDF grid initialized to `default_distance` (e.g., truncation distance).
    pub fn new(origin: [f32; 3], voxel_size: f32, dims: [usize; 3], default_distance: f32) -> Self {
        assert!(voxel_size > 0.0, "voxel_size must be positive");
        let total = dims[0] * dims[1] * dims[2];
        Self {
            origin,
            voxel_size,
            inv_voxel_size: 1.0 / voxel_size,
            dims,
            distances: vec![default_distance; total],
        }
    }

    /// Carves an analytical spherical obstacle `(center, radius)` into the ESDF grid
    /// via exact $\min(d_{\text{existing}}, \|\mathbf{p} - \mathbf{c}\|_2 - r)$ composition.
    pub fn add_sphere_obstacle(&mut self, center: [f32; 3], radius: f32) {
        let [nx, ny, nz] = self.dims;
        let stride_y = nx;
        let stride_z = nx * ny;

        for iz in 0..nz {
            let pz = self.origin[2] + (iz as f32 + 0.5) * self.voxel_size;
            let dz = pz - center[2];
            for iy in 0..ny {
                let py = self.origin[1] + (iy as f32 + 0.5) * self.voxel_size;
                let dy = py - center[1];
                let row_off = iz * stride_z + iy * stride_y;
                for ix in 0..nx {
                    let px = self.origin[0] + (ix as f32 + 0.5) * self.voxel_size;
                    let dx = px - center[0];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt() - radius;
                    let slot = &mut self.distances[row_off + ix];
                    if dist < *slot {
                        *slot = dist;
                    }
                }
            }
        }
    }

    /// Carves an axis-aligned box obstacle `(min_corner, max_corner)` into the ESDF grid.
    pub fn add_aabb_obstacle(&mut self, min_corner: [f32; 3], max_corner: [f32; 3]) {
        let half = [
            0.5 * (max_corner[0] - min_corner[0]),
            0.5 * (max_corner[1] - min_corner[1]),
            0.5 * (max_corner[2] - min_corner[2]),
        ];
        let center = [
            0.5 * (min_corner[0] + max_corner[0]),
            0.5 * (min_corner[1] + max_corner[1]),
            0.5 * (min_corner[2] + max_corner[2]),
        ];
        let [nx, ny, nz] = self.dims;
        let stride_y = nx;
        let stride_z = nx * ny;

        for iz in 0..nz {
            let qz =
                (self.origin[2] + (iz as f32 + 0.5) * self.voxel_size - center[2]).abs() - half[2];
            for iy in 0..ny {
                let qy = (self.origin[1] + (iy as f32 + 0.5) * self.voxel_size - center[1]).abs()
                    - half[1];
                let row_off = iz * stride_z + iy * stride_y;
                for ix in 0..nx {
                    let qx = (self.origin[0] + (ix as f32 + 0.5) * self.voxel_size - center[0])
                        .abs()
                        - half[0];
                    let out_x = qx.max(0.0);
                    let out_y = qy.max(0.0);
                    let out_z = qz.max(0.0);
                    let outside = (out_x * out_x + out_y * out_y + out_z * out_z).sqrt();
                    let inside = qx.max(qy.max(qz)).min(0.0);
                    let dist = outside + inside;
                    let slot = &mut self.distances[row_off + ix];
                    if dist < *slot {
                        *slot = dist;
                    }
                }
            }
        }
    }

    /// $O(1)$ branchless trilinear interpolation of signed distance and unit gradient `[gx, gy, gz]`
    /// at world point `p = [x, y, z]`.
    #[inline]
    pub fn sample_distance_and_gradient(&self, p: [f32; 3]) -> Option<(f32, [f32; 3])> {
        let gx = (p[0] - self.origin[0]) * self.inv_voxel_size - 0.5;
        let gy = (p[1] - self.origin[1]) * self.inv_voxel_size - 0.5;
        let gz = (p[2] - self.origin[2]) * self.inv_voxel_size - 0.5;

        if gx < 0.0 || gy < 0.0 || gz < 0.0 {
            return None;
        }
        let (ix, iy, iz) = (gx as usize, gy as usize, gz as usize);
        if ix + 1 >= self.dims[0] || iy + 1 >= self.dims[1] || iz + 1 >= self.dims[2] {
            return None;
        }

        let (tx, ty, tz) = (gx - ix as f32, gy - iy as f32, gz - iz as f32);
        let stride_y = self.dims[0];
        let stride_z = self.dims[0] * self.dims[1];
        let base = iz * stride_z + iy * stride_y + ix;

        let c000 = self.distances[base];
        let c100 = self.distances[base + 1];
        let c010 = self.distances[base + stride_y];
        let c110 = self.distances[base + stride_y + 1];
        let c001 = self.distances[base + stride_z];
        let c101 = self.distances[base + stride_z + 1];
        let c011 = self.distances[base + stride_z + stride_y];
        let c111 = self.distances[base + stride_z + stride_y + 1];

        let c00 = c000 * (1.0 - tx) + c100 * tx;
        let c10 = c010 * (1.0 - tx) + c110 * tx;
        let c01 = c001 * (1.0 - tx) + c101 * tx;
        let c11 = c011 * (1.0 - tx) + c111 * tx;

        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;
        let dist = c0 * (1.0 - tz) + c1 * tz;

        let dc_dx = ((c100 - c000) * (1.0 - ty) + (c110 - c010) * ty) * (1.0 - tz)
            + ((c101 - c001) * (1.0 - ty) + (c111 - c011) * ty) * tz;
        let dc_dy = (c10 - c00) * (1.0 - tz) + (c11 - c01) * tz;
        let dc_dz = c1 - c0;

        let raw_gx = dc_dx * self.inv_voxel_size;
        let raw_gy = dc_dy * self.inv_voxel_size;
        let raw_gz = dc_dz * self.inv_voxel_size;
        let norm = (raw_gx * raw_gx + raw_gy * raw_gy + raw_gz * raw_gz).sqrt();
        let grad = if norm > 1e-7 {
            [raw_gx / norm, raw_gy / norm, raw_gz / norm]
        } else {
            [0.0, 0.0, 1.0]
        };

        Some((dist, grad))
    }
}

/// Hardware-agnostic trait for batched `f32` Forward Kinematics and Collision Rollouts
/// (implemented by [`CpuSimdBatchBackend`] and [`WgpuBatchBackend`]).
pub trait BatchRolloutBackend: Send + Sync {
    /// Evaluates Forward Kinematics for a batch of $N$ configurations (`q_batch.len() == N * dof`),
    /// writing world-space spheres `[x, y, z, r]` (`len = N * num_spheres`)
    /// and end-effector poses `[tx, ty, tz, qx, qy, qz, qw]` (`len = N * num_ees`).
    fn forward_kinematics_batch_f32(
        &self,
        model: &FlatRobotModelF32,
        q_batch: &[f32],
        out_world_spheres: &mut [[f32; 4]],
        out_ee_poses: &mut [[f32; 7]],
    );

    /// Evaluates smooth CuRobo-style collision cost for a batch of $N$ states (`N = B * H` rollouts)
    /// against self/inter-arm sphere pairs and an [`EsdfGridF32`], writing one cost per state into `out_costs`.
    fn collision_cost_batch_f32(
        &self,
        model: &FlatRobotModelF32,
        esdf: Option<&EsdfGridF32>,
        q_batch: &[f32],
        activation_distance: f32,
        out_costs: &mut [f32],
    );
}

/// CPU backend using 8-wide `f32x8` SIMD and `rayon` multi-threading.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuSimdBatchBackend;

impl CpuSimdBatchBackend {
    /// Processes 8 configurations simultaneously in lockstep using 8-wide `f32x8` SIMD registers.
    #[inline]
    pub fn fk_8_wide(
        model: &FlatRobotModelF32,
        q_8: &[f32],
        out_spheres_8: &mut [[f32; 4]],
        out_ees_8: &mut [[f32; 7]],
    ) {
        let dof = model.dof;
        let num_links = model.num_links;

        let mut q_simd = vec![f32x8::ZERO; dof];
        for (d, slot) in q_simd.iter_mut().enumerate() {
            *slot = f32x8::from([
                q_8[d],
                q_8[dof + d],
                q_8[2 * dof + d],
                q_8[3 * dof + d],
                q_8[4 * dof + d],
                q_8[5 * dof + d],
                q_8[6 * dof + d],
                q_8[7 * dof + d],
            ]);
        }

        let mut link_t = vec![[f32x8::ZERO; 3]; num_links];
        let mut link_q =
            vec![[f32x8::ZERO, f32x8::ZERO, f32x8::ZERO, f32x8::ONE]; num_links];

        for joint in &model.joints {
            let (p_t, p_q) = if joint.parent_link == u32::MAX {
                (
                    [f32x8::ZERO; 3],
                    [f32x8::ZERO, f32x8::ZERO, f32x8::ZERO, f32x8::ONE],
                )
            } else {
                let p_idx = joint.parent_link as usize;
                (link_t[p_idx], link_q[p_idx])
            };

            let o_t = [
                f32x8::splat(joint.origin_xyz[0]),
                f32x8::splat(joint.origin_xyz[1]),
                f32x8::splat(joint.origin_xyz[2]),
            ];
            let o_q = [
                f32x8::splat(joint.origin_quat_xyzw[0]),
                f32x8::splat(joint.origin_quat_xyzw[1]),
                f32x8::splat(joint.origin_quat_xyzw[2]),
                f32x8::splat(joint.origin_quat_xyzw[3]),
            ];

            let base_t = simd_transform_point(&p_t, &p_q, &o_t);
            let base_q = simd_quat_mul(&p_q, &o_q);

            let q_val = if joint.q_index == u32::MAX {
                f32x8::ZERO
            } else {
                q_simd[joint.q_index as usize]
            };

            let ax = [
                f32x8::splat(joint.axis_xyz[0]),
                f32x8::splat(joint.axis_xyz[1]),
                f32x8::splat(joint.axis_xyz[2]),
            ];

            let (c_t, c_q) = match joint.kind {
                JOINT_KIND_REVOLUTE => {
                    let half = q_val * f32x8::splat(0.5);
                    let (s, c) = half.sin_cos();
                    let j_q = [ax[0] * s, ax[1] * s, ax[2] * s, c];
                    (base_t, simd_quat_mul(&base_q, &j_q))
                }
                JOINT_KIND_PRISMATIC => {
                    let local_disp = [ax[0] * q_val, ax[1] * q_val, ax[2] * q_val];
                    (
                        simd_transform_point(&base_t, &base_q, &local_disp),
                        base_q,
                    )
                }
                _ => (base_t, base_q),
            };

            let child_idx = joint.child_link as usize;
            link_t[child_idx] = c_t;
            link_q[child_idx] = c_q;
        }

        let num_spheres = model.spheres.len();
        for (s_idx, s) in model.spheres.iter().enumerate() {
            let l_idx = s.link_index as usize;
            let local_p = [
                f32x8::splat(s.local_center[0]),
                f32x8::splat(s.local_center[1]),
                f32x8::splat(s.local_center[2]),
            ];
            let world_p = simd_transform_point(&link_t[l_idx], &link_q[l_idx], &local_p);
            let wx: [f32; 8] = world_p[0].into();
            let wy: [f32; 8] = world_p[1].into();
            let wz: [f32; 8] = world_p[2].into();
            for lane in 0..8 {
                out_spheres_8[lane * num_spheres + s_idx] =
                    [wx[lane], wy[lane], wz[lane], s.radius];
            }
        }

        let num_ees = model.end_effectors.len();
        for (ee_idx, ee) in model.end_effectors.iter().enumerate() {
            let l_idx = ee.link_index as usize;
            let local_t = [
                f32x8::splat(ee.local_xyz[0]),
                f32x8::splat(ee.local_xyz[1]),
                f32x8::splat(ee.local_xyz[2]),
            ];
            let local_q = [
                f32x8::splat(ee.local_quat_xyzw[0]),
                f32x8::splat(ee.local_quat_xyzw[1]),
                f32x8::splat(ee.local_quat_xyzw[2]),
                f32x8::splat(ee.local_quat_xyzw[3]),
            ];
            let ee_t = simd_transform_point(&link_t[l_idx], &link_q[l_idx], &local_t);
            let ee_q = simd_quat_mul(&link_q[l_idx], &local_q);

            let tx: [f32; 8] = ee_t[0].into();
            let ty: [f32; 8] = ee_t[1].into();
            let tz: [f32; 8] = ee_t[2].into();
            let qx: [f32; 8] = ee_q[0].into();
            let qy: [f32; 8] = ee_q[1].into();
            let qz: [f32; 8] = ee_q[2].into();
            let qw: [f32; 8] = ee_q[3].into();

            for lane in 0..8 {
                out_ees_8[lane * num_ees + ee_idx] = [
                    tx[lane], ty[lane], tz[lane], qx[lane], qy[lane], qz[lane], qw[lane],
                ];
            }
        }
    }

    /// Scalar single-state FK fallback for remainder items (`N % 8`).
    #[inline]
    pub fn fk_scalar(
        model: &FlatRobotModelF32,
        q: &[f32],
        out_spheres: &mut [[f32; 4]],
        out_ees: &mut [[f32; 7]],
    ) {
        let num_links = model.num_links;
        let mut link_t = vec![[0.0_f32; 3]; num_links];
        let mut link_q = vec![[0.0_f32, 0.0, 0.0, 1.0]; num_links];

        for joint in &model.joints {
            let (p_t, p_q) = if joint.parent_link == u32::MAX {
                ([0.0; 3], [0.0, 0.0, 0.0, 1.0])
            } else {
                let p_idx = joint.parent_link as usize;
                (link_t[p_idx], link_q[p_idx])
            };
            let base_t = scalar_transform_point(&p_t, &p_q, &joint.origin_xyz);
            let base_q = scalar_quat_mul(&p_q, &joint.origin_quat_xyzw);
            let q_val = if joint.q_index == u32::MAX {
                0.0
            } else {
                q[joint.q_index as usize]
            };

            let (c_t, c_q) = match joint.kind {
                JOINT_KIND_REVOLUTE => {
                    let half = 0.5 * q_val;
                    let (s, c) = half.sin_cos();
                    let j_q = [
                        joint.axis_xyz[0] * s,
                        joint.axis_xyz[1] * s,
                        joint.axis_xyz[2] * s,
                        c,
                    ];
                    (base_t, scalar_quat_mul(&base_q, &j_q))
                }
                JOINT_KIND_PRISMATIC => {
                    let disp = [
                        joint.axis_xyz[0] * q_val,
                        joint.axis_xyz[1] * q_val,
                        joint.axis_xyz[2] * q_val,
                    ];
                    (scalar_transform_point(&base_t, &base_q, &disp), base_q)
                }
                _ => (base_t, base_q),
            };
            let child_idx = joint.child_link as usize;
            link_t[child_idx] = c_t;
            link_q[child_idx] = c_q;
        }

        for (s_idx, s) in model.spheres.iter().enumerate() {
            let l_idx = s.link_index as usize;
            let wp = scalar_transform_point(&link_t[l_idx], &link_q[l_idx], &s.local_center);
            out_spheres[s_idx] = [wp[0], wp[1], wp[2], s.radius];
        }

        for (ee_idx, ee) in model.end_effectors.iter().enumerate() {
            let l_idx = ee.link_index as usize;
            let t = scalar_transform_point(&link_t[l_idx], &link_q[l_idx], &ee.local_xyz);
            let q_ee = scalar_quat_mul(&link_q[l_idx], &ee.local_quat_xyzw);
            out_ees[ee_idx] = [t[0], t[1], t[2], q_ee[0], q_ee[1], q_ee[2], q_ee[3]];
        }
    }
}

impl BatchRolloutBackend for CpuSimdBatchBackend {
    fn forward_kinematics_batch_f32(
        &self,
        model: &FlatRobotModelF32,
        q_batch: &[f32],
        out_world_spheres: &mut [[f32; 4]],
        out_ee_poses: &mut [[f32; 7]],
    ) {
        let dof = model.dof.max(1);
        let batch_size = q_batch.len() / dof;
        let num_spheres = model.spheres.len();
        let num_ees = model.end_effectors.len();
        assert_eq!(out_world_spheres.len(), batch_size * num_spheres);
        assert_eq!(out_ee_poses.len(), batch_size * num_ees);

        let chunk_8 = 8;
        let num_chunks = batch_size / chunk_8;

        if num_chunks > 0 {
            q_batch[..num_chunks * chunk_8 * dof]
                .par_chunks_exact(chunk_8 * dof)
                .zip(
                    out_world_spheres[..num_chunks * chunk_8 * num_spheres]
                        .par_chunks_exact_mut(chunk_8.max(1) * num_spheres.max(1)),
                )
                .zip(
                    out_ee_poses[..num_chunks * chunk_8 * num_ees]
                        .par_chunks_exact_mut(chunk_8.max(1) * num_ees.max(1)),
                )
                .for_each(|((q_chunk, spheres_chunk), ees_chunk)| {
                    Self::fk_8_wide(model, q_chunk, spheres_chunk, ees_chunk);
                });
        }

        for b in (num_chunks * chunk_8)..batch_size {
            let q_slice = &q_batch[b * dof..(b + 1) * dof];
            let s_slice = &mut out_world_spheres[b * num_spheres..(b + 1) * num_spheres];
            let e_slice = &mut out_ee_poses[b * num_ees..(b + 1) * num_ees];
            Self::fk_scalar(model, q_slice, s_slice, e_slice);
        }
    }

    fn collision_cost_batch_f32(
        &self,
        model: &FlatRobotModelF32,
        esdf: Option<&EsdfGridF32>,
        q_batch: &[f32],
        activation_distance: f32,
        out_costs: &mut [f32],
    ) {
        let dof = model.dof.max(1);
        let batch_size = q_batch.len() / dof;
        assert_eq!(out_costs.len(), batch_size);

        let num_spheres = model.spheres.len();
        let num_ees = model.end_effectors.len();
        let mut world_spheres = vec![[0.0_f32; 4]; batch_size * num_spheres];
        let mut ee_poses = vec![[0.0_f32; 7]; batch_size * num_ees];

        self.forward_kinematics_batch_f32(model, q_batch, &mut world_spheres, &mut ee_poses);

        let act = activation_distance.max(1e-4);
        let inv_two_act = 0.5 / act;

        out_costs
            .par_iter_mut()
            .enumerate()
            .for_each(|(b, cost_out)| {
                let spheres_b = &world_spheres[b * num_spheres..(b + 1) * num_spheres];
                let mut total_cost = 0.0_f32;

                for pair in &model.active_self_pairs {
                    let sa = spheres_b[pair.sphere_a as usize];
                    let sb = spheres_b[pair.sphere_b as usize];
                    let dx = sa[0] - sb[0];
                    let dy = sa[1] - sb[1];
                    let dz = sa[2] - sb[2];
                    let d = (dx * dx + dy * dy + dz * dz).sqrt() - pair.combined_radius;
                    total_cost += smooth_hinge_cost(d, act, inv_two_act);
                }

                if let Some(grid) = esdf {
                    for s in spheres_b {
                        if let Some((voxel_d, _grad)) =
                            grid.sample_distance_and_gradient([s[0], s[1], s[2]])
                        {
                            let d = voxel_d - s[3];
                            total_cost += smooth_hinge_cost(d, act, inv_two_act);
                        }
                    }
                }

                *cost_out = total_cost;
            });
    }
}

/// Cross-platform GPU compute backend powered by `wgpu` (Metal on Apple Silicon, Vulkan on Linux/Windows, DX12, WebGPU).
///
/// Compiles and dispatches `rollout.wgsl` (`fk_main` and `collision_main` compute shaders).
/// On headless CI environments without a hardware GPU, `WgpuBatchBackend` initializes `wgpu`'s
/// `Noop` validation backend so the WGSL shader, pipeline layouts, buffer alignments, and dispatches
/// are strictly verified by `naga`/`wgpu-core`, while transparently populating results via [`CpuSimdBatchBackend`].
#[cfg(feature = "gpu")]
pub struct WgpuBatchBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
    group0_layout: wgpu::BindGroupLayout,
    group1_layout: wgpu::BindGroupLayout,
    fk_pipeline: wgpu::ComputePipeline,
    collision_pipeline: wgpu::ComputePipeline,
}

#[cfg(feature = "gpu")]
impl WgpuBatchBackend {
    /// Initializes the `wgpu` compute device, queue, and compiled `rollout.wgsl` pipelines synchronously.
    pub fn new() -> Result<Self, String> {
        pollster::block_on(Self::new_async())
    }

    /// Initializes the `wgpu` compute device, queue, and compiled `rollout.wgsl` pipelines asynchronously.
    pub async fn new_async() -> Result<Self, String> {
        // 1. Try hardware GPU backends first (Metal, Vulkan, DX12, GL)
        let hw_instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY | wgpu::Backends::GL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = match hw_instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
        {
            Ok(a) => a,
            Err(_) => {
                // Fall back to wgpu's Noop backend on headless CI machines so WGSL compilation,
                // bind-group layouts, and command encoding are still validated.
                let noop_instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::NOOP,
                    backend_options: wgpu::BackendOptions {
                        noop: wgpu::NoopBackendOptions {
                            enable: true,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    ..wgpu::InstanceDescriptor::new_without_display_handle()
                });
                noop_instance
                    .request_adapter(&wgpu::RequestAdapterOptions::default())
                    .await
                    .map_err(|e| format!("failed to acquire hardware or noop wgpu adapter: {e}"))?
            }
        };

        let adapter_info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("unrust-it-rollout-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("failed to create wgpu device: {e}"))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("unrust-it-rollout-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("rollout.wgsl").into()),
        });

        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("unrust-it-group0-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_ro(1), // joints
                storage_ro(2), // spheres
                storage_ro(3), // end_effectors
                storage_ro(4), // q_batch
                storage_rw(5), // out_world_spheres
                storage_rw(6), // out_ee_poses
            ],
        });

        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("unrust-it-group1-layout"),
            entries: &[
                storage_ro(0), // self_pairs
                storage_ro(1), // esdf_distances
            ],
        });

        let fk_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("unrust-it-fk-pipeline-layout"),
            bind_group_layouts: &[Some(&group0_layout)],
            immediate_size: 0,
        });

        let collision_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("unrust-it-collision-pipeline-layout"),
            bind_group_layouts: &[Some(&group0_layout), Some(&group1_layout)],
            immediate_size: 0,
        });

        let fk_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("unrust-it-fk-pipeline"),
            layout: Some(&fk_layout),
            module: &shader,
            entry_point: Some("fk_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let collision_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("unrust-it-collision-pipeline"),
            layout: Some(&collision_layout),
            module: &shader,
            entry_point: Some("collision_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            adapter_info,
            group0_layout,
            group1_layout,
            fk_pipeline,
            collision_pipeline,
        })
    }

    /// Returns information about the active `wgpu` adapter (e.g., Metal, Vulkan, DX12, or Noop).
    #[inline]
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Returns `true` if backed by a physical/driver GPU (Metal, Vulkan, DX12, GL) rather than `Backends::NOOP`.
    #[inline]
    pub fn is_hardware_accelerated(&self) -> bool {
        self.adapter_info.backend != wgpu::Backend::Noop
    }

    fn create_storage_buffer_from_bytes(&self, label: &str, bytes: &[u8], rw: bool) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        let non_empty = if bytes.is_empty() { &[0u8; 64] } else { bytes };
        let mut usage = wgpu::BufferUsages::STORAGE;
        if rw {
            usage |= wgpu::BufferUsages::COPY_SRC;
        }
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: non_empty,
                usage,
            })
    }

    fn read_buffer_sync(&self, src_buffer: &wgpu::Buffer, byte_len: u64, dst_bytes: &mut [u8]) {
        if byte_len == 0 {
            return;
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("unrust-it-staging-readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("unrust-it-readback-encoder"),
            });
        encoder.copy_buffer_to_buffer(src_buffer, 0, &staging, 0, byte_len);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .expect("wgpu map_async channel closed")
            .expect("failed to map staging buffer");

        let mapped = slice
            .get_mapped_range()
            .expect("failed to get mapped staging buffer range");
        dst_bytes.copy_from_slice(&mapped[..dst_bytes.len()]);
        drop(mapped);
        staging.unmap();
    }
}

#[cfg(feature = "gpu")]
impl BatchRolloutBackend for WgpuBatchBackend {
    fn forward_kinematics_batch_f32(
        &self,
        model: &FlatRobotModelF32,
        q_batch: &[f32],
        out_world_spheres: &mut [[f32; 4]],
        out_ee_poses: &mut [[f32; 7]],
    ) {
        use wgpu::util::DeviceExt;
        let dof = model.dof.max(1);
        let batch_size = q_batch.len() / dof;
        if batch_size == 0 {
            return;
        }

        let uniforms = RolloutUniformsF32 {
            batch_size: batch_size as u32,
            dof: model.dof as u32,
            num_links: model.num_links as u32,
            num_spheres: model.spheres.len() as u32,
            num_ees: model.end_effectors.len() as u32,
            num_self_pairs: model.active_self_pairs.len() as u32,
            has_esdf: 0,
            activation_distance: 0.1,
            esdf_origin: [0.0; 3],
            esdf_inv_voxel_size: 1.0,
            esdf_dims: [1, 1, 1],
            _pad: 0,
        };

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let joints_buf = self.create_storage_buffer_from_bytes(
            "joints",
            bytemuck::cast_slice(&model.joints),
            false,
        );
        let spheres_buf = self.create_storage_buffer_from_bytes(
            "spheres",
            bytemuck::cast_slice(&model.spheres),
            false,
        );
        let ees_buf = self.create_storage_buffer_from_bytes(
            "ees",
            bytemuck::cast_slice(&model.end_effectors),
            false,
        );
        let q_buf =
            self.create_storage_buffer_from_bytes("q_batch", bytemuck::cast_slice(q_batch), false);

        let spheres_byte_len = std::mem::size_of_val(out_world_spheres) as u64;
        let ees_byte_len = std::mem::size_of_val(out_ee_poses) as u64;

        let out_spheres_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_world_spheres"),
            size: spheres_byte_len.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_ees_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_ee_poses"),
            size: ees_byte_len.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let group0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fk_group0"),
            layout: &self.group0_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: joints_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: spheres_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ees_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: out_spheres_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: out_ees_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fk_encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fk_compute_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.fk_pipeline);
            cpass.set_bind_group(0, &group0, &[]);
            let workgroups = (batch_size as u32).div_ceil(64);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        if self.is_hardware_accelerated() {
            self.read_buffer_sync(
                &out_spheres_buf,
                spheres_byte_len,
                bytemuck::cast_slice_mut(out_world_spheres),
            );
            self.read_buffer_sync(
                &out_ees_buf,
                ees_byte_len,
                bytemuck::cast_slice_mut(out_ee_poses),
            );
        } else {
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            CpuSimdBatchBackend.forward_kinematics_batch_f32(
                model,
                q_batch,
                out_world_spheres,
                out_ee_poses,
            );
        }
    }

    fn collision_cost_batch_f32(
        &self,
        model: &FlatRobotModelF32,
        esdf: Option<&EsdfGridF32>,
        q_batch: &[f32],
        activation_distance: f32,
        out_costs: &mut [f32],
    ) {
        use wgpu::util::DeviceExt;
        let dof = model.dof.max(1);
        let batch_size = q_batch.len() / dof;
        if batch_size == 0 {
            return;
        }

        let (has_esdf, esdf_origin, esdf_inv, esdf_dims, esdf_slice): (
            u32,
            [f32; 3],
            f32,
            [u32; 3],
            &[f32],
        ) = match esdf {
            Some(g) => (
                1,
                g.origin,
                g.inv_voxel_size,
                [g.dims[0] as u32, g.dims[1] as u32, g.dims[2] as u32],
                &g.distances,
            ),
            None => (0, [0.0; 3], 1.0, [1, 1, 1], &[1.0_f32]),
        };

        let uniforms = RolloutUniformsF32 {
            batch_size: batch_size as u32,
            dof: model.dof as u32,
            num_links: model.num_links as u32,
            num_spheres: model.spheres.len() as u32,
            num_ees: 0,
            num_self_pairs: model.active_self_pairs.len() as u32,
            has_esdf,
            activation_distance,
            esdf_origin,
            esdf_inv_voxel_size: esdf_inv,
            esdf_dims,
            _pad: 0,
        };

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let joints_buf = self.create_storage_buffer_from_bytes(
            "joints",
            bytemuck::cast_slice(&model.joints),
            false,
        );
        let spheres_buf = self.create_storage_buffer_from_bytes(
            "spheres",
            bytemuck::cast_slice(&model.spheres),
            false,
        );
        let ees_buf = self.create_storage_buffer_from_bytes(
            "ees",
            bytemuck::cast_slice(&model.end_effectors),
            false,
        );
        let q_buf =
            self.create_storage_buffer_from_bytes("q_batch", bytemuck::cast_slice(q_batch), false);

        let spheres_byte_len = (batch_size * model.spheres.len() * 16) as u64;
        let costs_byte_len = std::mem::size_of_val(out_costs) as u64;

        let out_spheres_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_world_spheres"),
            size: spheres_byte_len.max(16),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let out_costs_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_costs"),
            size: costs_byte_len.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let pairs_buf = self.create_storage_buffer_from_bytes(
            "self_pairs",
            bytemuck::cast_slice(&model.active_self_pairs),
            false,
        );
        let esdf_buf = self.create_storage_buffer_from_bytes(
            "esdf_distances",
            bytemuck::cast_slice(esdf_slice),
            false,
        );

        let group0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rollout_group0"),
            layout: &self.group0_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: joints_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: spheres_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ees_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: out_spheres_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: out_costs_buf.as_entire_binding(),
                },
            ],
        });

        let group1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rollout_group1"),
            layout: &self.group1_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pairs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: esdf_buf.as_entire_binding(),
                },
            ],
        });

        // Dispatch both fk_main and collision_main back-to-back in a single command submission!
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("collision_rollout_encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fk_and_collision_pass"),
                timestamp_writes: None,
            });
            let workgroups = (batch_size as u32).div_ceil(64);
            cpass.set_pipeline(&self.fk_pipeline);
            cpass.set_bind_group(0, &group0, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);

            cpass.set_pipeline(&self.collision_pipeline);
            cpass.set_bind_group(0, &group0, &[]);
            cpass.set_bind_group(1, &group1, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        if self.is_hardware_accelerated() {
            self.read_buffer_sync(
                &out_costs_buf,
                costs_byte_len,
                bytemuck::cast_slice_mut(out_costs),
            );
        } else {
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            CpuSimdBatchBackend.collision_cost_batch_f32(
                model,
                esdf,
                q_batch,
                activation_distance,
                out_costs,
            );
        }
    }
}

/// Smooth CuRobo-style piecewise-quadratic collision hinge cost ($C^1$ continuous).
#[inline]
pub fn smooth_hinge_cost(signed_dist: f32, activation_dist: f32, inv_two_act: f32) -> f32 {
    if signed_dist >= activation_dist {
        0.0
    } else if signed_dist >= 0.0 {
        let diff = activation_dist - signed_dist;
        diff * diff * inv_two_act
    } else {
        0.5 * activation_dist - signed_dist
    }
}

#[inline(always)]
fn simd_quat_mul(a: &[f32x8; 4], b: &[f32x8; 4]) -> [f32x8; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

#[inline(always)]
fn simd_transform_point(t: &[f32x8; 3], q: &[f32x8; 4], p: &[f32x8; 3]) -> [f32x8; 3] {
    let two = f32x8::splat(2.0);
    let uv_x = two * (q[1] * p[2] - q[2] * p[1]);
    let uv_y = two * (q[2] * p[0] - q[0] * p[2]);
    let uv_z = two * (q[0] * p[1] - q[1] * p[0]);

    let uuv_x = q[1] * uv_z - q[2] * uv_y;
    let uuv_y = q[2] * uv_x - q[0] * uv_z;
    let uuv_z = q[0] * uv_y - q[1] * uv_x;

    [
        t[0] + p[0] + q[3] * uv_x + uuv_x,
        t[1] + p[1] + q[3] * uv_y + uuv_y,
        t[2] + p[2] + q[3] * uv_z + uuv_z,
    ]
}

#[inline(always)]
fn scalar_quat_mul(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

#[inline(always)]
fn scalar_transform_point(t: &[f32; 3], q: &[f32; 4], p: &[f32; 3]) -> [f32; 3] {
    let uv_x = 2.0 * (q[1] * p[2] - q[2] * p[1]);
    let uv_y = 2.0 * (q[2] * p[0] - q[0] * p[2]);
    let uv_z = 2.0 * (q[0] * p[1] - q[1] * p[0]);

    let uuv_x = q[1] * uv_z - q[2] * uv_y;
    let uuv_y = q[2] * uv_x - q[0] * uv_z;
    let uuv_z = q[0] * uv_y - q[1] * uv_x;

    [
        t[0] + p[0] + q[3] * uv_x + uuv_x,
        t[1] + p[1] + q[3] * uv_y + uuv_y,
        t[2] + p[2] + q[3] * uv_z + uuv_z,
    ]
}

impl From<&Point3<f64>> for EsdfGridF32 {
    fn from(origin: &Point3<f64>) -> Self {
        Self::new(
            [origin.x as f32, origin.y as f32, origin.z as f32],
            0.02,
            [64, 64, 64],
            1.0,
        )
    }
}

/// Converts a slice of `f64` joint values into a `Vec<f32>`.
#[inline]
pub fn config_to_f32(q: &[f64]) -> Vec<f32> {
    q.iter().map(|&v| v as f32).collect()
}

/// Helper to convert a `[tx, ty, tz, qx, qy, qz, qw]` `f32` pose back to `nalgebra::Isometry3<f64>`.
#[inline]
pub fn pose_f32_to_isometry(pose: &[f32; 7]) -> Isometry3<f64> {
    let t = Vector3::new(pose[0] as f64, pose[1] as f64, pose[2] as f64);
    let q = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
        pose[6] as f64,
        pose[3] as f64,
        pose[4] as f64,
        pose[5] as f64,
    ));
    Isometry3::from_parts(t.into(), q)
}
