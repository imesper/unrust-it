// WGSL Compute Shaders for CuRobo-style Batched f32 Forward Kinematics & ESDF/Self-Collision Rollouts
// Uses at most 8 storage buffers total to strictly satisfy WebGPU default limits (`max_storage_buffers_per_shader_stage = 8`).

struct RolloutParams {
    batch_size: u32,
    dof: u32,
    num_links: u32,
    num_spheres: u32,
    num_ees: u32,
    num_self_pairs: u32,
    has_esdf: u32,
    activation_distance: f32,
    esdf_origin: vec3<f32>,
    esdf_inv_voxel_size: f32,
    esdf_dims: vec3<u32>,
    _pad: u32,
};

struct FlatJoint {
    origin_xyz: vec3<f32>,
    q_index: u32,
    origin_quat: vec4<f32>,
    axis_xyz: vec3<f32>,
    kind: u32,
    parent_link: u32,
    child_link: u32,
    min_pos: f32,
    max_pos: f32,
};

struct FlatLinkSphere {
    local_center: vec3<f32>,
    radius: f32,
    link_index: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct FlatSpherePair {
    sphere_a: u32,
    sphere_b: u32,
    combined_radius: f32,
    pad: u32,
};

struct FlatEndEffector {
    local_xyz: vec3<f32>,
    link_index: u32,
    local_quat: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: RolloutParams;
@group(0) @binding(1) var<storage, read> joints: array<FlatJoint>;
@group(0) @binding(2) var<storage, read> spheres: array<FlatLinkSphere>;
@group(0) @binding(3) var<storage, read> end_effectors: array<FlatEndEffector>;
@group(0) @binding(4) var<storage, read> q_batch: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_world_spheres: array<vec4<f32>>;
// Shared scalar output buffer: holds `out_ee_poses` (7 floats per EE) in `fk_main`, or `out_costs` (1 float per state) in `collision_main`.
@group(0) @binding(6) var<storage, read_write> out_scalars: array<f32>;

@group(1) @binding(0) var<storage, read> self_pairs: array<FlatSpherePair>;
@group(1) @binding(1) var<storage, read> esdf_distances: array<f32>;

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

fn transform_point(t: vec3<f32>, q: vec4<f32>, p: vec3<f32>) -> vec3<f32> {
    let q_vec = q.xyz;
    let uv = 2.0 * cross(q_vec, p);
    let uuv = cross(q_vec, uv);
    return t + p + q.w * uv + uuv;
}

fn smooth_hinge_cost(signed_dist: f32, act: f32) -> f32 {
    if (signed_dist >= act) {
        return 0.0;
    } else if (signed_dist >= 0.0) {
        let diff = act - signed_dist;
        return (0.5 / act) * diff * diff;
    } else {
        return 0.5 * act - signed_dist;
    }
}

fn sample_esdf_distance(p: vec3<f32>) -> f32 {
    let g = (p - params.esdf_origin) * params.esdf_inv_voxel_size - vec3<f32>(0.5);
    if (g.x < 0.0 || g.y < 0.0 || g.z < 0.0) {
        return 1000.0;
    }
    let ix = u32(g.x);
    let iy = u32(g.y);
    let iz = u32(g.z);
    if (ix + 1u >= params.esdf_dims.x || iy + 1u >= params.esdf_dims.y || iz + 1u >= params.esdf_dims.z) {
        return 1000.0;
    }

    let tx = g.x - f32(ix);
    let ty = g.y - f32(iy);
    let tz = g.z - f32(iz);

    let stride_y = params.esdf_dims.x;
    let stride_z = params.esdf_dims.x * params.esdf_dims.y;
    let base = iz * stride_z + iy * stride_y + ix;

    let c000 = esdf_distances[base];
    let c100 = esdf_distances[base + 1u];
    let c010 = esdf_distances[base + stride_y];
    let c110 = esdf_distances[base + stride_y + 1u];
    let c001 = esdf_distances[base + stride_z];
    let c101 = esdf_distances[base + stride_z + 1u];
    let c011 = esdf_distances[base + stride_z + stride_y];
    let c111 = esdf_distances[base + stride_z + stride_y + 1u];

    let c00 = c000 * (1.0 - tx) + c100 * tx;
    let c10 = c010 * (1.0 - tx) + c110 * tx;
    let c01 = c001 * (1.0 - tx) + c101 * tx;
    let c11 = c011 * (1.0 - tx) + c111 * tx;

    let c0 = c00 * (1.0 - ty) + c10 * ty;
    let c1 = c01 * (1.0 - ty) + c11 * ty;
    return c0 * (1.0 - tz) + c1 * tz;
}

@compute @workgroup_size(64)
fn fk_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= params.batch_size) {
        return;
    }

    var link_t: array<vec3<f32>, 64>;
    var link_q: array<vec4<f32>, 64>;
    let q_offset = b * params.dof;

    for (var j_idx: u32 = 0u; j_idx < params.num_links; j_idx = j_idx + 1u) {
        let joint = joints[j_idx];
        var p_t = vec3<f32>(0.0);
        var p_q = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        if (joint.parent_link != 0xffffffffu) {
            p_t = link_t[joint.parent_link];
            p_q = link_q[joint.parent_link];
        }

        let base_t = transform_point(p_t, p_q, joint.origin_xyz);
        let base_q = quat_mul(p_q, joint.origin_quat);

        var q_val: f32 = 0.0;
        if (joint.q_index != 0xffffffffu) {
            q_val = q_batch[q_offset + joint.q_index];
        }

        var c_t = base_t;
        var c_q = base_q;
        if (joint.kind == 1u) {
            let half = 0.5 * q_val;
            let s = sin(half);
            let c = cos(half);
            let j_q = vec4<f32>(joint.axis_xyz * s, c);
            c_q = quat_mul(base_q, j_q);
        } else if (joint.kind == 2u) {
            let disp = joint.axis_xyz * q_val;
            c_t = transform_point(base_t, base_q, disp);
        }

        link_t[joint.child_link] = c_t;
        link_q[joint.child_link] = c_q;
    }

    let sphere_base = b * params.num_spheres;
    for (var s_idx: u32 = 0u; s_idx < params.num_spheres; s_idx = s_idx + 1u) {
        let s = spheres[s_idx];
        let wp = transform_point(link_t[s.link_index], link_q[s.link_index], s.local_center);
        out_world_spheres[sphere_base + s_idx] = vec4<f32>(wp, s.radius);
    }

    let ee_base = b * params.num_ees;
    for (var ee_idx: u32 = 0u; ee_idx < params.num_ees; ee_idx = ee_idx + 1u) {
        let ee = end_effectors[ee_idx];
        let t = transform_point(link_t[ee.link_index], link_q[ee.link_index], ee.local_xyz);
        let q = quat_mul(link_q[ee.link_index], ee.local_quat);
        let out_off = (ee_base + ee_idx) * 7u;
        out_scalars[out_off + 0u] = t.x;
        out_scalars[out_off + 1u] = t.y;
        out_scalars[out_off + 2u] = t.z;
        out_scalars[out_off + 3u] = q.x;
        out_scalars[out_off + 4u] = q.y;
        out_scalars[out_off + 5u] = q.z;
        out_scalars[out_off + 6u] = q.w;
    }
}

@compute @workgroup_size(64)
fn collision_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= params.batch_size) {
        return;
    }

    let sphere_base = b * params.num_spheres;
    let act = max(params.activation_distance, 1e-4);
    var total_cost: f32 = 0.0;

    for (var p_idx: u32 = 0u; p_idx < params.num_self_pairs; p_idx = p_idx + 1u) {
        let pair = self_pairs[p_idx];
        let sa = out_world_spheres[sphere_base + pair.sphere_a];
        let sb = out_world_spheres[sphere_base + pair.sphere_b];
        let d = distance(sa.xyz, sb.xyz) - pair.combined_radius;
        total_cost = total_cost + smooth_hinge_cost(d, act);
    }

    if (params.has_esdf != 0u) {
        for (var s_idx: u32 = 0u; s_idx < params.num_spheres; s_idx = s_idx + 1u) {
            let s = out_world_spheres[sphere_base + s_idx];
            let voxel_d = sample_esdf_distance(s.xyz);
            let d = voxel_d - s.w;
            total_cost = total_cost + smooth_hinge_cost(d, act);
        }
    }

    out_scalars[b] = total_cost;
}
