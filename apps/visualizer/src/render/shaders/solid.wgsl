// Shared camera & Blinn-Phong lighting pass for instanced opaque and Fresnel translucent geometry.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
    fill_dir: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> cam: CameraUniform;

struct VertexIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) model_col0: vec4<f32>,
    @location(3) model_col1: vec4<f32>,
    @location(4) model_col2: vec4<f32>,
    @location(5) model_col3: vec4<f32>,
    @location(6) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    let model = mat4x4<f32>(
        in.model_col0,
        in.model_col1,
        in.model_col2,
        in.model_col3,
    );
    let world_h = model * vec4<f32>(in.pos, 1.0);

    // Exact 3x3 cofactor normal matrix handles non-uniform cuboid/cylinder scales without extra instance bytes.
    let m0 = in.model_col0.xyz;
    let m1 = in.model_col1.xyz;
    let m2 = in.model_col2.xyz;
    let c0 = cross(m1, m2);
    let c1 = cross(m2, m0);
    let c2 = cross(m0, m1);
    let det_sign = select(-1.0, 1.0, dot(m0, c0) >= 0.0);
    let normal_mat = mat3x3<f32>(c0, c1, c2) * det_sign;

    var out: VertexOut;
    out.clip_pos = cam.view_proj * world_h;
    out.world_pos = world_h.xyz;
    out.world_normal = normalize(normal_mat * in.normal);
    out.color = in.color;
    return out;
}

fn shade_blinn_phong(in: VertexOut) -> vec3<f32> {
    let n = normalize(in.world_normal);
    let v = normalize(cam.camera_pos.xyz - in.world_pos);
    let l1 = normalize(cam.light_dir.xyz);
    let l2 = normalize(cam.fill_dir.xyz);

    // Z-up hemisphere ambient (cool sky, slate floor bounce)
    let hemi = 0.5 + 0.5 * n.z;
    let ambient = mix(vec3<f32>(0.10, 0.12, 0.16), vec3<f32>(0.24, 0.28, 0.35), hemi);

    let ndl1 = max(dot(n, l1), 0.0);
    let ndl2 = max(dot(n, l2), 0.0);
    let h1 = normalize(l1 + v);
    let spec1 = pow(max(dot(n, h1), 0.0), 48.0) * 0.35;

    let lit = ambient + vec3<f32>(0.85, 0.87, 0.90) * ndl1 + vec3<f32>(0.28, 0.34, 0.44) * ndl2;
    return in.color.rgb * lit + vec3<f32>(spec1);
}

@fragment
fn fs_opaque(in: VertexOut) -> @location(0) vec4<f32> {
    let rgb = shade_blinn_phong(in);
    return vec4<f32>(rgb, 1.0);
}

@fragment
fn fs_translucent(in: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let v = normalize(cam.camera_pos.xyz - in.world_pos);
    let fresnel = pow(1.0 - abs(dot(n, v)), 1.8);
    let rgb = shade_blinn_phong(in) + in.color.rgb * (0.25 * fresnel);
    let alpha = clamp(in.color.a * (0.38 + 0.62 * fresnel), 0.05, 0.95);
    return vec4<f32>(rgb, alpha);
}
