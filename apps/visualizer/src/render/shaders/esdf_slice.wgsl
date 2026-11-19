// Interactive 2D X/Y/Z slice quad visualizing EsdfGridF32 with manual bilinear filtering and fwidth iso-contours.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
    fill_dir: vec4<f32>,
};

struct EsdfSliceUniform {
    corner_origin: vec4<f32>,
    axis_u: vec4<f32>,
    axis_v: vec4<f32>,
    dims_and_act: vec4<f32>, // (width, height, activation_dist, opacity)
};

@group(0) @binding(0)
var<uniform> cam: CameraUniform;

@group(1) @binding(0)
var<uniform> slice_uni: EsdfSliceUniform;

@group(1) @binding(1)
var esdf_tex: texture_2d<f32>;

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
    );
    let uv = corners[vid];
    let world = slice_uni.corner_origin.xyz
        + slice_uni.axis_u.xyz * uv.x
        + slice_uni.axis_v.xyz * uv.y;

    var out: VertexOut;
    out.clip_pos = cam.view_proj * vec4<f32>(world, 1.0);
    out.uv = uv;
    return out;
}

fn sample_esdf_bilinear(uv: vec2<f32>, w: i32, h: i32) -> f32 {
    let gx = clamp(uv.x * f32(w) - 0.5, 0.0, f32(max(w - 1, 0)));
    let gy = clamp(uv.y * f32(h) - 0.5, 0.0, f32(max(h - 1, 0)));
    let x0 = i32(floor(gx));
    let y0 = i32(floor(gy));
    let x1 = min(x0 + 1, max(w - 1, 0));
    let y1 = min(y0 + 1, max(h - 1, 0));
    let fx = gx - f32(x0);
    let fy = gy - f32(y0);

    let d00 = textureLoad(esdf_tex, vec2<i32>(x0, y0), 0).r;
    let d10 = textureLoad(esdf_tex, vec2<i32>(x1, y0), 0).r;
    let d01 = textureLoad(esdf_tex, vec2<i32>(x0, y1), 0).r;
    let d11 = textureLoad(esdf_tex, vec2<i32>(x1, y1), 0).r;
    return mix(mix(d00, d10, fx), mix(d01, d11, fx), fy);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let w = max(i32(slice_uni.dims_and_act.x), 1);
    let h = max(i32(slice_uni.dims_and_act.y), 1);
    let act = max(slice_uni.dims_and_act.z, 0.01);
    let base_alpha = clamp(slice_uni.dims_and_act.w, 0.05, 0.95);

    let d = sample_esdf_bilinear(in.uv, w, h);
    let dd = max(fwidth(d), 1e-4);

    // Colormap: crimson inside (< 0), warm amber in [0, act], cool cyan/slate beyond act
    var color: vec3<f32>;
    if d < 0.0 {
        let t_in = clamp(-d / 0.05, 0.0, 1.0);
        color = mix(vec3<f32>(0.96, 0.25, 0.37), vec3<f32>(0.55, 0.08, 0.16), t_in);
    } else if d < act {
        let t_act = clamp(d / act, 0.0, 1.0);
        color = mix(vec3<f32>(0.96, 0.62, 0.14), vec3<f32>(0.22, 0.74, 0.97), t_act);
    } else {
        let t_far = clamp((d - act) / 0.25, 0.0, 1.0);
        color = mix(vec3<f32>(0.12, 0.45, 0.62), vec3<f32>(0.07, 0.12, 0.18), t_far);
    }

    // Crisp anti-aliased iso-contours at d=0 (obstacle surface), d=act (activation), and every 5cm
    let zero_line = 1.0 - smoothstep(0.0, 1.5 * dd, abs(d));
    let act_line = 1.0 - smoothstep(0.0, 1.25 * dd, abs(d - act));
    let spacing = 0.05;
    let dist_mod = abs(fract(d / spacing + 0.5) - 0.5) * spacing;
    let step_line = (1.0 - smoothstep(0.0, 1.15 * dd, dist_mod)) * 0.45;

    color = mix(color, vec3<f32>(1.0, 1.0, 1.0), clamp(zero_line, 0.0, 1.0));
    color = mix(color, vec3<f32>(0.98, 0.80, 0.15), clamp(act_line * 0.85, 0.0, 1.0));
    color = mix(color, vec3<f32>(0.85, 0.94, 1.0), step_line);

    let alpha = clamp(base_alpha + zero_line * 0.35 + act_line * 0.20, 0.08, 0.92);
    return vec4<f32>(color, alpha);
}
