//! Procedural static unit meshes (`unit_cube`, `unit_cylinder_y`, `unit_sphere`) for instanced rendering.

use bytemuck::{Pod, Zeroable};
use std::f32::consts::{PI, TAU};

/// Interleaved position + outward normal vertex (`24` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct VertexPN {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
}

/// Generates a unit UV sphere centered at the origin with radius `1.0`.
pub fn generate_unit_sphere(sectors: u32, stacks: u32) -> (Vec<VertexPN>, Vec<u32>) {
    let sec = sectors.max(8);
    let stk = stacks.max(4);
    let mut verts = Vec::with_capacity(((stk + 1) * (sec + 1)) as usize);
    let mut indices = Vec::with_capacity((stk * sec * 6) as usize);

    for i in 0..=stk {
        let phi = PI * (i as f32) / (stk as f32);
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        for j in 0..=sec {
            let theta = TAU * (j as f32) / (sec as f32);
            let nx = sin_phi * theta.cos();
            let ny = sin_phi * theta.sin();
            let nz = cos_phi;
            verts.push(VertexPN {
                pos: [nx, ny, nz],
                normal: [nx, ny, nz],
            });
        }
    }

    let stride = sec + 1;
    for i in 0..stk {
        for j in 0..sec {
            let k00 = i * stride + j;
            let k10 = (i + 1) * stride + j;
            let k01 = i * stride + j + 1;
            let k11 = (i + 1) * stride + j + 1;

            // CCW outward winding: (-Z) x (+theta) = +radial
            if i != 0 {
                indices.extend_from_slice(&[k00, k10, k01]);
            }
            if i + 1 != stk {
                indices.extend_from_slice(&[k01, k10, k11]);
            }
        }
    }

    (verts, indices)
}

/// Generates a closed Y-aligned cylinder (`y in [-1, +1]`, radius `1.0`) matching `parry3d::shape::Cylinder`.
pub fn generate_unit_cylinder_y(segments: u32) -> (Vec<VertexPN>, Vec<u32>) {
    let seg = segments.max(8);
    let mut verts = Vec::new();
    let mut indices = Vec::new();

    // 1. Side wall ring vertices (top & bottom)
    let side_base = 0_u32;
    for j in 0..=seg {
        let th = TAU * (j as f32) / (seg as f32);
        let nx = th.cos();
        let nz = th.sin();
        verts.push(VertexPN {
            pos: [nx, 1.0, nz],
            normal: [nx, 0.0, nz],
        });
        verts.push(VertexPN {
            pos: [nx, -1.0, nz],
            normal: [nx, 0.0, nz],
        });
    }
    for j in 0..seg {
        let t0 = side_base + 2 * j;
        let b0 = t0 + 1;
        let t1 = side_base + 2 * (j + 1);
        let b1 = t1 + 1;
        // Outward CCW: (+theta) x (-Y) = +radial
        indices.extend_from_slice(&[t0, t1, b0]);
        indices.extend_from_slice(&[t1, b1, b0]);
    }

    // 2. Top cap (y = +1, normal = +Y)
    let top_center = verts.len() as u32;
    verts.push(VertexPN {
        pos: [0.0, 1.0, 0.0],
        normal: [0.0, 1.0, 0.0],
    });
    let top_ring = verts.len() as u32;
    for j in 0..=seg {
        let th = TAU * (j as f32) / (seg as f32);
        verts.push(VertexPN {
            pos: [th.cos(), 1.0, th.sin()],
            normal: [0.0, 1.0, 0.0],
        });
    }
    for j in 0..seg {
        indices.extend_from_slice(&[top_center, top_ring + j + 1, top_ring + j]);
    }

    // 3. Bottom cap (y = -1, normal = -Y)
    let bot_center = verts.len() as u32;
    verts.push(VertexPN {
        pos: [0.0, -1.0, 0.0],
        normal: [0.0, -1.0, 0.0],
    });
    let bot_ring = verts.len() as u32;
    for j in 0..=seg {
        let th = TAU * (j as f32) / (seg as f32);
        verts.push(VertexPN {
            pos: [th.cos(), -1.0, th.sin()],
            normal: [0.0, -1.0, 0.0],
        });
    }
    for j in 0..seg {
        indices.extend_from_slice(&[bot_center, bot_ring + j, bot_ring + j + 1]);
    }

    (verts, indices)
}

/// Generates a centered unit cube (`[-1, +1]^3`) with crisp flat face normals (`24` vertices, `36` indices).
pub fn generate_unit_cube() -> (Vec<VertexPN>, Vec<u32>) {
    #[rustfmt::skip]
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        // +X
        ([1.0, 0.0, 0.0], [[1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0], [1.0, -1.0, 1.0]]),
        // -X
        ([-1.0, 0.0, 0.0], [[-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]]),
        // +Y
        ([0.0, 1.0, 0.0], [[1.0, 1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        // -Y
        ([0.0, -1.0, 0.0], [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]]),
        // +Z
        ([0.0, 0.0, 1.0], [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]]),
        // -Z
        ([0.0, 0.0, -1.0], [[-1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]),
    ];

    let mut verts = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (n, quad) in faces {
        let base = verts.len() as u32;
        for pos in quad {
            verts.push(VertexPN { pos, normal: n });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (verts, indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn assert_outward_winding(verts: &[VertexPN], indices: &[u32]) {
        assert!(!indices.is_empty() && indices.len().is_multiple_of(3));
        for tri in indices.as_chunks::<3>().0 {
            let a = Vector3::from(verts[tri[0] as usize].pos);
            let b = Vector3::from(verts[tri[1] as usize].pos);
            let c = Vector3::from(verts[tri[2] as usize].pos);
            let face_n = (b - a).cross(&(c - a));
            if face_n.norm_squared() < 1e-12 {
                continue;
            }
            let v_n = Vector3::from(verts[tri[0] as usize].normal);
            assert!(
                face_n.dot(&v_n) > 0.0,
                "triangle winding opposes vertex normal: face_n={face_n:?}, v_n={v_n:?}"
            );
        }
    }

    #[test]
    fn unit_meshes_have_consistent_outward_ccw_winding() {
        let (sv, si) = generate_unit_sphere(16, 10);
        assert_outward_winding(&sv, &si);
        let (cv, ci) = generate_unit_cylinder_y(16);
        assert_outward_winding(&cv, &ci);
        let (bv, bi) = generate_unit_cube();
        assert_outward_winding(&bv, &bi);
    }
}
