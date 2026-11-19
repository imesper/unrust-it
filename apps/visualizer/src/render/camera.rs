//! Right-handed Z-up orbit/pan/zoom camera with explicit WebGPU `[0, 1]` NDC depth mapping.

use nalgebra::{Isometry3, Matrix4, Point3, Vector3, Vector4};
use std::f32::consts::{FRAC_PI_2, PI};

/// Converts OpenGL `[-1, 1]` NDC Z (produced by `Matrix4::new_perspective`) to WebGPU `[0, 1]` NDC Z.
#[rustfmt::skip]
pub const OPENGL_TO_WGPU_NDC: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

/// Interactive Z-up spherical orbit camera with pan and raycast picking helpers.
#[derive(Debug, Clone, PartialEq)]
pub struct OrbitCamera {
    pub target: Vector3<f32>,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub fov_y_rad: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vector3::new(0.06, 0.0, 0.42),
            yaw: -0.68,
            pitch: 0.42,
            distance: 1.65,
            fov_y_rad: 45.0_f32.to_radians(),
            near: 0.01,
            far: 50.0,
        }
    }
}

impl OrbitCamera {
    /// World-space camera eye position.
    #[inline]
    pub fn eye_position(&self) -> Vector3<f32> {
        let cp = self.pitch.cos();
        self.target
            + self.distance
                * Vector3::new(cp * self.yaw.cos(), cp * self.yaw.sin(), self.pitch.sin())
    }

    /// World-to-camera right-handed Z-up view matrix.
    #[inline]
    pub fn view_matrix(&self) -> Matrix4<f32> {
        let eye = self.eye_position();
        let up = if self.pitch.abs() > 1.54 {
            Vector3::new(-self.yaw.cos(), -self.yaw.sin(), 0.0)
        } else {
            Vector3::z()
        };
        Isometry3::look_at_rh(&Point3::from(eye), &Point3::from(self.target), &up).to_homogeneous()
    }

    /// Camera-to-clip WebGPU `[0, 1]` depth projection matrix.
    #[inline]
    pub fn proj_matrix(&self, aspect: f32) -> Matrix4<f32> {
        let persp = Matrix4::new_perspective(aspect.max(1e-3), self.fov_y_rad, self.near, self.far);
        OPENGL_TO_WGPU_NDC * persp
    }

    /// Combined `projection * view` matrix in column-major layout for GPU uniforms.
    #[inline]
    pub fn view_proj(&self, aspect: f32) -> Matrix4<f32> {
        self.proj_matrix(aspect) * self.view_matrix()
    }

    /// Orbits around [`Self::target`] in yaw/pitch radians for pointer drag deltas `(dx, dy)`.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.0075;
        self.pitch = (self.pitch + dy * 0.0075).clamp(-1.52, 1.52);
    }

    /// Pans [`Self::target`] in the camera's local right/up plane scaled by distance.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let view_dir = (self.target - self.eye_position()).normalize();
        let right = view_dir.cross(&Vector3::z());
        let right = if right.norm_squared() > 1e-8 {
            right.normalize()
        } else {
            Vector3::x()
        };
        let cam_up = right.cross(&view_dir).normalize();
        let scale = self.distance * 0.0018;
        self.target += (-right * dx + cam_up * dy) * scale;
    }

    /// Exponentially zooms camera distance in or out for scroll wheel delta.
    pub fn zoom(&mut self, delta: f32) {
        let factor = (-delta * 0.0015).exp();
        self.distance = (self.distance * factor).clamp(0.18, 15.0);
    }

    /// Resets to the default isometric workcell framing.
    pub fn set_iso(&mut self) {
        *self = Self::default();
    }

    /// Switches to a top-down Z-axis overview.
    pub fn set_top(&mut self) {
        self.yaw = -FRAC_PI_2;
        self.pitch = 1.48;
    }

    /// Switches to a front view along `-X`.
    pub fn set_front(&mut self) {
        self.yaw = 0.0;
        self.pitch = 0.12;
    }

    /// Switches to a side view along `-Y`.
    pub fn set_side(&mut self) {
        self.yaw = -FRAC_PI_2;
        self.pitch = 0.12;
    }

    /// Focuses tightly on a world-space region of interest (e.g., workpiece/tool zone).
    pub fn focus_point(&mut self, center: Vector3<f32>, distance: f32) {
        self.target = center;
        self.distance = distance.clamp(0.18, 15.0);
        if self.pitch < 0.15 {
            self.pitch = 0.36;
        }
    }

    /// Computes a world-space picking ray `(origin, unit_dir)` from normalized screen coords `[0, 1]`.
    pub fn screen_ray(&self, norm_x: f32, norm_y: f32, aspect: f32) -> (Vector3<f32>, Vector3<f32>) {
        let ndc_x = 2.0 * norm_x - 1.0;
        let ndc_y = 1.0 - 2.0 * norm_y;
        let Some(inv_vp) = self.view_proj(aspect).try_inverse() else {
            let eye = self.eye_position();
            return (eye, (self.target - eye).normalize());
        };
        let p_near_h = inv_vp * Vector4::new(ndc_x, ndc_y, 0.0, 1.0);
        let p_far_h = inv_vp * Vector4::new(ndc_x, ndc_y, 1.0, 1.0);
        let p_near = p_near_h.xyz() / p_near_h.w.max(1e-6);
        let p_far = p_far_h.xyz() / p_far_h.w.max(1e-6);
        let dir = (p_far - p_near).normalize();
        (p_near, dir)
    }

    /// Wraps yaw cleanly into `[-PI, PI]`.
    pub fn normalize_yaw(&mut self) {
        while self.yaw > PI {
            self.yaw -= 2.0 * PI;
        }
        while self.yaw < -PI {
            self.yaw += 2.0 * PI;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_ndc_maps_near_to_zero_and_far_to_one() {
        let cam = OrbitCamera::default();
        let eye = cam.eye_position();
        let fwd = (cam.target - eye).normalize();
        let vp = cam.view_proj(16.0 / 9.0);

        let p_near = eye + fwd * cam.near;
        let c_near = vp * Vector4::new(p_near.x, p_near.y, p_near.z, 1.0);
        let z_near = c_near.z / c_near.w;
        assert!(
            z_near.abs() < 1e-4,
            "expected near plane to map to WGPU NDC z=0, got {z_near}"
        );

        let p_far = eye + fwd * cam.far;
        let c_far = vp * Vector4::new(p_far.x, p_far.y, p_far.z, 1.0);
        let z_far = c_far.z / c_far.w;
        assert!(
            (z_far - 1.0).abs() < 1e-4,
            "expected far plane to map to WGPU NDC z=1, got {z_far}"
        );
    }
}
