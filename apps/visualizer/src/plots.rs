//! Pure-Rust anti-aliased RGBA8 telemetry plot rasterizers (HiDPI-ready, zero external deps).

use crate::scene::{BuiltWorkcell, PlaybackTrajectory};

/// Rasterized RGBA8 plot image ready to upload into Slint `Image::from_rgba8(...)`.
#[derive(Debug, Clone)]
pub struct PlotImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

struct Canvas {
    w: usize,
    h: usize,
    rgba: Vec<u8>,
}

impl Canvas {
    fn new(width: u32, height: u32, bg: [u8; 4]) -> Self {
        let w = width.max(16) as usize;
        let h = height.max(16) as usize;
        let mut rgba = vec![0u8; w * h * 4];
        for px in rgba.as_chunks_mut::<4>().0 {
            px.copy_from_slice(&bg);
        }
        Self { w, h, rgba }
    }

    #[inline]
    fn blend_px(&mut self, x: i32, y: i32, col: [u8; 4], cov: f32) {
        if x < 0 || y < 0 || x >= self.w as i32 || y >= self.h as i32 {
            return;
        }
        let a = ((col[3] as f32) / 255.0) * cov.clamp(0.0, 1.0);
        if a <= 1e-4 {
            return;
        }
        let idx = ((y as usize) * self.w + (x as usize)) * 4;
        let inv_a = 1.0 - a;
        self.rgba[idx] = ((self.rgba[idx] as f32) * inv_a + (col[0] as f32) * a).round() as u8;
        self.rgba[idx + 1] =
            ((self.rgba[idx + 1] as f32) * inv_a + (col[1] as f32) * a).round() as u8;
        self.rgba[idx + 2] =
            ((self.rgba[idx + 2] as f32) * inv_a + (col[2] as f32) * a).round() as u8;
        self.rgba[idx + 3] = 255;
    }

    fn fill_rect(&mut self, x0: usize, y0: usize, x1: usize, y1: usize, col: [u8; 4]) {
        let xe = x1.min(self.w);
        let ye = y1.min(self.h);
        for y in y0..ye {
            for x in x0..xe {
                self.blend_px(x as i32, y as i32, col, 1.0);
            }
        }
    }

    fn draw_hline(&mut self, y: i32, col: [u8; 4], dashed: bool) {
        if y < 0 || y >= self.h as i32 {
            return;
        }
        for x in 0..self.w {
            if !dashed || (x / 5).is_multiple_of(2) {
                self.blend_px(x as i32, y, col, 1.0);
            }
        }
    }

    fn draw_vline(&mut self, x: i32, col: [u8; 4]) {
        if x < 0 || x >= self.w as i32 {
            return;
        }
        for y in 0..self.h {
            self.blend_px(x, y as i32, col, 1.0);
        }
    }

    fn draw_aa_segment(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, thickness: f32, col: [u8; 4]) {
        let pad = thickness.ceil() as i32 + 2;
        let min_x = (x0.min(x1).floor() as i32 - pad).max(0);
        let max_x = (x0.max(x1).ceil() as i32 + pad).min(self.w as i32 - 1);
        let min_y = (y0.min(y1).floor() as i32 - pad).max(0);
        let max_y = (y0.max(y1).ceil() as i32 + pad).min(self.h as i32 - 1);

        let dx = x1 - x0;
        let dy = y1 - y0;
        let len_sq = dx * dx + dy * dy;
        let half_t = 0.5 * thickness;

        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let fx = px as f32 + 0.5;
                let fy = py as f32 + 0.5;
                let d = if len_sq <= 1e-6 {
                    ((fx - x0).powi(2) + (fy - y0).powi(2)).sqrt()
                } else {
                    let t = (((fx - x0) * dx + (fy - y0) * dy) / len_sq).clamp(0.0, 1.0);
                    let cx = x0 + t * dx;
                    let cy = y0 + t * dy;
                    ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt()
                };
                if d <= half_t + 0.85 {
                    let cov = (half_t + 0.85 - d).clamp(0.0, 1.0);
                    self.blend_px(px, py, col, cov);
                }
            }
        }
    }

    fn draw_polyline(&mut self, pts: &[(f32, f32)], thickness: f32, col: [u8; 4]) {
        for seg in pts.windows(2) {
            self.draw_aa_segment(seg[0].0, seg[0].1, seg[1].0, seg[1].1, thickness, col);
        }
    }

    fn into_image(self) -> PlotImage {
        PlotImage {
            width: self.w as u32,
            height: self.h as u32,
            rgba: self.rgba,
        }
    }
}

const PLOT_BG: [u8; 4] = [13, 17, 23, 255];
const GRID_LINE: [u8; 4] = [38, 48, 66, 170];
const STAGE1_TINT: [u8; 4] = [56, 189, 248, 16];
const STAGE2_TINT: [u8; 4] = [167, 139, 250, 22];

fn setup_background(canvas: &mut Canvas, traj: &PlaybackTrajectory) {
    let w = canvas.w;
    let h = canvas.h;
    let dur = traj.metrics.duration_s.max(1e-6);

    if let Some(s1_end) = traj.metrics.stage1_end_time {
        let split_x = ((s1_end / dur).clamp(0.0, 1.0) * (w as f64)).round() as usize;
        canvas.fill_rect(0, 0, split_x, h, STAGE1_TINT);
        canvas.fill_rect(split_x, 0, w, h, STAGE2_TINT);
        canvas.draw_vline(split_x as i32, [167, 139, 250, 95]);
    }

    for k in 1..4 {
        let y = (h * k / 4) as i32;
        canvas.draw_hline(y, GRID_LINE, true);
    }
}

#[inline]
fn map_x(i: usize, total: usize, width: usize) -> f32 {
    if total <= 1 {
        return 0.5 * (width as f32);
    }
    let pad = 4.0_f32;
    let usable = (width as f32 - 2.0 * pad).max(1.0);
    pad + ((i as f32) / ((total - 1) as f32)) * usable
}

#[inline]
fn map_y(val: f64, y_min: f64, y_max: f64, height: usize) -> f32 {
    let pad = 6.0_f32;
    let usable = (height as f32 - 2.0 * pad).max(1.0);
    let span = (y_max - y_min).max(1e-6);
    let frac = ((val - y_min) / span).clamp(0.0, 1.0) as f32;
    (height as f32 - pad) - frac * usable
}

/// 1. Global Minimum Signed Clearance vs Safety Margin (`mm`).
pub fn render_min_clearance_plot(
    traj: &PlaybackTrajectory,
    safety_margin_m: f64,
    width: u32,
    height: u32,
) -> PlotImage {
    let mut c = Canvas::new(width, height, PLOT_BG);
    setup_background(&mut c, traj);

    let y_min = -10.0_f64;
    let margin_mm = safety_margin_m * 1000.0;
    let max_obs_mm = traj
        .frames
        .iter()
        .map(|f| f.min_clearance * 1000.0)
        .fold(margin_mm * 2.0, f64::max);
    let y_max = max_obs_mm.clamp(60.0, 350.0);

    let y_margin = map_y(margin_mm, y_min, y_max, c.h).round() as usize;
    c.fill_rect(0, y_margin.min(c.h), c.w, c.h, [244, 63, 94, 26]);
    c.draw_hline(y_margin as i32, [244, 63, 94, 210], true);

    let n = traj.frames.len();
    let pts_global: Vec<(f32, f32)> = traj
        .frames
        .iter()
        .enumerate()
        .map(|(i, f)| (map_x(i, n, c.w), map_y(f.min_clearance * 1000.0, y_min, y_max, c.h)))
        .collect();
    let pts_body: Vec<(f32, f32)> = traj
        .frames
        .iter()
        .enumerate()
        .filter_map(|(i, f)| {
            f.min_body_clearance
                .map(|b| (map_x(i, n, c.w), map_y(b * 1000.0, y_min, y_max, c.h)))
        })
        .collect();

    c.draw_polyline(&pts_body, 1.6, [245, 158, 11, 215]);
    c.draw_polyline(&pts_global, 2.2, [56, 189, 248, 255]);
    c.into_image()
}

/// 2. Zoomed Tool-to-Tool and Tool-to-Workpiece Clearance (`0..50 mm`).
pub fn render_tool_clearance_plot(
    traj: &PlaybackTrajectory,
    tool_tool_margin_m: f64,
    tool_wp_margin_m: f64,
    width: u32,
    height: u32,
) -> PlotImage {
    let mut c = Canvas::new(width, height, PLOT_BG);
    setup_background(&mut c, traj);

    let y_min = 0.0;
    let y_max = 50.0;
    let y_tt = map_y(tool_tool_margin_m * 1000.0, y_min, y_max, c.h) as i32;
    let y_tw = map_y(tool_wp_margin_m * 1000.0, y_min, y_max, c.h) as i32;
    c.draw_hline(y_tt, [52, 211, 153, 180], true);
    c.draw_hline(y_tw, [192, 132, 252, 180], true);

    let n = traj.frames.len();
    let pts_tt: Vec<(f32, f32)> = traj
        .frames
        .iter()
        .enumerate()
        .filter_map(|(i, f)| {
            f.min_tool_clearance
                .map(|d| (map_x(i, n, c.w), map_y(d * 1000.0, y_min, y_max, c.h)))
        })
        .collect();
    let pts_tw: Vec<(f32, f32)> = traj
        .frames
        .iter()
        .enumerate()
        .filter_map(|(i, f)| {
            f.min_tool_workpiece_clearance
                .map(|d| (map_x(i, n, c.w), map_y(d * 1000.0, y_min, y_max, c.h)))
        })
        .collect();

    c.draw_polyline(&pts_tt, 2.0, [52, 211, 153, 255]);
    c.draw_polyline(&pts_tw, 2.2, [192, 132, 252, 255]);
    c.into_image()
}

/// 3. Pose / Closed-Chain Grasp Convergence Error (`mm`).
pub fn render_pose_error_plot(
    traj: &PlaybackTrajectory,
    tol_mm: f64,
    width: u32,
    height: u32,
) -> PlotImage {
    let mut c = Canvas::new(width, height, PLOT_BG);
    setup_background(&mut c, traj);

    let use_grasp = traj.frames.iter().any(|f| f.grasp_error_mm > 1e-4)
        && traj.frames.iter().all(|f| (f.pose_error_mm - f.grasp_error_mm).abs() < 1e-3);

    let vals: Vec<f64> = traj
        .frames
        .iter()
        .map(|f| if use_grasp { f.grasp_error_mm } else { f.pose_error_mm })
        .collect();
    let max_v = vals.iter().copied().fold(tol_mm * 3.0, f64::max).max(10.0);

    let y_tol = map_y(tol_mm, 0.0, max_v, c.h) as i32;
    c.draw_hline(y_tol, [16, 185, 129, 200], true);

    let n = vals.len();
    let pts: Vec<(f32, f32)> = vals
        .iter()
        .enumerate()
        .map(|(i, &v)| (map_x(i, n, c.w), map_y(v, 0.0, max_v, c.h)))
        .collect();

    let stroke = if use_grasp {
        [167, 139, 250, 255]
    } else {
        [56, 189, 248, 255]
    };
    c.draw_polyline(&pts, 2.2, stroke);
    c.into_image()
}

/// 4. Per-Arm Maximum Joint Velocity (`rad/s`) vs Velocity Bound.
pub fn render_joint_vel_plot(
    workcell: &BuiltWorkcell,
    traj: &PlaybackTrajectory,
    width: u32,
    height: u32,
) -> PlotImage {
    let mut c = Canvas::new(width, height, PLOT_BG);
    setup_background(&mut c, traj);

    let max_limit = workcell
        .model
        .dof_limits()
        .iter()
        .map(|l| l.max_velocity)
        .fold(1.0_f64, f64::max);
    let y_max = (max_limit * 1.1).max(2.0);
    let y_lim = map_y(max_limit, 0.0, y_max, c.h) as i32;
    c.draw_hline(y_lim, [244, 63, 94, 165], true);

    let n = traj.frames.len();
    for arm in &workcell.arms {
        let col = [
            (arm.color_rgba[0] * 255.0).round() as u8,
            (arm.color_rgba[1] * 255.0).round() as u8,
            (arm.color_rgba[2] * 255.0).round() as u8,
            255,
        ];
        let pts: Vec<(f32, f32)> = traj
            .frames
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let vmax = f
                    .qd
                    .get(arm.q_range.clone())
                    .map_or(0.0, |sl| sl.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));
                (map_x(i, n, c.w), map_y(vmax, 0.0, y_max, c.h))
            })
            .collect();
        c.draw_polyline(&pts, 2.0, col);
    }

    c.into_image()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{preset_workcell, run_planner_sync, PresetId};

    #[test]
    fn renders_all_four_telemetry_plots_to_non_empty_rgba8() {
        let spec = preset_workcell(PresetId::DualArmRrtToppra);
        let wc = spec.compile(1).unwrap();
        let traj = run_planner_sync(&wc, &spec).unwrap();

        let p1 = render_min_clearance_plot(&traj, 0.04, 240, 84);
        let p2 = render_tool_clearance_plot(&traj, 0.004, 0.003, 240, 84);
        let p3 = render_pose_error_plot(&traj, 1.8, 240, 84);
        let p4 = render_joint_vel_plot(&wc, &traj, 240, 84);

        for p in [&p1, &p2, &p3, &p4] {
            assert_eq!(p.rgba.len(), (p.width * p.height * 4) as usize);
            let first = &p.rgba[0..4];
            assert!(
                p.rgba.as_chunks::<4>().0.iter().any(|px| px != first),
                "plot should contain drawn grid/curve pixels"
            );
        }
    }
}
