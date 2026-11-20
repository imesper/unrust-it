//! Interactive zero-copy 3D `wgpu 30` viewport panel with orbit/pan/zoom/ray-picking
//! and non-overlapping corner HUD overlays.

use crate::gui::panels::{
    rgba_to_color32, status_badge, BORDER_FOCUS, BORDER_SUBTLE, TEXT_MUTED, TEXT_PRIMARY,
};
use crate::gui::state::StudioState;
use crate::gui::viewport::GpuViewportBridge;

pub fn show(
    ui: &mut egui::Ui,
    state: &mut StudioState,
    gpu: Option<&mut GpuViewportBridge>,
    render_state: Option<&egui_wgpu::RenderState>,
) {
    let avail = ui.available_size().max(egui::vec2(1.0, 1.0));
    let (rect, response) = ui.allocate_exact_size(avail, egui::Sense::click_and_drag());

    let ppp = ui.ctx().pixels_per_point().max(1.0);
    state.viewport_logical_size = (
        rect.width().round().max(1.0) as u32,
        rect.height().round().max(1.0) as u32,
    );
    let phys_w = (rect.width() * ppp).round().max(1.0) as u32;
    let phys_h = (rect.height() * ppp).round().max(1.0) as u32;

    let frame_ms = if let (Some(bridge), Some(rs)) = (gpu, render_state) {
        let tex_id = bridge.ensure_rendered(
            rs,
            phys_w,
            phys_h,
            &state.camera,
            &state.render_scene,
            &mut state.viewport_dirty,
        );
        ui.painter().image(
            tex_id,
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
        Some(bridge.last_frame_ms())
    } else {
        ui.painter()
            .rect_filled(rect, 0.0, egui::Color32::from_rgb(0x0d, 0x11, 0x17));
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "wgpu 30 render state unavailable",
            egui::FontId::proportional(14.0),
            TEXT_MUTED,
        );
        None
    };

    // Handle pointer orbit / pan / zoom / double-click picking directly on the 3D rect
    if response.dragged() {
        let delta = response.drag_delta();
        if delta != egui::Vec2::ZERO {
            let shift = ui.input(|i| i.modifiers.shift);
            let pan_btn = response.dragged_by(egui::PointerButton::Secondary)
                || response.dragged_by(egui::PointerButton::Middle)
                || (response.dragged_by(egui::PointerButton::Primary) && shift);
            if pan_btn {
                state.camera.pan(delta.x, delta.y);
            } else if response.dragged_by(egui::PointerButton::Primary) {
                state.camera.orbit(delta.x, delta.y);
            }
            state.viewport_dirty = true;
        }
    }

    if response.hovered() {
        let scroll_y = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll_y.abs() > 0.01 {
            state.camera.zoom(scroll_y);
            state.viewport_dirty = true;
        }
    }

    if response.double_clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let nx = ((pos.x - rect.min.x) / rect.width().max(1.0)).clamp(0.0, 1.0);
            let ny = ((pos.y - rect.min.y) / rect.height().max(1.0)).clamp(0.0, 1.0);
            state.pick_ray(nx, ny);
        }
    }

    let hud_frame = egui::Frame::new()
        .fill(egui::Color32::from_rgba_unmultiplied(0x14, 0x19, 0x22, 0xd9))
        .stroke(egui::Stroke::new(1.0, BORDER_SUBTLE))
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::symmetric(8, 4));

    // Top-Left HUD Status Pill
    let clr_mm = state
        .last_snapshot
        .as_ref()
        .map_or(0.0, |s| s.min_clearance * 1000.0);
    let (safety_lvl, safety_lbl) = if clr_mm < 2.0 {
        (2_u8, format!("COLLISION · {clr_mm:.1} mm"))
    } else if clr_mm < 15.0 {
        (1_u8, format!("TIGHT · {clr_mm:.1} mm"))
    } else {
        (0_u8, format!("SAFE · {clr_mm:.1} mm"))
    };

    egui::Area::new(egui::Id::unique("vp_hud_top_left"))
        .order(egui::Order::Middle)
        .constrain_to(rect)
        .fixed_pos(rect.min + egui::vec2(10.0, 10.0))
        .show(ui.ctx(), |ui| {
            hud_frame.show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(&state.spec.title)
                            .size(11.0)
                            .strong()
                            .color(TEXT_PRIMARY),
                    );
                    status_badge(ui, &safety_lbl, safety_lvl);
                    if let Some(ms) = frame_ms {
                        ui.label(
                            egui::RichText::new(format!("{ms:.1} ms · wgpu 30"))
                                .size(10.0)
                                .color(TEXT_MUTED),
                        );
                    }
                });
            });
        });

    // Top-Right Camera Preset & Layers Bar (anchored dynamically inside `rect` top-right corner)
    let screen = ui.ctx().content_rect();
    let cam_y_pad = if rect.width() < 640.0 { 44.0 } else { 10.0 };
    let top_right_offset = egui::vec2(
        rect.max.x - screen.max.x - 10.0,
        rect.min.y - screen.min.y + cam_y_pad,
    );
    let cam_bar_resp = egui::Area::new(egui::Id::unique("vp_hud_top_right"))
        .order(egui::Order::Middle)
        .constrain_to(rect)
        .anchor(egui::Align2::RIGHT_TOP, top_right_offset)
        .show(ui.ctx(), |ui| {
            hud_frame.show(ui, |ui| {
                ui.horizontal(|ui| {
                    for (label, id) in [
                        ("Iso", "iso"),
                        ("Top", "top"),
                        ("Front", "front"),
                        ("Side", "side"),
                        ("Workpiece", "workpiece"),
                    ] {
                        if ui.small_button(label).clicked() {
                            state.set_camera_preset(id);
                        }
                    }
                    let layers_lbl = if state.layers_open { "Layers ▴" } else { "Layers ▾" };
                    if ui.selectable_label(state.layers_open, layers_lbl).clicked() {
                        state.layers_open = !state.layers_open;
                    }
                });
            });
        })
        .response;

    // Floating Layers Popover
    if state.layers_open {
        let pop_pos = egui::pos2(
            (cam_bar_resp.rect.right() - 252.0).max(rect.left() + 10.0),
            cam_bar_resp.rect.bottom() + 6.0,
        );
        egui::Area::new(egui::Id::unique("vp_layers_popover"))
            .order(egui::Order::Middle)
            .constrain_to(rect)
            .fixed_pos(pop_pos)
            .show(ui.ctx(), |ui| {
                egui::Frame::new()
                    .fill(egui::Color32::from_rgba_unmultiplied(0x14, 0x19, 0x22, 0xf2))
                    .stroke(egui::Stroke::new(1.0, BORDER_FOCUS))
                    .corner_radius(egui::CornerRadius::same(6))
                    .inner_margin(egui::Margin::same(10))
                    .show(ui, |ui| {
                        ui.set_width(236.0);
                        ui.label(
                            egui::RichText::new("RENDER LAYERS")
                                .size(10.0)
                                .strong()
                                .color(TEXT_MUTED),
                        );
                        let mut changed = false;
                        changed |= ui
                            .checkbox(&mut state.overlays.show_links, "Arm Links (Convex)")
                            .changed();
                        changed |= ui
                            .checkbox(
                                &mut state.overlays.show_tool_convex,
                                "Attached Tools (Tier-2)",
                            )
                            .changed();
                        changed |= ui
                            .checkbox(
                                &mut state.overlays.show_tier1_spheres,
                                "Tier-1 Spheres (SoA)",
                            )
                            .changed();
                        changed |= ui
                            .checkbox(
                                &mut state.overlays.show_esdf_slice,
                                "3D ESDF Slice Heatmap",
                            )
                            .changed();

                        if state.overlays.show_esdf_slice {
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new("Axis:").size(10.5).color(TEXT_MUTED));
                                for (ax_idx, ax_lbl) in [(0_u8, "X"), (1, "Y"), (2, "Z")] {
                                    if ui
                                        .selectable_label(
                                            state.overlays.esdf_slice_axis == ax_idx,
                                            ax_lbl,
                                        )
                                        .clicked()
                                    {
                                        state.overlays.esdf_slice_axis = ax_idx;
                                        state.clamp_esdf_slice_coord();
                                        changed = true;
                                    }
                                }
                            });
                            let (lo, hi) = state.esdf_axis_bounds();
                            changed |= ui
                                .add(
                                    egui::Slider::new(
                                        &mut state.overlays.esdf_slice_coord,
                                        lo..=hi,
                                    )
                                    .step_by(0.025)
                                    .text("Slice (m)"),
                                )
                                .changed();
                            changed |= ui
                                .add(
                                    egui::Slider::new(
                                        &mut state.overlays.esdf_opacity,
                                        0.1..=0.95,
                                    )
                                    .step_by(0.05)
                                    .text("Opacity"),
                                )
                                .changed();
                        }

                        changed |= ui
                            .checkbox(
                                &mut state.overlays.show_mppi_ribbons,
                                "MPPI Rollouts & Trails",
                            )
                            .changed();
                        changed |= ui
                            .checkbox(&mut state.overlays.show_witnesses, "Distance Witnesses")
                            .changed();
                        if state.overlays.show_witnesses {
                            let mut max_w = state.overlays.max_witnesses as i32;
                            if ui
                                .add(egui::Slider::new(&mut max_w, 1..=16).text("Max Witnesses"))
                                .changed()
                            {
                                state.overlays.max_witnesses = max_w as usize;
                                changed = true;
                            }
                        }
                        changed |= ui
                            .checkbox(
                                &mut state.overlays.witnesses_xray,
                                "X-Ray Witness Lines",
                            )
                            .changed();
                        changed |= ui
                            .checkbox(&mut state.overlays.show_tf_frames, "TCP / TF Frames")
                            .changed();
                        changed |= ui
                            .checkbox(
                                &mut state.overlays.show_target_ghosts,
                                "Target & Standoff Ghosts",
                            )
                            .changed();
                        changed |= ui
                            .checkbox(&mut state.overlays.show_floor_grid, "Floor Grid")
                            .changed();

                        if changed {
                            state.refresh_pose_and_scene();
                        }
                    });
            });
    }

    // Bottom-Left Dynamic N-Arm Legend & Controls Hint
    let bottom_left_offset = egui::vec2(
        rect.min.x - screen.min.x + 10.0,
        rect.max.y - screen.max.y - 10.0,
    );
    egui::Area::new(egui::Id::unique("vp_hud_bottom_left"))
        .order(egui::Order::Middle)
        .constrain_to(rect)
        .anchor(egui::Align2::LEFT_BOTTOM, bottom_left_offset)
        .show(ui.ctx(), |ui| {
            hud_frame.show(ui, |ui| {
                ui.horizontal(|ui| {
                    for arm in &state.workcell.arms {
                        let (dot_r, _) =
                            ui.allocate_exact_size(egui::vec2(9.0, 9.0), egui::Sense::hover());
                        ui.painter()
                            .circle_filled(dot_r.center(), 4.5, rgba_to_color32(arm.color_rgba));
                        ui.label(egui::RichText::new(&arm.label).size(10.0).color(TEXT_PRIMARY));
                    }
                    ui.label(
                        egui::RichText::new(
                            "LMB Orbit · Shift/RMB Pan · Wheel Zoom · DblClick Pick",
                        )
                        .size(10.0)
                        .color(TEXT_MUTED),
                    );
                });
            });
        });
}
