//! Bottom Dock: transport controls, stage-annotated alert scrubber, and 4 interactive `egui_plot` charts.

use crate::gui::panels::{
    rgba_to_color32, status_color, ACCENT_PRIMARY, ACCENT_STAGE2, BG_CANVAS, BORDER_SUBTLE,
    STATUS_DANGER, STATUS_SAFE, STATUS_WARN, TEXT_MUTED, TEXT_PRIMARY,
};
use crate::gui::state::{StudioState, SPEED_LABELS};
use crate::scene::{PlaybackTrajectory, TrajectoryStage};
use egui_plot::{HLine, HoverPosition, Line, Plot, PlotPoints, VLine};

pub fn show(ui: &mut egui::Ui, state: &mut StudioState) {
    let mut seek_req: Option<f32> = None;

    // Transport bar + scrubber + readouts
    ui.horizontal(|ui| {
        if ui.button("|<").clicked() {
            seek_req = Some(0.0);
        }
        if ui.button("<").clicked() {
            state.step_frame(-1);
        }
        if ui
            .button(if state.playing { "❚❚" } else { "▶" })
            .clicked()
        {
            state.toggle_play();
        }
        if ui.button(">").clicked() {
            state.step_frame(1);
        }
        if ui.button(">|").clicked() {
            seek_req = Some(1.0);
        }

        egui::ComboBox::from_id_salt("timeline_speed_combo")
            .width(64.0)
            .selected_text(SPEED_LABELS[state.speed_index])
            .show_ui(ui, |ui| {
                for (i, lbl) in SPEED_LABELS.iter().enumerate() {
                    ui.selectable_value(&mut state.speed_index, i, *lbl);
                }
            });

        if ui.selectable_label(state.looping, "Loop").clicked() {
            state.looping = !state.looping;
        }

        let readouts_reserved = 320.0;
        let scrub_w = (ui.available_width() - readouts_reserved).max(120.0);
        if let Some(frac) = show_stage_scrubber(ui, state, scrub_w) {
            seek_req = Some(frac);
        }

        let (time_txt, stage_txt, max_vel_txt) = if let Some(traj) = &state.trajectory {
            let dur = traj.metrics.duration_s;
            if let Some(f) = traj.frames.get(state.current_frame_idx) {
                let stage = match f.stage {
                    TrajectoryStage::SingleStage => "Planned",
                    TrajectoryStage::Stage1Coarse => "Stage 1",
                    TrajectoryStage::Stage2Refinement => "Stage 2",
                };
                let max_vel = f.qd.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
                (
                    format!("{:.2}/{:.2}s", f.time, dur),
                    stage,
                    format!("{max_vel:.2} r/s"),
                )
            } else {
                ("0.00/0.00s".to_string(), "Ready", "--".to_string())
            }
        } else {
            ("0.00/0.00s".to_string(), "Ready", "--".to_string())
        };

        let clr_mm = state
            .last_snapshot
            .as_ref()
            .map_or(0.0, |s| s.min_clearance * 1000.0);
        let clr_sev = if clr_mm < 2.0 {
            2
        } else if clr_mm < 15.0 {
            1
        } else {
            0
        };

        ui.label(
            egui::RichText::new(format!("{time_txt} · {stage_txt}"))
                .size(10.0)
                .strong()
                .color(TEXT_PRIMARY),
        );
        ui.label(
            egui::RichText::new(format!("Clr: {clr_mm:.1}mm"))
                .size(10.0)
                .strong()
                .color(status_color(clr_sev)),
        );
        ui.label(
            egui::RichText::new(format!("|dq|: {max_vel_txt}"))
                .size(10.0)
                .color(TEXT_MUTED),
        );

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let toggle_lbl = if state.plots_collapsed {
                "▲ Plots"
            } else {
                "▼ Plots"
            };
            if ui.button(toggle_lbl).clicked() {
                state.plots_collapsed = !state.plots_collapsed;
            }
        });
    });

    if let Some(frac) = seek_req {
        state.seek_fraction(frac);
        return;
    }

    if state.plots_collapsed {
        return;
    }

    let Some(traj) = state.trajectory.as_ref().map(std::sync::Arc::clone) else {
        return;
    };
    if traj.frames.is_empty() {
        return;
    }

    ui.add_space(4.0);
    let plot_h = ui.available_height().max(72.0);
    let play_t = traj
        .frames
        .get(state.current_frame_idx)
        .map_or(0.0, |f| f.time);

    let mut scrubbed_t: Option<f64> = None;
    ui.columns(4, |cols| {
        if let Some(t) = plot_min_clearance(&mut cols[0], &traj, state.spec.rules.default_body_margin, play_t, plot_h) {
            scrubbed_t = Some(t);
        }
        if let Some(t) = plot_tool_clearance(
            &mut cols[1],
            &traj,
            state.spec.rules.tool_tool_margin,
            state.spec.rules.tool_workpiece_margin,
            play_t,
            plot_h,
        ) {
            scrubbed_t = Some(t);
        }
        if let Some(t) = plot_pose_error(&mut cols[2], &traj, play_t, plot_h) {
            scrubbed_t = Some(t);
        }
        if let Some(t) = plot_joint_velocities(&mut cols[3], state, &traj, play_t, plot_h) {
            scrubbed_t = Some(t);
        }
    });

    if let Some(t) = scrubbed_t {
        let dur = traj.metrics.duration_s.max(1e-6);
        state.seek_fraction((t / dur).clamp(0.0, 1.0) as f32);
    }
}

fn show_stage_scrubber(ui: &mut egui::Ui, state: &StudioState, width: f32) -> Option<f32> {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, 22.0), egui::Sense::click_and_drag());
    let p = ui.painter();

    p.rect(
        rect,
        4.0,
        BG_CANVAS,
        egui::Stroke::new(1.0, BORDER_SUBTLE),
        egui::StrokeKind::Inside,
    );

    if let Some(traj) = &state.trajectory {
        let dur = traj.metrics.duration_s.max(1e-6);
        let s1_frac = traj
            .metrics
            .stage1_end_time
            .map_or(0.0_f32, |t| (t / dur).clamp(0.0, 1.0) as f32);

        let inner = rect.shrink2(egui::vec2(2.0, 4.0));
        if s1_frac > 0.001 {
            let split_x = inner.left() + s1_frac * inner.width();
            p.rect_filled(
                egui::Rect::from_min_max(inner.min, egui::pos2(split_x, inner.max.y)),
                2.0,
                egui::Color32::from_rgb(0x10, 0x32, 0x47),
            );
            p.rect_filled(
                egui::Rect::from_min_max(egui::pos2(split_x, inner.min.y), inner.max),
                2.0,
                egui::Color32::from_rgb(0x27, 0x1b, 0x45),
            );
        } else {
            p.rect_filled(inner, 2.0, egui::Color32::from_rgb(0x10, 0x32, 0x47));
        }

        let n = traj.frames.len().max(2);
        for (idx, f) in traj.frames.iter().enumerate() {
            let clr_mm = f.min_clearance * 1000.0;
            if clr_mm < 5.0 {
                let frac = (idx as f32) / ((n - 1) as f32);
                let x = rect.left() + frac * (rect.width() - 2.0);
                let col = if clr_mm < 1.5 {
                    STATUS_DANGER
                } else {
                    STATUS_WARN
                };
                p.line_segment(
                    [egui::pos2(x, rect.top() + 2.0), egui::pos2(x, rect.bottom() - 2.0)],
                    egui::Stroke::new(2.0, col),
                );
            }
        }

        let play_frac = if n > 1 {
            ((state.current_frame_idx as f32) / ((n - 1) as f32)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let thumb_x = rect.left() + play_frac * (rect.width() - 6.0);
        p.rect_filled(
            egui::Rect::from_min_size(
                egui::pos2(thumb_x, rect.top() + 1.0),
                egui::vec2(6.0, rect.height() - 2.0),
            ),
            3.0,
            egui::Color32::from_rgb(0xf8, 0xfa, 0xfc),
        );
    }

    if (resp.clicked() || resp.dragged()) && rect.width() > 1.0 {
        if let Some(pos) = resp.interact_pointer_pos() {
            return Some(((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0));
        }
    }
    None
}

fn configure_plot<'a>(id: &'a str, traj: &PlaybackTrajectory, height: f32) -> Plot<'a> {
    Plot::new(id)
        .height((height - 20.0).max(52.0))
        .include_x(0.0)
        .include_x(traj.metrics.duration_s.max(0.05))
        .allow_drag(false)
        .allow_scroll(false)
        .allow_zoom(false)
        .show_axes([false, true])
        .label_formatter(|pos| match pos {
            HoverPosition::NearDataPoint {
                plot_name,
                position,
                ..
            } if !plot_name.is_empty() => Some(format!(
                "{plot_name}: t={:.2}s, {:.2}",
                position.x, position.y
            )),
            HoverPosition::NearDataPoint { position, .. } | HoverPosition::Elsewhere { position } => {
                Some(format!("t={:.2}s, {:.2}", position.x, position.y))
            }
        })
}

fn plot_min_clearance(
    ui: &mut egui::Ui,
    traj: &PlaybackTrajectory,
    body_margin_m: f64,
    play_t: f64,
    height: f32,
) -> Option<f64> {
    ui.vertical(|ui| {
        ui.label(
            egui::RichText::new("MIN CLEARANCE VS MARGIN (mm)")
                .size(9.5)
                .strong()
                .color(ACCENT_PRIMARY),
        );
        let pts: PlotPoints = traj
            .frames
            .iter()
            .map(|f| [f.time, (f.min_clearance * 1000.0).clamp(-10.0, 250.0)])
            .collect();
        let resp = configure_plot("plot_min_clr", traj, height).show(ui, |plot_ui| {
            plot_ui.hline(
                HLine::new("Margin", body_margin_m * 1000.0)
                    .color(STATUS_WARN)
                    .width(1.2),
            );
            plot_ui.hline(HLine::new("Zero", 0.0).color(STATUS_DANGER).width(1.0));
            if let Some(s1) = traj.metrics.stage1_end_time {
                plot_ui.vline(VLine::new("S1/S2", s1).color(ACCENT_STAGE2).width(1.2));
            }
            plot_ui.line(Line::new("d_min", pts).color(ACCENT_PRIMARY).width(1.8));
            plot_ui.vline(
                VLine::new("Playhead", play_t)
                    .color(egui::Color32::WHITE)
                    .width(1.2),
            );
            if plot_ui.response().clicked() {
                return plot_ui.pointer_coordinate().map(|c| c.x);
            }
            None
        });
        resp.inner
    })
    .inner
}

fn plot_tool_clearance(
    ui: &mut egui::Ui,
    traj: &PlaybackTrajectory,
    tool_tool_m: f64,
    tool_wp_m: f64,
    play_t: f64,
    height: f32,
) -> Option<f64> {
    ui.vertical(|ui| {
        ui.label(
            egui::RichText::new("TOOL-TOOL & WORKPIECE (mm)")
                .size(9.5)
                .strong()
                .color(ACCENT_STAGE2),
        );
        let tt_pts: PlotPoints = traj
            .frames
            .iter()
            .filter_map(|f| f.min_tool_clearance.map(|d| [f.time, (d * 1000.0).clamp(-5.0, 60.0)]))
            .collect();
        let twp_pts: PlotPoints = traj
            .frames
            .iter()
            .filter_map(|f| {
                f.min_tool_workpiece_clearance
                    .map(|d| [f.time, (d * 1000.0).clamp(-5.0, 60.0)])
            })
            .collect();
        let resp = configure_plot("plot_tool_clr", traj, height)
            .include_y(0.0)
            .include_y(50.0)
            .show(ui, |plot_ui| {
                plot_ui.hline(
                    HLine::new("Tool-Tool Margin", tool_tool_m * 1000.0)
                        .color(ACCENT_STAGE2)
                        .width(1.0),
                );
                plot_ui.hline(
                    HLine::new("Tool-WP Margin", tool_wp_m * 1000.0)
                        .color(STATUS_WARN)
                        .width(1.0),
                );
                if let Some(s1) = traj.metrics.stage1_end_time {
                    plot_ui.vline(VLine::new("S1/S2", s1).color(ACCENT_STAGE2).width(1.2));
                }
                plot_ui.line(Line::new("Tool-Tool", tt_pts).color(ACCENT_STAGE2).width(1.8));
                plot_ui.line(Line::new("Tool-WP", twp_pts).color(STATUS_SAFE).width(1.8));
                plot_ui.vline(
                    VLine::new("Playhead", play_t)
                        .color(egui::Color32::WHITE)
                        .width(1.2),
                );
                if plot_ui.response().clicked() {
                    return plot_ui.pointer_coordinate().map(|c| c.x);
                }
                None
            });
        resp.inner
    })
    .inner
}

fn plot_pose_error(
    ui: &mut egui::Ui,
    traj: &PlaybackTrajectory,
    play_t: f64,
    height: f32,
) -> Option<f64> {
    ui.vertical(|ui| {
        ui.label(
            egui::RichText::new("POSE / GRASP ERROR (mm)")
                .size(9.5)
                .strong()
                .color(STATUS_SAFE),
        );
        let pts: PlotPoints = traj
            .frames
            .iter()
            .map(|f| [f.time, f.pose_error_mm.max(f.grasp_error_mm)])
            .collect();
        let resp = configure_plot("plot_pose_err", traj, height)
            .include_y(0.0)
            .show(ui, |plot_ui| {
                plot_ui.hline(HLine::new("1.8 mm Tol", 1.8).color(STATUS_SAFE).width(1.0));
                if let Some(s1) = traj.metrics.stage1_end_time {
                    plot_ui.vline(VLine::new("S1/S2", s1).color(ACCENT_STAGE2).width(1.2));
                }
                plot_ui.line(Line::new("Error", pts).color(STATUS_SAFE).width(1.8));
                plot_ui.vline(
                    VLine::new("Playhead", play_t)
                        .color(egui::Color32::WHITE)
                        .width(1.2),
                );
                if plot_ui.response().clicked() {
                    return plot_ui.pointer_coordinate().map(|c| c.x);
                }
                None
            });
        resp.inner
    })
    .inner
}

fn plot_joint_velocities(
    ui: &mut egui::Ui,
    state: &StudioState,
    traj: &PlaybackTrajectory,
    play_t: f64,
    height: f32,
) -> Option<f64> {
    ui.vertical(|ui| {
        ui.label(
            egui::RichText::new("PER-ARM MAX |dq/dt| (rad/s)")
                .size(9.5)
                .strong()
                .color(STATUS_WARN),
        );
        let resp = configure_plot("plot_joint_vel", traj, height)
            .include_y(0.0)
            .show(ui, |plot_ui| {
                if let Some(s1) = traj.metrics.stage1_end_time {
                    plot_ui.vline(VLine::new("S1/S2", s1).color(ACCENT_STAGE2).width(1.2));
                }
                for arm in &state.workcell.arms {
                    let pts: PlotPoints = traj
                        .frames
                        .iter()
                        .map(|f| {
                            let peak = f
                                .qd
                                .get(arm.q_range.clone())
                                .map_or(0.0, |s| s.iter().map(|v| v.abs()).fold(0.0, f64::max));
                            [f.time, peak]
                        })
                        .collect();
                    plot_ui.line(
                        Line::new(&arm.label, pts)
                            .color(rgba_to_color32(arm.color_rgba))
                            .width(1.6),
                    );
                }
                plot_ui.vline(
                    VLine::new("Playhead", play_t)
                        .color(egui::Color32::WHITE)
                        .width(1.2),
                );
                if plot_ui.response().clicked() {
                    return plot_ui.pointer_coordinate().map(|c| c.x);
                }
                None
            });
        resp.inner
    })
    .inner
}
