//! Frozen-header interactive 2D Allowed Collision Matrix (ACM) panel with live `d(t)`
//! readouts, pair filtering, single-click 3D witness ray highlight, and inline margin editor.

use crate::gui::panels::{
    rgba_to_color32, ACCENT_PRIMARY, BG_CANVAS, BG_PANEL, BG_SURFACE, BORDER_SUBTLE,
    STATUS_DANGER, STATUS_WARN, TEXT_MUTED, TEXT_PRIMARY,
};
use crate::gui::state::{AcmRuleKind, AcmSliceMode, StudioState};

const ROW_HDR_W: f32 = 108.0;
const COL_HDR_H: f32 = 26.0;
const CELL_W: f32 = 52.0;
const CELL_H: f32 = 28.0;

pub fn show(ui: &mut egui::Ui, state: &mut StudioState) {
    // Top slice selector & bulk margin buttons
    ui.horizontal_wrapped(|ui| {
        for (mode, label) in [
            (AcmSliceMode::InterArm, "Inter-Arm Body"),
            (AcmSliceMode::SelfCollision, "Self-Collision"),
            (AcmSliceMode::ToolsAndWorkpieces, "Tools & Workpieces"),
            (AcmSliceMode::LinksAndObstacles, "Links x Obstacles"),
        ] {
            if ui.selectable_label(state.acm_slice == mode, label).clicked() {
                state.acm_slice = mode;
            }
        }
        ui.separator();
        if ui.button("Body 40mm").clicked() {
            state.spec.rules.default_body_margin = 0.040;
            state.dispatch_compile_and_plan(crate::scene::DirtyFlags::RULES);
        }
        if ui.button("Tool-Tool 4mm").clicked() {
            state.spec.rules.tool_tool_margin = 0.004;
            state.dispatch_compile_and_plan(crate::scene::DirtyFlags::RULES);
        }
        if ui.button("Tool-WP 3mm").clicked() {
            state.spec.rules.tool_workpiece_margin = 0.003;
            state.dispatch_compile_and_plan(crate::scene::DirtyFlags::RULES);
        }
        if ui.button("Reset SRDF").clicked() {
            state.reset_acm_from_srdf();
        }
    });

    // Filter bar
    ui.horizontal(|ui| {
        ui.add(
            egui::TextEdit::singleline(&mut state.acm_name_filter)
                .hint_text("Filter links / tools / obstacles...")
                .desired_width((ui.available_width() - 180.0).max(120.0)),
        );
        ui.checkbox(&mut state.acm_only_active_or_custom, "Active / Custom Only");
    });

    ui.add_space(2.0);
    let view = state.build_acm_view();

    // Reserve height for bottom pair inspector strip
    let strip_h = 54.0;
    let matrix_h = (ui.available_height() - strip_h - 6.0).max(120.0);

    let mut clicked_pair: Option<(String, String)> = None;
    let mut toggled_pair: Option<(String, String)> = None;

    egui::Frame::new()
        .fill(BG_PANEL)
        .stroke(egui::Stroke::new(1.0, BORDER_SUBTLE))
        .corner_radius(egui::CornerRadius::same(5))
        .inner_margin(egui::Margin::same(2))
        .show(ui, |ui| {
            ui.set_min_height(matrix_h);
            ui.set_max_height(matrix_h);

            let scroll_id = egui::Id::unique("acm_matrix_scroll_offset");
            let saved_offset: egui::Vec2 = ui
                .data_mut(|d| d.get_temp::<egui::Vec2>(scroll_id))
                .unwrap_or(egui::Vec2::ZERO);

            // Top row: frozen corner + frozen column headers
            ui.horizontal(|ui| {
                egui::Frame::new()
                    .fill(BG_SURFACE)
                    .stroke(egui::Stroke::new(1.0, BORDER_SUBTLE))
                    .show(ui, |ui| {
                        ui.set_width(ROW_HDR_W);
                        ui.set_height(COL_HDR_H);
                        ui.centered_and_justified(|ui| {
                            ui.label(
                                egui::RichText::new("PAIR / d(t)")
                                    .size(9.0)
                                    .strong()
                                    .color(TEXT_MUTED),
                            );
                        });
                    });

                egui::ScrollArea::horizontal()
                    .id_salt("acm_col_hdr_scroll")
                    .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                    .scroll_offset(egui::vec2(saved_offset.x, 0.0))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing = egui::vec2(2.0, 2.0);
                            for hdr in &view.col_headers {
                                let (r, _) = ui.allocate_exact_size(
                                    egui::vec2(CELL_W, COL_HDR_H),
                                    egui::Sense::hover(),
                                );
                                ui.painter().rect_filled(r, 3.0, BG_SURFACE);
                                ui.painter()
                                    .text(
                                        r.center(),
                                        egui::Align2::CENTER_CENTER,
                                        &hdr.short_label,
                                        egui::FontId::proportional(10.0),
                                        rgba_to_color32(hdr.accent),
                                    );
                            }
                        });
                    });
            });

            // Bottom row: frozen row headers + scrollable matrix body
            let mut next_offset = saved_offset;
            ui.horizontal(|ui| {
                let body_h = (matrix_h - COL_HDR_H - 8.0).max(60.0);
                ui.allocate_ui(egui::vec2(ROW_HDR_W + 4.0, body_h), |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("acm_row_hdr_scroll")
                        .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                        .scroll_offset(egui::vec2(0.0, saved_offset.y))
                        .show(ui, |ui| {
                            ui.vertical(|ui| {
                                ui.spacing_mut().item_spacing = egui::vec2(2.0, 2.0);
                                for hdr in &view.row_headers {
                                    let (r, _) = ui.allocate_exact_size(
                                        egui::vec2(ROW_HDR_W, CELL_H),
                                        egui::Sense::hover(),
                                    );
                                    ui.painter().rect_filled(r, 3.0, BG_SURFACE);
                                    ui.painter().text(
                                        egui::pos2(r.left() + 6.0, r.center().y),
                                        egui::Align2::LEFT_CENTER,
                                        &hdr.short_label,
                                        egui::FontId::proportional(10.0),
                                        rgba_to_color32(hdr.accent),
                                    );
                                }
                            });
                        });
                });

                let body_out = egui::ScrollArea::both()
                    .id_salt("acm_body_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.vertical(|ui| {
                            ui.spacing_mut().item_spacing = egui::vec2(2.0, 2.0);
                            for row_cells in &view.cells {
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing = egui::vec2(2.0, 2.0);
                                    for cell in row_cells {
                                        let (rect, resp) = ui.allocate_exact_size(
                                            egui::vec2(CELL_W, CELL_H),
                                            egui::Sense::click(),
                                        );
                                        let bg = match cell.rule {
                                            AcmRuleKind::Diagonal => {
                                                egui::Color32::from_rgb(0x0f, 0x13, 0x1a)
                                            }
                                            AcmRuleKind::CustomMargin if cell.margin_mm < 10.0 => {
                                                egui::Color32::from_rgb(0x2e, 0x1f, 0x47)
                                            }
                                            AcmRuleKind::CustomMargin => {
                                                egui::Color32::from_rgb(0x3d, 0x2b, 0x12)
                                            }
                                            AcmRuleKind::Enabled => {
                                                egui::Color32::from_rgb(0x15, 0x30, 0x42)
                                            }
                                            AcmRuleKind::DisabledDynamic => {
                                                egui::Color32::from_rgb(0x38, 0x19, 0x24)
                                            }
                                            _ => egui::Color32::from_rgb(0x16, 0x1b, 0x22),
                                        };
                                        let (stroke_w, stroke_col) = if cell.near_violation {
                                            (2.0, STATUS_DANGER)
                                        } else if cell.selected {
                                            (2.0, ACCENT_PRIMARY)
                                        } else {
                                            (1.0, BORDER_SUBTLE)
                                        };
                                        ui.painter().rect(
                                            rect,
                                            3.0,
                                            bg,
                                            egui::Stroke::new(stroke_w, stroke_col),
                                            egui::StrokeKind::Inside,
                                        );

                                        let fg = match cell.rule {
                                            AcmRuleKind::Diagonal => {
                                                egui::Color32::from_rgb(0x30, 0x38, 0x46)
                                            }
                                            AcmRuleKind::CustomMargin if cell.margin_mm < 10.0 => {
                                                egui::Color32::from_rgb(0xd8, 0xb4, 0xfe)
                                            }
                                            AcmRuleKind::CustomMargin => {
                                                egui::Color32::from_rgb(0xfc, 0xd3, 0x4d)
                                            }
                                            AcmRuleKind::Enabled => {
                                                egui::Color32::from_rgb(0x7d, 0xd3, 0xfc)
                                            }
                                            _ => TEXT_MUTED,
                                        };
                                        let has_dist = cell.rule != AcmRuleKind::Diagonal
                                            && cell.live_dist_mm > -900.0;
                                        let top_y = if has_dist {
                                            rect.center().y - 5.0
                                        } else {
                                            rect.center().y
                                        };
                                        ui.painter().text(
                                            egui::pos2(rect.center().x, top_y),
                                            egui::Align2::CENTER_CENTER,
                                            &cell.label,
                                            egui::FontId::proportional(9.0),
                                            fg,
                                        );
                                        if has_dist {
                                            ui.painter().text(
                                                egui::pos2(rect.center().x, rect.center().y + 6.0),
                                                egui::Align2::CENTER_CENTER,
                                                format!("{}mm", cell.live_dist_mm.round() as i32),
                                                egui::FontId::proportional(8.0),
                                                if cell.near_violation {
                                                    STATUS_DANGER
                                                } else {
                                                    TEXT_MUTED
                                                },
                                            );
                                        }

                                        if resp.double_clicked() {
                                            toggled_pair =
                                                Some((cell.row_key.clone(), cell.col_key.clone()));
                                        } else if resp.clicked() {
                                            clicked_pair =
                                                Some((cell.row_key.clone(), cell.col_key.clone()));
                                        }
                                    }
                                });
                            }
                        });
                    });
                next_offset = body_out.state.clamped_offset();
            });

            if (next_offset - saved_offset).length_sq() > 0.25 {
                ui.data_mut(|d| d.insert_temp(scroll_id, next_offset));
                ui.ctx().request_repaint();
            }
        });

    if let Some((rk, ck)) = toggled_pair {
        state.select_acm_pair(rk.clone(), ck.clone());
        state.toggle_acm_cell_quick(&rk, &ck);
    } else if let Some((rk, ck)) = clicked_pair {
        state.select_acm_pair(rk, ck);
    }

    ui.add_space(4.0);

    struct PendingPairRule {
        rk: String,
        ck: String,
        enabled: bool,
        margin: Option<f64>,
        activation: Option<f64>,
    }

    // Bottom Pinned Pair Inspector Strip
    let mut apply_rule: Option<PendingPairRule> = None;
    let mut toggle_selected: Option<(String, String)> = None;

    egui::Frame::new()
        .fill(BG_CANVAS)
        .stroke(egui::Stroke::new(1.0, BORDER_SUBTLE))
        .corner_radius(egui::CornerRadius::same(5))
        .inner_margin(egui::Margin::symmetric(10, 6))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            if let Some(sel) = view.selected_pair.as_ref().filter(|s| s.valid) {
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "{} ({})",
                                sel.pair_title, sel.pair_kind_badge
                            ))
                            .size(11.0)
                            .strong()
                            .color(TEXT_PRIMARY),
                        );
                        let dist_txt = if sel.live_dist_mm > -900.0 {
                            format!("{:.1} mm", sel.live_dist_mm)
                        } else {
                            "Outside witness batch".to_string()
                        };
                        let near = sel.live_dist_mm > -900.0
                            && sel.live_dist_mm <= sel.margin_mm + 5.0;
                        ui.label(
                            egui::RichText::new(format!(
                                "Live d(t): {dist_txt} · Active Margin: {:.1} mm · CBF Act: {:.0} mm",
                                sel.margin_mm, sel.activation_mm
                            ))
                            .size(10.0)
                            .color(if near { STATUS_WARN } else { TEXT_MUTED }),
                        );
                    });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add_enabled(!sel.locked, egui::Button::new("Toggle Active"))
                            .clicked()
                        {
                            toggle_selected = Some((sel.row_key.clone(), sel.col_key.clone()));
                        }
                        if ui
                            .add_enabled(!sel.locked, egui::Button::new("Apply"))
                            .clicked()
                        {
                            apply_rule = Some(PendingPairRule {
                                rk: sel.row_key.clone(),
                                ck: sel.col_key.clone(),
                                enabled: true,
                                margin: Some((state.acm_edit_margin_mm.max(0.0) as f64) * 0.001),
                                activation: Some((state.acm_edit_act_mm.max(0.0) as f64) * 0.001),
                            });
                        }
                        ui.add_enabled(
                            !sel.locked,
                            egui::DragValue::new(&mut state.acm_edit_act_mm)
                                .speed(1.0)
                                .range(0.0..=500.0)
                                .suffix(" mm"),
                        );
                        ui.label(egui::RichText::new("Act:").size(10.0).color(TEXT_MUTED));
                        ui.add_enabled(
                            !sel.locked,
                            egui::DragValue::new(&mut state.acm_edit_margin_mm)
                                .speed(0.5)
                                .range(0.0..=200.0)
                                .suffix(" mm"),
                        );
                        ui.label(egui::RichText::new("Margin:").size(10.0).color(TEXT_MUTED));
                    });
                });
            } else {
                ui.label(
                    egui::RichText::new(
                        "Click any cell to inspect pair & 3D distance ray · Double-click to toggle Enabled/Disabled",
                    )
                    .size(11.0)
                    .color(TEXT_MUTED),
                );
            }
        });

    if let Some(r) = apply_rule {
        state.apply_acm_pair_rule(&r.rk, &r.ck, r.enabled, r.margin, r.activation);
    } else if let Some((rk, ck)) = toggle_selected {
        state.toggle_acm_cell_quick(&rk, &ck);
    }
}
