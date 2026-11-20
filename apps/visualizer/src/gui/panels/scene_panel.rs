//! Left Dock: N-Arm Scene Tree hierarchy and Arm / Attached Tool / Obstacle Inspector.

use crate::gui::panels::{
    rgba_to_color32, section_card, ACCENT_PRIMARY, ACCENT_STAGE2, BG_CANVAS, BG_SURFACE,
    BORDER_SUBTLE, STATUS_WARN, TEXT_MUTED, TEXT_PRIMARY,
};
use crate::gui::state::{PrimitiveKind, SceneTab, StudioState};
use crate::scene::{DirtyFlags, ObstacleRole, ToolPreset, UrdfSource};

pub fn show(ui: &mut egui::Ui, state: &mut StudioState) {
    // Summary bar
    egui::Frame::new()
        .fill(BG_SURFACE)
        .stroke(egui::Stroke::new(1.0, BORDER_SUBTLE))
        .corner_radius(egui::CornerRadius::same(4))
        .inner_margin(egui::Margin::symmetric(8, 5))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(
                egui::RichText::new(format!(
                    "Arms: {} · DOF: {} · Tools: {}",
                    state.workcell.arms.len(),
                    state.workcell.model.dof(),
                    state.workcell.env.attached_payloads().len(),
                ))
                .size(11.0)
                .strong()
                .color(TEXT_PRIMARY),
            );
        });

    ui.add_space(4.0);
    ui.horizontal(|ui| {
        if ui
            .selectable_label(state.scene_tab == SceneTab::Tree, "Scene Tree")
            .clicked()
        {
            state.scene_tab = SceneTab::Tree;
        }
        if ui
            .selectable_label(state.scene_tab == SceneTab::Inspector, "Inspector")
            .clicked()
        {
            state.scene_tab = SceneTab::Inspector;
        }
    });
    ui.separator();

    match state.scene_tab {
        SceneTab::Tree => show_scene_tree(ui, state),
        SceneTab::Inspector => show_inspector(ui, state),
    }
}

fn show_scene_tree(ui: &mut egui::Ui, state: &mut StudioState) {
    ui.columns(2, |cols| {
        if cols[0].button("+ 6-DOF Arm").clicked() {
            state.add_arm_preset(false);
        }
        if cols[1].button("+ 4-DOF Arm").clicked() {
            state.add_arm_preset(true);
        }
    });
    ui.columns(2, |cols| {
        if cols[0].button("+ Sphere Obs").clicked() {
            state.add_obstacle(PrimitiveKind::Sphere, false);
        }
        if cols[1].button("+ Workpiece Box").clicked() {
            state.add_workpiece_box();
        }
    });
    ui.add_space(4.0);

    let scroll_h = (ui.available_height() - 32.0).max(80.0);
    egui::ScrollArea::vertical()
        .max_height(scroll_h)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let robots_open = state.expanded_keys.contains("group:robots");
            if tree_row(
                ui,
                "group:robots",
                0,
                !state.workcell.arms.is_empty(),
                robots_open,
                ACCENT_PRIMARY,
                &format!("Robots ({})", state.workcell.arms.len()),
                &format!("{} DOF", state.workcell.model.dof()),
                state.selected_tree_key == "group:robots",
            ) == RowClick::ToggleExpand
            {
                toggle_key(&mut state.expanded_keys, "group:robots");
            }

            if robots_open {
                let arm_count = state.workcell.arms.len();
                for arm_idx in 0..arm_count {
                    let arm_key = format!("arm:{arm_idx}");
                    let arm_open = state.expanded_keys.contains(&arm_key);
                    let (arm_label, arm_dof, arm_color, link_ids, payload_name) = {
                        let arm = &state.workcell.arms[arm_idx];
                        (
                            arm.label.clone(),
                            arm.q_range.len(),
                            rgba_to_color32(arm.color_rgba),
                            arm.handle.links.clone(),
                            arm.payload_name.clone(),
                        )
                    };
                    let act = tree_row(
                        ui,
                        &arm_key,
                        1,
                        true,
                        arm_open,
                        arm_color,
                        &arm_label,
                        &format!("{arm_dof} DOF"),
                        state.selected_tree_key == arm_key,
                    );
                    match act {
                        RowClick::ToggleExpand => toggle_key(&mut state.expanded_keys, &arm_key),
                        RowClick::Select => {
                            state.selected_arm_idx = arm_idx;
                            state.selected_tree_key = arm_key.clone();
                            state.selected_link_id = None;
                            state.selected_payload_name = None;
                            state.selected_obstacle_idx = None;
                            state.selected_info_summary = format!("{arm_label} [{arm_dof} DOF]");
                            state.refresh_pose_and_scene();
                        }
                        RowClick::None => {}
                    }

                    if arm_open {
                        for lid in link_ids {
                            let link_name = state.workcell.model.link(lid).name.clone();
                            let sph_cnt = state
                                .workcell
                                .env
                                .link_geometry(lid)
                                .map_or(0, |g| g.local_spheres.len());
                            let link_key = format!("link:{}", lid.0);
                            if tree_row(
                                ui,
                                &link_key,
                                2,
                                false,
                                false,
                                arm_color,
                                &link_name,
                                &format!("{sph_cnt} sph"),
                                state.selected_link_id == Some(lid),
                            ) == RowClick::Select
                            {
                                state.selected_arm_idx = arm_idx;
                                state.selected_link_id = Some(lid);
                                state.selected_payload_name = None;
                                state.selected_obstacle_idx = None;
                                state.selected_tree_key = link_key;
                                state.selected_info_summary = format!("{link_name} [{sph_cnt} sph]");
                                state.refresh_pose_and_scene();
                            }
                        }

                        if let Some(pname) = payload_name {
                            let tool_key = format!("tool:{arm_idx}");
                            if tree_row(
                                ui,
                                &tool_key,
                                2,
                                false,
                                false,
                                ACCENT_STAGE2,
                                &pname,
                                "TOOL",
                                state.selected_payload_name.as_deref() == Some(pname.as_str()),
                            ) == RowClick::Select
                            {
                                state.selected_arm_idx = arm_idx;
                                state.selected_link_id = None;
                                state.selected_payload_name = Some(pname.clone());
                                state.selected_obstacle_idx = None;
                                state.selected_tree_key = tool_key;
                                state.selected_info_summary = format!("{pname} [TOOL]");
                                state.refresh_pose_and_scene();
                            }
                        }
                    }
                }
            }

            let obs_open = state.expanded_keys.contains("group:obstacles");
            if tree_row(
                ui,
                "group:obstacles",
                0,
                !state.workcell.obstacles.is_empty(),
                obs_open,
                STATUS_WARN,
                &format!("Obstacles ({})", state.workcell.obstacles.len()),
                "",
                state.selected_tree_key == "group:obstacles",
            ) == RowClick::ToggleExpand
            {
                toggle_key(&mut state.expanded_keys, "group:obstacles");
            }

            if obs_open {
                let obs_count = state.workcell.obstacles.len();
                for idx in 0..obs_count {
                    let (obs_label, badge, col) = {
                        let o = &state.workcell.obstacles[idx];
                        let badge = match o.role {
                            ObstacleRole::Workpiece => "WP",
                            ObstacleRole::Pedestal => "FIX",
                            ObstacleRole::DynamicSweep { .. } => "DYN",
                            ObstacleRole::StaticObstacle => "OBS",
                        };
                        (o.label.clone(), badge, rgba_to_color32(o.color_rgba))
                    };
                    let obs_key = format!("obs:{idx}");
                    if tree_row(
                        ui,
                        &obs_key,
                        1,
                        false,
                        false,
                        col,
                        &obs_label,
                        badge,
                        state.selected_obstacle_idx == Some(idx),
                    ) == RowClick::Select
                    {
                        state.selected_link_id = None;
                        state.selected_payload_name = None;
                        state.selected_obstacle_idx = Some(idx);
                        state.selected_tree_key = obs_key;
                        state.selected_info_summary = format!("{obs_label} [{badge}]");
                        state.refresh_pose_and_scene();
                    }
                }
            }
        });

    ui.add_space(4.0);
    if ui.button("Arrange Arms on Ring (0.56 m)").clicked() {
        state.arrange_arms_ring(0.56);
    }
}

fn show_inspector(ui: &mut egui::Ui, state: &mut StudioState) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let arm_count = state.spec.arms.len();
            if arm_count > 0 {
                let arm_idx = state.selected_arm_idx.min(arm_count - 1);
                let (arm_name, src_lbl, dof, link_cnt, mut bx, mut by, mut yaw_deg, tool_cur, default_tcp_z) = {
                    let arm_s = &state.spec.arms[arm_idx];
                    let arm_c = &state.workcell.arms[arm_idx];
                    let src = match &arm_s.urdf_source {
                        UrdfSource::Builtin4Dof => "Built-in 4-DOF",
                        UrdfSource::Builtin6Dof => "Built-in 6-DOF",
                        UrdfSource::InlineXml { .. } => "Imported XML",
                        UrdfSource::FilePath { .. } => "URDF File",
                    };
                    (
                        arm_s.label.clone(),
                        src,
                        arm_c.q_range.len(),
                        arm_c.handle.links.len(),
                        arm_s.base_pose.xyz[0] as f32,
                        arm_s.base_pose.xyz[1] as f32,
                        arm_s.base_pose.rpy[2].to_degrees() as f32,
                        arm_s.tool.clone(),
                        arm_s.default_tcp_z,
                    )
                };

                section_card(ui, &format!("SELECTED ARM · {arm_name}"), |ui| {
                    ui.label(
                        egui::RichText::new(format!("{src_lbl} · {dof} DOF · {link_cnt} links"))
                            .size(10.5)
                            .color(TEXT_MUTED),
                    );
                    let mut pose_changed = false;
                    pose_changed |= labeled_slider(ui, "Base X", &mut bx, -1.5..=1.5, 0.02, " m");
                    pose_changed |= labeled_slider(ui, "Base Y", &mut by, -1.5..=1.5, 0.02, " m");
                    pose_changed |=
                        labeled_slider(ui, "Base Yaw", &mut yaw_deg, -180.0..=180.0, 5.0, " deg");
                    if pose_changed {
                        let arm = &mut state.spec.arms[arm_idx];
                        arm.base_pose.xyz[0] = bx as f64;
                        arm.base_pose.xyz[1] = by as f64;
                        arm.base_pose.rpy[2] = (yaw_deg as f64).to_radians();
                        state.schedule_debounced(DirtyFlags::TOPO);
                    }
                    ui.horizontal(|ui| {
                        if ui.button("Face Origin").clicked() {
                            state.face_arm_origin(arm_idx);
                        }
                        if ui
                            .add_enabled(arm_count > 1, egui::Button::new("Remove Arm"))
                            .clicked()
                        {
                            state.remove_arm(arm_idx);
                        }
                    });
                });

                ui.add_space(6.0);

                // Attached tool card
                let mut tool_idx = match &tool_cur {
                    ToolPreset::None => 0,
                    ToolPreset::ParallelGripper => 1,
                    ToolPreset::Screwdriver => 2,
                    ToolPreset::InspectionProbe => 3,
                    ToolPreset::Custom(_) => 4,
                };
                let prev_tool_idx = tool_idx;
                let tool_names = [
                    "None (Detached)",
                    "Parallel Gripper",
                    "Precision Screwdriver",
                    "Inspection Probe",
                    "Custom Tool",
                ];

                let payload_info = state.workcell.arms.get(arm_idx).and_then(|arm_c| {
                    let pid = arm_c.payload_id?;
                    let pay = state.workcell.env.attached_payload(pid)?;
                    let touch_list: Vec<(String, bool)> = arm_c
                        .handle
                        .links
                        .iter()
                        .map(|&lid| {
                            (
                                state.workcell.model.link(lid).name.clone(),
                                pay.touch_links.contains(&lid),
                            )
                        })
                        .collect();
                    Some((
                        pay.id.clone(),
                        (pay.safety_margin.unwrap_or(state.spec.rules.tool_tool_margin) * 1000.0)
                            as f32,
                        (pay.activation_distance
                            .unwrap_or(state.spec.rules.tool_tool_activation)
                            * 1000.0) as f32,
                        touch_list,
                    ))
                });

                let mut toggle_link_action: Option<(String, bool)> = None;
                let mut tool_params_changed: Option<(f32, f32)> = None;

                section_card(ui, "ATTACHED TOOL (PAYLOAD)", |ui| {
                    egui::ComboBox::from_id_salt("inspector_tool_preset")
                        .width(ui.available_width())
                        .selected_text(tool_names[tool_idx])
                        .show_ui(ui, |ui| {
                            for (i, label) in tool_names.iter().enumerate().take(4) {
                                ui.selectable_value(&mut tool_idx, i, *label);
                            }
                        });

                    if let Some((t_name, mut margin_mm, mut act_mm, touch_list)) = payload_info {
                        ui.label(
                            egui::RichText::new(format!(
                                "Tool: {t_name} · Margin: {:.1} mm",
                                margin_mm
                            ))
                            .size(10.5)
                            .color(TEXT_MUTED),
                        );
                        let mut changed = false;
                        changed |=
                            labeled_slider(ui, "Tool Safety Margin", &mut margin_mm, 0.0..=30.0, 0.5, " mm");
                        changed |=
                            labeled_slider(ui, "Tool CBF Activation", &mut act_mm, 0.0..=80.0, 1.0, " mm");
                        if changed {
                            tool_params_changed = Some((margin_mm, act_mm));
                        }

                        if !touch_list.is_empty() {
                            ui.add_space(2.0);
                            ui.label(
                                egui::RichText::new("TOUCH LINKS (SELF-EXEMPT)")
                                    .size(9.5)
                                    .strong()
                                    .color(TEXT_MUTED),
                            );
                            for (link_name, mut exempt) in touch_list {
                                if ui.checkbox(&mut exempt, &link_name).changed() {
                                    toggle_link_action = Some((link_name, exempt));
                                }
                            }
                        }
                    } else {
                        ui.label(egui::RichText::new("No attached payload").color(TEXT_MUTED));
                    }
                });

                if tool_idx != prev_tool_idx {
                    if let Some(arm) = state.spec.arms.get_mut(arm_idx) {
                        arm.tool = match tool_idx {
                            0 => ToolPreset::None,
                            1 => ToolPreset::ParallelGripper,
                            2 => ToolPreset::Screwdriver,
                            _ => ToolPreset::InspectionProbe,
                        };
                        state.dispatch_compile_and_plan(DirtyFlags::TOPO);
                    }
                } else if let Some((m_mm, a_mm)) = tool_params_changed {
                    if let Some(arm) = state.spec.arms.get_mut(arm_idx) {
                        if let Some(mut custom) = arm.tool.expand(&arm.uid, default_tcp_z) {
                            custom.safety_margin = Some((m_mm.max(0.0) as f64) * 0.001);
                            custom.activation_distance = Some((a_mm.max(0.0) as f64) * 0.001);
                            arm.tool = ToolPreset::Custom(Box::new(custom));
                            state.schedule_debounced(DirtyFlags::TOPO);
                        }
                    }
                } else if let Some((link_name, exempt)) = toggle_link_action {
                    state.toggle_tool_touch_link(arm_idx, &link_name, exempt);
                }
            }

            if let Some(obs_idx) = state
                .selected_obstacle_idx
                .filter(|&i| i < state.spec.obstacles.len())
            {
                ui.add_space(6.0);
                let obs_label = state.spec.obstacles[obs_idx].label.clone();
                let mut remove_clicked = false;
                let mut obs_changed = false;

                section_card(ui, &format!("OBSTACLE ({obs_label})"), |ui| {
                    let obs = &mut state.spec.obstacles[obs_idx];
                    let mut is_wp = matches!(obs.role, ObstacleRole::Workpiece);
                    if ui.checkbox(&mut is_wp, "Workpiece Role (WP)").changed() {
                        obs.role = if is_wp {
                            ObstacleRole::Workpiece
                        } else {
                            ObstacleRole::StaticObstacle
                        };
                        obs_changed = true;
                    }
                    if ui
                        .checkbox(&mut obs.carve_esdf, "Carve Into ESDF Grid")
                        .changed()
                    {
                        obs_changed = true;
                    }
                    let mut x = obs.pose.xyz[0] as f32;
                    let mut y = obs.pose.xyz[1] as f32;
                    let mut z = obs.pose.xyz[2] as f32;
                    let mut m_mm =
                        (obs.safety_margin.unwrap_or(state.spec.rules.default_body_margin) * 1000.0)
                            as f32;
                    let mut a_mm = (obs
                        .activation_distance
                        .unwrap_or(state.spec.rules.default_body_activation)
                        * 1000.0) as f32;
                    obs_changed |= labeled_slider(ui, "Pos X", &mut x, -1.0..=1.0, 0.01, " m");
                    obs_changed |= labeled_slider(ui, "Pos Y", &mut y, -1.0..=1.0, 0.01, " m");
                    obs_changed |= labeled_slider(ui, "Pos Z", &mut z, 0.0..=1.4, 0.01, " m");
                    obs_changed |=
                        labeled_slider(ui, "Obstacle Margin", &mut m_mm, 0.0..=50.0, 1.0, " mm");
                    obs_changed |=
                        labeled_slider(ui, "CBF Activation", &mut a_mm, 0.0..=150.0, 2.0, " mm");
                    if obs_changed {
                        obs.pose.xyz = [x as f64, y as f64, z as f64];
                        obs.safety_margin = Some((m_mm.max(0.0) as f64) * 0.001);
                        obs.activation_distance = Some((a_mm.max(0.0) as f64) * 0.001);
                    }
                    if ui.button("Remove Obstacle").clicked() {
                        remove_clicked = true;
                    }
                });

                if remove_clicked {
                    state.remove_obstacle(obs_idx);
                } else if obs_changed {
                    state.schedule_debounced(DirtyFlags::OBSTACLES);
                }
            }

            ui.add_space(6.0);
            section_card(ui, "DETAILS", |ui| {
                ui.label(
                    egui::RichText::new(&state.selected_info_summary)
                        .size(11.0)
                        .color(TEXT_PRIMARY),
                );
            });
        });
}

#[derive(PartialEq, Eq)]
enum RowClick {
    None,
    Select,
    ToggleExpand,
}

#[allow(clippy::too_many_arguments)]
fn tree_row(
    ui: &mut egui::Ui,
    id_key: &str,
    depth: usize,
    has_children: bool,
    expanded: bool,
    accent: egui::Color32,
    label: &str,
    badge: &str,
    selected: bool,
) -> RowClick {
    let mut action = RowClick::None;
    let fill = if selected {
        egui::Color32::from_rgb(0x17, 0x34, 0x4d)
    } else {
        egui::Color32::TRANSPARENT
    };
    let stroke = if selected {
        egui::Stroke::new(1.0, ACCENT_PRIMARY)
    } else {
        egui::Stroke::NONE
    };
    egui::Frame::new()
        .fill(fill)
        .stroke(stroke)
        .corner_radius(egui::CornerRadius::same(4))
        .inner_margin(egui::Margin::symmetric(4, 2))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.add_space((depth * 12) as f32);
                if has_children {
                    if ui
                        .small_button(if expanded { "▾" } else { "▸" })
                        .clicked()
                    {
                        action = RowClick::ToggleExpand;
                    }
                } else {
                    ui.add_space(18.0);
                }

                let (dot_rect, _) =
                    ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                ui.painter().circle_filled(dot_rect.center(), 4.0, accent);

                let resp = ui.add(
                    egui::Label::new(
                        egui::RichText::new(label).size(11.0).color(TEXT_PRIMARY),
                    )
                    .sense(egui::Sense::click()),
                );
                if resp.clicked() && matches!(action, RowClick::None) {
                    action = if depth == 0 && has_children {
                        RowClick::ToggleExpand
                    } else {
                        RowClick::Select
                    };
                }

                if !badge.is_empty() {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        egui::Frame::new()
                            .fill(BG_CANVAS)
                            .stroke(egui::Stroke::new(1.0, BORDER_SUBTLE))
                            .corner_radius(egui::CornerRadius::same(3))
                            .inner_margin(egui::Margin::symmetric(5, 1))
                            .show(ui, |ui| {
                                ui.label(egui::RichText::new(badge).size(9.0).color(TEXT_MUTED));
                            });
                    });
                }
                let _ = id_key;
            });
        });
    action
}

fn toggle_key(set: &mut std::collections::HashSet<String>, key: &str) {
    if !set.remove(key) {
        set.insert(key.to_string());
    }
}

pub fn labeled_slider(
    ui: &mut egui::Ui,
    label: &str,
    val: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    step: f64,
    suffix: &str,
) -> bool {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).size(10.5).color(TEXT_MUTED));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                egui::RichText::new(format!("{:.2}{suffix}", *val))
                    .size(10.5)
                    .color(TEXT_PRIMARY),
            );
        });
    });
    ui.add(
        egui::Slider::new(val, range)
            .step_by(step)
            .show_value(false),
    )
    .changed()
}
