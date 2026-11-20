//! Right Dock: Plan & CBF parameters, Targets & IK Preview, Live Joint Jogging, and pinned Execution Footer.

use crate::gui::panels::scene_panel::labeled_slider;
use crate::gui::panels::{
    rgba_to_color32, section_card, ACCENT_STAGE2, BG_CANVAS, BG_SURFACE, BORDER_SUBTLE,
    STATUS_SAFE, TEXT_MUTED, TEXT_PRIMARY,
};
use crate::gui::state::{PlannerTab, SolverStatus, StudioState};
use crate::scene::{ArmTargetSpec, DirtyFlags, PlannerSpec};
use unrust_it_io::SubframeRole;

pub fn show(ui: &mut egui::Ui, state: &mut StudioState) {
    ui.horizontal(|ui| {
        if ui
            .selectable_label(state.planner_tab == PlannerTab::PlanCbf, "Plan & CBF")
            .clicked()
        {
            state.planner_tab = PlannerTab::PlanCbf;
        }
        if ui
            .selectable_label(state.planner_tab == PlannerTab::TargetsIk, "Targets & IK")
            .clicked()
        {
            state.planner_tab = PlannerTab::TargetsIk;
        }
        if ui
            .selectable_label(state.planner_tab == PlannerTab::JointsJog, "Joints / Jog")
            .clicked()
        {
            state.planner_tab = PlannerTab::JointsJog;
        }
    });
    ui.separator();

    let footer_h = 154.0;
    let body_h = (ui.available_height() - footer_h - 6.0).max(80.0);

    egui::ScrollArea::vertical()
        .max_height(body_h)
        .auto_shrink([false, false])
        .show(ui, |ui| match state.planner_tab {
            PlannerTab::PlanCbf => show_plan_cbf_tab(ui, state),
            PlannerTab::TargetsIk => show_targets_ik_tab(ui, state),
            PlannerTab::JointsJog => show_joints_jog_tab(ui, state),
        });

    ui.add_space(6.0);
    show_execution_footer(ui, state);
}

fn show_plan_cbf_tab(ui: &mut egui::Ui, state: &mut StudioState) {
    show_imported_trajectories_card(ui, state);
    ui.add_space(6.0);

    let mut prio_changed = false;
    section_card(ui, "ARM PRIORITY WEIGHTS (QP-CBF)", |ui| {
        for arm in &mut state.spec.arms {
            let mut p = arm.priority as f32;
            if labeled_slider(ui, &arm.label, &mut p, 0.2..=6.0, 0.1, "x") {
                arm.priority = p as f64;
                prio_changed = true;
            }
        }
    });
    if prio_changed {
        state.schedule_debounced(DirtyFlags::PLAN);
    }

    ui.add_space(6.0);

    let mut cbf_changed = false;
    let mut margin_mm = (state.spec.rules.default_body_margin * 1000.0) as f32;
    let mut act_mm = (state.spec.rules.default_body_activation * 1000.0) as f32;
    let (mut gamma, mut curl, has_servo) = match &state.spec.planner {
        PlannerSpec::TwoStageProximity { servo, .. }
        | PlannerSpec::ThreeArmCrossing { servo, .. }
        | PlannerSpec::ClosedChainCbfServo { servo, .. } => {
            (servo.cbf_gamma as f32, servo.tangential_curl_gain as f32, true)
        }
        _ => (8.0, 0.35, false),
    };

    section_card(ui, "CBF & DEADLOCK AVOIDANCE", |ui| {
        cbf_changed |=
            labeled_slider(ui, "Default Body Margin", &mut margin_mm, 5.0..=100.0, 1.0, " mm");
        cbf_changed |=
            labeled_slider(ui, "CBF Activation Dist", &mut act_mm, 30.0..=350.0, 5.0, " mm");
        if has_servo {
            cbf_changed |=
                labeled_slider(ui, "CBF Decay (gamma)", &mut gamma, 1.0..=20.0, 0.5, " 1/s");
            cbf_changed |=
                labeled_slider(ui, "Tangential Curl Gain", &mut curl, 0.0..=1.5, 0.05, "");
        }
    });
    if cbf_changed {
        state.spec.rules.default_body_margin = (margin_mm as f64) * 0.001;
        state.spec.rules.default_body_activation = (act_mm as f64) * 0.001;
        match &mut state.spec.planner {
            PlannerSpec::TwoStageProximity { servo, .. }
            | PlannerSpec::ThreeArmCrossing { servo, .. }
            | PlannerSpec::ClosedChainCbfServo { servo, .. } => {
                servo.safety_margin = (margin_mm as f64) * 0.001;
                servo.cbf_activation_distance = (act_mm as f64) * 0.001;
                servo.cbf_gamma = gamma as f64;
                servo.tangential_curl_gain = curl as f64;
            }
            _ => {}
        }
        state.schedule_debounced(DirtyFlags::RULES);
    }

    if let PlannerSpec::TwoStageProximity { mppi, .. } | PlannerSpec::BatchMppiEsdf { mppi, .. } =
        &mut state.spec.planner
    {
        ui.add_space(6.0);
        let mut mppi_changed = false;
        let mut b = mppi.num_rollouts as f32;
        let mut h = mppi.horizon as f32;
        let mut it = mppi.num_iterations as f32;
        section_card(ui, "BATCHED MPPI PARAMETERS", |ui| {
            mppi_changed |= labeled_slider(ui, "Rollouts (B)", &mut b, 16.0..=128.0, 8.0, "");
            mppi_changed |= labeled_slider(ui, "Horizon (H)", &mut h, 8.0..=40.0, 2.0, "");
            mppi_changed |= labeled_slider(ui, "Iterations", &mut it, 5.0..=60.0, 5.0, "");
        });
        if mppi_changed {
            mppi.num_rollouts = (b.round() as usize).max(8);
            mppi.horizon = (h.round() as usize).max(4);
            mppi.num_iterations = (it.round() as usize).max(1);
            state.schedule_debounced(DirtyFlags::PLAN);
        }
    }
}

fn show_targets_ik_tab(ui: &mut egui::Ui, state: &mut StudioState) {
    let target_indices: Vec<usize> = state
        .spec
        .arms
        .iter()
        .enumerate()
        .filter_map(|(i, a)| {
            matches!(
                a.target,
                ArmTargetSpec::ApproachPose { .. } | ArmTargetSpec::DirectPose(_)
            )
            .then_some(i)
        })
        .collect();

    if !target_indices.is_empty() && !target_indices.contains(&state.selected_target_arm) {
        state.selected_target_arm = target_indices[0];
    }

    let mut copy_arm_idx: Option<usize> = None;
    let mut solve_ik_clicked = false;
    let mut pose_edited: Option<(usize, [f64; 3], [f64; 3], f64)> = None;

    section_card(
        ui,
        &format!("ARM TCP TARGETS ({})", target_indices.len()),
        |ui| {
            if target_indices.is_empty() {
                ui.label(
                    egui::RichText::new(
                        "This scenario uses direct joint goals or closed-chain constraints.",
                    )
                    .size(10.5)
                    .color(TEXT_MUTED),
                );
                return;
            }

            ui.horizontal_wrapped(|ui| {
                for &r_idx in &target_indices {
                    if ui
                        .selectable_label(
                            state.selected_target_arm == r_idx,
                            &state.spec.arms[r_idx].label,
                        )
                        .clicked()
                    {
                        state.selected_target_arm = r_idx;
                    }
                }
            });

            let sel = state.selected_target_arm;
            if let (Some(arm_s), Some(arm_c)) =
                (state.spec.arms.get(sel), state.workcell.arms.get(sel))
            {
                let tcp_name = state
                    .workcell
                    .model
                    .frame_graph()
                    .node(arm_c.tcp_frame)
                    .map_or("tcp", |n| n.name.as_str());
                ui.label(
                    egui::RichText::new(format!("Target TCP: {tcp_name}"))
                        .size(11.0)
                        .strong()
                        .color(rgba_to_color32(arm_s.color_rgba)),
                );

                let (pose, mut standoff_mm) = match &arm_s.target {
                    ArmTargetSpec::ApproachPose {
                        final_pose,
                        standoff_local,
                    } => (*final_pose, (standoff_local[2].abs() * 1000.0) as f32),
                    ArmTargetSpec::DirectPose(p) => (*p, 0.0),
                    _ => return,
                };
                let mut x = pose.xyz[0] as f32;
                let mut y = pose.xyz[1] as f32;
                let mut z = pose.xyz[2] as f32;
                let mut r_deg = pose.rpy[0].to_degrees() as f32;
                let mut p_deg = pose.rpy[1].to_degrees() as f32;
                let mut y_deg = pose.rpy[2].to_degrees() as f32;

                let mut chg = false;
                chg |= labeled_slider(ui, "Target X", &mut x, -0.9..=0.9, 0.01, " m");
                chg |= labeled_slider(ui, "Target Y", &mut y, -0.9..=0.9, 0.01, " m");
                chg |= labeled_slider(ui, "Target Z", &mut z, 0.05..=1.25, 0.01, " m");
                chg |= labeled_slider(ui, "Roll", &mut r_deg, -180.0..=180.0, 1.0, " deg");
                chg |= labeled_slider(ui, "Pitch", &mut p_deg, -180.0..=180.0, 1.0, " deg");
                chg |= labeled_slider(ui, "Yaw", &mut y_deg, -180.0..=180.0, 1.0, " deg");
                chg |= labeled_slider(ui, "Standoff", &mut standoff_mm, 0.0..=200.0, 2.0, " mm");

                if chg {
                    pose_edited = Some((
                        sel,
                        [x as f64, y as f64, z as f64],
                        [
                            (r_deg as f64).to_radians(),
                            (p_deg as f64).to_radians(),
                            (y_deg as f64).to_radians(),
                        ],
                        standoff_mm as f64,
                    ));
                }

                if ui.button("Copy Current TCP Pose -> Target").clicked() {
                    copy_arm_idx = Some(sel);
                }
            }

            ui.add_space(4.0);
            if ui.button("Solve IK Preview (Batch 32 Seeds)").clicked() {
                solve_ik_clicked = true;
            }
        },
    );

    if let Some((sel, xyz, rpy, standoff_mm)) = pose_edited {
        if let Some(arm) = state.spec.arms.get_mut(sel) {
            let new_pose = crate::scene::PoseSpec { xyz, rpy };
            match &mut arm.target {
                ArmTargetSpec::ApproachPose {
                    final_pose,
                    standoff_local,
                } => {
                    *final_pose = new_pose;
                    standoff_local[2] = -standoff_mm.max(0.0) * 0.001;
                }
                ArmTargetSpec::DirectPose(p) => {
                    if standoff_mm > 0.5 {
                        arm.target = ArmTargetSpec::ApproachPose {
                            final_pose: new_pose,
                            standoff_local: [0.0, 0.0, -standoff_mm * 0.001],
                        };
                    } else {
                        *p = new_pose;
                    }
                }
                _ => {}
            }
        }
        state.refresh_pose_and_scene();
        state.schedule_debounced(DirtyFlags::PLAN);
    } else if let Some(idx) = copy_arm_idx {
        state.copy_current_tcp_to_target(idx);
    } else if solve_ik_clicked {
        state.preview_ik_targets();
    }
}

fn show_joints_jog_tab(ui: &mut egui::Ui, state: &mut StudioState) {
    ui.checkbox(&mut state.live_jog_mode, "Live FK & Convex Witness Mode");
    ui.separator();

    let limits = state.workcell.model.dof_limits();
    let mut jogged = false;
    for arm in &state.workcell.arms {
        ui.label(
            egui::RichText::new(&arm.label)
                .size(10.5)
                .strong()
                .color(rgba_to_color32(arm.color_rgba)),
        );
        for (local_i, global_q) in arm.q_range.clone().enumerate() {
            if global_q >= state.current_q.len() || global_q >= limits.len() {
                continue;
            }
            let name = arm
                .joint_names
                .get(local_i)
                .map_or("q", String::as_str);
            let lim = limits[global_q];
            let lo = lim.position.min.max(-std::f64::consts::TAU) as f32;
            let hi = lim.position.max.min(std::f64::consts::TAU) as f32;
            let mut val = state.current_q[global_q] as f32;
            if labeled_slider(ui, name, &mut val, lo..=hi, 0.02, " rad") {
                state.current_q[global_q] = val as f64;
                state.playing = false;
                jogged = true;
            }
        }
    }
    if jogged {
        state.refresh_pose_and_scene();
    }
}

fn show_execution_footer(ui: &mut egui::Ui, state: &mut StudioState) {
    let running = matches!(state.solver_status, SolverStatus::Running);
    egui::Frame::new()
        .fill(BG_SURFACE)
        .stroke(egui::Stroke::new(1.0, BORDER_SUBTLE))
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                let run_lbl = if running {
                    "Planning..."
                } else {
                    "▶ Plan / Re-run"
                };
                if ui.button(run_lbl).clicked() {
                    state.dispatch_compile_and_plan(DirtyFlags::RULES);
                }
                if running && ui.button("Stop").clicked() {
                    state.cancel_plan();
                }
            });

            let bar_val = if running {
                state.solver_progress.clamp(0.0, 1.0)
            } else {
                1.0
            };
            ui.add(egui::ProgressBar::new(bar_val).desired_height(6.0));
            ui.label(
                egui::RichText::new(&state.solver_message)
                    .size(10.0)
                    .color(TEXT_MUTED),
            );

            let (solve_txt, err_txt, body_txt, tool_txt) = if let Some(traj) = &state.trajectory {
                (
                    format!("{:.1} ms", traj.metrics.solve_time_ms),
                    format!("{:.2} mm", traj.metrics.max_pos_error_mm),
                    format!("{:.1} mm", traj.metrics.min_clearance_mm),
                    traj.metrics
                        .min_tool_clearance_mm
                        .map_or_else(|| "--".to_string(), |v| format!("{v:.1} mm")),
                )
            } else {
                (
                    "--".to_string(),
                    "--".to_string(),
                    "--".to_string(),
                    "--".to_string(),
                )
            };

            ui.columns(2, |cols| {
                metric_tile(&mut cols[0], "SOLVE", &solve_txt, TEXT_PRIMARY);
                metric_tile(&mut cols[1], "POSE ERR", &err_txt, TEXT_PRIMARY);
            });
            ui.columns(2, |cols| {
                metric_tile(&mut cols[0], "BODY CLR", &body_txt, STATUS_SAFE);
                metric_tile(&mut cols[1], "TOOL CLR", &tool_txt, ACCENT_STAGE2);
            });
        });
}

fn metric_tile(ui: &mut egui::Ui, label: &str, value: &str, val_col: egui::Color32) {
    egui::Frame::new()
        .fill(BG_CANVAS)
        .corner_radius(egui::CornerRadius::same(4))
        .inner_margin(egui::Margin::symmetric(6, 4))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(
                egui::RichText::new(label)
                    .size(8.0)
                    .strong()
                    .color(TEXT_MUTED),
            );
            ui.label(
                egui::RichText::new(value)
                    .size(10.5)
                    .strong()
                    .color(val_col),
            );
        });
}

fn show_imported_trajectories_card(ui: &mut egui::Ui, state: &mut StudioState) {
    enum Action {
        None,
        Import,
        LoadSample,
        ExportSample,
        ReEval,
        Clear,
        SelectNode(Option<String>),
    }

    let mut action = Action::None;

    section_card(ui, "IMPORTED TRAJECTORIES", |ui| {
        if state.imported_trajectory_set.is_none() {
            ui.label(
                egui::RichText::new("Import per-robot ordered frames & subframes (.toml / .json).")
                    .size(10.5)
                    .color(TEXT_MUTED),
            );
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                if ui.button("Import...").clicked() {
                    action = Action::Import;
                }
                if ui.button("Load Sample").clicked() {
                    action = Action::LoadSample;
                }
                if ui.button("Export Sample").clicked() {
                    action = Action::ExportSample;
                }
            });
            return;
        }

        let source_lbl = state
            .imported_source_label
            .clone()
            .unwrap_or_else(|| "imported".to_string());
        let set_name = state
            .imported_trajectory_set
            .as_ref()
            .map_or_else(String::new, |s| s.name.clone());

        ui.horizontal_wrapped(|ui| {
            ui.label(
                egui::RichText::new(format!("{set_name} ({source_lbl})"))
                    .size(11.0)
                    .strong()
                    .color(TEXT_PRIMARY),
            );
        });

        ui.horizontal_wrapped(|ui| {
            if ui
                .checkbox(&mut state.imported_cbf_filter, "QP-CBF Filter")
                .changed()
            {
                action = Action::ReEval;
            }
            if ui.button("Re-Eval").clicked() {
                action = Action::ReEval;
            }
            if ui.button("Import...").clicked() {
                action = Action::Import;
            }
            if ui.button("Clear").clicked() {
                action = Action::Clear;
            }
        });

        ui.add_space(2.0);
        let selected_copy = state.selected_imported_node.clone();
        let resolved_ref = state.imported_resolved.as_ref();

        if let Some(ref mut set) = state.imported_trajectory_set {
            for r_spec in &mut set.robots {
                let r_uid = r_spec.robot_uid.clone();
                let header_txt = format!("Robot: {r_uid} ({} traj)", r_spec.trajectories.len());
                egui::CollapsingHeader::new(header_txt)
                    .id_salt(("imported_robot", &r_uid))
                    .default_open(true)
                    .show(ui, |ui| {
                        let res_robot = resolved_ref
                            .and_then(|res| res.robots.iter().find(|rr| rr.robot_uid == r_uid));

                        for traj_spec in &mut r_spec.trajectories {
                            let label_txt = if traj_spec.label.trim().is_empty() {
                                traj_spec.id.clone()
                            } else {
                                traj_spec.label.clone()
                            };
                            ui.horizontal(|ui| {
                                if ui.checkbox(&mut traj_spec.enabled, "").changed() {
                                    action = Action::ReEval;
                                }
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{label_txt} ({} frames)",
                                        traj_spec.frames.len()
                                    ))
                                    .size(10.5)
                                    .color(if traj_spec.enabled {
                                        TEXT_PRIMARY
                                    } else {
                                        TEXT_MUTED
                                    }),
                                );
                            });

                            if let Some(res_traj) = res_robot.and_then(|rr| {
                                rr.trajectories.iter().find(|rt| rt.id == traj_spec.id)
                            }) {
                                for wp in &res_traj.waypoints {
                                    let indent = "  ".repeat(wp.depth);
                                    let role_tag = match wp.role {
                                        SubframeRole::Approach => "APP",
                                        SubframeRole::Waypoint => "WPT",
                                        SubframeRole::Contact => "CNT",
                                        SubframeRole::Retract => "RET",
                                        SubframeRole::ReferenceOnly => "REF",
                                    };
                                    let key = format!(
                                        "{}/{}/{}",
                                        wp.robot_uid, wp.trajectory_id, wp.node_id
                                    );
                                    let is_sel = selected_copy.as_deref() == Some(key.as_str());
                                    let time_txt = wp
                                        .time_s
                                        .map_or_else(String::new, |t| format!(" @{t:.2}s"));
                                    let row_txt = format!(
                                        "{indent}[{role_tag}] {}{time_txt}",
                                        wp.label
                                    );
                                    let rich = egui::RichText::new(row_txt).size(10.0).color(
                                        if wp.execute_ik {
                                            rgba_to_color32(wp.color_rgba)
                                        } else {
                                            TEXT_MUTED
                                        },
                                    );
                                    if ui.selectable_label(is_sel, rich).clicked() {
                                        action = Action::SelectNode(if is_sel {
                                            None
                                        } else {
                                            Some(key)
                                        });
                                    }
                                }
                            }
                        }
                    });
            }
        }
    });

    match action {
        Action::None => {}
        Action::Import => state.import_trajectories_dialog(),
        Action::LoadSample => state.load_sample_trajectories(),
        Action::ExportSample => state.export_sample_trajectories_dialog(),
        Action::ReEval => state.apply_imported_trajectories(),
        Action::Clear => state.clear_imported_trajectories(),
        Action::SelectNode(node) => state.select_imported_node(node),
    }
}
