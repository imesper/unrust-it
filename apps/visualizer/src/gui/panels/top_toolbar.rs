//! Top toolbar with dock toggles, scenario preset picker, workspace mode switcher, and URDF/export actions.

use crate::gui::panels::{status_badge, ACCENT_PRIMARY, BORDER_SUBTLE};
use crate::gui::state::{CenterViewMode, StudioState};
use crate::scene::PresetId;

pub fn show(ui: &mut egui::Ui, state: &mut StudioState) {
    ui.horizontal(|ui| {
        let left_lbl = if state.left_collapsed { "▸ Tree" } else { "◂" };
        if ui.button(left_lbl).clicked() {
            state.left_collapsed = !state.left_collapsed;
        }

        egui::Frame::new()
            .fill(egui::Color32::from_rgb(0x10, 0x2a, 0x3c))
            .stroke(egui::Stroke::new(1.0, ACCENT_PRIMARY))
            .corner_radius(egui::CornerRadius::same(4))
            .inner_margin(egui::Margin::symmetric(10, 4))
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("unrust-it Studio")
                        .size(12.0)
                        .strong()
                        .color(ACCENT_PRIMARY),
                );
            });

        let mut selected_idx = state.active_preset_index;
        let current_title = PresetId::ALL
            .get(selected_idx)
            .map_or("Preset", PresetId::title);
        egui::ComboBox::from_id_salt("scenario_preset_combo")
            .width(245.0)
            .selected_text(current_title)
            .show_ui(ui, |ui| {
                for (i, preset) in PresetId::ALL.iter().enumerate() {
                    ui.selectable_value(&mut selected_idx, i, preset.title());
                }
            });
        if selected_idx != state.active_preset_index {
            state.load_preset(selected_idx);
        }

        ui.separator();

        if ui
            .selectable_label(state.center_mode == CenterViewMode::Viewport, "3D View")
            .clicked()
        {
            state.center_mode = CenterViewMode::Viewport;
        }
        if ui
            .selectable_label(state.center_mode == CenterViewMode::AcmMatrix, "ACM Matrix")
            .clicked()
        {
            state.center_mode = CenterViewMode::AcmMatrix;
        }
        if ui
            .selectable_label(state.center_mode == CenterViewMode::Split, "Split 3D | ACM")
            .clicked()
        {
            state.center_mode = CenterViewMode::Split;
        }

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let right_lbl = if state.right_collapsed { "Plan ◂" } else { "▸" };
            if ui.button(right_lbl).clicked() {
                state.right_collapsed = !state.right_collapsed;
            }
            status_badge(ui, &state.gpu_adapter_label, 3);
            ui.menu_button("File ▾", |ui| {
                if ui.button("Open World (.toml / .json)...").clicked() {
                    ui.close();
                    state.open_world_dialog();
                }
                if ui.button("Save World (.toml / .json)...").clicked() {
                    ui.close();
                    state.save_world_dialog();
                }
                ui.separator();
                if ui.button("Import Trajectories (.toml / .json)...").clicked() {
                    ui.close();
                    state.import_trajectories_dialog();
                }
                if ui.button("Load Sample Trajectories").clicked() {
                    ui.close();
                    state.load_sample_trajectories();
                }
                if ui.button("Export Sample Trajectories...").clicked() {
                    ui.close();
                    state.export_sample_trajectories_dialog();
                }
                ui.separator();
                if ui.button("Import URDF...").clicked() {
                    ui.close();
                    state.import_urdf_dialog();
                }
                if ui.button("Export Legacy JSON...").clicked() {
                    ui.close();
                    state.export_legacy_json();
                }
                if ui.button("Export Legacy HTML...").clicked() {
                    ui.close();
                    state.export_legacy_html();
                }
            });
            if ui
                .button("Import Trajectories")
                .on_hover_text("Import per-robot ordered trajectories with frames & subframes (.toml / .json)")
                .clicked()
            {
                state.import_trajectories_dialog();
            }
        });
    });
    ui.add_space(1.0);
    let rect = ui.min_rect();
    ui.painter().line_segment(
        [rect.left_bottom(), rect.right_bottom()],
        egui::Stroke::new(1.0, BORDER_SUBTLE),
    );
}
