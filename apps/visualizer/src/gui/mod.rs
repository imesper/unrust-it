//! Native desktop Engineering Studio built with `eframe 0.37` + `egui 0.37` + `egui-wgpu 0.37` + `egui_plot 0.38`.

pub mod panels;
pub mod state;
pub mod viewport;

use panels::{
    acm_panel, apply_studio_theme, planner_panel, scene_panel, timeline_dock, top_toolbar,
    viewport_panel, BG_CANVAS, BG_PANEL, BORDER_SUBTLE,
};
use state::{CenterViewMode, StudioState};
use std::sync::Arc;
use viewport::GpuViewportBridge;

/// Root `eframe` application owning direct-mode [`StudioState`] and zero-copy [`GpuViewportBridge`].
pub struct StudioApp {
    state: StudioState,
    gpu: Option<GpuViewportBridge>,
}

impl StudioApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        apply_studio_theme(&cc.egui_ctx);

        let gpu = cc.wgpu_render_state.as_ref().map(GpuViewportBridge::new);
        let adapter_label = gpu
            .as_ref()
            .map_or_else(|| "wgpu 30".to_string(), |g| g.adapter_label().to_string());

        let ctx_wakeup = cc.egui_ctx.clone();
        let wakeup = Arc::new(move || {
            ctx_wakeup.request_repaint();
        });

        let state = StudioState::new(Some(wakeup), adapter_label);
        Self { state, gpu }
    }
}

impl eframe::App for StudioApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        self.state.drain_worker_events();
        self.state.tick_timers(ui.ctx());

        if !ui.ctx().egui_wants_keyboard_input() {
            let (space, left, right) = ui.input(|i| {
                (
                    i.key_pressed(egui::Key::Space),
                    i.key_pressed(egui::Key::ArrowLeft),
                    i.key_pressed(egui::Key::ArrowRight),
                )
            });
            if space {
                self.state.toggle_play();
            } else if left {
                self.state.step_frame(-1);
            } else if right {
                self.state.step_frame(1);
            }
        }

        let panel_frame = egui::Frame::new()
            .fill(BG_PANEL)
            .stroke(egui::Stroke::new(1.0, BORDER_SUBTLE))
            .inner_margin(egui::Margin::same(8));

        egui::Panel::top("top_toolbar")
            .frame(
                egui::Frame::new()
                    .fill(BG_PANEL)
                    .inner_margin(egui::Margin::symmetric(8, 5)),
            )
            .show(ui, |ui| top_toolbar::show(ui, &mut self.state));

        let bottom_height = if self.state.plots_collapsed { 42.0 } else { 188.0 };
        egui::Panel::bottom("timeline_dock")
            .resizable(!self.state.plots_collapsed)
            .min_size(if self.state.plots_collapsed { 42.0 } else { 120.0 })
            .max_size(360.0)
            .default_size(bottom_height)
            .frame(panel_frame)
            .show(ui, |ui| timeline_dock::show(ui, &mut self.state));

        if !self.state.left_collapsed {
            egui::Panel::left("scene_panel")
                .resizable(true)
                .min_size(220.0)
                .max_size(440.0)
                .default_size(285.0)
                .frame(panel_frame)
                .show(ui, |ui| scene_panel::show(ui, &mut self.state));
        }

        if !self.state.right_collapsed {
            egui::Panel::right("planner_panel")
                .resizable(true)
                .min_size(250.0)
                .max_size(460.0)
                .default_size(315.0)
                .frame(panel_frame)
                .show(ui, |ui| planner_panel::show(ui, &mut self.state));
        }

        let render_state = frame.wgpu_render_state();
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(BG_CANVAS).inner_margin(egui::Margin::ZERO))
            .show(ui, |ui| match self.state.center_mode {
                CenterViewMode::Viewport => {
                    viewport_panel::show(ui, &mut self.state, self.gpu.as_mut(), render_state);
                }
                CenterViewMode::AcmMatrix => {
                    egui::Frame::new()
                        .inner_margin(egui::Margin::same(8))
                        .show(ui, |ui| acm_panel::show(ui, &mut self.state));
                }
                CenterViewMode::Split => {
                    let avail = ui.available_size();
                    let left_w = (avail.x * 0.54).max(160.0);
                    let right_w = (avail.x - left_w - 6.0).max(160.0);
                    ui.horizontal(|ui| {
                        ui.allocate_ui(egui::vec2(left_w, avail.y), |ui| {
                            viewport_panel::show(
                                ui,
                                &mut self.state,
                                self.gpu.as_mut(),
                                render_state,
                            );
                        });
                        ui.separator();
                        ui.allocate_ui(egui::vec2(right_w, avail.y), |ui| {
                            egui::Frame::new()
                                .inner_margin(egui::Margin::same(6))
                                .show(ui, |ui| acm_panel::show(ui, &mut self.state));
                        });
                    });
                }
            });
    }
}

/// Launches the interactive native `eframe` / `egui 0.37` + zero-copy `wgpu 30` Desktop Engineering Studio.
pub fn run_gui() -> Result<(), Box<dyn std::error::Error>> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("unrust-it Studio — Multi-Arm Path Planning & Collision Avoidance")
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([1080.0, 680.0]),
        ..Default::default()
    };

    eframe::run_native(
        "unrust-it Studio",
        native_options,
        Box::new(|cc| Ok(Box::new(StudioApp::new(cc)))),
    )?;
    Ok(())
}
