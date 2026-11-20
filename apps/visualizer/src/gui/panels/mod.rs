//! Shared dark engineering theme tokens, helper widgets, and panel submodules.

pub mod acm_panel;
pub mod planner_panel;
pub mod scene_panel;
pub mod timeline_dock;
pub mod top_toolbar;
pub mod viewport_panel;

pub const BG_CANVAS: egui::Color32 = egui::Color32::from_rgb(0x0d, 0x11, 0x17);
pub const BG_PANEL: egui::Color32 = egui::Color32::from_rgb(0x14, 0x19, 0x22);
pub const BG_SURFACE: egui::Color32 = egui::Color32::from_rgb(0x1b, 0x22, 0x30);
pub const BG_HOVER: egui::Color32 = egui::Color32::from_rgb(0x23, 0x2d, 0x40);
pub const BORDER_SUBTLE: egui::Color32 = egui::Color32::from_rgb(0x26, 0x30, 0x42);
pub const BORDER_FOCUS: egui::Color32 = egui::Color32::from_rgb(0x38, 0xbd, 0xf8);
pub const TEXT_PRIMARY: egui::Color32 = egui::Color32::from_rgb(0xe6, 0xed, 0xf3);
pub const TEXT_MUTED: egui::Color32 = egui::Color32::from_rgb(0x8b, 0x94, 0x9e);
pub const ACCENT_PRIMARY: egui::Color32 = egui::Color32::from_rgb(0x38, 0xbd, 0xf8);
pub const ACCENT_STAGE2: egui::Color32 = egui::Color32::from_rgb(0xa7, 0x8b, 0xfa);
pub const STATUS_SAFE: egui::Color32 = egui::Color32::from_rgb(0x10, 0xb9, 0x81);
pub const STATUS_WARN: egui::Color32 = egui::Color32::from_rgb(0xf5, 0x9e, 0x0b);
pub const STATUS_DANGER: egui::Color32 = egui::Color32::from_rgb(0xf4, 0x3f, 0x5e);

pub fn apply_studio_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = BG_PANEL;
    visuals.window_fill = BG_PANEL;
    visuals.extreme_bg_color = BG_CANVAS;
    visuals.faint_bg_color = BG_SURFACE;
    visuals.widgets.noninteractive.bg_fill = BG_SURFACE;
    visuals.widgets.noninteractive.weak_bg_fill = BG_SURFACE;
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, BORDER_SUBTLE);
    visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);

    visuals.widgets.inactive.bg_fill = BG_SURFACE;
    visuals.widgets.inactive.weak_bg_fill = BG_SURFACE;
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, BORDER_SUBTLE);
    visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);

    visuals.widgets.hovered.bg_fill = BG_HOVER;
    visuals.widgets.hovered.weak_bg_fill = BG_HOVER;
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, BORDER_FOCUS);
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);

    visuals.widgets.active.bg_fill = egui::Color32::from_rgb(0x17, 0x34, 0x4d);
    visuals.widgets.active.weak_bg_fill = egui::Color32::from_rgb(0x17, 0x34, 0x4d);
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, ACCENT_PRIMARY);
    visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);

    visuals.selection.bg_fill = egui::Color32::from_rgb(0x17, 0x34, 0x4d);
    visuals.selection.stroke = egui::Stroke::new(1.0, ACCENT_PRIMARY);
    ctx.set_theme(egui::Theme::Dark);
    ctx.set_visuals_of(egui::Theme::Dark, visuals);

    ctx.all_styles_mut(|style| {
        style.spacing.item_spacing = egui::vec2(6.0, 5.0);
        style.spacing.button_padding = egui::vec2(8.0, 4.0);
    });
}

#[inline]
pub fn rgba_to_color32(c: [f32; 4]) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(
        (c[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (c[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (c[2].clamp(0.0, 1.0) * 255.0).round() as u8,
        (c[3].clamp(0.0, 1.0) * 255.0).round() as u8,
    )
}

#[inline]
pub fn status_color(level: u8) -> egui::Color32 {
    match level {
        0 => STATUS_SAFE,
        1 => STATUS_WARN,
        2 => STATUS_DANGER,
        _ => ACCENT_PRIMARY,
    }
}

pub fn section_card(ui: &mut egui::Ui, title: &str, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::new()
        .fill(BG_SURFACE)
        .stroke(egui::Stroke::new(1.0, BORDER_SUBTLE))
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(
                egui::RichText::new(title)
                    .size(10.0)
                    .strong()
                    .color(TEXT_MUTED),
            );
            ui.add_space(2.0);
            add_contents(ui);
        });
}

pub fn status_badge(ui: &mut egui::Ui, text: &str, level: u8) {
    let fg = status_color(level);
    let bg = egui::Color32::from_rgba_unmultiplied(fg.r(), fg.g(), fg.b(), 36);
    egui::Frame::new()
        .fill(bg)
        .stroke(egui::Stroke::new(1.0, fg))
        .corner_radius(egui::CornerRadius::same(4))
        .inner_margin(egui::Margin::symmetric(6, 2))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(text).size(10.5).strong().color(fg));
        });
}
