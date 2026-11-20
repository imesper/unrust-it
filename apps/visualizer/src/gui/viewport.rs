//! Zero-copy `wgpu 30` offscreen texture bridge into `egui-wgpu 0.37`.

use crate::render::{OrbitCamera, RenderSceneData, SceneRenderer};

/// Owns the pure-`wgpu 30` [`SceneRenderer`] and a resident `Rgba8Unorm` texture registered
/// with `egui_wgpu::Renderer` for zero-readback presentation inside an `egui` panel.
pub struct GpuViewportBridge {
    renderer: SceneRenderer,
    target_tex: Option<wgpu::Texture>,
    target_view: Option<wgpu::TextureView>,
    target_dims: (u32, u32),
    texture_id: Option<egui::TextureId>,
    adapter_label: String,
    last_frame_ms: f32,
}

impl GpuViewportBridge {
    /// Creates a [`GpuViewportBridge`] bound to `eframe`'s active `wgpu 30` render state.
    pub fn new(render_state: &egui_wgpu::RenderState) -> Self {
        let info = render_state.adapter.get_info();
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let msaa_supported = render_state
            .adapter
            .get_texture_format_features(format)
            .flags
            .contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X4);
        let sample_count = if msaa_supported && info.backend != wgpu::Backend::Noop {
            4
        } else {
            1
        };

        let renderer = SceneRenderer::new(
            &render_state.device,
            &render_state.queue,
            format,
            sample_count,
        );

        let backend_name = match info.backend {
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Vulkan => "Vulkan",
            wgpu::Backend::Dx12 => "DX12",
            wgpu::Backend::Gl => "GL",
            wgpu::Backend::BrowserWebGpu => "WebGPU",
            wgpu::Backend::Noop => "Noop",
        };
        let compute_name = if cfg!(feature = "gpu-compute") {
            "GPU"
        } else {
            "SIMD"
        };

        Self {
            renderer,
            target_tex: None,
            target_view: None,
            target_dims: (0, 0),
            texture_id: None,
            adapter_label: format!("{backend_name} · {compute_name}"),
            last_frame_ms: 0.0,
        }
    }

    /// Returns a concise GPU adapter + batch compute label for the Studio toolbar.
    #[inline]
    pub fn adapter_label(&self) -> &str {
        &self.adapter_label
    }

    /// Returns the last 3D pass render duration in milliseconds.
    #[inline]
    pub fn last_frame_ms(&self) -> f32 {
        self.last_frame_ms
    }

    /// Reallocates the offscreen texture if needed and renders `scene` into it when `dirty` is set.
    ///
    /// Returns the registered [`egui::TextureId`] ready for `egui` painting.
    pub fn ensure_rendered(
        &mut self,
        render_state: &egui_wgpu::RenderState,
        phys_w: u32,
        phys_h: u32,
        camera: &OrbitCamera,
        scene: &RenderSceneData,
        dirty: &mut bool,
    ) -> egui::TextureId {
        let w = phys_w.max(1);
        let h = phys_h.max(1);

        if self.target_dims != (w, h) || self.target_tex.is_none() {
            let tex = render_state.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("unrust_egui_viewport_target"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

            let mut egui_renderer = render_state.renderer.write();
            if let Some(id) = self.texture_id {
                egui_renderer.update_egui_texture_from_wgpu_texture(
                    &render_state.device,
                    &view,
                    wgpu::FilterMode::Linear,
                    id,
                );
            } else {
                self.texture_id = Some(egui_renderer.register_native_texture(
                    &render_state.device,
                    &view,
                    wgpu::FilterMode::Linear,
                ));
            }

            self.target_tex = Some(tex);
            self.target_view = Some(view);
            self.target_dims = (w, h);
            *dirty = true;
        }

        if *dirty {
            if let Some(view) = self.target_view.as_ref() {
                let t0 = std::time::Instant::now();
                self.renderer.render(
                    &render_state.device,
                    &render_state.queue,
                    view,
                    w,
                    h,
                    camera,
                    scene,
                );
                self.last_frame_ms = t0.elapsed().as_secs_f32() * 1000.0;
            }
            *dirty = false;
        }

        self.texture_id
            .expect("viewport texture id always registered before return")
    }
}
