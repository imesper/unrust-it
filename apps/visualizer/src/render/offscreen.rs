//! Standalone headless `wgpu 30` offscreen viewport + CPU RGBA8 readback and CI smoke tests.

use crate::render::{OrbitCamera, RenderSceneData, SceneRenderer};

/// Standalone offscreen `wgpu 30` render target with `Backends::NOOP` headless CI fallback
/// and 256-byte-aligned RGBA8 pixel readback.
pub struct OffscreenViewport {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
    renderer: SceneRenderer,
    color_tex: wgpu::Texture,
    color_view: wgpu::TextureView,
    staging_buf: wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
}

impl OffscreenViewport {
    /// Initializes an offscreen `wgpu 30` viewport of size `(width, height)`.
    pub fn new(width: u32, height: u32) -> Result<Self, String> {
        pollster::block_on(Self::new_async(width, height))
    }

    /// Asynchronously initializes an offscreen `wgpu 30` viewport of size `(width, height)`.
    pub async fn new_async(width: u32, height: u32) -> Result<Self, String> {
        let hw_instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY | wgpu::Backends::GL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = match hw_instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
        {
            Ok(a) => a,
            Err(_) => {
                let noop_instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::NOOP,
                    backend_options: wgpu::BackendOptions {
                        noop: wgpu::NoopBackendOptions {
                            enable: true,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    ..wgpu::InstanceDescriptor::new_without_display_handle()
                });
                noop_instance
                    .request_adapter(&wgpu::RequestAdapterOptions::default())
                    .await
                    .map_err(|e| format!("failed to acquire hardware or noop wgpu adapter: {e}"))?
            }
        };

        let adapter_info = adapter.get_info();
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let msaa_supported = adapter
            .get_texture_format_features(format)
            .flags
            .contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X4);
        let sample_count = if msaa_supported && adapter_info.backend != wgpu::Backend::Noop {
            4
        } else {
            1
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("unrust_offscreen_render_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("failed to create wgpu device: {e}"))?;

        let renderer = SceneRenderer::new(&device, &queue, format, sample_count);
        let w = width.max(1);
        let h = height.max(1);
        let (color_tex, color_view, staging_buf, padded_bytes_per_row) =
            Self::create_targets(&device, w, h, format);

        Ok(Self {
            device,
            queue,
            adapter_info,
            renderer,
            color_tex,
            color_view,
            staging_buf,
            width: w,
            height: h,
            padded_bytes_per_row,
        })
    }

    /// Underlying `wgpu::AdapterInfo` (can be used to detect hardware vs `Noop` execution).
    #[inline]
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Resizes the offscreen color texture and staging readback buffer if needed.
    pub fn resize(&mut self, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if (w, h) == (self.width, self.height) {
            return;
        }
        let (color_tex, color_view, staging_buf, padded_bytes_per_row) =
            Self::create_targets(&self.device, w, h, wgpu::TextureFormat::Rgba8Unorm);
        self.color_tex = color_tex;
        self.color_view = color_view;
        self.staging_buf = staging_buf;
        self.padded_bytes_per_row = padded_bytes_per_row;
        self.width = w;
        self.height = h;
    }

    /// Renders `scene` and reads back an unpadded tightly-packed `width * height * 4` RGBA8 pixel buffer.
    pub fn render_to_rgba8(&mut self, camera: &OrbitCamera, scene: &RenderSceneData) -> Vec<u8> {
        self.renderer.render(
            &self.device,
            &self.queue,
            &self.color_view,
            self.width,
            self.height,
            camera,
            scene,
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("unrust_offscreen_copy_encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.color_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = self.staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .expect("wgpu readback channel closed")
            .expect("failed to map offscreen staging buffer");

        let mapped = slice
            .get_mapped_range()
            .expect("failed to map offscreen staging range");
        let row_bytes = (self.width * 4) as usize;
        let stride = self.padded_bytes_per_row as usize;
        let mut out = vec![0u8; row_bytes * (self.height as usize)];
        for row in 0..(self.height as usize) {
            let src = &mapped[row * stride..row * stride + row_bytes];
            let dst = &mut out[row * row_bytes..(row + 1) * row_bytes];
            dst.copy_from_slice(src);
        }
        drop(mapped);
        self.staging_buf.unmap();
        out
    }

    fn create_targets(
        device: &wgpu::Device,
        w: u32,
        h: u32,
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::Buffer, u32) {
        let color_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("unrust_offscreen_color"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let padded_bytes_per_row = (w * 4).div_ceil(256) * 256;
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("unrust_offscreen_staging"),
            size: (padded_bytes_per_row as u64) * (h as u64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (color_tex, color_view, staging_buf, padded_bytes_per_row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{extract_render_scene, OverlayFlags};
    use crate::scene::{evaluate_frame_snapshot, preset_workcell, run_planner_sync, PresetId};

    #[test]
    fn offscreen_renderer_validates_all_shaders_and_presets() {
        let mut viewport = OffscreenViewport::new(320, 200)
            .expect("offscreen wgpu viewport should initialize on GPU or Noop backend");
        let cam = OrbitCamera::default();

        // 1. 3-Arm Close Proximity preset with attached compound tools, ESDF Z-slice, Tier-1 spheres & witnesses
        let spec3 = preset_workcell(PresetId::ThreeArmCloseProximity);
        let wc3 = spec3.compile(1).unwrap();
        let mut ws3 = wc3.env.create_workspace(&wc3.model);
        let snap3 = evaluate_frame_snapshot(&wc3, wc3.q_start.as_slice(), &[], 8, None, &mut ws3);
        let overlays3 = OverlayFlags {
            show_tier1_spheres: true,
            show_esdf_slice: true,
            esdf_slice_coord: 0.42,
            ..OverlayFlags::default()
        };
        let scene3 = extract_render_scene(&wc3, &spec3, &snap3, None, cam.eye_position(), &overlays3);
        assert!(!scene3.cubes.is_empty());
        assert!(!scene3.cylinders.is_empty());
        assert!(!scene3.spheres.is_empty());
        assert!(!scene3.translucent_spheres.is_empty());
        assert!(!scene3.grid_and_frame_lines.is_empty());
        assert!(scene3.esdf_slice.is_some());

        let pixels3 = viewport.render_to_rgba8(&cam, &scene3);
        assert_eq!(pixels3.len(), 320 * 200 * 4);
        if viewport.adapter_info().backend != wgpu::Backend::Noop {
            assert!(
                pixels3
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .any(|px| px[0] > 20 || px[1] > 20 || px[2] > 20),
                "hardware GPU render should produce non-black scene pixels"
            );
        }

        // 2. Dual-Arm MPPI + ESDF preset with real rollout ribbons and resize check
        let spec2 = preset_workcell(PresetId::DualArmMppiEsdf);
        let wc2 = spec2.compile(2).unwrap();
        let traj2 = run_planner_sync(&wc2, &spec2).unwrap();
        let mut ws2 = wc2.env.create_workspace(&wc2.model);
        let snap2 = evaluate_frame_snapshot(&wc2, &traj2.frames[0].q, &[], 5, None, &mut ws2);
        let scene2 = extract_render_scene(
            &wc2,
            &spec2,
            &snap2,
            Some(&traj2),
            cam.eye_position(),
            &OverlayFlags::default(),
        );
        assert!(!scene2.overlay_lines.is_empty());

        viewport.resize(257, 150); // non-256-aligned row stride test
        let pixels2 = viewport.render_to_rgba8(&cam, &scene2);
        assert_eq!(pixels2.len(), 257 * 150 * 4);
    }
}
