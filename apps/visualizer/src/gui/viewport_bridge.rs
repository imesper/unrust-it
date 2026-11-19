//! Zero-copy `wgpu 30` (and optional readback fallback) presentation bridge into Slint `Image`.

use crate::gui::{MainWindow, ViewportState};
use crate::render::{OrbitCamera, OverlayFlags, RenderSceneData};
use slint::ComponentHandle;
use std::cell::RefCell;
use std::rc::Rc;

#[cfg(feature = "gui")]
use crate::render::SceneRenderer;
#[cfg(all(feature = "gui-readback", not(feature = "gui")))]
use crate::render::OffscreenViewport;

/// Shared UI-thread viewport state read during `BeforeRendering` and mutated by camera/layer callbacks.
pub struct ViewportRenderState {
    pub camera: OrbitCamera,
    pub overlays: OverlayFlags,
    pub scene: RenderSceneData,
    pub logical_size: (u32, u32),
    pub scale_factor: f32,
    pub dirty: bool,
    pub hud_frame_ms: f32,
}

impl Default for ViewportRenderState {
    fn default() -> Self {
        Self {
            camera: OrbitCamera::default(),
            overlays: OverlayFlags::default(),
            scene: RenderSceneData::default(),
            logical_size: (800, 600),
            scale_factor: 1.0,
            dirty: true,
            hud_frame_ms: 0.0,
        }
    }
}

/// Initializes Slint's windowing & graphics backend (requiring `wgpu 30` when `gui` is active).
pub fn init_slint_backend() -> Result<(), String> {
    #[cfg(feature = "gui")]
    {
        slint::BackendSelector::new()
            .require_wgpu_30(slint::wgpu_30::WGPUConfiguration::default())
            .select()
            .map_err(|e| format!("failed to initialize Slint wgpu 30 backend: {e}"))?;
    }
    Ok(())
}

/// Attaches the 3D `wgpu 30` rendering hook onto `window`.
pub fn attach_viewport_renderer(
    window: &MainWindow,
    state: Rc<RefCell<ViewportRenderState>>,
) -> Result<(), String> {
    #[cfg(feature = "gui")]
    {
        attach_wgpu30_zero_copy(window, state)
    }
    #[cfg(all(feature = "gui-readback", not(feature = "gui")))]
    {
        attach_offscreen_readback(window, state)
    }
}

#[cfg(feature = "gui")]
struct GpuViewportResources {
    renderer: SceneRenderer,
    targets: Option<[wgpu::Texture; 2]>,
    target_dims: (u32, u32),
    slot: usize,
}

#[cfg(feature = "gui")]
fn attach_wgpu30_zero_copy(
    window: &MainWindow,
    state: Rc<RefCell<ViewportRenderState>>,
) -> Result<(), String> {
    let win_weak = window.as_weak();
    let gpu_res: Rc<RefCell<Option<GpuViewportResources>>> = Rc::new(RefCell::new(None));

    window
        .window()
        .set_rendering_notifier(move |render_state, graphics_api| {
            let slint::GraphicsAPI::WGPU30 { device, queue, .. } = graphics_api else {
                return;
            };

            match render_state {
                slint::RenderingState::RenderingSetup => {
                    // Rgba8Unorm is required by Slint's Image::try_from(wgpu::Texture); 4x MSAA is
                    // guaranteed on standard Core WebGPU adapters.
                    let renderer =
                        SceneRenderer::new(device, queue, wgpu::TextureFormat::Rgba8Unorm, 4);
                    *gpu_res.borrow_mut() = Some(GpuViewportResources {
                        renderer,
                        targets: None,
                        target_dims: (0, 0),
                        slot: 0,
                    });
                }
                slint::RenderingState::BeforeRendering => {
                    let Some(ui) = win_weak.upgrade() else {
                        return;
                    };
                    let mut st = state.borrow_mut();
                    let scale = ui.window().scale_factor().max(1.0);
                    st.scale_factor = scale;
                    let phys_w = ((st.logical_size.0 as f32) * scale).round().max(1.0) as u32;
                    let phys_h = ((st.logical_size.1 as f32) * scale).round().max(1.0) as u32;

                    let mut res_opt = gpu_res.borrow_mut();
                    let Some(res) = res_opt.as_mut() else {
                        return;
                    };

                    let need_realloc = res.target_dims != (phys_w, phys_h) || res.targets.is_none();
                    if !st.dirty && !need_realloc {
                        return;
                    }

                    if need_realloc {
                        let desc = wgpu::TextureDescriptor {
                            label: Some("unrust_slint_viewport_target"),
                            size: wgpu::Extent3d {
                                width: phys_w,
                                height: phys_h,
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                                | wgpu::TextureUsages::TEXTURE_BINDING,
                            view_formats: &[],
                        };
                        res.targets =
                            Some([device.create_texture(&desc), device.create_texture(&desc)]);
                        res.target_dims = (phys_w, phys_h);
                    }

                    let slot = res.slot;
                    let tex_clone = {
                        let tex = &res.targets.as_ref().unwrap()[slot];
                        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                        let t0 = std::time::Instant::now();
                        res.renderer.render(
                            device,
                            queue,
                            &view,
                            phys_w,
                            phys_h,
                            &st.camera,
                            &st.scene,
                        );
                        st.hud_frame_ms = t0.elapsed().as_secs_f32() * 1000.0;
                        tex.clone()
                    };
                    res.slot ^= 1;
                    st.dirty = false;

                    if let Ok(img) = slint::Image::try_from(tex_clone) {
                        ui.global::<ViewportState>().set_frame_image(img);
                    }
                }
                slint::RenderingState::RenderingTeardown => {
                    *gpu_res.borrow_mut() = None;
                }
                _ => {}
            }
        })
        .map_err(|e| format!("failed to install Slint wgpu 30 rendering notifier: {e}"))
}

#[cfg(all(feature = "gui-readback", not(feature = "gui")))]
fn attach_offscreen_readback(
    window: &MainWindow,
    state: Rc<RefCell<ViewportRenderState>>,
) -> Result<(), String> {
    let win_weak = window.as_weak();
    let offscreen: Rc<RefCell<Option<OffscreenViewport>>> = Rc::new(RefCell::new(None));

    window
        .window()
        .set_rendering_notifier(move |render_state, _api| match render_state {
            slint::RenderingState::RenderingSetup => {
                *offscreen.borrow_mut() = OffscreenViewport::new(800, 600).ok();
            }
            slint::RenderingState::BeforeRendering => {
                let Some(ui) = win_weak.upgrade() else {
                    return;
                };
                let mut st = state.borrow_mut();
                if !st.dirty {
                    return;
                }
                let scale = ui.window().scale_factor().max(1.0);
                let phys_w = ((st.logical_size.0 as f32) * scale).round().max(1.0) as u32;
                let phys_h = ((st.logical_size.1 as f32) * scale).round().max(1.0) as u32;
                if let Some(vp) = offscreen.borrow_mut().as_mut() {
                    vp.resize(phys_w, phys_h);
                    let pixels = vp.render_to_rgba8(&st.camera, &st.scene);
                    let buf =
                        slint::SharedPixelBuffer::<slint::Rgba8Pixel>::clone_from_slice(
                            &pixels, phys_w, phys_h,
                        );
                    ui.global::<ViewportState>()
                        .set_frame_image(slint::Image::from_rgba8(buf));
                    st.dirty = false;
                }
            }
            slint::RenderingState::RenderingTeardown => {
                *offscreen.borrow_mut() = None;
            }
            _ => {}
        })
        .map_err(|e| format!("failed to install Slint readback rendering notifier: {e}"))
}
