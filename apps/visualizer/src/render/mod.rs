//! Pure-`wgpu 30` 3D viewport renderer for `unrust-it` (zero GUI/windowing dependencies).

pub mod camera;
pub mod extract;
pub mod meshes;
pub mod offscreen;

pub use camera::OrbitCamera;
pub use extract::{
    extract_esdf_slice, extract_render_scene, extract_render_scene_with_highlights,
    pick_entity_along_ray, EsdfSliceData, ImportedFrameMarker, LineVertex, OverlayFlags,
    PickedEntity, RenderSceneData, SceneHighlights, SolidInstance,
};
pub use offscreen::OffscreenViewport;

use bytemuck::{Pod, Zeroable};
use meshes::{generate_unit_cube, generate_unit_cylinder_y, generate_unit_sphere, VertexPN};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    light_dir: [f32; 4],
    fill_dir: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct EsdfSliceUniform {
    corner_origin: [f32; 4],
    axis_u: [f32; 4],
    axis_v: [f32; 4],
    dims_and_act: [f32; 4],
}

struct MeshGpu {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    num_indices: u32,
}

impl MeshGpu {
    fn from_data(device: &wgpu::Device, label: &str, verts: &[VertexPN], indices: &[u32]) -> Self {
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            vertex_buf,
            index_buf,
            num_indices: indices.len() as u32,
        }
    }
}

struct DynamicBuffer {
    buffer: wgpu::Buffer,
    capacity_bytes: u64,
    usage: wgpu::BufferUsages,
    label: &'static str,
}

impl DynamicBuffer {
    fn new(device: &wgpu::Device, label: &'static str, usage: wgpu::BufferUsages, init_bytes: u64) -> Self {
        let cap = init_bytes.max(256);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: cap,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            capacity_bytes: cap,
            usage,
            label,
        }
    }

    fn upload<T: Pod>(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[T]) -> u32 {
        if data.is_empty() {
            return 0;
        }
        let bytes = bytemuck::cast_slice::<T, u8>(data);
        let needed = bytes.len() as u64;
        if needed > self.capacity_bytes {
            let new_cap = (needed * 3 / 2).next_multiple_of(4096);
            self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: new_cap,
                usage: self.usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.capacity_bytes = new_cap;
        }
        queue.write_buffer(&self.buffer, 0, bytes);
        data.len() as u32
    }
}

/// Pure-`wgpu 30` multi-layer 3D scene renderer.
///
/// Renders into any supplied [`wgpu::TextureView`] (`color_format`) with optional MSAA (`1` or `4`).
pub struct SceneRenderer {
    color_format: wgpu::TextureFormat,
    sample_count: u32,

    camera_uniform_buf: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    lines_depth_pipeline: wgpu::RenderPipeline,
    lines_xray_pipeline: wgpu::RenderPipeline,
    solid_opaque_pipeline: wgpu::RenderPipeline,
    solid_translucent_pipeline: wgpu::RenderPipeline,
    esdf_pipeline: wgpu::RenderPipeline,

    cube_mesh: MeshGpu,
    cylinder_mesh: MeshGpu,
    sphere_mesh: MeshGpu,

    cube_instances: DynamicBuffer,
    cylinder_instances: DynamicBuffer,
    sphere_instances: DynamicBuffer,
    translucent_instances: DynamicBuffer,
    grid_lines_buf: DynamicBuffer,
    overlay_lines_buf: DynamicBuffer,
    xray_lines_buf: DynamicBuffer,

    esdf_layout: wgpu::BindGroupLayout,
    esdf_uniform_buf: wgpu::Buffer,
    esdf_texture: wgpu::Texture,
    esdf_tex_dims: (u32, u32),
    esdf_bind_group: wgpu::BindGroup,

    depth_tex: Option<wgpu::Texture>,
    depth_view: Option<wgpu::TextureView>,
    msaa_tex: Option<wgpu::Texture>,
    msaa_view: Option<wgpu::TextureView>,
    target_dims: (u32, u32),
}

impl SceneRenderer {
    /// Creates all WGSL render pipelines and static unit meshes on `(device, queue)`.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Self {
        let sample_count = sample_count.max(1);

        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("unrust_camera_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let camera_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("unrust_camera_uniform"),
            contents: bytemuck::bytes_of(&CameraUniform {
                view_proj: nalgebra::Matrix4::identity().into(),
                camera_pos: [1.0, 1.0, 1.0, 1.0],
                light_dir: [0.55, 0.35, 0.76, 0.0],
                fill_dir: [-0.65, -0.45, 0.30, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("unrust_camera_bg"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_uniform_buf.as_entire_binding(),
            }],
        });

        let esdf_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("unrust_esdf_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let solid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("unrust_solid_wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/solid.wgsl").into()),
        });
        let lines_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("unrust_lines_wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/lines.wgsl").into()),
        });
        let esdf_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("unrust_esdf_slice_wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/esdf_slice.wgsl").into()),
        });

        let cam_only_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("unrust_cam_only_pl"),
            bind_group_layouts: &[Some(&camera_layout)],
            immediate_size: 0,
        });
        let esdf_pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("unrust_esdf_pl"),
            bind_group_layouts: &[Some(&camera_layout), Some(&esdf_layout)],
            immediate_size: 0,
        });

        let vertex_pn_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VertexPN>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        };
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SolidInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 64,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };
        let line_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let solid_buffers = [Some(vertex_pn_layout), Some(instance_layout)];
        let line_buffers = [Some(line_layout)];
        let solid_opaque_pipeline = create_solid_pipeline(
            device,
            &cam_only_layout,
            &solid_shader,
            &solid_buffers,
            color_format,
            sample_count,
            false,
        );
        let solid_translucent_pipeline = create_solid_pipeline(
            device,
            &cam_only_layout,
            &solid_shader,
            &solid_buffers,
            color_format,
            sample_count,
            true,
        );
        let lines_depth_pipeline = create_lines_pipeline(
            device,
            &cam_only_layout,
            &lines_shader,
            &line_buffers,
            color_format,
            sample_count,
            false,
        );
        let lines_xray_pipeline = create_lines_pipeline(
            device,
            &cam_only_layout,
            &lines_shader,
            &line_buffers,
            color_format,
            sample_count,
            true,
        );
        let esdf_pipeline = create_esdf_pipeline(
            device,
            &esdf_pipe_layout,
            &esdf_shader,
            color_format,
            sample_count,
        );

        let (sv, si) = generate_unit_sphere(24, 16);
        let (cv, ci) = generate_unit_cylinder_y(24);
        let (bv, bi) = generate_unit_cube();

        let esdf_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("unrust_esdf_slice_uni"),
            contents: bytemuck::bytes_of(&EsdfSliceUniform {
                corner_origin: [0.0; 4],
                axis_u: [1.0, 0.0, 0.0, 0.0],
                axis_v: [0.0, 1.0, 0.0, 0.0],
                dims_and_act: [1.0, 1.0, 0.1, 0.7],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let (esdf_texture, esdf_bind_group) =
            create_esdf_texture_and_bg(device, queue, &esdf_layout, &esdf_uniform_buf, 2, 2, &[1.0; 4]);

        Self {
            color_format,
            sample_count,
            camera_uniform_buf,
            camera_bind_group,
            lines_depth_pipeline,
            lines_xray_pipeline,
            solid_opaque_pipeline,
            solid_translucent_pipeline,
            esdf_pipeline,
            cube_mesh: MeshGpu::from_data(device, "unit_cube", &bv, &bi),
            cylinder_mesh: MeshGpu::from_data(device, "unit_cylinder_y", &cv, &ci),
            sphere_mesh: MeshGpu::from_data(device, "unit_sphere", &sv, &si),
            cube_instances: DynamicBuffer::new(device, "cube_inst", wgpu::BufferUsages::VERTEX, 4096),
            cylinder_instances: DynamicBuffer::new(device, "cyl_inst", wgpu::BufferUsages::VERTEX, 8192),
            sphere_instances: DynamicBuffer::new(device, "sph_inst", wgpu::BufferUsages::VERTEX, 8192),
            translucent_instances: DynamicBuffer::new(
                device,
                "trans_inst",
                wgpu::BufferUsages::VERTEX,
                8192,
            ),
            grid_lines_buf: DynamicBuffer::new(device, "grid_lines", wgpu::BufferUsages::VERTEX, 8192),
            overlay_lines_buf: DynamicBuffer::new(
                device,
                "overlay_lines",
                wgpu::BufferUsages::VERTEX,
                16384,
            ),
            xray_lines_buf: DynamicBuffer::new(device, "xray_lines", wgpu::BufferUsages::VERTEX, 4096),
            esdf_layout,
            esdf_uniform_buf,
            esdf_texture,
            esdf_tex_dims: (2, 2),
            esdf_bind_group,
            depth_tex: None,
            depth_view: None,
            msaa_tex: None,
            msaa_view: None,
            target_dims: (0, 0),
        }
    }

    /// Renders `scene` into `target_view` using `camera` for viewport dimensions `(width, height)`.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        camera: &OrbitCamera,
        scene: &RenderSceneData,
    ) {
        let w = width.max(1);
        let h = height.max(1);
        self.ensure_attachments(device, w, h);

        let aspect = (w as f32) / (h as f32);
        let eye = camera.eye_position();
        let cam_uni = CameraUniform {
            view_proj: camera.view_proj(aspect).into(),
            camera_pos: [eye.x, eye.y, eye.z, 1.0],
            light_dir: [0.55, 0.35, 0.76, 0.0],
            fill_dir: [-0.65, -0.45, 0.30, 0.0],
        };
        queue.write_buffer(&self.camera_uniform_buf, 0, bytemuck::bytes_of(&cam_uni));

        let n_cubes = self.cube_instances.upload(device, queue, &scene.cubes);
        let n_cyls = self
            .cylinder_instances
            .upload(device, queue, &scene.cylinders);
        let n_sphs = self.sphere_instances.upload(device, queue, &scene.spheres);
        let n_trans = self
            .translucent_instances
            .upload(device, queue, &scene.translucent_spheres);
        let n_grid = self
            .grid_lines_buf
            .upload(device, queue, &scene.grid_and_frame_lines);
        let n_overlay = self
            .overlay_lines_buf
            .upload(device, queue, &scene.overlay_lines);
        let n_xray = self.xray_lines_buf.upload(device, queue, &scene.xray_lines);

        let draw_esdf = if let Some(slice) = &scene.esdf_slice {
            self.upload_esdf_slice(device, queue, slice);
            true
        } else {
            false
        };

        let (color_view, resolve_target) = if self.sample_count > 1 {
            (
                self.msaa_view
                    .as_ref()
                    .expect("MSAA view initialized for sample_count > 1"),
                Some(target_view),
            )
        } else {
            (target_view, None)
        };
        let depth_view = self
            .depth_view
            .as_ref()
            .expect("depth view initialized before render pass");

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("unrust_scene_render_encoder"),
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("unrust_scene_main_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    depth_slice: None,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.051,
                            g: 0.067,
                            b: 0.090,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // 1. Floor grid & axis triads
            if n_grid > 0 {
                pass.set_pipeline(&self.lines_depth_pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, self.grid_lines_buf.buffer.slice(..));
                pass.draw(0..n_grid, 0..1);
            }

            // 2. Opaque solid geometry (links, attached tools, obstacles, 3D witness rods)
            if n_cubes > 0 || n_cyls > 0 || n_sphs > 0 {
                pass.set_pipeline(&self.solid_opaque_pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                draw_mesh_instances(&mut pass, &self.cube_mesh, &self.cube_instances.buffer, n_cubes);
                draw_mesh_instances(
                    &mut pass,
                    &self.cylinder_mesh,
                    &self.cylinder_instances.buffer,
                    n_cyls,
                );
                draw_mesh_instances(
                    &mut pass,
                    &self.sphere_mesh,
                    &self.sphere_instances.buffer,
                    n_sphs,
                );
            }

            // 3. 3D ESDF slice quad
            if draw_esdf {
                pass.set_pipeline(&self.esdf_pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_bind_group(1, &self.esdf_bind_group, &[]);
                pass.draw(0..6, 0..1);
            }

            // 4. MPPI rollout ribbons, TCP trails & X-ray witnesses
            if n_overlay > 0 {
                pass.set_pipeline(&self.lines_depth_pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, self.overlay_lines_buf.buffer.slice(..));
                pass.draw(0..n_overlay, 0..1);
            }
            if n_xray > 0 {
                pass.set_pipeline(&self.lines_xray_pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, self.xray_lines_buf.buffer.slice(..));
                pass.draw(0..n_xray, 0..1);
            }

            // 5. Back-to-front sorted Fresnel translucent Tier-1 spheres & target ghosts
            if n_trans > 0 {
                pass.set_pipeline(&self.solid_translucent_pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                draw_mesh_instances(
                    &mut pass,
                    &self.sphere_mesh,
                    &self.translucent_instances.buffer,
                    n_trans,
                );
            }
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    fn ensure_attachments(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        if self.target_dims == (w, h) && self.depth_view.is_some() {
            return;
        }
        self.target_dims = (w, h);
        let extent = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };

        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("unrust_scene_depth"),
            size: extent,
            mip_level_count: 1,
            sample_count: self.sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
        self.depth_tex = Some(depth_tex);
        self.depth_view = Some(depth_view);

        if self.sample_count > 1 {
            let msaa_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("unrust_scene_msaa_color"),
                size: extent,
                mip_level_count: 1,
                sample_count: self.sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: self.color_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let msaa_view = msaa_tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.msaa_tex = Some(msaa_tex);
            self.msaa_view = Some(msaa_view);
        } else {
            self.msaa_tex = None;
            self.msaa_view = None;
        }
    }

    fn upload_esdf_slice(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slice: &EsdfSliceData,
    ) {
        let uni = EsdfSliceUniform {
            corner_origin: [
                slice.corner_origin[0],
                slice.corner_origin[1],
                slice.corner_origin[2],
                1.0,
            ],
            axis_u: [slice.axis_u[0], slice.axis_u[1], slice.axis_u[2], 0.0],
            axis_v: [slice.axis_v[0], slice.axis_v[1], slice.axis_v[2], 0.0],
            dims_and_act: [
                slice.width as f32,
                slice.height as f32,
                slice.activation_dist,
                slice.opacity,
            ],
        };
        queue.write_buffer(&self.esdf_uniform_buf, 0, bytemuck::bytes_of(&uni));

        let dims = (slice.width.max(1), slice.height.max(1));
        if self.esdf_tex_dims != dims {
            let (tex, bg) = create_esdf_texture_and_bg(
                device,
                queue,
                &self.esdf_layout,
                &self.esdf_uniform_buf,
                dims.0,
                dims.1,
                &slice.values,
            );
            self.esdf_texture = tex;
            self.esdf_bind_group = bg;
            self.esdf_tex_dims = dims;
        } else if slice.values.len() == (dims.0 * dims.1) as usize {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.esdf_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&slice.values),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(dims.0 * 4),
                    rows_per_image: Some(dims.1),
                },
                wgpu::Extent3d {
                    width: dims.0,
                    height: dims.1,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}

fn draw_mesh_instances<'a>(
    pass: &mut wgpu::RenderPass<'a>,
    mesh: &'a MeshGpu,
    instance_buf: &'a wgpu::Buffer,
    count: u32,
) {
    if count == 0 {
        return;
    }
    pass.set_vertex_buffer(0, mesh.vertex_buf.slice(..));
    pass.set_vertex_buffer(1, instance_buf.slice(..));
    pass.set_index_buffer(mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..mesh.num_indices, 0, 0..count);
}

fn create_esdf_texture_and_bg(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    uniform_buf: &wgpu::Buffer,
    width: u32,
    height: u32,
    values: &[f32],
) -> (wgpu::Texture, wgpu::BindGroup) {
    let w = width.max(1);
    let h = height.max(1);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("unrust_esdf_slice_tex"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    if values.len() == (w * h) as usize {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(values),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("unrust_esdf_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            },
        ],
    });
    (tex, bg)
}

fn create_solid_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
    color_format: wgpu::TextureFormat,
    sample_count: u32,
    translucent: bool,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(if translucent {
            "unrust_solid_translucent_pipe"
        } else {
            "unrust_solid_opaque_pipe"
        }),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            front_face: wgpu::FrontFace::Ccw,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(!translucent),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(if translucent {
                "fs_translucent"
            } else {
                "fs_opaque"
            }),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(if translucent {
                    wgpu::BlendState::ALPHA_BLENDING
                } else {
                    wgpu::BlendState::REPLACE
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn create_lines_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
    color_format: wgpu::TextureFormat,
    sample_count: u32,
    xray: bool,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(if xray {
            "unrust_lines_xray_pipe"
        } else {
            "unrust_lines_depth_pipe"
        }),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(false),
            depth_compare: Some(if xray {
                wgpu::CompareFunction::Always
            } else {
                wgpu::CompareFunction::LessEqual
            }),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn create_esdf_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    color_format: wgpu::TextureFormat,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("unrust_esdf_slice_pipe"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}
