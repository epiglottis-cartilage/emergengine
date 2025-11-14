#![feature(const_trait_impl)]
#![feature(iterator_try_collect)]
#![feature(unsafe_cell_access)]
#![feature(lazy_type_alias)]
// #![allow(dead_code)]
mod camera;
mod error;
mod instance;
pub mod model;
mod texture;
pub use error::{ErrorLogger, Result};
use glam::{Quat, Vec3};
pub use model::*;
use parking_lot::Mutex;
use std::{sync::Arc, time};
use wgpu::include_wgsl;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalPosition,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

use crate::instance::{Instance, LoadedInstance};

pub struct App {
    window: Arc<Window>,
    context: RenderContext,
    models: Vec<(Model, LoadedInstance)>,
    frame_interval: time::Duration,
    camera_controller: camera::CameraLookingAt,
}
struct RenderContext {
    device: wgpu::Device,
    surface: wgpu::Surface<'static>,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    depth_texture: texture::ZbufferTexture,
    camera: camera::Camera,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    size: winit::dpi::PhysicalSize<u32>,
    size_changed: bool,
    default_texture: Arc<Texture>,
    default_material: Arc<Material>,
}
impl RenderContext {
    pub async fn new(window: &Arc<Window>) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).log()?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .log()?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .log()?;
        let mut size = window.inner_size();
        size.width = size.width.max(1);
        size.height = size.height.max(1);

        let cap = surface.get_capabilities(&adapter);
        let config = wgpu::SurfaceConfiguration {
            alpha_mode: if cap
                .alpha_modes
                .contains(&wgpu::CompositeAlphaMode::PreMultiplied)
            {
                wgpu::CompositeAlphaMode::PreMultiplied
            } else {
                wgpu::CompositeAlphaMode::Opaque
            },
            present_mode: wgpu::PresentMode::Fifo,
            ..surface
                .get_default_config(&adapter, size.width, size.height)
                .log()?
        };
        surface.configure(&device, &config);

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let shader = device.create_shader_module(include_wgsl!("shader.wgsl"));

        let camera = camera::Camera {
            // 将摄像机向上移动 1 个单位，向后移动 2 个单位
            // +z 朝向屏幕外
            eye: (0.0, 1.0, 2.0).into(),
            // 摄像机看向原点
            facing: (0.0, 0.0, -1.0).into(),
            // 定义哪个方向朝上
            up: glam::Vec3::Y,
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(
                &camera.build_view_projection_matrix().to_cols_array_2d(),
            ),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX, // 1
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, // 2
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let depth_texture =
            texture::ZbufferTexture::create_depth_texture(&device, &config, "depth_texture");

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
                push_constant_ranges: &[],
            });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                compilation_options: Default::default(),
                entry_point: Some("vs_main"),
                buffers: &[ModelVertex::DESC, Instance::DESC],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                compilation_options: Default::default(),
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // 将此设置为 Fill 以外的任何值都要需要开启 Feature::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // 需要开启 Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // 需要开启 Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::ZbufferTexture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None, // 5.
            cache: None,
        });

        let default_texture = Arc::new(Texture::new_default(
            &device,
            &queue,
            &texture_bind_group_layout,
        ));
        let default_material = Arc::new(Material::new_default(default_texture.clone()));

        Ok(Self {
            surface,
            device,
            queue,
            config,
            texture_bind_group_layout,
            depth_texture,
            camera,
            camera_buffer,
            camera_bind_group,
            render_pipeline,
            size,
            size_changed: false,
            default_texture,
            default_material,
        })
    }
    fn set_window_resized(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if self.size != new_size {
            self.size = new_size;
            self.size_changed = true;
        }
    }
    /// Resizes the surface if the window size has changed.
    fn resize_surface_checked(&mut self) {
        if self.size_changed {
            self.config.width = self.size.width;
            self.config.height = self.size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = texture::ZbufferTexture::create_depth_texture(
                &self.device,
                &self.config,
                "depth_texture",
            );
            self.size_changed = false;
        }
    }
    pub fn load_model<P>(&mut self, path: P) -> Result<Vec<Model>>
    where
        P: AsRef<std::path::Path>,
    {
        model::Model::load(
            path,
            &self.device,
            &self.queue,
            &self.texture_bind_group_layout,
            self.default_texture.clone(),
            self.default_material.clone(),
        )
    }
    pub fn create_instance(&self, instances: Vec<Instance>) -> LoadedInstance {
        LoadedInstance::new(&self.device, instances)
    }

    fn render<'a, I>(&mut self, models: I) -> Result<()>
    where
        I: Iterator<Item = &'a (Model, LoadedInstance)>,
    {
        if self.size.width == 0 || self.size.height == 0 {
            return Ok(());
        }
        self.resize_surface_checked();

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0,
                            g: 1.0,
                            b: 1.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            for (model, instance) in models {
                render_pass.set_vertex_buffer(1, instance.buffer.slice(..));
                render_pass.draw_model(model, 0..instance.instances.len() as _);
            }
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
    }
    fn update_camera_buffer(&mut self) {
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(
                &self
                    .camera
                    .build_view_projection_matrix()
                    .to_cols_array_2d(),
            ),
        );
    }
}

impl App {
    async fn new(window: Arc<Window>) -> Result<Self> {
        Ok(Self {
            context: RenderContext::new(&window).await?,
            models: Vec::new(),
            window,
            frame_interval: time::Duration::from_millis(16),
            camera_controller: camera::CameraLookingAt::new(0.2),
        })
    }
    pub fn load_model<P>(&mut self, path: P) -> Result<()>
    where
        P: AsRef<std::path::Path>,
    {
        let models = self.context.load_model(path)?;
        let instance = Instance {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        };
        self.models = models
            .into_iter()
            .map(|m| (m, self.context.create_instance(vec![instance])))
            .collect();
        Ok(())
    }
}
#[derive(Default)]
pub struct AppHandler {
    app: Arc<Mutex<Option<App>>>,
    cursor_position: Option<LogicalPosition<f64>>,
}

impl ApplicationHandler for AppHandler {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // 恢复事件
        let mut app = self.app.as_ref().lock();
        match *app {
            Some(_) => {}
            ref mut app @ None => {
                let window_attributes = Window::default_attributes()
                    .with_title("showsomething")
                    .with_transparent(true);

                let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
                let mut wgpu_app = pollster::block_on(App::new(window)).unwrap();
                wgpu_app.load_model("./resource/models/2.glb").unwrap();
                app.replace(wgpu_app);
            }
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {}

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        use winit::event_loop::ControlFlow;
        let mut app = self.app.lock();
        let app = app.as_mut().unwrap();

        match event {
            WindowEvent::CursorMoved { position, .. } => match self.cursor_position {
                Some(ref mut p) => {
                    let new_p = position.to_logical(app.window.scale_factor());
                    app.camera_controller
                        .process_mouse(new_p.x - p.x, new_p.y - p.y);
                    *p = new_p;
                }
                None => {
                    self.cursor_position = Some(position.to_logical(app.window.scale_factor()));
                }
            },
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                if physical_size.width == 0 || physical_size.height == 0 {
                    log::info!("Window minimized!");
                } else {
                    log::info!("Window resized: {:?}", physical_size);
                    app.context.set_window_resized(physical_size);
                    app.context.camera.aspect =
                        physical_size.width as f32 / physical_size.height as f32;
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                app.camera_controller.process_keyevents(&event);
            }
            WindowEvent::RedrawRequested => {
                app.window.pre_present_notify();
                app.camera_controller.update_camera(&mut app.context.camera);
                app.context.update_camera_buffer();
                match app.context.render(app.models.iter().filter(|_| true)) {
                    Ok(_) => {}
                    // 当展示平面的上下文丢失，就需重新配置
                    // Err(wgpu::SurfaceError::Lost) => eprintln!("Surface is lost"),
                    // 所有其他错误（过期、超时等）应在下一帧解决
                    Err(e) => eprintln!("{e:?}"),
                }
                event_loop.set_control_flow(ControlFlow::wait_duration(app.frame_interval));
                app.window.request_redraw();
            }
            _ => (),
        }
    }
}
