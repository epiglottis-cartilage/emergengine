mod camera;
mod error;
mod texture;
mod vertex;
pub use error::{ErrorLogger, Result};
use parking_lot::Mutex;
use std::{sync::Arc, time};
pub use vertex::Vertex;
use wgpu::include_wgsl;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

pub struct App {
    window: Arc<Window>,
    context: RenderContext,
    frame_interval: time::Duration,
    camra_controller: camera::CameraController,
}
struct RenderContext {
    device: wgpu::Device,
    surface: wgpu::Surface<'static>,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    diffuse_bind_group: wgpu::BindGroup,
    camera: camera::Camera,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    size: winit::dpi::PhysicalSize<u32>,
    size_changed: bool,
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
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(vertex::VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(vertex::INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let diffuse = texture::Texture::default(&device, &queue);
        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse.sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        let camera = camera::Camera {
            // 将摄像机向上移动 1 个单位，向后移动 2 个单位
            // +z 朝向屏幕外
            eye: (0.0, 1.0, 2.0).into(),
            // 摄像机看向原点
            target: (0.0, 0.0, 0.0).into(),
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
                buffers: &[Vertex::desc()],
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None, // 5.
            cache: None,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            vertex_buffer,
            index_buffer,
            camera,
            camera_buffer,
            camera_bind_group,
            diffuse_bind_group,
            render_pipeline,
            size,
            size_changed: false,
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
            self.size_changed = false;
        }
    }

    fn render(&mut self) -> Result<()> {
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
                            r: 0.6,
                            g: 0.2,
                            b: 0.3,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..vertex::INDICES.len() as _, 0, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
    }
    fn update_camera(&mut self) {
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
            window,
            frame_interval: time::Duration::from_millis(16),
            camra_controller: camera::CameraController::new(0.4),
        })
    }
}
#[derive(Default)]
pub struct AppHandler {
    app: Arc<Mutex<Option<App>>>,
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
                let wgpu_app = pollster::block_on(App::new(window)).unwrap();
                app.replace(wgpu_app);
            }
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        // 暂停事件
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        use winit::event_loop::ControlFlow;
        let mut app = self.app.lock();
        let app = app.as_mut().unwrap();

        // 窗口事件
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                if physical_size.width == 0 || physical_size.height == 0 {
                    // 处理最小化窗口的事件
                    log::info!("Window minimized!");
                } else {
                    log::info!("Window resized: {:?}", physical_size);

                    app.context.set_window_resized(physical_size);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                app.camra_controller.process_events(&event);
            }
            WindowEvent::RedrawRequested => {
                app.window.pre_present_notify();
                app.camra_controller.update_camera(&mut app.context.camera);
                app.context.update_camera();
                match app.context.render() {
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
