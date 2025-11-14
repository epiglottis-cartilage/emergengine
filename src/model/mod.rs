mod vertex;
use crate::{Result, error::ErrorLogger};
use glam::Mat4;
use image::GenericImageView;
use std::{cell::UnsafeCell, collections::HashMap, fmt::Debug, path::Path, sync::Arc};
pub use vertex::{ModelVertex, Vertex};
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub struct Model {
    pub nodes: Vec<Node>,
}
pub struct Node {
    pub name: String,
    pub mesh: Option<Mesh>,
    children: Vec<Node>,
    transform: Mat4,
    transform_acc: Mat4,
}
impl Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("name", &self.name)
            .field("mash", &self.mesh)
            .field("children", &self.children)
            .finish()
    }
}
pub struct Mesh {
    pub name: String,
    pub primitives: Vec<Arc<Primitive>>,
    transform_acc: Mat4,
}
impl Debug for Mesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mesf")
            .field("name", &self.name)
            .field("primitives", &self.primitives)
            .finish()
    }
}
#[derive(Clone)]
pub struct Material {
    pub name: Option<String>,
    pub base_color_texture: Option<Arc<Texture>>,
    pub base_color_factor: [f32; 4],
}
impl Debug for Material {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Material")
            .field("name", &self.name)
            .field("base", &self.base_color_texture)
            .finish()
    }
}
impl Material {
    pub fn new_default(texture: Arc<Texture>) -> Self {
        Self {
            name: Some(String::from("default")),
            base_color_texture: Some(texture),
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
        }
    }
}
#[derive(Clone)]
pub struct Primitive {
    vertex_buf: wgpu::Buffer,
    index_buf: Option<(wgpu::Buffer, u32)>,
    topo: wgpu::PrimitiveTopology,
    pub material: Arc<Material>,
}
impl Debug for Primitive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Primitive")
            .field("material", &self.material)
            .finish()
    }
}
#[derive(Clone)]
pub struct Texture {
    pub name: Option<String>,
    image: Arc<Image>,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    bind_group: wgpu::BindGroup,
}
impl Debug for Texture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Texture").field("name", &self.name).finish()
    }
}
impl Texture {
    pub fn new_default(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let label = "default".to_string();
        const SZIE: u32 = 5;
        let mut rgba = image::ImageBuffer::new(SZIE, SZIE);
        rgba.pixels_mut().enumerate().for_each(|(i, pixel)| {
            *pixel = image::Rgba(if i & 1 == 0 {
                [0u8, 0, 0, 255]
            } else {
                [137, 100, 204, 255]
            });
        });
        let image = Image::from_image(
            device,
            queue,
            &image::DynamicImage::from(rgba),
            Some(&label),
        );
        let view = image
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::MirrorRepeat,
            address_mode_v: wgpu::AddressMode::MirrorRepeat,
            address_mode_w: wgpu::AddressMode::MirrorRepeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });
        Self {
            name: Some("default".into()),
            image: Arc::new(image),
            view,
            sampler,
            bind_group,
        }
    }
}

pub struct Image {
    texture: wgpu::Texture,
}
impl Image {
    pub fn from_file(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        label: Option<&str>,
    ) -> Result<Self> {
        let img = image::load_from_memory(bytes)?;
        Ok(Self::from_image(device, queue, &img, label))
    }
    pub fn from_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &image::DynamicImage,
        label: Option<&str>,
    ) -> Self {
        let mut binding = None;
        let rgba = image.as_rgba8().unwrap_or_else(|| {
            binding = Some(image.to_rgba8());
            binding.as_ref().unwrap()
        });
        let dimensions = image.dimensions();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size_of::<image::Rgba<u8>>() as u32 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            size,
        );
        Self { texture }
    }
}

impl Model {
    pub fn load<P>(
        path: P,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_layout: &wgpu::BindGroupLayout,
        default_texture: Arc<Texture>,
        default_material: Arc<Material>,
    ) -> Result<Vec<Self>>
    where
        P: AsRef<Path>,
    {
        log::info!("Loading model: {}", path.as_ref().display());
        let (document, buffers, images) = gltf::import(path).log()?;
        ModelLoader::new(
            device,
            queue,
            document,
            buffers,
            images,
            texture_layout,
            default_texture,
            default_material,
        )
        .load()
    }
}

struct ModelLoader<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    document: gltf::Document,
    data_buf: Vec<gltf::buffer::Data>,
    image_buf: Vec<gltf::image::Data>,
    // Following parts are unsafe, stay private
    primitives: UnsafeCell<HashMap<usize, Arc<Primitive>>>,
    materials: UnsafeCell<HashMap<Option<usize>, Arc<Material>>>,
    textures: UnsafeCell<HashMap<usize, Arc<Texture>>>,
    images: UnsafeCell<HashMap<usize, Arc<Image>>>,
    // layouts
    texture_layout: &'a wgpu::BindGroupLayout,
    // default
    default_texture: Arc<Texture>,
}
impl<'a> ModelLoader<'a> {
    fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        document: gltf::Document,
        buffers: Vec<gltf::buffer::Data>,
        images: Vec<gltf::image::Data>,
        texture_layout: &'a wgpu::BindGroupLayout,
        default_texture: Arc<Texture>,
        default_material: Arc<Material>,
    ) -> Self {
        let mut materials = HashMap::with_capacity(document.materials().len());
        materials.insert(None, default_material);
        Self {
            primitives: UnsafeCell::new(HashMap::with_capacity(buffers.len())),
            materials: UnsafeCell::new(materials),
            textures: UnsafeCell::new(HashMap::with_capacity(document.textures().len())),
            images: UnsafeCell::new(HashMap::with_capacity(document.images().len())),
            device,
            queue,
            document,
            data_buf: buffers,
            image_buf: images,
            texture_layout,
            default_texture,
        }
    }
    fn load(self) -> Result<Vec<Model>> {
        self.document
            .scenes()
            .map(|s| dbg!(self.load_scene(s)))
            .try_collect()
            .log()
    }
    fn load_scene(&self, scene: gltf::Scene) -> Result<Model> {
        log::debug!("Loading scene {:?}", scene.name());
        let nodes: Vec<Node> = scene
            .nodes()
            .map(|n| self.load_node(n, Mat4::IDENTITY))
            .try_collect()?;
        Ok(Model { nodes })
    }
    fn load_node(&self, node: gltf::Node, mut transform_acc: Mat4) -> Result<Node> {
        log::debug!("Loading node {:?}", node.name());
        let transform = Mat4::from_cols_array_2d(&node.transform().matrix());
        transform_acc = transform_acc.mul_mat4(&transform);
        let children = node
            .children()
            .map(|n| self.load_node(n, transform_acc))
            .try_collect()?;
        let mesh = match node.mesh() {
            None => None,
            Some(m) => Some(self.load_mesh(m, transform_acc)?),
        };
        Ok(Node {
            name: node.name().unwrap_or_default().to_string(),
            mesh,
            children,
            transform,
            transform_acc,
        })
    }
    fn load_mesh(&self, mesh: gltf::Mesh, transform_acc: Mat4) -> Result<Mesh> {
        log::debug!(
            "Loading mesh {:?} with {} primitives",
            mesh.name(),
            mesh.primitives().len()
        );
        let mesh = Mesh {
            name: mesh.name().unwrap_or_default().to_string(),
            primitives: mesh
                .primitives()
                .map(|p| self.load_primitive(p))
                .try_collect()?,
            transform_acc,
        };
        Ok(mesh)
    }
    fn load_primitive(&self, primitive: gltf::Primitive) -> Result<Arc<Primitive>> {
        use std::collections::hash_map::Entry;
        use std::iter::repeat_n;
        let material = self.load_materials(primitive.material())?;
        let primitives = unsafe { self.primitives.as_mut_unchecked() };
        match primitives.entry(primitive.indices().unwrap().index()) {
            Entry::Occupied(e) => Ok(e.get().to_owned()),
            Entry::Vacant(entry) => {
                let reader = primitive.reader(|buffer| Some(&self.data_buf[buffer.index()]));
                let positons = reader.read_positions().ok_or("no positions")?;
                let n = positons.len();

                let normals: Box<dyn ExactSizeIterator<Item = _>> = reader
                    .read_normals()
                    .map(|x| Box::new(x) as _)
                    .unwrap_or(Box::new(repeat_n([0f32, 0., 1.], n)));

                let uv: Box<dyn ExactSizeIterator<Item = _>> = reader
                    .read_tex_coords(0)
                    .map(|x| Box::new(x.into_f32()) as _)
                    .unwrap();
                // .unwrap_or(Box::new(repeat_n([0f32, 0.], n)));

                let vertex_buf =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Vertex Buffer"),
                            contents: bytemuck::cast_slice(
                                positons
                                    .zip(uv)
                                    .zip(normals)
                                    .map(|((position, uv), normal)| ModelVertex {
                                        position,
                                        uv,
                                        normal,
                                    })
                                    .collect::<Vec<_>>()
                                    .as_slice(),
                            ),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                let index_buf = match reader.read_indices() {
                    None => None,
                    Some(indices) => {
                        let indices = indices.into_u32();
                        let len = indices.len();
                        Some((
                            self.device
                                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some(&format!("Index Buffer")),
                                    contents: bytemuck::cast_slice(
                                        indices.collect::<Vec<_>>().as_slice(),
                                    ),
                                    usage: wgpu::BufferUsages::INDEX,
                                }),
                            len as _,
                        ))
                    }
                };
                use gltf::mesh::Mode;
                let topo = match primitive.mode() {
                    Mode::Triangles => wgpu::PrimitiveTopology::TriangleList,
                    // Mode::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
                    _ => unimplemented!(),
                };
                let vertex = Arc::new(Primitive {
                    vertex_buf,
                    index_buf,
                    topo,
                    material: material,
                });
                entry.insert(vertex.clone());
                Ok(vertex)
            }
        }
    }
    fn load_materials(&self, material: gltf::Material) -> Result<Arc<Material>> {
        use std::collections::hash_map::Entry;
        let materials = unsafe { self.materials.as_mut_unchecked() };
        match materials.entry(material.index()) {
            Entry::Occupied(x) => Ok(x.get().clone()),
            Entry::Vacant(entry) => {
                let pbr = material.pbr_metallic_roughness();
                let base_color_factor = pbr.base_color_factor();
                let base_color_texture = match pbr.base_color_texture() {
                    None => Some(self.default_texture.clone()),
                    Some(t) => Some(self.load_texture(t.texture())?),
                };
                let material = Arc::new(Material {
                    name: material.name().map(Into::into),
                    base_color_factor,
                    base_color_texture,
                });
                entry.insert(material.clone());
                Ok(material)
            }
        }
    }
    fn load_texture(&self, texture: gltf::Texture) -> Result<Arc<Texture>> {
        use std::collections::hash_map::Entry;
        let textures = unsafe { self.textures.as_mut_unchecked() };
        match textures.entry(texture.index()) {
            Entry::Occupied(x) => Ok(x.get().clone()),
            Entry::Vacant(entry) => {
                let image = self.load_image(texture.source())?;
                let view = image
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let sampler = self.load_sample(texture.sampler());
                let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: self.texture_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                    ],
                    label: Some("diffuse_bind_group"),
                });
                let texture = Arc::new(Texture {
                    name: texture.name().map(Into::into),
                    image,
                    sampler,
                    view,
                    bind_group,
                });
                entry.insert(texture.clone());
                Ok(texture)
            }
        }
    }
    fn load_image(&self, image: gltf::Image) -> Result<Arc<Image>> {
        use std::collections::hash_map::Entry;
        let images = unsafe { self.images.as_mut_unchecked() };
        match images.entry(image.index()) {
            Entry::Occupied(x) => Ok(x.get().clone()),
            Entry::Vacant(entry) => {
                let data = &self.image_buf[image.index()];
                let img = match data.format {
                    gltf::image::Format::R8G8B8 => image::DynamicImage::from(
                        image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(
                            data.width,
                            data.height,
                            data.pixels.clone(),
                        )
                        .ok_or("E")?,
                    )
                    .to_rgba8(),
                    gltf::image::Format::R8G8B8A8 => {
                        image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
                            data.width,
                            data.height,
                            data.pixels.clone(),
                        )
                        .ok_or("E")?
                    }
                    _ => unimplemented!(),
                };
                let size = wgpu::Extent3d {
                    width: data.width,
                    height: data.height,
                    depth_or_array_layers: 1,
                };

                let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: image.name(),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        aspect: wgpu::TextureAspect::All,
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                    },
                    &img.into_vec().as_slice(),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(size_of::<image::Rgba<u8>>() as u32 * data.width),
                        rows_per_image: Some(data.height),
                    },
                    size,
                );
                let image = Arc::new(Image { texture });
                entry.insert(image.clone());
                Ok(image)
            }
        }
    }
    fn load_sample(&self, sampler: gltf::texture::Sampler) -> wgpu::Sampler {
        let mag_filter = match sampler.mag_filter() {
            Some(f) => match f {
                gltf::texture::MagFilter::Nearest => wgpu::FilterMode::Nearest,
                gltf::texture::MagFilter::Linear => wgpu::FilterMode::Linear,
            },
            None => wgpu::FilterMode::Nearest,
        };
        let min_filter = match sampler.min_filter() {
            Some(f) => match f {
                gltf::texture::MinFilter::Nearest => wgpu::FilterMode::Nearest,
                gltf::texture::MinFilter::Linear => wgpu::FilterMode::Linear,
                _ => wgpu::FilterMode::Linear,
            },
            None => wgpu::FilterMode::Nearest,
        };
        let wrap_s = match sampler.wrap_s() {
            gltf::texture::WrappingMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            gltf::texture::WrappingMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
            gltf::texture::WrappingMode::Repeat => wgpu::AddressMode::Repeat,
        };
        let wrap_t = match sampler.wrap_t() {
            gltf::texture::WrappingMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            gltf::texture::WrappingMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
            gltf::texture::WrappingMode::Repeat => wgpu::AddressMode::Repeat,
        };
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wrap_s,
            address_mode_v: wrap_t,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter,
            min_filter,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        sampler
    }
}

use core::ops::Range;
pub trait DrawModel {
    fn draw_model(&mut self, model: &Model, instances: Range<u32>);
}
impl DrawModel for wgpu::RenderPass<'_> {
    fn draw_model(&mut self, model: &Model, instances: Range<u32>) {
        Draw::draw_model(self, model, instances);
    }
}

trait Draw {
    fn draw_model(&mut self, model: &Model, instances: Range<u32>);
    fn draw_node(&mut self, node: &Node, instances: Range<u32>);
    fn draw_mesh(&mut self, mesh: &Mesh, instances: Range<u32>);
    fn draw_primitive(&mut self, primitive: &Primitive, transform: Mat4, instances: Range<u32>);
    fn bind_material(&mut self, material: &Material);
}
impl Draw for wgpu::RenderPass<'_> {
    fn draw_model(&mut self, model: &Model, instances: Range<u32>) {
        for node in &model.nodes {
            self.draw_node(node, instances.clone());
        }
    }

    fn draw_node(&mut self, node: &Node, instances: Range<u32>) {
        for node in &node.children {
            self.draw_node(node, instances.clone());
        }
        if let Some(mesh) = &node.mesh {
            self.draw_mesh(mesh, instances.clone());
        }
    }

    fn draw_mesh(&mut self, mesh: &Mesh, instances: Range<u32>) {
        for primitive in &mesh.primitives {
            self.draw_primitive(primitive, mesh.transform_acc, instances.clone());
        }
    }

    fn draw_primitive(&mut self, primitive: &Primitive, transform: Mat4, instances: Range<u32>) {
        match &primitive.index_buf {
            None => {
                todo!()
            }
            Some((index_buf, len)) => {
                self.bind_material(&primitive.material);
                self.set_vertex_buffer(0, primitive.vertex_buf.slice(..));
                self.set_index_buffer(index_buf.slice(..), wgpu::IndexFormat::Uint32);
                self.draw_indexed(0..*len, 0, instances);
            }
        }
    }
    fn bind_material(&mut self, material: &Material) {
        if let Some(texture) = &material.base_color_texture {
            self.set_bind_group(0, &texture.bind_group, &[]);
        } else {
            todo!()
        }
    }
}
