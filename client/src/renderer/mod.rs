use std::{collections::BTreeMap, mem::size_of, sync::Arc};

use nalgebra::Matrix4;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo,
    },
    descriptor_set::{
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, DeviceOwned, Features, Queue, QueueCreateInfo,
        QueueFlags,
    },
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::{InputAssemblyState},
            multisample::MultisampleState,
            rasterization::{CullMode, FrontFace, RasterizationState},
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::{
            PipelineLayoutCreateInfo, PushConstantRange,
        }, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::{EntryPoint, ShaderStages},
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    Validated, VulkanError, VulkanLibrary,
};

use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

mod lighting;
mod mesh;

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct UniformBufferContents {
    #[format(R32G32_SFLOAT)]
    pub projection_matrix: [[f32; 4]; 4],
}

#[allow(dead_code)]
pub struct Renderer {
    library: Arc<VulkanLibrary>,
    required_extensions: InstanceExtensions,
    instance: Arc<Instance>,
    surface: Arc<Surface>,
    physical_device: Arc<PhysicalDevice>,
    queue_family_index: u32,
    device: Arc<Device>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<Image>>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    vertex_buffer: Subbuffer<[mesh::VertexBufferItem]>,
    face_buffer: Subbuffer<[lighting::LightMapEntry]>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    descriptor_set: Arc<PersistentDescriptorSet>,
    framebuffers: Vec<Arc<Framebuffer>>,
    command_buffer_allocator: StandardCommandBufferAllocator,

    vs: EntryPoint,
    fs: EntryPoint,

    recreate_swapchain: bool,
    previous_frame_end: Option<Box<(dyn GpuFuture + 'static)>>,
}

impl Renderer {
    pub fn new(
        display: &impl HasRawDisplayHandle,
        window: Arc<impl HasRawWindowHandle + HasRawDisplayHandle + std::any::Any + Send + Sync>,
        inner_size: [u32; 2],
        object: &crate::world::Object<bool>,
    ) -> Result<Renderer, Box<dyn std::error::Error>> {
        let library = VulkanLibrary::new().unwrap();
        // The first step of any Vulkan program is to create an instance.
        //
        // When we create an instance, we have to pass a list of extensions that we want to enable.
        //
        // All the window-drawing functionalities are part of non-core extensions that we need to
        // enable manually. To do so, we ask `Surface` for the list of extensions required to draw to
        // a window.
        let required_extensions = Surface::required_extensions(display);

        // Now creating the instance.
        let instance = Instance::new(
            library.clone(),
            InstanceCreateInfo {
                // Enable enumerating devices that use non-conformant Vulkan implementations.
                // (e.g. MoltenVK)
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

        // Choose device extensions that we're going to use. In order to present images to a surface,
        // we need a `Swapchain`, which is provided by the `khr_swapchain` extension.
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        // We then choose which physical device to use. First, we enumerate all the available physical
        // devices, then apply filters to narrow them down to those that can support our needs.
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| {
                // Some devices may not support the extensions or features that your application, or
                // report properties and limits that are not sufficient for your application. These
                // should be filtered out here.
                p.supported_extensions().contains(&device_extensions)
            })
            .filter_map(|p| {
                // For each physical device, we try to find a suitable queue family that will execute
                // our draw commands.
                //
                // Devices can provide multiple queues to run commands in parallel (for example a draw
                // queue and a compute queue), similar to CPU threads. This is something you have to
                // have to manage manually in Vulkan. Queues of the same type belong to the same queue
                // family.
                //
                // Here, we look for a single queue family that is suitable for our purposes. In a
                // real-world application, you may want to use a separate dedicated transfer queue to
                // handle data transfers in parallel with graphics operations. You may also need a
                // separate queue for compute operations, if your application uses those.
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        // We select a queue family that supports graphics operations. When drawing to
                        // a window surface, as we do in this example, we also need to check that
                        // queues in this queue family are capable of presenting images to the surface.
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    // The code here searches for the first queue family that is suitable. If none is
                    // found, `None` is returned to `filter_map`, which disqualifies this physical
                    // device.
                    .map(|i| (p, i as u32))
            })
            // All the physical devices that pass the filters above are suitable for the application.
            // However, not every device is equal, some are preferred over others. Now, we assign each
            // physical device a score, and pick the device with the lowest ("best") score.
            //
            // In this example, we simply select the best-scoring device to use in the application.
            // In a real-world setting, you may want to use the best-scoring device only as a "default"
            // or "recommended" device, and let the user choose the device themself.
            .min_by_key(|(p, _)| {
                // We assign a lower score to device types that are likely to be faster/better.
                match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                    _ => 5,
                }
            })
            .expect("no suitable physical device found");

        // Some little debug infos.
        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        // Now initializing the device. This is probably the most important object of Vulkan.
        //
        // An iterator of created queues is returned by the function alongside the device.
        let (device, mut queues) = Device::new(
            // Which physical device to connect to.
            physical_device.clone(),
            DeviceCreateInfo {
                // A list of optional features and extensions that our program needs to work correctly.
                // Some parts of the Vulkan specs are optional and must be enabled manually at device
                // creation. In this example the only thing we are going to need is the `khr_swapchain`
                // extension that allows us to draw to a window.
                enabled_extensions: device_extensions,

                // The list of queues that we are going to use. Here we only use one queue, from the
                // previously chosen queue family.
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],

                enabled_features: {
                    let mut features = Features::empty();
                    features.fill_mode_non_solid = true;
                    features
                },

                ..Default::default()
            },
        )
        .unwrap();

        // Since we can request multiple queues, the `queues` variable is in fact an iterator. We only
        // use one queue here, so we just retrieve the first and only element of the iterator.
        let queue = queues.next().unwrap();

        // Before we can draw on the surface, we have to create what is called a swapchain. Creating a
        // swapchain allocates the color buffers that will contain the image that will ultimately be
        // visible on the screen. These images are returned alongside the swapchain.
        let (swapchain, images) = {
            // Querying the capabilities of the surface. When we create the swapchain we can only pass
            // values that are allowed by the capabilities.
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            // Choosing the internal format that the images will have.
            let image_formats = device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap();
            let mut image_format = image_formats[0].0;
            dbg!(&image_formats);
            for (other_format, _color_space) in image_formats {
                if other_format == Format::A2R10G10B10_UNORM_PACK32 {
                    image_format = other_format;
                }
            }

            println!("Using image format {image_format:?}");

            Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    // Some drivers report an `min_image_count` of 1, but fullscreen mode requires at
                    // least 2. Therefore we must ensure the count is at least 2, otherwise the program
                    // would crash when entering fullscreen mode on those drivers.
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: inner_size,
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo {
                ..Default::default()
            },
        ));

        mod vs {
            vulkano_shaders::shader! {
                ty: "vertex",
                path: "src/renderer/shaders/vert.glsl",
            }
        }

        mod fs {
            vulkano_shaders::shader! {
                ty: "fragment",
                path: "src/renderer/shaders/frag.glsl",
            }
        }

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                depth_stencil: {
                    format: Format::D16_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {depth_stencil},
            },
        )
        .unwrap();

        let vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let (pipeline, framebuffers) = Self::window_size_dependent_setup(
            memory_allocator.clone(),
            vs.clone(),
            fs.clone(),
            &images,
            render_pass.clone(),
        );

        let (vertex_buffer, face_buffer) = Self::build_object(&memory_allocator, object);

        let descriptor_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::buffer(0, face_buffer.clone())],
            [],
        )
        .unwrap();

        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());

        Ok(Renderer {
            recreate_swapchain: false,
            previous_frame_end: Some(sync::now(device.clone()).boxed()),
            library,
            required_extensions,
            instance,
            surface,
            physical_device,
            queue_family_index,
            device,
            queue,
            swapchain,
            images,
            memory_allocator,
            vertex_buffer,
            face_buffer,
            descriptor_set,
            render_pass,
            pipeline,
            framebuffers,
            command_buffer_allocator,
            vs,
            fs,
        })
    }

    fn perspective(size: [u32; 2]) -> Matrix4<f32> {
        let aspect = size[0] as f32 / size[1] as f32;
        let fovy = 90.0f32.to_radians();
        let near = 0.1;

        let focal_length = 1.0 / (fovy / 2.0).tan();

        #[rustfmt::skip]
        let projection_matrix = Matrix4::from_row_slice(&[
            focal_length / aspect, 0.0, 0.0, 0.0,
            0.0, -focal_length, 0.0, 0.0,
            0.0, 0.0, 0.0, near,
            0.0, 0.0, -1.0, 0.0,
        ]);

        projection_matrix
    }

    fn build_object(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        object: &crate::world::Object<bool>,
    ) -> (
        Subbuffer<[mesh::VertexBufferItem]>,
        Subbuffer<[lighting::LightMapEntry]>,
    ) {
        let mut faces = Vec::new();
        let vertices = mesh::mesh(
            object,
            |_, _, _, _, _| {},
            |x, y, z, face, vertices| {
                for vertex in vertices {
                    vertex.set_data(faces.len() as u32);
                }
                let entry = lighting::light_face(x as isize, y as isize, z as isize, face, object);
                faces.push(entry);
            },
            |_, _, _, _| {},
        );

        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();

        let face_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            faces,
        )
        .unwrap();

        (vertex_buffer, face_buffer)
    }

    pub fn draw(&mut self, image_extent: [u32; 2], view: Matrix4<f32>) {
        // It is important to call this function from time to time, otherwise resources
        // will keep accumulating and you will eventually reach an out of memory error.
        // Calling this function polls various fences in order to determine what the GPU
        // has already processed, and frees the resources that are no longer needed.
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        // Whenever the window resizes we need to recreate everything dependent on the
        // window size. In this example that includes the swapchain, the framebuffers and
        // the dynamic state viewport.
        if self.recreate_swapchain {
            let (new_swapchain, new_images) = self
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent,
                    ..self.swapchain.create_info()
                })
                .expect("failed to recreate swapchain");

            self.swapchain = new_swapchain;
            let (new_pipeline, new_framebuffers) = Self::window_size_dependent_setup(
                self.memory_allocator.clone(),
                self.vs.clone(),
                self.fs.clone(),
                &new_images,
                self.render_pass.clone(),
            );
            self.pipeline = new_pipeline;
            self.framebuffers = new_framebuffers;
            self.recreate_swapchain = false;
        }

        // Before we can draw on the output, we have to *acquire* an image from the
        // swapchain. If no image is available (which happens if you submit draw commands
        // too quickly), then the function will block. This operation returns the index of
        // the image that we are allowed to draw upon.
        //
        // This function can block if no image is available. The parameter is an optional
        // timeout after which the function call will return an error.
        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        // `acquire_next_image` can be successful, but suboptimal. This means that the
        // swapchain image will still work, but it may not display correctly. With some
        // drivers this can be when the window resizes, but it may not cause the swapchain
        // to become out of date.
        if suboptimal {
            self.recreate_swapchain = true;
        }

        // In order to draw, we have to build a *command buffer*. The command buffer object
        // holds the list of commands that are going to be executed.
        //
        // Building a command buffer is an expensive operation (usually a few hundred
        // microseconds), but it is known to be a hot path in the driver and is expected to
        // be optimized.
        //
        // Note that we have to pass a queue family when we create the command buffer. The
        // command buffer will only be executable on that given queue family.
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some([128.0 / 255.0, 218.0 / 255.0, 235.0 / 255.0, 1.0].into()),
                        Some(1f32.into()),
                    ],
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index as usize].clone(),
                    )
                },
                Default::default(),
            )
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                self.descriptor_set.clone(),
            )
            .unwrap()
            .push_constants(
                self.pipeline.layout().clone(),
                0,
                UniformBufferContents {
                    projection_matrix: (Self::perspective(image_extent) * view).into(),
                },
            )
            .unwrap()
            .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap()
            .end_render_pass(Default::default())
            .unwrap();

        // Finish building the command buffer by calling `build`.
        let command_buffer = builder.build().unwrap();

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            // This function does not actually present the image immediately. Instead it
            // submits a present command at the end of the queue. This means that it will
            // only be presented once the GPU has finished executing the command buffer
            // that draws the triangle.
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                self.recreate_swapchain = true;
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                panic!("failed to flush future: {e}");
                // previous_frame_end = Some(sync::now(device.clone()).boxed());
            }
        }
    }

    pub fn recreate_swapchain(&mut self) {
        self.recreate_swapchain = true;
    }

    /// This function is called once during initialization, then again whenever the window is resized.
    fn window_size_dependent_setup(
        memory_allocator: Arc<StandardMemoryAllocator>,
        vs: EntryPoint,
        fs: EntryPoint,
        images: &[Arc<Image>],
        render_pass: Arc<RenderPass>,
    ) -> (Arc<GraphicsPipeline>, Vec<Arc<Framebuffer>>) {
        let device = memory_allocator.device().clone();
        let extent = images[0].extent();

        let depth_buffer = ImageView::new_default(
            Image::new(
                memory_allocator,
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::D16_UNORM,
                    extent: images[0].extent(),
                    usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        let framebuffers = images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view, depth_buffer.clone()],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let pipeline = {
            let vertex_input_state = mesh::VertexBufferItem::per_vertex()
                .definition(&vs.info().input_interface)
                .unwrap();
            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineLayoutCreateInfo {
                    push_constant_ranges: vec![PushConstantRange {
                        stages: ShaderStages::VERTEX,
                        offset: 0,
                        size: size_of::<UniformBufferContents>() as u32,
                    }],
                    set_layouts: vec![DescriptorSetLayout::new(
                        device.clone(),
                        DescriptorSetLayoutCreateInfo {
                            bindings: BTreeMap::from([(
                                0,
                                DescriptorSetLayoutBinding {
                                    stages: ShaderStages::VERTEX,
                                    ..DescriptorSetLayoutBinding::descriptor_type(
                                        DescriptorType::StorageBuffer,
                                    )
                                },
                            )]),
                            ..Default::default()
                        },
                    )
                    .unwrap()],
                    ..Default::default()
                },
            )
            .unwrap();
            let subpass = Subpass::from(render_pass, 0).unwrap();

            GraphicsPipeline::new(
                device,
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState {
                        viewports: [Viewport {
                            offset: [0.0, 0.0],
                            extent: [extent[0] as f32, extent[1] as f32],
                            depth_range: 1.0..=0.0,
                        }]
                        .into_iter()
                        .collect(),
                        ..Default::default()
                    }),
                    rasterization_state: Some(RasterizationState {
                        cull_mode: CullMode::Back,
                        front_face: FrontFace::Clockwise,
                        //polygon_mode: PolygonMode::Line,
                        ..Default::default()
                    }),
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    }),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        (pipeline, framebuffers)
    }
}
