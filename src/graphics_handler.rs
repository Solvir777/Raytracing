use vulkano::DeviceSize;
use vulkano::buffer::BufferContents;
use nalgebra::Vector3;
use std::{
	sync::Arc,
	time::SystemTime
};
use std::time::Duration;
use vulkano::{
	buffer::{Buffer, BufferCreateInfo, BufferUsage},
	command_buffer::{
		ClearColorImageInfo,
		CopyBufferToImageInfo,
		allocator::StandardCommandBufferAllocator,
		AutoCommandBufferBuilder,
		CommandBufferUsage
	},
	format::{ClearColorValue, Format},
	image::{
		sampler::{Sampler, SamplerCreateInfo},
		ImageType,
		ImageAspects,
		ImageCreateInfo,
		view::{ImageView, ImageViewCreateInfo},
		Image,
		ImageUsage
	},
	pipeline::{
		layout::PipelineDescriptorSetLayoutCreateInfo,
		compute::ComputePipelineCreateInfo,
		layout::{PipelineLayoutCreateInfo, PushConstantRange},
		ComputePipeline,
		Pipeline,
		PipelineBindPoint,
		PipelineLayout,
		PipelineShaderStageCreateInfo
	},
	descriptor_set::{
		allocator::StandardDescriptorSetAllocator,
		PersistentDescriptorSet, WriteDescriptorSet,
	},
	device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags},
	instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
	memory::{
		allocator::{
			StandardMemoryAllocator,
			AllocationCreateInfo,
			MemoryTypeFilter
		}
	},
	shader::ShaderStages,
	swapchain::{
		acquire_next_image, ColorSpace, Surface, Swapchain, SwapchainCreateFlags,
		SwapchainCreateInfo, SwapchainPresentInfo,
	},
	sync::{
		self,
		GpuFuture
	},
	Validated
};
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract};
use vulkano::device::DeviceOwned;
use vulkano::image::sampler::{Filter, SamplerMipmapMode};
use winit::{event_loop::EventLoop, window::Window, window::WindowBuilder};
use winit::dpi::LogicalPosition;
use winit::window::CursorGrabMode;
use crate::player::Player;
use crate::tree::{CHUNK_SIZE, CHUNK_SIZE_3};
use crate::tree::Tree;
use crate::shaders::df_cs;
use crate::shaders::rendering_cs;


const RENDER_DIST: usize = 3;

#[repr(C)]
#[derive(BufferContents)]
pub struct SetBlockData{
    pos: Vector3::<i32>,
    value: u32,
}
impl SetBlockData {
    pub fn new(pos: Vector3::<i32>, value: u32) -> Self{
        Self{
            pos,
            value
        }
    }
}
pub struct GraphicsHandler {
    window: Arc<Window>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<Image>>,
    render_pipeline: Arc<ComputePipeline>,
    distance_field_pipeline: Arc<ComputePipeline>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    recreate_swapchain: bool,
    memory_allocator: Arc<StandardMemoryAllocator>,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    distance_field_image_view: Vec<Arc<ImageView>>,
}


impl GraphicsHandler {
    pub fn grab_cursor(&self) {
        self.window.set_cursor_grab(CursorGrabMode::Confined).unwrap();
        self.window.set_cursor_visible(false);
    }
    fn setup(
        img_size: (u32, u32),
    ) -> (
        Arc<Window>,
        Arc<Device>,
        Arc<Queue>,
        Arc<Swapchain>,
        Vec<Arc<Image>>,
        Arc<StandardMemoryAllocator>,
        Arc<StandardDescriptorSetAllocator>,
        EventLoop<()>,
    ) {
        let event_loop = EventLoop::new();
        let required_extensions = vulkano::instance::InstanceExtensions {
            ..Surface::required_extensions(&event_loop)
        };
        let library = vulkano::VulkanLibrary::new().expect("no local Vulkan library/DLL");
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..InstanceCreateInfo::default()
            },
        )
        .expect("failed to create instance");

        let physical = instance
            .enumerate_physical_devices()
            .expect("could not enumerate devices")
            .next()
            .expect("no devices available");

        let queue_family_index = physical
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_, q)| q.queue_flags.contains(QueueFlags::GRAPHICS))
            .expect("couldn't find a graphical queue family")
            as u32;

        let mut window = Arc::new(
            WindowBuilder::new()
                .with_inner_size(winit::dpi::LogicalSize::new(img_size.0, img_size.1))
                .build(&event_loop)
                .unwrap(),
        );
        
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            khr_swapchain_mutable_format: true,
            khr_storage_buffer_storage_class: true,
            ..DeviceExtensions::default()
        };

        let (device, mut queues) = Device::new(
            physical.clone(),
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("failed to create device");

        let queue = queues.next().unwrap();

        let (swapchain, images) = {
            let image_format = device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[1]
                .0;
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            Swapchain::new(
                device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT
                        | ImageUsage::TRANSFER_DST
                        | ImageUsage::STORAGE,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    image_color_space: ColorSpace::SrgbNonLinear,
                    flags: SwapchainCreateFlags::MUTABLE_FORMAT,
                    image_view_formats: vec![image_format],
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        (
            window,
            device,
            queue,
            swapchain,
            images,
            memory_allocator,
            descriptor_set_allocator,
            event_loop,
        )
    }

    fn setup_render_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
        let rendering_compute_shader =
            rendering_cs::load(device.clone()).expect("failed to create shader module");

        let render_cs = rendering_compute_shader.entry_point("main").unwrap();
        let render_stage = PipelineShaderStageCreateInfo {
            ..PipelineShaderStageCreateInfo::new(render_cs)
        };
        
        let render_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                push_constant_ranges: vec![PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    offset: 0u32,
                    size: 64u32,
                }],
                ..PipelineDescriptorSetLayoutCreateInfo::from_stages([&render_stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap()
            },
        )
        .unwrap();

        ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(render_stage, render_pipeline_layout),
        )
        .expect("failed to create rendering pipeline")
    }

    fn setup_distance_field(device: Arc<Device>, memory_allocator: Arc<StandardMemoryAllocator>) -> (Vec<Arc<ImageView>>, Arc<ComputePipeline>) {
        let mut df_image_views: Vec<Arc<ImageView>> = vec!();
        for _ in 0..(RENDER_DIST * RENDER_DIST * RENDER_DIST) {
            let distance_field_image_view = {
                let image = Image::new(
                    memory_allocator.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim3d,
                        format: Format::R32_SINT,
                        extent: [CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE],
                        samples: vulkano::image::SampleCount::Sample1,
                        tiling: vulkano::image::ImageTiling::Optimal,
                        usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                        ..Default::default()
                    },
                )
                    .unwrap();
                
                ImageView::new(
                    image.clone(),
                    ImageViewCreateInfo {
                        view_type: vulkano::image::view::ImageViewType::Dim3d,
                        format: Format::R32_SINT,
                        subresource_range: vulkano::image::ImageSubresourceRange {
                            aspects: ImageAspects::COLOR,
                            mip_levels: 0..1,
                            array_layers: 0..1,
                        },
                        usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
                        ..Default::default()
                    },
                )
                    .unwrap()
            };
            df_image_views.push(distance_field_image_view);
        }

        let df_compute_shader = df_cs::load(device.clone()).unwrap();

        let df_cs = df_compute_shader.entry_point("main").unwrap();

        let df_stage = PipelineShaderStageCreateInfo::new(df_cs);

        let df_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&df_stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let df_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(df_stage.clone(), df_pipeline_layout.clone()),
        )
        .expect("failed to create distance field pipeline");
        
        
        (df_image_views, df_pipeline)
    }
    pub fn initialize(img_size: (u32, u32)) -> (GraphicsHandler, EventLoop<()>) {
        let (
            window,
            device,
            queue,
            swapchain,
            images,
            memory_allocator,
            descriptor_set_allocator,
            event_loop,
        ) = Self::setup(img_size);

        let render_pipeline = Self::setup_render_pipeline(device.clone());
        let (df_images, df_pipeline) =
            Self::setup_distance_field(device.clone(), memory_allocator.clone());

        let previous_frame_end = Some(sync::now(device.clone()).boxed());
        let mut s = Self {
            distance_field_image_view: df_images,
            previous_frame_end,
            window,
            device,
            queue,
            swapchain,
            images,
            render_pipeline,
            distance_field_pipeline: df_pipeline,
            descriptor_set_allocator,
            recreate_swapchain: true,
            memory_allocator,
        };
        s.clear_images();
        (
            s,
            event_loop
        )
    }
    fn clear_images(&mut self) {
        for i in 0..(RENDER_DIST * RENDER_DIST * RENDER_DIST) {
            
            let mut builder = AutoCommandBufferBuilder::primary(
                &StandardCommandBufferAllocator::new(
                    self.device.clone(),
                    Default::default()
                ),
                self.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
                .unwrap();
            builder
                .clear_color_image(ClearColorImageInfo {
                    clear_value: ClearColorValue::Int([CHUNK_SIZE as i32; 4]),
                    ..ClearColorImageInfo::image(self.distance_field_image_view[i].image().clone())
                })
                .unwrap();
            
            let command_buffer = builder.build().unwrap();
            let future = sync::now(self.device.clone())
                .then_execute(self.queue.clone(), command_buffer)
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap();
            future.wait(None).unwrap();
        }
    }
    pub fn redraw(&mut self, push_constant_data: Player) {
        if self.recreate_swapchain {
            (self.swapchain, self.images) = self
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent: [
                        self.window.inner_size().width,
                        self.window.inner_size().height,
                    ],
                    ..self.swapchain.create_info()
                })
                .expect("Failed to recreate Swapchain");

            self.recreate_swapchain = false;
        }

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(e) => panic!("failed to acquire next image: {}", e),
            };

        if suboptimal {
            self.recreate_swapchain = true;
        }
        let command_buffer = {
            let command_buffer_allocator = StandardCommandBufferAllocator::new(
                self.render_pipeline.device().clone(),
                Default::default(),
            );

            let mut builder = AutoCommandBufferBuilder::primary(
                &command_buffer_allocator,
                self.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            let render_target = &self.images[image_index as usize];

            let swapchain_image_view = ImageView::new(
                render_target.clone(),
                ImageViewCreateInfo::from_image(&*render_target),
            )
            .unwrap();

	        
	        let textures = self.create_block_textures();
	        
	        let sampler = Sampler::new(self.device.clone(), SamplerCreateInfo{
                min_filter: Filter::Nearest,
                mag_filter: Filter::Nearest,
                mipmap_mode: SamplerMipmapMode::Nearest,
                ..SamplerCreateInfo::simple_repeat_linear()
            }).unwrap();
	        
            let set = PersistentDescriptorSet::new(
                &self.descriptor_set_allocator,
                self.render_pipeline.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::image_view(0, swapchain_image_view),
                    WriteDescriptorSet::image_view_array(1, 0, self.distance_field_image_view.clone()),
	                WriteDescriptorSet::image_view(2, textures),
	                WriteDescriptorSet::sampler(3, sampler),
                ],
                [],
            )
            .unwrap();

            builder
                .bind_pipeline_compute(self.render_pipeline.clone())
                .unwrap()
                .push_constants(self.render_pipeline.layout().clone(), 0, push_constant_data)
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.render_pipeline.layout().clone(),
                    0,
                    set.clone(),
                )
                .unwrap()
                .dispatch([
                    self.window.inner_size().width,
                    self.window.inner_size().height,
                    1,
                ])
                .unwrap();

            builder.build().unwrap()
        };

        let future = sync::now(self.device.clone())
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer.clone())
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();
    }
    pub fn recreate_swapchain(&mut self) {self.recreate_swapchain = true}
    pub fn set_block(&self, info: SetBlockData) {
        let command_buffer_allocator = StandardCommandBufferAllocator::new(
            self.device.clone(),
            Default::default(),
        );
        
        let mut builder = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            self.queue.clone().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
            .unwrap();
        
        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            self.distance_field_pipeline.clone().layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view_array(0, 0, self.distance_field_image_view.clone()),
            ],
            [],
        )
            .unwrap();
        
        builder
            .bind_pipeline_compute(self.distance_field_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.distance_field_pipeline.clone().layout().clone(),
                0,
                set.clone(),
            )
            .unwrap()
            .push_constants(self.distance_field_pipeline.layout().clone(), 0, info)
            .unwrap()
            .dispatch([CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE])
            .unwrap();
        
        let command_buffer = builder.build().unwrap();
        
        let future = sync::now(self.device.clone())
            .then_execute(
                self.queue.clone(),
                command_buffer
            )
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        
        future.wait(None).unwrap();
    }
    

    fn create_block_textures(&mut self) -> Arc<ImageView> {
        let texture = {
            // Replace with your actual image array dimensions.
            let format = Format::R8G8B8A8_SRGB;
            let extent: [u32; 3] = [8, 8, 1];
            let array_layers = 9u32;

            let buffer_size = format.block_size()
                * extent
                    .into_iter()
                    .map(|e| e as vulkano::DeviceSize)
                    .product::<vulkano::DeviceSize>()
                * array_layers as vulkano::DeviceSize;
            let upload_buffer = Buffer::new_slice(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                buffer_size,
            )
            .unwrap();

            {
                let mut image_data = &mut *upload_buffer.write().unwrap();

                for png_bytes in [
                    include_bytes!("textures/grass_side.png").as_slice(),
                    include_bytes!("textures/grass_top.png").as_slice(),
                    include_bytes!("textures/grass_bottom.png").as_slice(),
                    include_bytes!("textures/stone_side.png").as_slice(),
                    include_bytes!("textures/stone_top.png").as_slice(),
                    include_bytes!("textures/stone_bottom.png").as_slice(),
                    include_bytes!("textures/mirror_side.png").as_slice(),
                    include_bytes!("textures/mirror_top.png").as_slice(),
                    include_bytes!("textures/mirror_bottom.png").as_slice(),
                ] {
                    let decoder = png::Decoder::new(png_bytes);
                    let mut reader = decoder.read_info().unwrap();
                    reader.next_frame(image_data).unwrap();
                    let info = reader.info();
                    image_data = &mut image_data[(info.width * info.height * 4) as usize..];
                }
            }

            let image = Image::new(
                self.memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format,
                    extent,
                    array_layers,
                    usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap();

            let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
                self.device.clone(),
                Default::default(),
            ));

            let mut uploads = AutoCommandBufferBuilder::primary(
                &command_buffer_allocator.clone(),
                self.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            uploads
                .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                    upload_buffer,
                    image.clone(),
                ))
                .unwrap();
	        
	        
	        let future = uploads
		        .build()
		        .unwrap()
		        .execute(self.queue.clone()).unwrap().
                then_signal_fence_and_flush().unwrap();
	        future.wait(None).unwrap();
            
            ImageView::new_default(image).unwrap()
        };

        texture
    }
}