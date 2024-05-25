use std::{
	sync::Arc
};

use vulkano::{
	sync,
	Validated,
	command_buffer::{
		AutoCommandBufferBuilder,
		CommandBufferUsage,
		allocator::StandardCommandBufferAllocator
	},
	descriptor_set::{
		PersistentDescriptorSet,
		WriteDescriptorSet,
		allocator::StandardDescriptorSetAllocator,
		layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateFlags, DescriptorSetLayoutCreateInfo, DescriptorType}
	},
	device::{
		Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
	},
	image::{
		Image,
		ImageUsage,
		view::{ImageView, ImageViewCreateInfo}
	},
	instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
	memory::allocator::StandardMemoryAllocator,
	pipeline::{
		ComputePipeline,
		Pipeline,
		PipelineBindPoint,
		PipelineLayout,
		PipelineShaderStageCreateInfo,
		compute::ComputePipelineCreateInfo,
		layout::{PipelineLayoutCreateInfo, PushConstantRange}
	},
	shader::ShaderStages,
	swapchain::{acquire_next_image, ColorSpace, Surface, Swapchain, SwapchainCreateFlags, SwapchainCreateInfo, SwapchainPresentInfo},
	sync::GpuFuture,
	buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
	memory::allocator::{AllocationCreateInfo, MemoryTypeFilter}
};
use winit::{
	event_loop::EventLoop,
	window::Window,
	window::WindowBuilder
};

use crate::player::Player;
use crate::terrain_tree::TerrainTree;

mod cs {
	vulkano_shaders::shader! {
        ty: "compute",
        path: r"cs.glsl",
    }
}

pub struct GraphicsHandler{
	window: Arc<Window>,
	device: Arc<Device>,
	queue: Arc<Queue>,
	swapchain: Arc<Swapchain>,
	images: Vec<Arc<Image>>,
	compute_pipeline: Arc<ComputePipeline>,
	descriptor_set_allocator: StandardDescriptorSetAllocator,
	recreate_swapchain: bool,
	memory_allocator: Arc<StandardMemoryAllocator>,
	terrain_buffer: Subbuffer<[u32]>,
	previous_frame_end: Option<Box<dyn GpuFuture>>,
}


impl GraphicsHandler{
	pub fn setup(terrain: TerrainTree) -> (GraphicsHandler, EventLoop<()>) {
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
			.expect("couldn't find a graphical queue family") as u32;
		
		let window = Arc::new(
			WindowBuilder::new()
				.with_inner_size(winit::dpi::LogicalSize::new(800, 600))
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
		
		let compute_shader = cs::load(device.clone()).expect("failed to create shader module");
		
		let cs = compute_shader.entry_point("main").unwrap();
		let stage = PipelineShaderStageCreateInfo {
			..PipelineShaderStageCreateInfo::new(cs)
		};
		
		let bindings: std::collections::BTreeMap<u32, DescriptorSetLayoutBinding> = {
			let mut bindings = std::collections::BTreeMap::default();
			bindings.insert(0, DescriptorSetLayoutBinding {
				stages: ShaderStages::COMPUTE,
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
			});
			bindings.insert(1, DescriptorSetLayoutBinding {
				stages: ShaderStages::COMPUTE,
				..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
			});
			bindings
		};
		
		
		
		let pipeline_layout = PipelineLayout::new(
			device.clone(),
			PipelineLayoutCreateInfo{
				set_layouts: vec![
					DescriptorSetLayout::new(
						device.clone(),
						DescriptorSetLayoutCreateInfo{
							flags: DescriptorSetLayoutCreateFlags::empty(),
							bindings,
							..Default::default()
						}
					).expect("Failed to create DescriptorSetLayout")
				],
				push_constant_ranges: vec!(
					PushConstantRange{
						stages: ShaderStages::COMPUTE,
						offset: 0u32,
						size: 64u32,
					}
				),
				..Default::default()
			}
		)
			.unwrap();
		
		let compute_pipeline = ComputePipeline::new(
			device.clone(),
			None,
			ComputePipelineCreateInfo {
				..ComputePipelineCreateInfo::stage_layout(stage, pipeline_layout)
			},
		)
			.expect("failed to create compute pipeline");
		
		let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone(), Default::default());
		
		let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
		
		
		let terrain_data = terrain.get_gpu_data();
		println!("full_gpu: {:?}", terrain_data);
		
		let terrain_buffer: Subbuffer<[u32]> = Buffer::from_iter(
			memory_allocator.clone(),
			BufferCreateInfo{
				usage: BufferUsage::STORAGE_BUFFER,
				..Default::default()
			},
			AllocationCreateInfo{
				memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_RANDOM_ACCESS,
				..Default::default()
			},
			terrain_data.iter().cloned()
		).unwrap();
		
		let previous_frame_end = Some(sync::now(device.clone()).boxed());
		
		(
			Self{
				previous_frame_end,
				window,
				device,
				queue,
				swapchain,
				images,
				compute_pipeline,
				descriptor_set_allocator,
				recreate_swapchain: true,
				memory_allocator,
				terrain_buffer
			},
			event_loop
		)
	}
	
	pub fn redraw(&mut self, push_constant_data: Player){
		
		if self.recreate_swapchain {
			(self.swapchain, self.images) = self.swapchain
				.recreate(SwapchainCreateInfo {
					image_extent: [self.window.inner_size().width, self.window.inner_size().height],
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
				self.compute_pipeline.device().clone(),
				Default::default(),
			);
			
			let mut builder = AutoCommandBufferBuilder::primary(
				&command_buffer_allocator,
				self.queue.queue_family_index(),
				CommandBufferUsage::MultipleSubmit,
			).unwrap();
			
			
			let img = &self.images[image_index as usize];
			
			let swapchain_image_view =
				ImageView::new(img.clone(), ImageViewCreateInfo::from_image(&*img)).unwrap();
			
			let set = PersistentDescriptorSet::new(
				&self.descriptor_set_allocator,
				self.compute_pipeline.layout().set_layouts()[0].clone(),
				[
					WriteDescriptorSet::image_view(0, swapchain_image_view),
					WriteDescriptorSet::buffer(1, self.terrain_buffer.clone()),
				],
				[],
			)
				.unwrap();
			
			builder
				.bind_pipeline_compute(self.compute_pipeline.clone())
				.unwrap()
				.push_constants(self.compute_pipeline.layout().clone(), 0, push_constant_data)
				.unwrap()
				.bind_descriptor_sets(
					PipelineBindPoint::Compute,
					self.compute_pipeline.layout().clone(),
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
	pub fn recreate_swapchain(&mut self) {
		self.recreate_swapchain = true;
	}
}