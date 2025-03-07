mod storage_buffers;

use std::sync::Arc;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::image::view::{ImageView, ImageViewCreateInfo};
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::swapchain::{
    acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::{sync, Validated, VulkanError, VulkanLibrary};
use vulkano::format::Format;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};
use crate::graphics::storage_buffers::StorageBuffers;

pub fn create_descriptor_sets(
    images: &Vec<Arc<Image>>,
    descriptor_set_allocator: &StandardDescriptorSetAllocator,
    pipeline: &Arc<ComputePipeline>,
    buffers: &StorageBuffers,
) -> Vec<Arc<PersistentDescriptorSet>> {
    let swapchain_image_views = images
        .iter()
        .map(|x| ImageView::new(x.clone(), ImageViewCreateInfo::from_image(x)).unwrap())
        .collect::<Vec<_>>();

    swapchain_image_views
        .iter()
        .map(|x| {
            PersistentDescriptorSet::new(
                descriptor_set_allocator,
                pipeline.layout().set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::image_view(0, x.clone()),
                    WriteDescriptorSet::image_view(1, buffers.terrain_image.clone()),
                ],
                [],
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

pub fn create_swapchain(
    window: &Arc<Window>,
    surface: &Arc<Surface>,
    device: &Arc<Device>,
) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
    let (swapchain, images) = {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();


        let image_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format,
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::STORAGE,
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),

                ..Default::default()
            },
        )
        .unwrap()
    };
    (swapchain, images)
}

pub fn get_physical_device(
    instance: Arc<Instance>,
    surface: &Arc<Surface>,
    device_extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .expect("no suitable physical device found");
    (physical_device, queue_family_index)
}

pub fn create_compute_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    use crate::shaders;
    use vulkano::pipeline::compute::ComputePipelineCreateInfo;
    use vulkano::pipeline::{PipelineLayout, PipelineShaderStageCreateInfo};

    let compute_shader = shaders::rt_cs::load(device.clone()).unwrap();
    let entry_point = compute_shader.entry_point("main").unwrap();
    let stage_info = PipelineShaderStageCreateInfo::new(entry_point);

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&[stage_info.clone()])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
        .unwrap();

    ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage_info, layout),
    )
    .unwrap()
}

pub struct RenderCore {
    window: Arc<Window>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    descriptor_set_allocator: StandardDescriptorSetAllocator,
    pipeline: Arc<ComputePipeline>,
    descriptorsets: Vec<Arc<PersistentDescriptorSet>>,
    command_buffer_allocator: StandardCommandBufferAllocator,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    buffers: StorageBuffers,
    memory_allocator: Arc<StandardMemoryAllocator>,
}

impl RenderCore {
    pub(crate) fn new() -> (EventLoop<()>, RenderCore) {
        let event_loop = EventLoop::new();

        let library = VulkanLibrary::new().unwrap();

        let required_extensions = Surface::required_extensions(&event_loop);

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) =
            get_physical_device(instance, &surface, &device_extensions);

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],

                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let (swapchain, images) = create_swapchain(&window, &surface, &device);

        let descriptor_set_allocator =
            StandardDescriptorSetAllocator::new(device.clone(), Default::default());

        let pipeline = create_compute_pipeline(device.clone());

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(
            device.clone(),
        ));


        let terrain_image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo{
                image_type: ImageType::Dim3d,
                format: Format::R16_UINT,
                extent: [32, 32, 32],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::STORAGE,
                ..Default::default()
            },
            AllocationCreateInfo{
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            }
        ).unwrap();


        let buffers = StorageBuffers::new(terrain_image);



        let descriptorsets =
            create_descriptor_sets(&images, &descriptor_set_allocator, &pipeline, &buffers);

        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());

        let recreate_swapchain = false;
        let previous_frame_end = Some(sync::now(device.clone()).boxed());

        (
            event_loop,
            Self {
            buffers,
            window,
            device,
            queue,
            swapchain,
            descriptor_set_allocator,
            pipeline,
            descriptorsets,
            command_buffer_allocator,
            recreate_swapchain,
            previous_frame_end,
            memory_allocator,
        },
        )
    }

    pub fn run(mut self, event_loop: EventLoop<()>) -> ! {
        event_loop.run(move |event, _, control_flow| {
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    *control_flow = ControlFlow::Exit;
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(_),
                    ..
                } => {
                    self.recreate_swapchain = true;
                }
                Event::RedrawEventsCleared => {



                    if self.redraw() { return; }
                }
                _ => (),
            }
        });
    }

    fn redraw(&mut self) -> bool {
        let image_extent: [u32; 2] = self.window.inner_size().into();

        if image_extent.contains(&0) {
            return true;
        }

        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if self.recreate_swapchain {
            println!("Recreating swapchain");
            let (new_swapchain, new_images) = self
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent,
                    ..self.swapchain.create_info()
                })
                .expect("failed to recreate swapchain");

            self.descriptorsets = create_descriptor_sets(
                &new_images,
                &self.descriptor_set_allocator,
                &self.pipeline,
                &self.buffers,
            );

            self.recreate_swapchain = false;
            self.swapchain = new_swapchain;
        }

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), None)
                .map_err(Validated::unwrap)
            {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    return true;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        if suboptimal {
            self.recreate_swapchain = true;
        }
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
            .unwrap();

        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                self.descriptorsets[image_index as usize].clone(),
            )
            .unwrap()
            .dispatch([image_extent[0], image_extent[1], 1])
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.swapchain.clone(),
                    image_index,
                ),
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
        false
    }
}
