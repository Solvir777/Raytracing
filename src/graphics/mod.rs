mod storage_buffers;
mod graphics_settings;
pub mod push_constants;
mod shaders;
mod pipelines;

use std::sync::Arc;
use std::time::Duration;
use nalgebra::Vector3;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, BlitImageInfo, BufferImageCopy, ClearColorImageInfo, CommandBufferUsage, CopyBufferToImageInfo, CopyImageToBufferInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, DeviceOwned, Queue, QueueCreateInfo, QueueFlags};
use vulkano::image::view::{ImageView, ImageViewCreateInfo};
use vulkano::image::{Image, ImageAspects, ImageCreateInfo, ImageSubresourceLayers, ImageSubresourceRange, ImageType, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::swapchain::{
    acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::{sync, DeviceSize, Validated, VulkanError, VulkanLibrary};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::format::Format;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use winit::event::{DeviceEvent, Event, MouseScrollDelta, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::wayland::WindowExtWayland;
use winit::platform::x11::{EventLoopWindowTargetExtX11, WindowBuilderExtX11, WindowExtX11};
use winit::window::{CursorGrabMode, Window, WindowBuilder};
use crate::GameState;
use crate::graphics::graphics_settings::GraphicsSettings;
use crate::graphics::pipelines::Pipelines;
use crate::graphics::push_constants::PushConstants;
use crate::graphics::storage_buffers::StorageBuffers;
use crate::terrain::block_ids::BlockType;
use crate::terrain::chunk::Chunk;

fn create_raytrace_descriptor_sets(
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
                    WriteDescriptorSet::image_view(1, ImageView::new_default(buffers.block_type_image.clone()).unwrap()),
                    WriteDescriptorSet::image_view(2, ImageView::new_default(buffers.distance_field_image.clone()).unwrap())
                ],
                [],
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

fn create_swapchain(
    window: &Arc<Window>,
    surface: &Arc<Surface>,
    device: &Arc<Device>,
) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
    let (swapchain, images) = {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format: Format::B8G8R8A8_UNORM,
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

pub struct RenderCore {
    settings: GraphicsSettings,
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    raytrace_descriptorsets: Vec<Arc<PersistentDescriptorSet>>,
    recreate_swapchain: bool,
    render_command_buffer: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
}

pub struct GraphicsCore {
    render_struct: RenderCore,
    pub(crate) device: Arc<Device>,
    queue: Arc<Queue>,
    descriptor_set_allocator: StandardDescriptorSetAllocator,
    command_buffer_allocator: StandardCommandBufferAllocator,
    pub(crate) previous_frame_end: Option<Box<dyn GpuFuture>>,
    pub(crate) buffers: StorageBuffers,
    memory_allocator: Arc<StandardMemoryAllocator>,
    pipelines: Pipelines,
}

impl GraphicsCore {
    pub const CHUNK_SIZE: u32 = 32;
    pub const CHUNK_SIZE_3: u32 = Self::CHUNK_SIZE * Self::CHUNK_SIZE * Self::CHUNK_SIZE;

    pub(crate) fn calculate_distance_field(&mut self, chunk_position: Vector3<i32>) {
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

        let descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            self.pipelines.distance_field_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view(0, ImageView::new_default(self.buffers.block_type_image.clone()).unwrap()),
                WriteDescriptorSet::image_view(1, ImageView::new_default(self.buffers.distance_field_image.clone()).unwrap()),
            ],
            [],
        ).unwrap();

        let clear_color_info = ClearColorImageInfo{
            regions: vec!(
                ImageSubresourceRange{
                    aspects: ImageAspects::COLOR,
                    mip_levels: 0..1,
                    array_layers: 0..1,
                }
            ).into(),
            ..ClearColorImageInfo::image(self.buffers.distance_field_image.clone())
        };

        builder
            .bind_pipeline_compute(
                self.pipelines.distance_field_pipeline.clone(),
            ).unwrap()
            .push_constants(
                self.pipelines.distance_field_pipeline.layout().clone(),
                0,
                chunk_position
            ).unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipelines.distance_field_pipeline.layout().clone(),
                0,
                descriptor_set,
            ).unwrap()
            .clear_color_image(
                clear_color_info
            ).unwrap()
            .dispatch(
                [Self::CHUNK_SIZE; 3]
            ).unwrap();

        let command_buffer = builder.build().unwrap();


        self.previous_frame_end = Some(self
            .previous_frame_end
            .take()
            .unwrap()
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap().boxed());
    }
    pub fn upload_chunk_gpu(&mut self, chunk_pos: Vector3<i32>, game_state: &mut GameState) {
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
            .unwrap();

        let descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            self.pipelines.terrain_generator_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view(0, ImageView::new_default(self.buffers.block_type_image.clone()).unwrap()),
            ],
            [],
        )
            .unwrap();



        builder
            .bind_pipeline_compute(self.pipelines.terrain_generator_pipeline.clone())
            .unwrap()
            .push_constants(
                self.pipelines.terrain_generator_pipeline.layout().clone(),
                0,
                chunk_pos
            ).unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipelines.terrain_generator_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .dispatch([Self::CHUNK_SIZE / 8; 3])
            .unwrap();

        let command_buffer = builder.build().unwrap();

        self.previous_frame_end = Some(self
            .previous_frame_end
            .take()
            .unwrap()
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap().boxed());
    }
    pub(crate) fn new() -> (EventLoop<()>, GraphicsCore) {
        
        let settings = GraphicsSettings::default();

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

        println!("uses xlib:{}", window.xlib_window().is_some());

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


        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(
            device.clone(),
        ));

        let terrain_image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo{
                image_type: ImageType::Dim3d,
                format: Format::R16_UINT,
                extent: [GraphicsCore::CHUNK_SIZE * (2*settings.render_distance + 3) as u32; 3],
                usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo{
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            }
        ).unwrap();

        let distance_field_image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo{
                image_type: ImageType::Dim3d,
                format: Format::R16_UINT,
                extent: [GraphicsCore::CHUNK_SIZE * (2*settings.render_distance + 3) as u32; 3],
                usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo{
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            }
        ).unwrap();

        let buffers = StorageBuffers::new(terrain_image, distance_field_image);
        
        let pipelines = Pipelines::new(device.clone());
        
        let raytrace_descriptorsets =
            create_raytrace_descriptor_sets(&images, &descriptor_set_allocator, &pipelines.raytrace_pipeline, &buffers);

        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());

        let recreate_swapchain = false;
        let previous_frame_end = Some(sync::now(device.clone()).boxed());


        let render_command_buffer = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        ).unwrap();

        (
            event_loop,
            Self {
                render_struct: RenderCore {
                    render_command_buffer,
                    settings,
                    window,
                    swapchain,
                    raytrace_descriptorsets,
                    recreate_swapchain,
                },
                buffers,
                device: device.clone(),
                queue,
                descriptor_set_allocator,
                command_buffer_allocator,
                previous_frame_end,
                memory_allocator,
                pipelines,
            },
        )
    }
    pub fn get_settings(&self) -> GraphicsSettings {
        self.render_struct.settings
    }

    pub fn run(mut self, event_loop: EventLoop<()>, mut game_state: GameState, game_update: fn(&mut GraphicsCore, &mut GameState) -> PushConstants) -> ! {
        event_loop.run(move |event, _, control_flow| {
            match event {
                Event::WindowEvent { event: WindowEvent::CursorEntered { .. }, .. } => {
                    self.render_struct.window.set_cursor_grab(CursorGrabMode::Confined).expect("TODO: panic message");
                    self.render_struct.window.set_cursor_visible(false);
                }
                Event::WindowEvent {event: WindowEvent::KeyboardInput {input, ..}, .. } => {
                    if input.virtual_keycode == Some(VirtualKeyCode::Escape) {
                        *control_flow = ControlFlow::Exit;
                        return;
                    }
                    game_state.key_states.update(input);
                }
                Event::DeviceEvent {event: DeviceEvent::MouseMotion {delta}, ..} => {
                    game_state.player_look_dir(delta, self.render_struct.settings.mouse_sensitivity);
                }
                Event::WindowEvent {event: WindowEvent::MouseWheel {delta, .. }, .. } => {

                    let modifier = match delta{
                        MouseScrollDelta::LineDelta(_, y) => {y}
                        MouseScrollDelta::PixelDelta(delta) => { delta.y as f32}
                    };
                    game_state.speed_modifier = (game_state.speed_modifier + modifier * 0.3).clamp(0.1, 100.);
                }
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
                    self.render_struct.recreate_swapchain = true;
                }
                Event::RedrawEventsCleared => {
                    let push_constants = game_update(&mut self, &mut game_state);
                    if self.redraw(push_constants) { return; }
                    game_state.key_states.reset();
                }
                _ => (),
            }
        });
    }

    pub(crate) fn update_terrain(&mut self, chunk: &Chunk, chunk_position: Vector3<i32>) {
        /*let staging_buffer = self.buffers.get_staging_buffer();
        self.previous_frame_end
            .take()
            .unwrap()
            .join(sync::now(self.device.clone()))
            .flush().unwrap();

        let mut guard = staging_buffer.write().unwrap();

        guard.copy_from_slice(&chunk.data.map(|x|
            if(x == BlockType::Air){0}else{1}
        ));
        drop(guard);
        self.previous_frame_end = Some(Box::new(sync::now(self.device.clone())));

        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

        let dest = Self::chunk_pos_to_image_dest(chunk_position, self.render_struct.settings.render_distance + 1);

        let buffer_image_copy = BufferImageCopy {
            image_subresource: ImageSubresourceLayers{
                aspects: ImageAspects::COLOR,
                mip_level: 0,
                array_layers: 0..1,
            },
            image_offset: [dest.x, dest.y, dest.z],
            image_extent: [Self::CHUNK_SIZE; 3],
            ..Default::default()
        };

        builder
            .copy_buffer_to_image(
                CopyBufferToImageInfo {
                    regions: vec!(buffer_image_copy).into(),
                    ..CopyBufferToImageInfo::buffer_image(staging_buffer, self.buffers.terrain_image.clone())
                }
            ).unwrap();

        let command_buffer = builder.build().unwrap();
        let future = self
            .previous_frame_end.take().unwrap()
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        self.previous_frame_end  = Some(future.boxed());*/
    }

    fn redraw(&mut self, push_constants: PushConstants) -> bool {
        let image_extent: [u32; 2] = self.render_struct.window.inner_size().into();

        if image_extent.contains(&0) {
            return true;
        }
        println!("debug");

        if self.render_struct.recreate_swapchain {
            let (new_swapchain, new_images) = self.render_struct
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent,
                    ..self.render_struct.swapchain.create_info()
                })
                .expect("failed to recreate swapchain");

            self.render_struct.raytrace_descriptorsets = create_raytrace_descriptor_sets(
                &new_images,
                &self.descriptor_set_allocator,
                &self.pipelines.raytrace_pipeline,
                &self.buffers,
            );

            self.render_struct.recreate_swapchain = false;
            self.render_struct.swapchain = new_swapchain;
        }
        println!("debug");

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(self.render_struct.swapchain.clone(), None)
                .map_err(Validated::unwrap)
            {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    self.render_struct.recreate_swapchain = true;
                    return true;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        println!("debug 1");
        if suboptimal {
            self.render_struct.recreate_swapchain = true;
        }
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
            .unwrap();

        println!("debug 1");
        builder
            .bind_pipeline_compute(self.pipelines.raytrace_pipeline.clone())
            .unwrap()
            .push_constants(
                self.pipelines.raytrace_pipeline.layout().clone(),
                0,
                push_constants.transform
            ).unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipelines.raytrace_pipeline.layout().clone(),
                0,
                self.render_struct.raytrace_descriptorsets[image_index as usize].clone(),
            )
            .unwrap()
            .dispatch([image_extent[0] / 16, image_extent[1] / 16, 1])
            .unwrap();

        println!("debug 1");
        let command_buffer = builder.build().unwrap();


        println!("debug a");
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
                    self.render_struct.swapchain.clone(),
                    image_index,
                ),
            )
            .then_signal_fence_and_flush();

        println!("debug 2");
        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                self.render_struct.recreate_swapchain = true;
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                panic!("failed to flush future: {e}");
                // previous_frame_end = Some(sync::now(device.clone()).boxed());
            }
        }

        println!("debug 3");
        false
    }
    fn chunk_pos_to_image_dest(chunk_pos: Vector3<i32>, gen_dist: u8) -> Vector3<u32> {
        chunk_pos.map(|x| x.rem_euclid((gen_dist as i32) * 2 + 1) as u32 * Self::CHUNK_SIZE)
    }
}
