use std::borrow::Cow;
use wgpu::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BufferBindingType, BufferUsages, Limits,
    ShaderStages,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};
use nalgebra::{Matrix4, Vector3};
use winapi::um::winuser::{GetKeyState};
use winit::event::DeviceEvent;
use winit::window::Fullscreen;

mod octree;
mod key_codes;

use crate::key_codes::KeyCode;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
struct UBO {
    transform_matrix: Matrix4<f32>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
struct Player {
    transform_matrix: Matrix4<f32>,
}

impl Player{
    const SPEED: f32 = 5.;
}
async fn run(event_loop: EventLoop<()>, window: Window) {
    let player_transform = Matrix4::identity().append_translation(&Vector3::<f32>::new(0., -5., 0.));

    let mut player_view_direction:(f32, f32) = (0.0, 0.0);

    let mut player = Player {
        transform_matrix: player_transform
    };
    window.set_fullscreen(Some(Fullscreen::Borderless(None)));
    let mut size = window.inner_size();
    size.width = size.width.max(1);
    size.height = size.height.max(1);

    let instance = wgpu::Instance::default();

    let surface = instance
        .create_surface(&window)
        .expect("Failed to create Surface!!");
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                required_limits: Limits::default(),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    // Load the shaders from disk
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Uniform_Buffer"),
        contents: bytemuck::bytes_of(&player),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    let uniform_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("uniform_bind_group_layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<UBO>() as _),
            },
            count: None,
        }],
    });

    let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &uniform_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
        label: Some("uniform_bind_group"),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&uniform_bind_group_layout],
        push_constant_ranges: &[],
    });

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(swapchain_format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();
    surface.configure(&device, &config);

    let window = &window;
    let mut last_time = std::time::SystemTime::now();
    let start_time = std::time::SystemTime::now();
    event_loop
        .run(move |event, target| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            let _ = (&instance, &adapter, &shader, &pipeline_layout);

            match event {
                Event::DeviceEvent{event, .. } => {
                    match event {
                        DeviceEvent::MouseMotion {delta} => {
                            player_view_direction.0 -= (delta.0 as f32 * 0.005) % std::f32::consts::PI;
                            player_view_direction.1 = (player_view_direction.1 + delta.1 as f32 * 0.005).max(-1.4).min(1.4);
                        },
                        _ => {}
                    }
                }
                Event::WindowEvent { window_id: _, event } => {
                    match event {
                        WindowEvent::Resized(new_size) => {
                            // Reconfigure the surface with the new size
                            config.width = new_size.width.max(1);
                            config.height = new_size.height.max(1);
                            surface.configure(&device, &config);
                            window.request_redraw();
                        }
                        WindowEvent::CloseRequested => target.exit(),
                        WindowEvent::RedrawRequested => {
                            let frame = surface
                                .get_current_texture()
                                .expect("Failed to acquire next swap chain texture");
                            let view = frame
                                .texture
                                .create_view(&wgpu::TextureViewDescriptor::default());
                            let mut encoder =
                                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: None,
                                });

                            {
                                let mut rpass =
                                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                        label: None,
                                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                            view: &view,
                                            resolve_target: None,
                                            ops: wgpu::Operations {
                                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                                store: wgpu::StoreOp::Store,
                                            },
                                        })],
                                        depth_stencil_attachment: None,
                                        timestamp_writes: None,
                                        occlusion_query_set: None,
                                    });
                                rpass.set_bind_group(0, &uniform_bind_group, &[]);
                                rpass.set_pipeline(&render_pipeline);
                                rpass.draw(0..3, 0..1);
                            }

                            queue.submit(Some(encoder.finish()));
                            frame.present();
                        },
                        _ => {}
                    }
                }
                Event::AboutToWait => {
                    //Game Update Logic
                    let delta_time = std::time::SystemTime::now().duration_since(last_time).unwrap().as_secs_f32();
                    #[allow(unused_variables)] let time_since_start = std::time::SystemTime::now().duration_since(start_time).unwrap().as_secs_f32();
                    if get_key_down(KeyCode::ESCAPE) {target.exit()};


                    let up = Vector3::<f32>::new(0., -1., 0.);
                    let local_z: Vector3<f32> = player.transform_matrix.try_inverse().unwrap().transform_vector(&Vector3::<f32>::new(0., 0., 1.));

                    let forward = Vector3::<f32>::new(local_z.x, 0., local_z.z).normalize();
                    let right = Vector3::<f32>::new(local_z.z, 0., -local_z.x).normalize();


                    let movement: Vector3<f32> = {
                        let (w, a, s, d, shift, space) = (get_key_down(KeyCode::KEY_W), get_key_down(KeyCode::KEY_A), get_key_down(KeyCode::KEY_S), get_key_down(KeyCode::KEY_D), get_key_down(KeyCode::SHIFT), get_key_down(KeyCode::SPACE));
                        if (w != s) || (a != d) || (shift != space){
                            (Vector3::<f32>::zeros()
                                + if w { forward } else { Vector3::<f32>::zeros() }
                                + if a { -right } else { Vector3::<f32>::zeros() }
                                + if s { -forward } else { Vector3::<f32>::zeros() }
                                + if d { right } else { Vector3::<f32>::zeros() }
                                + if shift { -up } else { Vector3::<f32>::zeros() }
                                + if space { up } else { Vector3::<f32>::zeros() })
                                .normalize() * Player::SPEED * delta_time
                        } else { Vector3::<f32>::zeros() }
                    } as Vector3<f32>;
                    fn create_rotation_mat( angles: (f32, f32)) -> Matrix4<f32>{
                        let rotation_x = Matrix4::new_rotation(Vector3::new(1., 0., 0.) * angles.0);
                        let rotation_y = Matrix4::new_rotation(Vector3::new(0., 1., 0.) * angles.1);
                        rotation_x * rotation_y
                    }
                    player.transform_matrix = create_rotation_mat((player_view_direction.1, player_view_direction.0))
                        .append_translation(&(&(player.transform_matrix.column(3).xyz()) + movement));
                    println!("{}", player.transform_matrix);



                    last_time = std::time::SystemTime::now();
                    queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&player));
                    window.request_redraw();
                }
                _ => {}
            }
        }).unwrap();
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let builder = winit::window::WindowBuilder::new();
    let window = builder.build(&event_loop).unwrap();

    env_logger::init();
    pollster::block_on(run(event_loop, window));
}

fn get_key_down(key: KeyCode) -> bool { unsafe { GetKeyState(key as std::ffi::c_int) < 0 }}



