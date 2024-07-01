use crate::graphics_handler::GraphicsHandler;
use crate::input_helper::InputHelper;
use crate::player::Player;
use std::time::SystemTime;
use nalgebra::{Vector3, Vector4};
use winit::event::{DeviceEvent, Event, WindowEvent};
use winit::event_loop::ControlFlow;

mod graphics_handler;
mod input_helper;
mod player;
mod player_settings;
mod time_utility;
mod terrain_tree;
const WINDOW_DIMENSIONS: (u32, u32) = (800, 600);

fn main() {
    //let terrain_tree = TerrainTree::new_by_function(2);
    let mut terrain_pos = Vector3::new(0, 0, 0);
    let mut timer = time_utility::Timer::new(true, true, true, 0.8);
    let mut player_view_direction = (-std::f32::consts::PI / 4., -0.2);
    let mut input_helper = InputHelper::new();
    let mut last_frame_time = SystemTime::now();
    let mut player = Player::new();
    let (mut graphics_handler, event_loop) = GraphicsHandler::initialize(WINDOW_DIMENSIONS);
    

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        
        match event {
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::Key(input) => {
                    input_helper.update(input);
                    if input_helper.exit_key_pressed() {
                        control_flow.set_exit();
                    }
                }
                DeviceEvent::MouseMotion { delta } => {
                    player_view_direction.0 -= (delta.0 as f32 * 0.005) % std::f32::consts::PI;
                    player_view_direction.1 = (player_view_direction.1 - delta.1 as f32 * 0.005)
                        .max(-1.4)
                        .min(1.4);
                },
                DeviceEvent::Button {button, state } => {
                    input_helper.update_mouse(button, state);
                }
                _ => {}
            },
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
                graphics_handler.recreate_swapchain();
            }
            Event::MainEventsCleared => {
                let delta_time = SystemTime::now()
                    .duration_since(last_frame_time)
                    .unwrap()
                    .as_secs_f32();
                last_frame_time = SystemTime::now();
                player.update_player(&input_helper, player_view_direction, delta_time);

                timer.print_status(input_helper.time_key_pressed());
                if input_helper.position_key_pressed() {player.debug()}
                
                if input_helper.mouse_button_down(1) {
                    player.transformation_matrix.try_inverse().unwrap() * Vector4::new(0., 0., 1., 1.);
                    
                    terrain_pos = Vector3::new(1, 1, 0);
                    
                    
                    graphics_handler.update_distance_field(terrain_pos, 1);
                }
                
                graphics_handler.redraw(player);
            }
            _ => {}
        };
    })
}