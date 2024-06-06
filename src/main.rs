use crate::graphics_handler::GraphicsHandler;
use crate::input_helper::InputHelper;
use crate::player::Player;
use std::time::SystemTime;
use winit::event::{DeviceEvent, ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;
use crate::terrain_tree::TerrainTree;

mod graphics_handler;
mod input_helper;
mod player;
mod player_settings;
mod time_utility;
mod terrain_tree;

fn main() {
    let terrain_tree = TerrainTree::new_by_function(2);
    let mut timer = time_utility::Timer::new(true, true, true, 0.8);
    let mut player_view_direction = (-std::f32::consts::PI / 4., -std::f32::consts::PI / 6.);
    let mut input_helper = InputHelper::new();
    let mut last_frame_time = SystemTime::now();
    let mut player = Player::new();
    let (mut graphics_handler, event_loop) = GraphicsHandler::setup(terrain_tree);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        
        match event {
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::Key(input) => {
                    if input.virtual_keycode.unwrap() == VirtualKeyCode::Escape && input.state == ElementState::Pressed {control_flow.set_exit()};
                    input_helper.update(input);
                }
                DeviceEvent::MouseMotion { delta } => {
                    player_view_direction.0 -= (delta.0 as f32 * 0.005) % std::f32::consts::PI;
                    player_view_direction.1 = (player_view_direction.1 - delta.1 as f32 * 0.005)
                        .max(-1.4)
                        .min(1.4);
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
                graphics_handler.redraw(player);
            }
            _ => {}
        };
    })
}