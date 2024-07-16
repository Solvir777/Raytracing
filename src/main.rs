use crate::graphics_handler::{GraphicsHandler, SetBlockData};
use crate::input_helper::InputHelper;
use crate::player::Player;
use std::time::SystemTime;
use nalgebra::Vector3;
use winit::event::{DeviceEvent, Event, MouseScrollDelta, WindowEvent};
use winit::event_loop::ControlFlow;
use crate::tree::{CHUNK_SIZE, Tree};

mod graphics_handler;
mod input_helper;
mod player;
mod player_settings;
mod time_utility;
mod terrain;
mod tree;
mod shaders;

const WINDOW_DIMENSIONS: (u32, u32) = (800, 600);

fn main() {
    println!("start");
    let mut terrain = Tree::new();
    
    let mut selected_slot = 0;
    let mut player_view_direction = (-std::f32::consts::PI / 4., -0.2);
    let mut input_helper = InputHelper::new();
    let mut player = Player::new();
    let (mut graphics_handler, event_loop) = GraphicsHandler::initialize(WINDOW_DIMENSIONS);
    
    graphics_handler.set_block(SetBlockData::new(Vector3::new(16, 15, 16), 1));
    graphics_handler.set_block(SetBlockData::new(Vector3::new(16, 16, 16), 1));
    
    let mut last_frame_time = SystemTime::now();
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        
        match event {
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseWheel {
                    delta
                } => {
                    match delta {
                        MouseScrollDelta::LineDelta(_, y) => {selected_slot += (y < 0.) as i32 * 2 - 1}
                        MouseScrollDelta::PixelDelta(delta) => {selected_slot += (delta.y < 0.) as i32 * 2 - 1}
                    }
                },
                DeviceEvent::Key(input) => {
                    input_helper.update(input);
                    if input_helper.exit_key_pressed() {
                        control_flow.set_exit();
                    }
                },
                DeviceEvent::MouseMotion { delta } => {
                    player_view_direction.0 -= (delta.0 as f32 * 0.005) % std::f32::consts::PI;
                    player_view_direction.1 = (player_view_direction.1 - delta.1 as f32 * 0.005)
                        .max(-1.4)
                        .min(1.4);
                },
                _ => {}
            },
            Event::WindowEvent {
                event: e,
                ..
            } => {
                match e {
                    WindowEvent::Resized{ .. } => {graphics_handler.recreate_swapchain();}
                    WindowEvent::CloseRequested => {*control_flow = ControlFlow::Exit;}
                    WindowEvent::CursorEntered { .. } => {
                        graphics_handler.grab_cursor();
                    }
                    /*WindowEvent::MouseInput {state, button, ..} => {
                        match (button, state) {
                            (MouseButton::Left, ElementState::Pressed) => {
                                terrain.raycast_destroy_block(&player, &mut graphics_handler);
                            },
                            (MouseButton::Right, ElementState::Pressed) => {
                                terrain.raycast_place_block(&player, (selected_slot % 3 + 1) as u32, &mut graphics_handler);
                            }
                            _ => {}
                        }
                    }*/
                    _ => {}
                }
            }
            Event::MainEventsCleared => {
                let delta_time = SystemTime::now()
                    .duration_since(last_frame_time)
                    .unwrap()
                    .as_secs_f32();
                last_frame_time = SystemTime::now();
                player.update_player(&input_helper, player_view_direction, delta_time);

                if input_helper.position_key_pressed() {player.debug()}
                
                graphics_handler.redraw(player);
            }
            _ => {}
        };
    })
}