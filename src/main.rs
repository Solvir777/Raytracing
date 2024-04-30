use std::time::SystemTime;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;
use crate::graphics_handler::GraphicsHandler;

mod graphics_handler;

fn main() {
    let start_time = SystemTime::now();
    let (mut graphics_handler, event_loop) = GraphicsHandler::setup();
    
    event_loop.run(move |event, _, control_flow|
    match event{
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
            graphics_handler.recreate_swapchain = true;
        }
        Event::RedrawEventsCleared => {
            let time = SystemTime::now().duration_since(start_time).unwrap().as_secs_f32();
            println!("time: {}", time);
            graphics_handler.redraw();
        }
        _ => {}
    });
}
