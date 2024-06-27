use std::collections::HashSet;
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode};

pub struct InputHelper {
	pressed_keys: HashSet<VirtualKeyCode>
}

impl InputHelper {
	pub fn new() -> Self{
		Self{
			pressed_keys: HashSet::new(),
		}
	}
	pub fn update(&mut self, input: KeyboardInput) {
		match input.state {
			ElementState::Pressed => {
				match input.virtual_keycode {
					Some(keycode) => {
						self.pressed_keys.insert(keycode);
					},
					None => {
						println!("unknown keypress!");
					}
				}
			}
			ElementState::Released => {
				match input.virtual_keycode {
					Some(keycode) => {
						self.pressed_keys.remove(&keycode);
					},
					None => {
						println!("unknown key released!");
					}
				}
			}
		}
	}
	
	pub fn is_pressed(&self, key_code: VirtualKeyCode) -> bool {
		self.pressed_keys.contains(&key_code)
	}
	
	pub fn get_wasd_up_down(&self) -> (bool, bool, bool, bool, bool, bool) {
		(
			self.pressed_keys.contains(&VirtualKeyCode::W),
			self.pressed_keys.contains(&VirtualKeyCode::A),
			self.pressed_keys.contains(&VirtualKeyCode::S),
			self.pressed_keys.contains(&VirtualKeyCode::D),
			self.pressed_keys.contains(&VirtualKeyCode::Space),
			self.pressed_keys.contains(&VirtualKeyCode::LShift),
		)
	}
	
	pub fn time_key_pressed(&self) -> bool {
		self.pressed_keys.contains(&VirtualKeyCode::T)
	}
	pub fn exit_key_pressed(&self) -> bool {
		self.pressed_keys.contains(&VirtualKeyCode::Escape)
	}
	pub fn position_key_pressed(&self) -> bool {
		self.pressed_keys.contains(&VirtualKeyCode::P)
	}
}