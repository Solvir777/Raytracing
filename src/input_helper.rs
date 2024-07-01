use std::collections::HashSet;
use winit::event::{ButtonId, ElementState, KeyboardInput, VirtualKeyCode};

pub struct InputHelper {
	pressed_keys: HashSet<VirtualKeyCode>,
	mouse_buttons: [bool; 4],
}

impl InputHelper {
	pub fn new() -> Self{
		Self{
			pressed_keys: HashSet::new(),
			mouse_buttons: [false; 4],
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
	pub fn update_mouse(&mut self, b: ButtonId, s: ElementState) {
		self.mouse_buttons[b as usize] = s == ElementState::Pressed;
	}
	
	pub fn mouse_button_down(&self, id: usize) -> bool {
		self.mouse_buttons[id]
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