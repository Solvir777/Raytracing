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
			ElementState::Pressed => {self.pressed_keys.insert(input.virtual_keycode.unwrap());}
			ElementState::Released => {self.pressed_keys.remove(&input.virtual_keycode.unwrap());}
		}
	}
	
	pub fn is_pressed(&self, key_code: VirtualKeyCode) -> bool {
		self.pressed_keys.contains(&key_code)
	}
	
	pub fn get_wasd_up_down(&self) -> (bool, bool, bool, bool, bool, bool) {
		println!("{:?}", self.pressed_keys.clone());
		(
			self.pressed_keys.contains(&VirtualKeyCode::W),
			self.pressed_keys.contains(&VirtualKeyCode::A),
			self.pressed_keys.contains(&VirtualKeyCode::S),
			self.pressed_keys.contains(&VirtualKeyCode::D),
			self.pressed_keys.contains(&VirtualKeyCode::Space),
			self.pressed_keys.contains(&VirtualKeyCode::LShift),
		)
	}
}