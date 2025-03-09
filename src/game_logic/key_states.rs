use std::collections::HashSet;
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode};

pub struct KeyStates {
    pub currently_pressed: HashSet<VirtualKeyCode>,
    pub new_pressed: HashSet<VirtualKeyCode>
}


impl KeyStates {
    pub fn new() -> Self {
        Self {
            currently_pressed: HashSet::new(),
            new_pressed: HashSet::new(),
        }
    }

    pub fn update(&mut self, input: KeyboardInput) {
        if let Some(keycode) = input.virtual_keycode {
            if input.state == ElementState::Pressed {
                if self.currently_pressed.insert(keycode) {
                    self.new_pressed.insert(input.virtual_keycode.unwrap());
                }
            } else {
                self.currently_pressed.remove(&keycode);
            }
        } else {
            println!("unknown keypress!");
        }
    }

    pub fn reset(&mut self) {
        self.new_pressed.clear();
    }
}