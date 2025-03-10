use std::time::Instant;
use nalgebra::Matrix4;
use crate::game_logic::key_states::KeyStates;
use crate::game_logic::player::Player;
use crate::tree::Tree;


pub struct GameState {
    player: Player,
    pub(crate) terrain: Tree,
    start_time: Instant,
    pub(crate) key_states: KeyStates,
    pub speed_modifier: f32
}


impl GameState {
    pub(crate) fn player_look_dir(&mut self, delta: (f64, f64), sens: (f32, f32)) {
        self.player.look_direction.0 = self.player.look_direction.0 - delta.0 as f32 * sens.0;
        self.player.look_direction.1 = (self.player.look_direction.1 - delta.1 as f32 * sens.1).clamp(-1.7, 1.7);
    }
    pub(crate) fn new() -> Self {
        GameState {
            player: Player::spawn(),
            terrain: Tree::new(),
            start_time: Instant::now(),
            key_states: KeyStates::new(),
            speed_modifier: 1.0,
        }
    }

    pub fn update_player(&mut self, speed_modifier: f32) {
        self.player.apply_rotation();
        self.player.update_pos(&self.key_states.currently_pressed, speed_modifier);
    }

    pub fn get_player_transform(&self) -> Matrix4<f32> {
        self.player.get_transform()
    }
}