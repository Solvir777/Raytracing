use std::f32::consts::PI;
use std::time::Instant;
use nalgebra::{Matrix4, Vector3};
use winit::event::KeyboardInput;
use crate::game_logic::game_state::GameState;
use crate::game_logic::key_states::KeyStates;
use crate::game_logic::player::Player;
use crate::graphics::push_constants::PushConstants;
use crate::graphics::RenderCore;
use crate::tree::Tree;

mod shaders;
mod graphics;
mod tree;
mod game_logic;



fn main() {
    let (event_loop, mut core) = graphics::RenderCore::new();
    let mut game_state = GameState::new();

    generate_spawn(&mut core);

    let data = game_state.terrain.get_chunk(Vector3::new(0, 0, 0) as Vector3<i32>);

    core.update_terrain(data, Vector3::new(0, 0, 0));

    core.run(event_loop, game_state, move |core, game_state| {
        game_state.update_player();
        return PushConstants{
            transform: game_state.get_player_transform()
        }
    });
}

fn generate_spawn(core: &mut RenderCore) {

}
