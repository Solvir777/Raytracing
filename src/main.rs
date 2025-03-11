use std::f32::consts::PI;
use std::time::Instant;
use nalgebra::{Matrix4, Vector3};
use winit::event::{KeyboardInput, VirtualKeyCode};
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

    generate_spawn(&mut core, &mut game_state);
    game_state.test();

    core.run(event_loop, game_state, move |core, game_state| {
        if(game_state.key_states.new_pressed.contains(&VirtualKeyCode::T)) {
            chunk_movement(core, game_state, Vector3::new(0, 0, 1));
        }
        if(game_state.key_states.new_pressed.contains(&VirtualKeyCode::R)) {
            generate_spawn(core, game_state);
        }

        game_state.update_player(game_state.speed_modifier);
        return PushConstants{
            transform: game_state.get_player_transform()
        }
    });
}

fn generate_spawn(core: &mut RenderCore, game_state: &mut GameState) {
    let render_dist = core.get_settings().render_distance as i32;
    for x in -render_dist..=render_dist {
        for y in -render_dist..=render_dist {
            for z in -render_dist..=render_dist {
                core.upload_chunk(Vector3::new(x, y, z), game_state);
            }
        }
    }
}

fn chunk_movement(core: &mut RenderCore, game_state: &mut GameState, direction: Vector3<i32>) {
    let render_dist = core.get_settings().render_distance as i32;
    for x in -render_dist..=render_dist {
        for y in -render_dist..=render_dist {
            for z in -render_dist..=render_dist {
                core.upload_chunk(game_state.current_player_chunk() + Vector3::new(x, y, z), game_state);
            }
        }
    }
}
