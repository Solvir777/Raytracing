use winit::event::VirtualKeyCode;
use crate::game_logic::game_state::GameState;
use crate::graphics::push_constants::PushConstants;

mod graphics;
mod game_logic;
mod terrain;

fn main() {
    let (event_loop, mut core) = graphics::GraphicsCore::new();

    let game_state = GameState::new(&mut core);


    core.run(event_loop, game_state, move |core, game_state| {
        if(game_state.key_states.new_pressed.contains(&VirtualKeyCode::T)) {
        }
        if(game_state.key_states.new_pressed.contains(&VirtualKeyCode::R)) {
        }

        game_state.update_player(game_state.speed_modifier, core);

        return PushConstants{
            transform: game_state.get_player_transform()
        }
    });
}


