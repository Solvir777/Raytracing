use std::time::Instant;
use nalgebra::{Matrix4, Vector3};
use vulkano::sync::GpuFuture;
use crate::game_logic::key_states::KeyStates;
use crate::game_logic::player::Player;
use crate::graphics::GraphicsCore;
use crate::terrain::Terrain;


pub struct GameState {
    pub(crate) player: Player,
    pub(crate) terrain: Terrain,
    start_time: Instant,
    pub(crate) key_states: KeyStates,
    pub speed_modifier: f32
}


impl GameState {
    pub fn current_player_chunk(&self) -> Vector3<i32> {
        self.player.get_position().map(|x| (x / GraphicsCore::CHUNK_SIZE as f32).floor() as i32)
    }
    pub(crate) fn player_look_dir(&mut self, delta: (f64, f64), sens: (f32, f32)) {
        self.player.look_direction.0 = self.player.look_direction.0 - delta.0 as f32 * sens.0;
        self.player.look_direction.1 = (self.player.look_direction.1 - delta.1 as f32 * sens.1).clamp(-1.7, 1.7);
    }
    pub(crate) fn new(render_core: &mut GraphicsCore) -> Self {

        let mut gs = GameState {
            player: Player::spawn(),
            terrain: Terrain::new(),
            start_time: Instant::now(),
            key_states: KeyStates::new(),
            speed_modifier: 1.0,
        };

        gs.upload_chunks_around_player(render_core, None);
        gs
    }

    pub fn update_player(&mut self, speed_modifier: f32, render_core: &mut GraphicsCore){
        let old_chunk = self.current_player_chunk();

        self.player.apply_rotation();
        self.player.update_pos(&self.key_states.currently_pressed, speed_modifier);

        self.upload_chunks_around_player(render_core, Some(old_chunk));
    }

    pub fn upload_chunks_around_player(&mut self, core: &mut GraphicsCore, old_chunk: Option<Vector3<i32>>) {
        let t = Instant::now();
        let new_chunk = self.current_player_chunk();
        if let Some(old_pos) = old_chunk{
            let chunk_diff = new_chunk - old_pos;
            let max_diff = chunk_diff.amax();

            if max_diff == 0 {
                return;
            }
        }
        let gen_dist = core.get_settings().render_distance as i32 + 1;
        for x in -gen_dist..=gen_dist {
            for y in -gen_dist..=gen_dist {
                for z in -gen_dist..=gen_dist {
                    let chunk_position = new_chunk + Vector3::new(x, y, z);
                    if let Some(old_chunk_pos) = old_chunk {
                        if (chunk_position - old_chunk_pos).amax() <= gen_dist {
                            continue;
                        }
                    }
                    core.upload_chunk_gpu(chunk_position, self);
                    //core.calculate_distance_field(chunk_position);
                }
            }
        }
        println!("waiting");
        core.previous_frame_end.take().unwrap().then_signal_fence_and_flush().unwrap().wait(None).unwrap();
        println!("finished");
        core.previous_frame_end = Some(vulkano::sync::now(core.device.clone()).boxed());
        println!("time to generate chunks: {} secs", t.elapsed().as_secs_f64());

        core.reset_distance_field();
    }


    pub fn get_player_transform(&self) -> Matrix4<f32> {
        self.player.get_transform()
    }
}