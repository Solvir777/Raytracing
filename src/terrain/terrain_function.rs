use libnoise::{Generator, Simplex};
use nalgebra::Vector3;
use crate::terrain::block_ids::{BlockType, SolidBlock};

pub fn generate_block(pos: Vector3<i32>, simplex_noise: &Simplex<3>) -> BlockType {
    let npos = pos.cast() * 0.005;
    let noise = simplex_noise
        .sample([npos.x, npos.y, npos.z])
        - pos.y as f64 * 0.025;

    if noise > 0. {
        return BlockType::Air;
    }
    BlockType::SolidBlock(SolidBlock::Stone)
}