use libnoise::Simplex;
use crate::terrain::block_ids::{BlockType, SolidBlock};
use crate::terrain::Terrain;
use nalgebra::Vector3;
use crate::graphics::GraphicsCore;

#[derive(Copy, Clone)]
pub struct Chunk {
    pub(crate) data: [BlockType; Terrain::CHUNK_SIZE_3 as usize],
}

impl Chunk {
    pub(crate) fn generate_chunk(chunk_position: Vector3<i32>) -> Self {
        let simplex = Simplex::new(3);
        let mut data = [BlockType::Air; Terrain::CHUNK_SIZE_3 as usize];
        for z in 0..Terrain::CHUNK_SIZE {
            for y in 0..Terrain::CHUNK_SIZE {
                for x in 0..Terrain::CHUNK_SIZE {
                    data[pos_to_index(Vector3::new(x, y, z))] =
                        crate::terrain::terrain_function::generate_block(
                            Vector3::new(x as i32, y as i32, z as i32) + chunk_position * Terrain::CHUNK_SIZE as i32,
                            &simplex
                        );
                }
            }
        }
        Self{
            data
        }
    }

    pub(crate) fn new(data: &[u16]) -> Self {
        let mut data_array = [BlockType::Air; GraphicsCore::CHUNK_SIZE_3 as usize];
        data_array.copy_from_slice(&*data.iter().map(
            |x| {
                if (*x == 0) {
                    BlockType::Air
                } else {
                    BlockType::SolidBlock(SolidBlock::Stone)
                }
            }
        ).collect::<Vec<_>>());
        Self{
            data: data_array
        }
    }
}

fn pos_to_index(pos: Vector3<u32>) -> usize {
    (pos.x + pos.y * Terrain::CHUNK_SIZE + pos.z * Terrain::CHUNK_SIZE * Terrain::CHUNK_SIZE) as usize
}