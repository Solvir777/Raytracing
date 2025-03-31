use std::collections::HashMap;
use crate::graphics::GraphicsCore;
use libnoise::{Generator, Simplex};
use nalgebra::Vector3;
use vulkano::pipeline::Pipeline;
use crate::terrain::chunk::Chunk;

pub(crate) mod block_ids;
pub(crate) mod terrain_gen;
pub(crate) mod chunk;
pub(crate) mod terrain_function;

pub struct Terrain {
    generator: Simplex<3>,
    pub(crate) chunks: HashMap<Vector3<i32>, Chunk>,
}


impl Terrain {
    const CHUNK_SIZE: u32 = GraphicsCore::CHUNK_SIZE;
    const CHUNK_SIZE_3: u32 = GraphicsCore::CHUNK_SIZE * GraphicsCore::CHUNK_SIZE * GraphicsCore::CHUNK_SIZE;
    pub(crate) fn get_chunk(&mut self, chunk_pos: Vector3<i32>) -> &Chunk {
        if self.chunks.contains_key(&chunk_pos) {
            return self.chunks.get(&chunk_pos).unwrap()
        }

        let data = Chunk::generate_chunk(chunk_pos);
        self.chunks.insert(chunk_pos, data.clone());
        self.chunks.get(&chunk_pos).unwrap()
    }
    pub(crate) fn write_chunk(&mut self, data: &[u16], chunk_pos: Vector3<i32>) {
        self.chunks.insert(
            chunk_pos,
            Chunk::new(data)
        );
    }
    pub(crate) fn new() -> Self {
        let generator = Simplex::new(3);
        Self {
            generator,
            chunks: HashMap::new(),
        }
    }
}