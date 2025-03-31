use std::sync::Arc;
use vulkano::buffer::{Subbuffer};
use vulkano::image::Image;

pub struct StorageBuffers{
    pub terrain_image: Arc<Image>,
    pub staging_buffers: Vec<Subbuffer<[u16]>>,
}

impl StorageBuffers{
    pub fn new(image: Arc<Image>, buffers: Vec<Subbuffer<[u16]>>) -> Self {
        Self{
            terrain_image: image,
            staging_buffers: buffers,
        }
    }
    pub fn get_staging_buffer(&mut self) -> Subbuffer<[u16]> {
        self.staging_buffers.pop().unwrap()
    }
}