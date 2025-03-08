use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::image::Image;

pub struct StorageBuffers{
    pub image: Arc<Image>,
    pub staging_buffer: Subbuffer<[u16]>,
}

impl StorageBuffers{
    pub fn new(image: Arc<Image>, staging_buffer: Subbuffer<[u16]>) -> Self {
        Self{
            image,
            staging_buffer
        }
    }
}