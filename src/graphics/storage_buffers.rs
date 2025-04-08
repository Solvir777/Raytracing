use std::sync::Arc;
use vulkano::buffer::{Subbuffer};
use vulkano::image::Image;

pub struct StorageBuffers{
    pub block_type_image: Arc<Image>,
    pub distance_field_image: Arc<Image>,
}

impl StorageBuffers{
    pub fn new(block_type_image: Arc<Image>, distance_field_image: Arc<Image>) -> Self {
        Self{
            block_type_image,
            distance_field_image,
        }
    }
}