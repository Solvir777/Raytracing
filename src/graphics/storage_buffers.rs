use std::sync::Arc;
use vulkano::image::Image;
use vulkano::image::view::ImageView;

pub struct StorageBuffers{
    pub terrain_image: Arc<ImageView>,
}

impl StorageBuffers{
    pub fn new(img: Arc<Image>) -> Self {
        let view = ImageView::new_default(img).unwrap();
        Self{
            terrain_image: view,
        }
    }
}