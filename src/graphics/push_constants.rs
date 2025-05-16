use nalgebra::Matrix4;
use vulkano::buffer::BufferContents;

#[repr(C)]
#[derive(BufferContents)]
pub struct PushConstants {
    pub transform: Matrix4<f32>,
    pub image_index: u32,
}