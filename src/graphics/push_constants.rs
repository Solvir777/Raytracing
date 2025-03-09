use nalgebra::Matrix4;

pub struct PushConstants {
    pub transform: Matrix4<f32>,
}