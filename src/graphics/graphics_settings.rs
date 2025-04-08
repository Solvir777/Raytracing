#[derive(Copy, Clone)]
pub struct GraphicsSettings {
    pub(crate) render_distance: u8,
    fov: f64,
    pub mouse_sensitivity: (f32, f32),
}

impl GraphicsSettings {
    pub fn default() -> Self {
        Self{
            // as in distance in view direction
            render_distance: 1,
            fov: 90.0,
            mouse_sensitivity: (0.002, 0.002),
        }
    }
}