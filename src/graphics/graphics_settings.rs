pub struct GraphicsSettings {
    pub(crate) render_distance: u8,
    fov: f64,
}

impl GraphicsSettings {
    pub fn default() -> Self {
        Self{
            /// as in distance in view direction
            render_distance: 3,
            fov: 90.0,
        }
    }
}