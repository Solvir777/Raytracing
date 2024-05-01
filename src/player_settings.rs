pub struct PlayerSettings {
	sensitivity: (f32, f32),
	fov: f32,
}


impl PlayerSettings{
	pub fn default() -> Self{
		Self{
			sensitivity: (1., 1.),
			fov: 90.,
		}
	}
}
