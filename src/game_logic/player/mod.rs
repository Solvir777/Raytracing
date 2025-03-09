use std::collections::HashSet;
use nalgebra::{Transform3, Vector3, Matrix4};
use winit::event::VirtualKeyCode;

pub struct Player {
    matrix: Matrix4<f32>,
    pub(crate) look_direction: (f32, f32),
}

impl Player {
    const BASE_SPEED: f32 = 0.02;
    pub fn get_transform(&self) -> Matrix4<f32> {
        self.matrix
    }
    pub fn spawn() -> Self {
        Self{
            matrix: Matrix4::identity(),
            look_direction: (0., 0.),
        }
    }

    pub fn apply_rotation(&mut self) {
        self.reset_rotation();
        let dir = self.look_direction;
        let transform = Matrix4::from_scaled_axis(Vector3::new(dir.1, 0., 0.)) * Matrix4::from_scaled_axis(Vector3::new(0., dir.0, 0.));
        self.matrix *= transform;
    }

    fn reset_rotation(&mut self) {
        let translation = self.matrix.fixed_slice::<3, 1>(0, 3).into_owned();
        self.matrix = Matrix4::identity();
        self.matrix.fixed_slice_mut::<3, 1>(0, 3).copy_from(&translation);
    }
    pub fn update_pos(&mut self, keys: &HashSet<VirtualKeyCode>) {
        let mut mov = Vector3::zeros();
        let forward = Vector3::new(-self.look_direction.0.sin() , 0., self.look_direction.0.cos());
        let right = Vector3::new(self.look_direction.0.cos() , 0., self.look_direction.0.sin());
        let up = Vector3::new(0., 1., 0.);

        if keys.contains(&VirtualKeyCode::W) {
            mov += forward;
        }
        if keys.contains(&VirtualKeyCode::A) {
            mov -= right;
        }
        if keys.contains(&VirtualKeyCode::S) {
            mov -= forward;
        }
        if keys.contains(&VirtualKeyCode::D) {
            mov += right;
        }
        if keys.contains(&VirtualKeyCode::Space) {
            mov += up;
        }
        if keys.contains(&VirtualKeyCode::LShift) {
            mov -= up;
        }
        let mov = mov.cap_magnitude(Self::BASE_SPEED);
        self.matrix.append_translation_mut(&mov);
    }
}
