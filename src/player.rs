use nalgebra::{Matrix4, Vector3};
use vulkano::buffer::BufferContents;
use crate::input_helper::InputHelper;

#[repr(C)]
#[derive(Copy, Clone, BufferContents)]
pub struct Player {
	pub transformation_matrix: Matrix4<f32>,
}
impl Player {
	const SPEED: f32 = 5.;
	pub fn new() -> Self{
		Self{
			transformation_matrix: Matrix4::<f32>::identity().append_translation(&Vector3::<f32>::new(-5., 5., -5.)),
		}
	}
	
	pub fn update_player(&mut self, input_helper: &InputHelper, view_direction: (f32, f32), delta_time: f32) {
		let movement = {
			let up = Vector3::new(0., 1., 0.);
			let (forward, right) = {
				let local_z = self.transformation_matrix.try_inverse().unwrap().transform_vector(&Vector3::new(0., 0., 1.));
				(
					Vector3::<f32>::new(local_z.x, 0., local_z.z).normalize(),
					Vector3::<f32>::new(local_z.z, 0., -local_z.x).normalize()
				)
			};
			
			let (w, a, s, d, space, shift) = input_helper.get_wasd_up_down();
			if (w != s) || (a != d) || (shift != space) {
				(Vector3::<f32>::zeros()
					+ if w { forward } else { Vector3::<f32>::zeros() }
					+ if a { -right } else { Vector3::<f32>::zeros() }
					+ if s { -forward } else { Vector3::<f32>::zeros() }
					+ if d { right } else { Vector3::<f32>::zeros() }
					+ if shift { -up } else { Vector3::<f32>::zeros() }
					+ if space { up } else { Vector3::<f32>::zeros() }
				).normalize() * Player::SPEED * delta_time
			}
			else {Vector3::<f32>::zeros()}
		} as Vector3<f32>;
		fn create_rotation_mat( angles: (f32, f32)) -> Matrix4<f32>{
			let rotation_x = Matrix4::new_rotation(Vector3::new(1., 0., 0.) * angles.0);
			let rotation_y = Matrix4::new_rotation(Vector3::new(0., 1., 0.) * angles.1);
			rotation_x * rotation_y
		}
		
		
		self.transformation_matrix = create_rotation_mat((view_direction.1, view_direction.0))
			.append_translation(&(&(self.transformation_matrix.column(3).xyz()) + movement));
		
		//println!("player position: {}", self.transformation_matrix);
		//println!("but could also be: {:?}", (self.transformation_matrix.m41, self.transformation_matrix.m42, self.transformation_matrix.m43));
	}
}