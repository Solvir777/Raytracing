use std::ops::Div;
use nalgebra::{min, Vector3};
use opensimplex_noise_rs::OpenSimplexNoise;
use crate::graphics_handler::GraphicsHandler;
use crate::player::Player;

pub const CHUNK_SIZE: u32 = 16;

pub struct Terrain {
	data: [u32; (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize],
}
impl Terrain{
	pub fn new() -> Self{
		Self{
			data: world_function(Vector3::new(0, 0, 0)),
		}
	}
	
	pub fn get_data(&self) -> [u32; (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize]{
		self.data
	}
	
	pub fn place_block(
		&mut self,
		block_pos: Vector3<i32>,
		block_type: u32,
		graphics_handler: &mut GraphicsHandler
	) {
		let index = block_pos.x as usize * CHUNK_SIZE as usize * CHUNK_SIZE as usize + block_pos.y as usize * CHUNK_SIZE as usize + block_pos.z as usize;
		
		graphics_handler.update_distance_field(index, block_type);
		
		self.data[index] = block_type;
	}
	pub fn raycast_place_block(
		&mut self,
		player: &Player,
		block_type: u32,
		graphics_handler: &mut GraphicsHandler,
	) {
		fn is_in_bounds(p: Vector3<i32>, lower: i32, upper: i32) -> bool{
			lower <= p.min() && p.max() < upper
		}
		fn vector_one() -> Vector3<i32>{
			Vector3::new(1, 1, 1)
		}
		
		let rd = player.forward();
		let ro = player.position();
		let (t_near, t_far, intersection_index) = aabb_intersection(ro, rd, 0., CHUNK_SIZE as f32);
		if t_far < t_near {
			return;
		}
		
		let ioctand01 = Vector3::new((rd.x > 0.) as i32, (rd.y > 0.) as i32, (rd.z > 0.) as i32) as Vector3<i32>;
		let ioctand11 = ioctand01 * 2 - Vector3::new(1, 1, 1);
		
		let f_octand11 = (ioctand01 * 2 - Vector3::new(1, 1, 1)).map(|x| x as f32) as Vector3<f32>;
		let mut int_point = {
			let first_contact = ro + rd * 0f32.max(t_near);
			
			let mut ipos = first_contact.map(|l| l.floor() as i32) as Vector3<i32>;
			ipos[intersection_index] = first_contact[intersection_index].round() as i32 - 1 + ioctand01[intersection_index];
			(ipos + ioctand01) as Vector3<i32>
		};
		
		let mut dir_values = div(int_point.map(|x| x as f32) - ro, rd).zip_map(&f_octand11, |x, y| x * y) as Vector3<f32>;
		
		
		let ray_step = rd.map(|el| 1. / el) as Vector3<f32>;
		let increments = rd.map(|el| (0. < el) as i32 * 2 - 1) as Vector3<i32>;
		
		for i in 0..200 {
			
			let minindex = (dir_values.zip_map(&f_octand11, |x, y| x * y)).imin();
			
			let grid_p = int_point - ioctand01;
			let next = {
				let mut a = grid_p;
				a[minindex] += increments[minindex];
				a
			};
			if(!is_in_bounds(grid_p, 0, 16) || !is_in_bounds(next, 0, 16)) {
				return;
			}
			if(self.data[index(next)] != 0) {
				self.place_block(grid_p, block_type, graphics_handler);
				return;
			}
			int_point[minindex] += increments[minindex];
			dir_values[minindex] += ray_step[minindex];
			
		}
	}
	
	pub fn raycast_destroy_block(
		&mut self,
		player: &Player,
		block_type: u32,
		graphics_handler: &mut GraphicsHandler,
	) {
		fn is_in_bounds(p: Vector3<i32>, lower: i32, upper: i32) -> bool{
			lower <= p.min() && p.max() < upper
		}
		fn vector_one() -> Vector3<i32>{
			Vector3::new(1, 1, 1)
		}
		
		let rd = player.forward();
		let ro = player.position();
		let (t_near, t_far, intersection_index) = aabb_intersection(ro, rd, 0., CHUNK_SIZE as f32);
		if t_far < t_near {
			return;
		}
		
		let ioctand01 = Vector3::new((rd.x > 0.) as i32, (rd.y > 0.) as i32, (rd.z > 0.) as i32) as Vector3<i32>;
		let ioctand11 = ioctand01 * 2 - Vector3::new(1, 1, 1);
		
		let f_octand11 = (ioctand01 * 2 - Vector3::new(1, 1, 1)).map(|x| x as f32) as Vector3<f32>;
		let mut int_point = {
			let first_contact = ro + rd * 0f32.max(t_near);
			
			let mut ipos = first_contact.map(|l| l.floor() as i32) as Vector3<i32>;
			ipos[intersection_index] = first_contact[intersection_index].round() as i32 - 1 + ioctand01[intersection_index];
			(ipos + ioctand01) as Vector3<i32>
		};
		
		let mut dir_values = div(int_point.map(|x| x as f32) - ro, rd).zip_map(&f_octand11, |x, y| x * y) as Vector3<f32>;
		
		
		let ray_step = rd.map(|el| 1. / el) as Vector3<f32>;
		let increments = rd.map(|el| (0. < el) as i32 * 2 - 1) as Vector3<i32>;
		
		for i in 0..200 {
			
			let minindex = (dir_values.zip_map(&f_octand11, |x, y| x * y)).imin();
			
			let grid_p = int_point - ioctand01;
			if(!is_in_bounds(grid_p, 0, 16)) {
				return;
			}
			if(self.data[index(grid_p)] != 0) {
				self.place_block(grid_p, block_type, graphics_handler);
				return;
			}
			int_point[minindex] += increments[minindex];
			dir_values[minindex] += ray_step[minindex];
		}
	}
}



fn world_function(
	chunk_pos: Vector3<i32>,
) -> [u32; (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize] {
	const SCALE: f64 = 0.05;
	let noise_gen: OpenSimplexNoise = OpenSimplexNoise::new(Some(883_279_212_983_182_319));
	let mut terrain_data = [0u32; (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize];
	for i in 0..(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) {
		let (x, y, z) = (
			i / (CHUNK_SIZE * CHUNK_SIZE),
			i % (CHUNK_SIZE * CHUNK_SIZE) / CHUNK_SIZE,
			i % CHUNK_SIZE,
		);
		
		let noise_pos = Vector3::<f64>::new(
			(chunk_pos.x + x as i32) as f64,
			(chunk_pos.y + y as i32) as f64,
			(chunk_pos.z + z as i32) as f64,
		) * SCALE;
		let mut value = (noise_gen.eval_3d(noise_pos.x, noise_pos.y, noise_pos.z) < 0.) as u32;
		let up_value = (noise_gen.eval_3d(noise_pos.x, noise_pos.y + 1. * SCALE, noise_pos.z) < 0.) as u32;
		if value == 1 {
			if up_value == 1 {
				value = 2;
			}
		}
		value = (noise_pos.y < 0.5) as u32;
		terrain_data[(x * CHUNK_SIZE * CHUNK_SIZE + y * CHUNK_SIZE + z) as usize] = value;
	}
	terrain_data
}

fn div(a: Vector3<f32>, b: Vector3<f32>) -> Vector3<f32> {
	Vector3::new(a.x / b.x, a.y / b.y, a.z / b.z)
}
fn aabb_intersection(ro: Vector3<f32>, rd: Vector3<f32>, bb_min: f32, bb_max: f32) -> (f32, f32, usize) {
	let bb_min = Vector3::new(bb_min, bb_min, bb_min);
	let bb_max = Vector3::new(bb_max, bb_max, bb_max);
	
	let t_min = div(bb_min - ro, rd);
	let t_max = div(bb_max - ro, rd);
	let t1 = t_min.zip_map(&t_max, |a, b| a.min(b)) as Vector3<f32>;
	let t2 = t_min.zip_map(&t_max, |a, b| a.max(b)) as Vector3<f32>;
	
	(t1.max(), t2.min(), t1.imax())
}

fn index(pos: Vector3<i32>) -> usize {
	let index = pos.x as usize * CHUNK_SIZE as usize * CHUNK_SIZE as usize + pos.y as usize * CHUNK_SIZE as usize + pos.z as usize;
	if index >= 4096 {
		println!("tried to place block at {}", pos);
	}
	index
}