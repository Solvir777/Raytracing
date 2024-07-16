use std::array::IntoIter;
use nalgebra::Vector3;
pub const CHUNK_SIZE: u32 = 64;
pub const CHUNK_SIZE_3: usize = (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize;

pub(crate) struct Tree {
	size: u32,
	pos: Vector3<i32>,
	root: Node,
}

#[derive(Debug)]
pub struct Chunk {
	data: Box<[i32; CHUNK_SIZE_3]>,
}

#[derive(Debug)]
enum Node {
	Leaf(Chunk),
	Branch([Box<Node>; 8])
}

impl Chunk {
	pub fn get_data(&self) -> Box<[i32; CHUNK_SIZE_3]> {self.data.clone()}
}
impl Tree{
	pub fn new() -> Self{
		let ret = Self{
			size: 0,
			pos: Vector3::new(0, 0, 0),
			root: Node::Leaf(Chunk::new(Vector3::new(0, 0, 0))),
		};
		ret.node_at(Vector3::new(0, 0, 0));
		ret
	}
	pub fn get_data_at(&self, pos: Vector3<i32>) -> Box<[i32; CHUNK_SIZE_3]> {
		if let Node::Leaf(chunk) = self.node_at(pos) {
			return chunk.get_data();
		}
		panic!();
	}
	
	fn node_at(&self, in_pos: Vector3<i32>) -> &Node {
		let pos = (in_pos - self.pos * CHUNK_SIZE as i32) as Vector3<i32>;
		let mut current_node = &self.root;
		for i in (0..self.size).rev() {
			let (xbit, ybit, zbit) = ((pos.x >> i) & 1, (pos.y >> i) & 1, (pos.z >> i) & 1);
			let index = xbit * 4 + ybit * 2 + zbit;
			if let Node::Branch(children)  = current_node{
				current_node = &*children[index as usize];
			} else{panic!("this shouldn't be a Leaf but it is!");}
		}
		if let Node::Branch(_) = current_node {panic!("this should be a leaf, but isn't")}
		current_node
	}
}

impl Chunk {
	fn empty() -> Self {
		Self{data: [0i32; CHUNK_SIZE_3].to_vec().try_into().unwrap()}
	}
	fn new(
		chunk_pos: Vector3<i32>,
	) -> Self {
		use opensimplex_noise_rs::OpenSimplexNoise;
		const SCALE: f64 = 0.05;
		let noise_gen: OpenSimplexNoise = OpenSimplexNoise::new(Some(883_279_212_983_182_319));
		let mut terrain_data: Box<[i32; CHUNK_SIZE_3]> = [0i32; CHUNK_SIZE_3].to_vec().try_into().unwrap();
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
			terrain_data[(x * CHUNK_SIZE * CHUNK_SIZE + y * CHUNK_SIZE + z) as usize] = value as i32;
		}
		
		Chunk{data: terrain_data}
	}
}