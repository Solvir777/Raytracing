mod node;

use nalgebra::Vector3;
use crate::terrain_tree::node::Node;

pub struct TerrainTree{
	size: u32,
	pos: Vector3<i32>,
	root: Node,
}

impl TerrainTree {
	pub fn get_gpu_data(&self) -> Vec<u32> {
		let mut data: Vec<u32> = vec!(
			self.size,
			self.pos.x as u32,
			self.pos.y as u32,
			self.pos.z as u32,
		);
		
		let root_node_data = self.root.get_gpu_data();
		println!("root node data: \n{:?}", root_node_data);
		data.extend(root_node_data);
		data
	}
	
	pub fn new_empty_origin() -> TerrainTree {
		Self{
			size: 0,
			pos: Vector3::<i32>::zeros(),
			root: Node::Leaf(0)
		}
	}
	
	pub fn new_by_function(size: u32) -> TerrainTree {
		let pos = Vector3::<i32>::new(0, 0, 0);
		let root = Node::generate_by_size(size, pos);
		Self{
			size,
			pos,
			root
		}
	}
}
