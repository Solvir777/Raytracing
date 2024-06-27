mod node;

use crate::terrain_tree::node::Node;
use nalgebra::Vector3;

pub struct TerrainTree{
	size: u32,
	pos: Vector3<i32>,
	root: Node,
}

impl TerrainTree {
	pub fn new_by_function(size: u32) -> TerrainTree {
		let pos = Vector3::<i32>::new(16, -32, 16);
		let root = Node::generate_by_size(size, pos);
		Self{
			size,
			pos,
			root
		}
	}
}
