use libnoise::Simplex;
use crate::tree::node::Node;
use crate::tree::ivec3::IVec3;

pub mod node;
mod block_ids;
mod ivec3;

#[derive(Debug)]
pub struct Tree{
    root: Node,
    size: u32,
    position: IVec3,
}

impl Tree {
    fn new(size: u32) -> Self {
        let generator = Simplex::new(10);
        let root = Node::generate(IVec3::new(0,0,0), size, &generator);
        Self{
            root,
            size,
            position: IVec3::new(0, 0, 0),
        }
    }
}