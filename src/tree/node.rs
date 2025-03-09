use std::cmp::PartialEq;
use crate::tree::block_ids::{BlockType, SolidBlock};
use libnoise::{Generator, Simplex};
use nalgebra::Vector3;

#[derive(PartialEq, Debug)]
pub enum Node{
    Leaf(Option<BlockType>),
    Branch(Box<[Node; 64]>)
}
impl Node {
    pub fn default_branch() -> Self {
        Self::Branch(Box::new([const { Node::Leaf(None) }; 64]))
    }
    pub fn generate_block(pos: Vector3<i32>, generator: &Simplex<3>) -> BlockType {
        let noise = generator.sample([pos.x as f64, pos.y as f64, pos.z as f64]);
        if noise > 0.25 {
            return BlockType::Air;
        }
        BlockType::SolidBlock(SolidBlock::Stone)
    }
    pub(crate) fn generate(pos: Vector3<i32>, size: u32, generator: &Simplex<3>) -> Self {
        if size == 0 {
            return Self::Leaf(Some(Self::generate_block(pos, generator)));
        }

        let children: [Node; 64] = core::array::from_fn(|i| {
            let z = (i >> 4) as i32;
            let y = ((i >> 2) & 0b11) as i32;
            let x = (i & 0b11) as i32;

            Self::generate(pos + Vector3::new(x, y, z) * (1 << (2 * (size-1))), size - 1, generator)
        });

        if let Some(val) = same_content(&children) {
            return Self::Leaf(Some(val))
        }
        Self::Branch(Box::new(children))
    }
}

fn same_content(nodes: &[Node; 64]) -> Option<BlockType> {
    if let Node::Leaf(a) = nodes[0] {
        if nodes.iter().skip(1).all(|x| {
            if let Node::Leaf(b) = x {
                a == *b
            }
            else {false}
        }) {
            a
        }
        else{None}
    }
    else {None}
}
