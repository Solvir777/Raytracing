use std::cmp::PartialEq;
use crate::tree::block_ids::{BlockType, SolidBlock};

use libnoise::{Generator, Simplex};
use crate::tree::ivec3::IVec3;


#[derive(PartialEq, Debug)]
pub enum Node{
    Leaf(BlockType),
    Branch(Box<[Node; 64]>)
}


impl Node {
    fn block_at(pos: IVec3, generator: &Simplex<3>) -> BlockType {
        let noise = generator.sample(pos.into());
        if noise < 0. {
            return BlockType::Air;
        }
        BlockType::SolidBlock(SolidBlock::Stone)
    }
    pub(crate) fn generate(pos: IVec3, size: u32, generator: &Simplex<3>) -> Self {
        if size == 0 {
            return Self::Leaf(Self::block_at(pos, generator));
        }

        let children: [Node; 64] = core::array::from_fn(|i| {
            let x = (i >> 4) as i32;
            let y = ((i >> 2) & 0b11) as i32;
            let z = (i & 0b11) as i32;

            Self::generate(pos + IVec3::new(x, y, z) * (1 << (size-1)), size - 1, generator)
        });

        if let Some(val) = same_content(&children) {
            return Self::Leaf(val)
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
            Some(a)
        }
        else{None}
    }
    else {None}
}