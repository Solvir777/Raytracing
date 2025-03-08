use std::ops::{Shl, ShlAssign, Shr, ShrAssign, Sub};
use libnoise::Simplex;
use crate::tree::block_ids::{BlockType, SolidBlock};
use crate::tree::node::Node;
use crate::tree::ivec3::IVec3;

pub mod node;
mod block_ids;
pub mod ivec3;

#[derive(Debug)]
pub struct Tree{
    root: Node,
    size: u32,
    position: IVec3,
}

pub(crate) fn test() {
    let tree = Tree::new(3);

    let pos = IVec3::new(4, 0, 1);

    let value = tree.value_at(pos);
    println!("node at {:?}: {:?}", pos, value);
}

impl Tree {
    const CHUNK_SIZE: u32 = 32;
    pub(crate) fn get_chunk(&self, pos: IVec3) -> Vec<u16> {
        let mut ret = vec!();
        for x in 0..Self::CHUNK_SIZE {
            for y in 0..Self::CHUNK_SIZE {
                for z in 0..Self::CHUNK_SIZE {
                    let val = self.value_at(pos + IVec3(x as i32, y as i32, z as i32));
                    ret.push(
                        match val {
                            BlockType::Air => {0}
                            BlockType::TransparentBlock(_) => {1}
                            BlockType::SolidBlock(_) => {1}
                        }
                    );
                }
            }
        }
        ret
    }
    pub(crate) fn new(size: u32) -> Self {
        let generator = Simplex::new(3);
        let root = Node::generate(IVec3::new(0,0,0), size, &generator);
        Self{
            root,
            size,
            position: IVec3::new(0, 0, 0),
        }
    }

    fn out_of_bounds(&self, abs_pos: IVec3) -> bool {
        let rel_pos = abs_pos - self.position;
        rel_pos.0 < 0 || rel_pos.1 < 0 || rel_pos.2 < 0 || rel_pos.0 >= (1 << (self.size * 2)) || rel_pos.1 >= (1 << (self.size * 2)) || rel_pos.2 >= (1 << (self.size * 2))
    }

    pub fn value_at(&self, absolute_pos: IVec3) -> BlockType {
        //if self.out_of_bounds(absolute_pos) {
        //    return BlockType::Air;
        //}
        let pos = absolute_pos - self.position;
        let mut current = &self.root;
        for i in (0..=self.size).rev() {
            match current {
                Node::Leaf(content) => { return *content; }
                Node::Branch(children) => {
                    let t_pos = pos >> ((i-1) * 2);
                    let index = as_index(&t_pos);
                    println!("pos: {:?}", t_pos);
                    println!("index: {}", index);
                    current = &children[index];
                }
            }
        }
        panic!("shouldnt happen");
    }
}

fn as_index(p: &IVec3) -> usize {
    let (x, y, z) = (p.0 & 3, p.1 & 3, p.2 & 3);
    (x + (y << 2) + (z << 4)) as usize
}
