use nalgebra::{abs, Vector3};
use libnoise::{Generator, Simplex};
use crate::tree::block_ids::{BlockType, SolidBlock};
use crate::tree::node::Node;

pub mod node;
mod block_ids;

#[derive(Debug)]
pub struct Tree{
    generator: Simplex<3>,
    root: Node,
    pub(crate) size: u32,
    pub(crate) position: Vector3<i32>,
}
impl Tree {
    const CHUNK_SIZE: u32 = 32;
    pub(crate) fn get_chunk(&mut self, chunk_pos: Vector3<i32>) -> Vec<u16> {
        let mut ret = vec!();
        for x in 0..Self::CHUNK_SIZE {
            for y in 0..Self::CHUNK_SIZE {
                for z in 0..Self::CHUNK_SIZE {
                    let pos = chunk_pos * Self::CHUNK_SIZE as i32 + Vector3::new(x as i32, y as i32, z as i32);
                    let val = self.value_at(pos);
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
    pub(crate) fn new() -> Self {
        let generator = Simplex::new(3);
        Self{
            generator,
            root: Node::Leaf(None),
            size: 0,
            position: Vector3::new(0, 0, 0),
        }
    }

    fn out_of_bounds(&self, abs_pos: Vector3<i32>) -> bool {
        let rel_pos = abs_pos - self.position;
        rel_pos.x < 0 || rel_pos.y < 0 || rel_pos.z < 0 || rel_pos.x >= (1 << (self.size * 2)) || rel_pos.y >= (1 << (self.size * 2)) || rel_pos.z >= (1 << (self.size * 2))
    }


    pub fn value_at(&mut self, absolute_pos: Vector3<i32>) -> BlockType {
        self.fit(absolute_pos);

        let pos = absolute_pos - self.position;
        let mut current = &mut self.root;
        for i in (0..=self.size).rev() {
            if let Node::Leaf(content) = current {
                if let Some(block) = content {
                    return *block;
                }
                if i == 0 {
                    let block_id = generate_block(absolute_pos, &self.generator);
                    *current = Node::Leaf(Some(block_id));
                    return block_id;
                }
                std::mem::swap(current, &mut Node::default_branch());
            }
            if let Node::Branch(children) = current {
                let t_pos = Vector3::new(pos.x >> ((i-1) * 2), pos.y >> ((i-1) * 2), pos.z >> ((i-1) * 2));
                let index = as_index(&t_pos);
                current = &mut children[index];
            }
        }
        panic!("shouldnt happen");
    }

    fn fit(&mut self, position: Vector3<i32>) {
        while self.out_of_bounds(position) {
            if self.size % 2 == 0 {
                self.position -= Vector3::new(1, 1, 1) * (1 << (self.size * 2));
            }
            else {
                self.position -= Vector3::new(1, 1, 1) * (1 << (self.size * 2 + 1))
            }

            //expand the tree so that the old root becomes a child of the new root
            self.expand();
            self.size += 1;
        }
    }

    /// Expands the tree by one level without changing size or position
    fn expand(&mut self) {
        const OPTION_A_INDEX: usize = 21;
        const OPTION_B_INDEX: usize = 42;

        if self.root != Node::Leaf(None) {
            let old_root = std::mem::replace(&mut self.root, Node::default_branch());
            if let Node::Branch(children) = &mut self.root {
                children[if self.size % 2 == 0 { OPTION_A_INDEX } else { OPTION_B_INDEX }] = old_root;
            } else { panic!("shouldnt happen"); }
        }
    }
}

fn as_index(p: &Vector3<i32>) -> usize {
    let (x, y, z) = (p.x & 3, p.y & 3, p.z & 3);
    (x + (y << 2) + (z << 4)) as usize
}


fn generate_block(abs_pos: Vector3<i32>, generator: &Simplex<3>) -> BlockType {
    let npos = abs_pos.cast() * 0.025;
    let noise = generator.sample([npos.x, npos.y, npos.z]);
    if noise > 0.75 {
        return BlockType::Air;
    }
    BlockType::SolidBlock(SolidBlock::Stone)
}