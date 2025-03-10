use std::cmp::PartialEq;
use crate::tree::block_ids::BlockType;

#[derive(PartialEq, Debug)]
pub enum Node{
    Leaf(Option<BlockType>),
    Branch(Box<[Node; 64]>)
}
impl Node {
    pub fn default_branch() -> Self {
        Self::Branch(Box::new([const { Node::Leaf(None) }; 64]))
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
