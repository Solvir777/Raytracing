pub(crate) struct Octree {
    root: Node,
    size: i32,
}

impl Octree {
    pub fn new() -> Octree {
        let a: [Box<Node>; 8] = [
            Box::from(Node::Leaf(true)),
            Box::from(Node::Leaf(false)),
            Box::from(Node::Leaf(false)),
            Box::from(Node::Leaf(false)),
            Box::from(Node::Leaf(true)),
            Box::from(Node::Leaf(true)),
            Box::from(Node::Leaf(false)),
            Box::from(Node::Leaf(true)),
        ];
        Octree {
            root: Node::Branch { 0: a },
            size: 1,
        }
    }
}
enum Node {
    Leaf(bool),
    Branch([Box<Node>; 8]),
}
