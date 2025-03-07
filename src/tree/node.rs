pub enum Node{
    Leaf(u32),
    Branch(Box<[Node; 64]>)
}

pub enum BlockType {
    Air,
    TransparentBlock(TransparentId),
    SolidBlock(SolidId),
}

enum SolidId {
    Stone,
    Grass
}

enum TransparentId{

}
