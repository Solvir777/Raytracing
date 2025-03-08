
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BlockType {
    Air,
    TransparentBlock(TransparentBlock),
    SolidBlock(SolidBlock),
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SolidBlock {
    Stone,
    Grass,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum TransparentBlock {
    Glass,
    Water,
}
