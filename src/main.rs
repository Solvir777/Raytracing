use crate::tree::ivec3::IVec3;
use crate::tree::Tree;

mod shaders;
mod graphics;
mod tree;

fn main() {
    //tree::test();
    //return;
    let (event_loop, mut core) = graphics::RenderCore::new();

    let terrain = Tree::new(3);

    let data = terrain.get_chunk(IVec3::new(0, 0, 0));

    core.update_terrain(data);

    core.run(event_loop);
}