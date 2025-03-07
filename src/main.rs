mod shaders;
mod graphics;
mod tree;

fn main() {
    let (event_loop, core) = graphics::RenderCore::new();

    core.run(event_loop);
}