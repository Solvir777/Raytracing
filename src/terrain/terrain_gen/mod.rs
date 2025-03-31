pub mod generator_cs {
    vulkano_shaders::shader!{
        ty: "compute",
        path: r"src/terrain/terrain_gen/terrain_gen.comp",
    }
}