pub mod rt_cs {
    vulkano_shaders::shader!{
        ty: "compute",
        path: r"src/shaders/render.comp",
    }
}