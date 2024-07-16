pub mod rendering_cs {
	vulkano_shaders::shader! {
        ty: "compute",
        path: r"src/shaders/raytracer.glsl",
    }
}

pub mod df_cs {
	vulkano_shaders::shader! {
        ty: "compute",
        path: r"src/shaders/d_field_generator.glsl",
    }
}