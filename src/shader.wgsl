@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    var fullscreentriangle = array<vec2f, 3>(vec2f(-3., 1.), vec2f(1., 1.), vec2f(1., -3.));
    return vec4<f32>(fullscreentriangle[in_vertex_index], 0.0, 1.0);
}

struct UniformData{
    transform_mat: mat4x4<f32>
}

const Resolution: vec2<f32> = vec2(1920., 1080.);
@group(0) @binding(0) var<uniform> uniforms: UniformData;

fn plane_ray(origin: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    if(direction.y > 0.) {
        return vec3(0., 100000., 0.);
    }
    return origin + direction / direction.y * origin.y;
}


@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
	var ray_origin: vec3<f32> = uniforms.transform_mat.w.xyz;
	var pixel_rd: vec4<f32> = vec4<f32>(normalize(vec3<f32>((fragCoord.xy - (Resolution / 2)) / Resolution.x, 1.0)), 1.);
	var rd = pixel_rd * uniforms.transform_mat;

    let impact = plane_ray(ray_origin, rd.xyz);

    let evenOrNot = floor(impact.x) + floor(impact.z);
    let mult = 25. / length(ray_origin - impact);
    if evenOrNot % 2 == 0 {
        return vec4<f32>(1., 1., 1., 1.) * mult;
    }
    return vec4<f32>(0., 0., 0., 1.) * mult;


    /*let a = fract(ray_origin.x) / rd.x;
    let b = fract(ray_origin.y) / rd.y;
    let c = fract(ray_origin.z) / rd.z;

    if(a > b && a > c) {
        return vec4<f32>(1., 0., 0., 1.);
    }
    if(b > c) {
        return vec4<f32>(0., 1., 0., 1.);
    }
    return vec4<f32>(0., 0., 1., 1.);*/
}