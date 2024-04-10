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
    if(direction.y < 0. != origin.y < 0.) {
        return vec3(0., 100000., 0.);
    }
    return origin + direction / direction.y * origin.y;
}

fn cast_ray_checker_board(ray_origin: vec3<f32>, ray_direction: vec3<f32>) -> vec3<f32> {
    let impact = plane_ray(ray_origin, ray_direction);

    let evenOrNot = floor(impact.x) + floor(impact.z);
    let mult = 25. / length(ray_origin - impact);
    if evenOrNot % 2 == 0 {
        return vec3<f32>(1., 1., 1.) * mult;
    }
    return vec3<f32>(0., 0., 0.) * mult;
}

fn min_index(in: vec3<f32>) -> i32 {
    let input = in;
    if (input.x <= input.y && input.x <= input.z) {
        return 0;
    } else if (input.y <= input.x && input.y <= input.z) {
        return 1;
    } else {
        return 2;
    }
}
fn get_noise(ipos: vec3<i32>) -> bool {
    let fpos = vec3<f32>(ipos);
    if fpos.y * 0.2 - sin(fpos.x * fpos.x + fpos.z * fpos.z) * 3. >= 0 {
        return true;
    }
    return false;
}

fn to_int(in: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(round(in));
}

fn cast_ray_voxel(ray_origin: vec3<f32>, ray_direction: vec3<f32>) -> vec3<f32> {
    let inv_dir = 1. / ray_direction;
    let octand_01: vec3<f32> = step(vec3<f32>(0., 0., 0.), ray_direction);
    let octand_11: vec3<f32> = octand_01 * 2. - 1.;
    var pivot: vec3<i32> = to_int(floor(ray_origin) + octand_01);
    let start_values = (vec3<f32>(pivot) - ray_origin) * inv_dir * octand_11;



    var dir_values = start_values;
    var min_index = min_index(dir_values);
    for(var i = 0; i < 100; i++) {
        min_index = min_index(dir_values * octand_11);
        dir_values[min_index] += inv_dir[min_index];

        let noisepos = pivot - to_int(octand_01);
        if(get_noise(noisepos)){
            if ((noisepos.x + noisepos.z) % 2 == 0) {
                if(noisepos.x == 0 && noisepos.z == 0) {
                    return vec3(1., 1., 0.);
                }
                return vec3(0., 0., 0.);
            }
            return vec3(1., 1., 1.);
            //let gray = 1. / length(ray_origin - vec3<f32>(pivot));
            //return vec3<f32>(gray, gray, gray);
        }

        pivot[min_index] += i32(octand_11[min_index]);
    }
    return vec3<f32>(0., 0., 0.);

}

@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
	var ray_origin: vec3<f32> = uniforms.transform_mat.w.xyz;
	var pixel_rd: vec4<f32> = vec4<f32>(normalize(vec3<f32>((fragCoord.xy - (Resolution / 2)) / Resolution.x * (((2.))), 1.0)), 1.);
	var rd: vec3<f32> = (pixel_rd * uniforms.transform_mat).xyz;
    return vec4<f32>(cast_ray_voxel(ray_origin, rd), 1.);
}