#version 460

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


layout(set = 0, binding = 0) writeonly uniform image2D img;
layout(push_constant) readonly uniform PushConstantsPlayerData{
    mat4 transform;
} player;

const vec3 colors[3] = {
    vec3(1., 0., 0.),
    vec3(0., 1., 0.),
    vec3(0., 0., 1.)
};

int min_index(vec3 i) {
    if(i.x <= i.y && i.x <= i.z) {
        return 0;
    }
    if(i.y <= i.x && i.y <= i.z) {
        return 1;
    }
    return 2;
}
vec3 cast_ray() {
    const vec2 img_size = imageSize(img);
    vec2 norm_coordinates = (gl_GlobalInvocationID.xy / vec2(img_size.x)) - vec2(0.5, img_size.y / img_size.x * 0.5);
    vec3 rd = (vec4(norm_coordinates, 1., 1.) * player.transform).xyz;
    vec3 ro = player.transform[3].xyz;


    vec3 inv_dir = 1. / rd;
    vec3 octand01 = step(vec3(0.), rd);
    vec3 octand11 = octand01 * 2. - 1.;
    vec3 pivot = floor(ro) + octand01;
    vec3 start_values = (pivot - ro) * inv_dir * octand11;

    vec3 dir_values = start_values;

    int m_index = 0;
    for (int i = 0; i < 50; i++) {
        m_index = min_index(dir_values * octand11);
        dir_values[m_index] += inv_dir[m_index];

        vec3 impact_point = ro + (((dir_values - start_values) * octand11)[m_index] - 0.001) * rd;
        if(length(floor(impact_point)) > 10.5)
            return colors[m_index];
    }
    return vec3(0.);
}

void main() {

    vec4 to_write = vec4(cast_ray(), 1.);
    imageStore(img, ivec2(gl_GlobalInvocationID.x, imageSize(img).y - gl_GlobalInvocationID.y), to_write);
}
