#version 460
const uint CHUNK_SIZE = 64;
const uint RENDER_DIST = 1;
const uint RENDER_DIST_ALL_DIRECTIONS = 2 * RENDER_DIST + 1;
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


layout(set = 0, binding = 0) writeonly uniform image2D render_target;

layout(set = 0, binding = 1, r32i) readonly uniform iimage3D[27] distance_field;


layout(set = 0, binding = 2) uniform texture2DArray textures;
layout(set = 0, binding = 3) uniform sampler texture_sampler;


layout(push_constant) readonly uniform PushConstantsPlayerData{
    mat4 transform;
} player;


const vec3 colors[3] = {
vec3(1., 0., 0.),
vec3(0., 1., 0.),
vec3(0., 0., 1.)
};

vec3 get_player_pos() {
    return player.transform[3].xyz;
}

int min_index(vec3 i) {
    if(i.x <= i.y && i.x <= i.z) {
        return 0;
    }
    if(i.y <= i.x && i.y <= i.z) {
        return 1;
    }
    return 2;
}

vec3 aabb_intersection(vec3 ro, vec3 rd, vec3 bb_min, vec3 bb_max) {
    vec3 t_min = (bb_min - ro) / rd;
    vec3 t_max = (bb_max - ro) / rd;
    vec3 t1 = min(t_min, t_max);
    vec3 t2 = max(t_min, t_max);
    float t_near = max(max(t1.x, t1.y), t1.z);
    float t_far = min(min(t2.x, t2.y), t2.z);
    return vec3(t_near, t_far, min_index(-t1));
}


vec3 along_ray(vec3 ro, vec3 rd, ivec3 corner) {
    return (vec3(corner) - ro) / rd;
}



vec3 cast_ray() {
    const vec2 img_size = imageSize(render_target);
    const uint RENDER_GRID_SIZE = RENDER_DIST * 2 + 1;
    const vec2 norm_coordinates = (gl_GlobalInvocationID.xy / vec2(img_size.x)) - vec2(0.5, img_size.y / img_size.x * 0.5);
    if(length(norm_coordinates) < 0.002) {
        return vec3(1., 0., 0.);
    }
    vec3 rd = (vec4(norm_coordinates, 1., 1.) * player.transform).xyz;
    vec3 ro = player.transform[3].xyz;
    ivec3 chunk = ivec3(0);//ivec3(floor((ro + rd * 10.) / CHUNK_SIZE));
    chunk = (chunk);// - ivec3(RENDER_DIST);



    const ivec3 VMASK = ivec3(step((0.), rd));
    const ivec3 octand = VMASK * 2 - 1;


    vec3 aabb = aabb_intersection(ro, rd, vec3(chunk), vec3(chunk) + vec3(CHUNK_SIZE));
    float overall_distance_along_ray = max(aabb.x, 0.);

    vec3 inv_dir = 1. / rd;
    vec3 octand01 = step(vec3(0.), rd);
    vec3 octand11 = octand01 * 2. - 1.;
    vec3 pivot = floor(ro / CHUNK_SIZE) * CHUNK_SIZE + octand01 * CHUNK_SIZE;

    vec3 dir_values = (pivot - ro) * inv_dir * octand11;

    uint intersection_index = min_index(octand11 * fract(-octand11 * ro) / rd);


    vec3 f_point = ro + overall_distance_along_ray * rd;
    ivec3 p = ivec3(f_point);
    uint d = imageLoad(distance_field[0], p).x;

    if (d < 2147483648) {
        vec3 distance_to = along_ray(ro, rd, p + octand * ivec3(d) + 1 - VMASK);
        intersection_index = min_index(distance_to);
        overall_distance_along_ray = distance_to[intersection_index];
    }


    for(int i = 0; i < 100 && overall_distance_along_ray < aabb.y; i++) {

        f_point = ro + overall_distance_along_ray * rd;
        p = ivec3(f_point);

        p[intersection_index] = int(round(f_point)[intersection_index]) - 1 + VMASK[intersection_index];

        d = imageLoad(distance_field[0], p).x;
        if(2147483648 <= d) {
            uint texture_index = ((intersection_index != 1) ? 0: (VMASK[intersection_index] + 1));
            vec2 tex_p = texture_index != 0?
            fract(vec2(f_point[0], f_point[2])):
            fract(1. - vec2(f_point[2 - intersection_index], f_point[1]));


            vec4 material = texture(sampler2DArray(textures, texture_sampler), vec3(tex_p, (d - 2147483649) * 3 + texture_index));
            return material.xyz;
        }

        vec3 distance_to = along_ray(ro, rd, p + octand * ivec3(d) + 1 - VMASK);
        intersection_index = min_index(distance_to);
        overall_distance_along_ray = distance_to[intersection_index];
    }
    return vec3(0.);
}





void main() {
    imageLoad(distance_field[0], ivec3(0));
    texture(sampler2DArray(textures, texture_sampler), vec3(0));

    vec4 to_write = vec4(cast_ray(), 1.);
    imageStore(render_target, ivec2(gl_GlobalInvocationID.x, imageSize(render_target).y - gl_GlobalInvocationID.y), to_write);
}
