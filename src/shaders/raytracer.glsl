#version 460
const uint CHUNK_SIZE = 64;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


layout(set = 0, binding = 0) writeonly uniform image2D render_target;

layout(set = 0, binding = 1, r32ui) readonly uniform uimage3D distance_field;

layout(set = 0, binding = 2) uniform texture2DArray textures;
//layout(set = 0, binding = 3) uniform texture2DArray multi_block_textures;
layout(set = 0, binding = 3) uniform sampler texture_sampler;


layout(push_constant) readonly uniform PushConstantsPlayerData{
    mat4 transform;
} player;


int min_index(vec3 i) {
    if(i.x <= i.y && i.x <= i.z) {
        return 0;
    }
    if(i.y <= i.x && i.y <= i.z) {
        return 1;
    }
    return 2;
}

float far_plane(vec3 ro, vec3 rd, vec3 bb_min, vec3 bb_max) {
    vec3 t_min = (bb_min - ro) * rd;
    vec3 t_max = (bb_max - ro) * rd;
    vec3 t2 = max(t_min, t_max);
    float t_far = min(min(t2.x, t2.y), t2.z);
    return t_far;
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

bool in_bounds(ivec3 p) {
    const ivec2 b = ivec2(0, CHUNK_SIZE);
    return b.x <= p.x && p.x < b.y && b.x <= p.y && p.y < b.y && b.x <= p.z && p.z < b.y;
}



vec3 along_ray(vec3 ro, vec3 rd, ivec3 corner) {
    return (vec3(corner) - ro) / rd;
}

vec3 reflection(vec3 ro, vec3 earlier_rd, uint ii) {
    const uint max_bounces = 5;


    vec3 rd = earlier_rd;
    rd[ii] = -rd[ii];
    ivec3 VMASK = ivec3(step((0.), rd));
    ivec3 octand = VMASK * 2 - 1;
    uint intersection_index = ii;

    for(int b = 0; b < max_bounces; b++) {

        float overall_distance_along_ray = 0.0;

        vec3 t_far = aabb_intersection(ro, rd, vec3(0.), vec3(CHUNK_SIZE));

        for (int i = 0; i < 100 && overall_distance_along_ray < t_far.y; i++) {

            vec3 f_point = ro + overall_distance_along_ray * rd;
            ivec3 p = ivec3(f_point);

            p[intersection_index] = int(round(f_point)[intersection_index]) - 1 + VMASK[intersection_index];

            uint d = imageLoad(distance_field, p).x;
            if (2147483648 < d) {
                uint texture_index = ((intersection_index != 1) ? 0 : (VMASK[intersection_index] + 1));
                vec2 tex_p = texture_index != 0 ?
                fract(vec2(f_point[0], f_point[2])) :
                fract(1. - vec2(f_point[2 - intersection_index], f_point[1]));


                vec4 material = texture(sampler2DArray(textures, texture_sampler), vec3(tex_p, ((d &~ 3221225472) - 1) * 3 + texture_index));
                if(material.z < 0.9) {
                    return material.xyz;
                }

                ro = f_point;
                rd[intersection_index] = -rd[intersection_index];
                VMASK = ivec3(step((0.), rd));
                octand = VMASK * 2 - 1;
                break;
            }

            vec3 distance_to = along_ray(ro, rd, p + octand * ivec3(d) + 1 - VMASK);
            intersection_index = min_index(distance_to);
            overall_distance_along_ray = distance_to[intersection_index];
        }
    }
    return vec3(0.1);
}

vec3 cast_ray() {
    const vec2 render_img_size = imageSize(render_target);

    const vec2 norm_coordinates = (gl_GlobalInvocationID.xy / vec2(render_img_size.x)) - vec2(0.5, render_img_size.y / render_img_size.x * 0.5);

    if(length(norm_coordinates) < 0.002) {
        return vec3(1., 0., 0.);
    }

    const vec3 rd = (vec4(norm_coordinates, 1., 1.) * player.transform).xyz;
    const vec3 ro = player.transform[3].xyz;
    const ivec3 VMASK = ivec3(step((0.), rd));
    const ivec3 octand = VMASK * 2 - 1;
    const vec3 inv_rd = 1. / rd;
    vec3 aabb = aabb_intersection(ro, rd, vec3(0.), vec3(CHUNK_SIZE));


    if(aabb.x > aabb.y || aabb.y < 0.) {
        return vec3(0.1);
    }

    float overall_distance_along_ray = max(aabb.x, 0);
    uint intersection_index = uint(aabb.z);



    int i = 0;//todo

    for(;i < 100 && overall_distance_along_ray < aabb.y; i++) {

        vec3 f_point = ro + overall_distance_along_ray * rd;
        ivec3 p = ivec3(f_point);

        p[intersection_index] = int(round(f_point)[intersection_index]) - 1 + VMASK[intersection_index];

        uint d = imageLoad(distance_field, p).x;
        if(2147483648 < d) {
            uint texture_index = ((intersection_index != 1)? 0: (VMASK[intersection_index] + 1));
            vec2 tex_p = texture_index != 0?
            fract(vec2(f_point[0], f_point[2])):
            fract(1. - vec2(f_point[2 - intersection_index], f_point[1]));


            vec4 material = texture(sampler2DArray(textures, texture_sampler), vec3(tex_p, (d - 2147483649) * 3 + texture_index));
            if(material.z < 0.9) {
                return material.xyz;
            }
            return reflection(f_point, rd, intersection_index);
        }

        vec3 distance_to = along_ray(ro, rd, p + octand * ivec3(d) + 1 - VMASK);
        intersection_index = min_index(distance_to);
        overall_distance_along_ray = distance_to[intersection_index];
    }




    return vec3(0.1);
}

void main() {
    vec4 to_write = vec4(cast_ray(), 1.);
    imageStore(render_target, ivec2(gl_GlobalInvocationID.x, imageSize(render_target).y - gl_GlobalInvocationID.y), to_write);
}
