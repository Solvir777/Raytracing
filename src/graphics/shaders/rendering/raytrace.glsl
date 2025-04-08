layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(r16ui, set = 0, binding = 1) readonly uniform uimage3D block_data;
layout(r16ui, set = 0, binding = 2) readonly uniform uimage3D distance_data;

layout(push_constant) uniform PushConstants {
    mat4 cam_transform;
} push;

const int CHUNK_SIZE = 32;

#include "util.glsl"

const vec3[] debug_colors = {vec3(1., 0., 0.), vec3(0., 1., 0.), vec3(0., 0., 1.)};

vec3 raycast() {
    imageLoad(distance_data, ivec3(0));

    const vec2 render_img_size = imageSize(render_target);
    const vec2 norm_coordinates = vec2(((gl_GlobalInvocationID.xy) / render_img_size) - vec2(0.5, render_img_size.y / render_img_size.x * 0.5));
    const vec3 rd = normalize((vec4(norm_coordinates, 1., 1.) * push.cam_transform).xyz);
    const vec3 ro = push.cam_transform[3].xyz;
    const vec3 inv_rd = 1. / rd;

    int voxel_data_size = imageSize(block_data).x;

    if(imageLoad(block_data, rem_euclid_ivec3(ivec3(floor(ro)), voxel_data_size)).r > 0) {
        return vec3(0., 0.8, 0.5);
    }

    ivec3 oct_rd01 = ivec3(greaterThan(rd, vec3(0.)));
    ivec3 oct_rd11 = (oct_rd01 * 2) - ivec3(1);
    int render_dist = (voxel_data_size - CHUNK_SIZE) / (2 * CHUNK_SIZE);

    ivec3 to_chunk_middle = ivec3((vec3(0.5) + floor(ro / CHUNK_SIZE)) * CHUNK_SIZE) - ivec3(floor(ro));

    vec3 t_dist_to_next = ((floor(ro) + oct_rd01) - ro) * inv_rd;
    for(ivec3 offset = ivec3(0); all(lessThan(offset + ivec3(3)/*todo*/, ivec3(voxel_data_size / 2 - 1) + to_chunk_middle * oct_rd11));) {
        int next_xyz = argmin(t_dist_to_next);
        t_dist_to_next[next_xyz] += abs(inv_rd[next_xyz]);
        offset[next_xyz] += 1;


        ivec3 pos = ivec3(floor(ro)) + offset * oct_rd11;

        uint block_type = imageLoad(block_data, rem_euclid_ivec3(pos, voxel_data_size)).r;
        uint dist = imageLoad(distance_data,rem_euclid_ivec3(pos, voxel_data_size)).r;
        if(block_type > 0) {
            return debug_colors[next_xyz] + 0.5 * vec3(sin(pos.x * 2.), sin(pos.y * 2.), sin(pos.z * 2.));
        }
        if(dist == 2) {
            return debug_colors[next_xyz] * 0.3 + debug_colors[pos.x%3] * 0.2 + debug_colors[pos.y%3] * 0.2 + debug_colors[pos.z%3] * 0.2;
        }

    }

    return vec3(0.5);
}