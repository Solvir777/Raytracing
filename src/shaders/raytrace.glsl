layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(r16ui, set = 0, binding = 1) readonly uniform uimage3D voxeldata;
layout(push_constant) uniform ColorBlock {
    mat4 cam_transform;
} push;

const int CHUNK_SIZE = 32;

#include "util.glsl"

bool ray_chunk_intersection(ivec3 chunk_pos, vec3 ro, vec3 rd, out vec3 hit_pos) {
    int voxel_arr_size = imageSize(voxeldata).x / CHUNK_SIZE;
    ivec3 index_pos = rem_euclid_ivec3(chunk_pos, voxel_arr_size);

    vec2 near_far = chunk_AABB_test(chunk_pos, ro, rd);
    if (near_far.x < near_far.y) {
        return false;
    }


    return true;
}

vec3 raycast() {
    vec2 screen_pos_05 = (vec2(gl_GlobalInvocationID.xy)/imageSize(render_target)) - vec2(0.5);

    const vec2 img_size = imageSize(render_target);
    const vec2 norm_coordinates = vec2((gl_GlobalInvocationID.xy / vec2(img_size.x)) - vec2(0.5, img_size.y / img_size.x * 0.5));
    const vec3 rd = normalize((vec4(norm_coordinates, 1., 1.) * push.cam_transform).xyz);

    const vec3 ro = push.cam_transform[3].xyz;

    for (int i = 0; i < 250; i++) {
        vec3 current_pos = ro + rd * float(i) * 0.3;
        ivec3 v_pos = ivec3(current_pos);
        uint value = imageLoad(voxeldata, v_pos).x;
        if(value == 0) {
            return vec3(sin(v_pos.x * 5.), cos(v_pos.y * 0.3415), sin(2.4 + v_pos.z * 0.03));
        }
    }

    return vec3(0.5);
}