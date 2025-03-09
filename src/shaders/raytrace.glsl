layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(r16ui, set = 0, binding = 1) readonly uniform uimage3D voxeldata;

layout(push_constant) uniform ColorBlock {
    mat4 cam_transform;
} push;


vec3 raycast() {
    imageLoad(voxeldata, ivec3(0));
    vec2 screen_pos_05 = (vec2(gl_GlobalInvocationID.xy)/imageSize(render_target)) - vec2(0.5);

    const vec2 img_size = imageSize(render_target);
    const vec2 norm_coordinates = vec2((gl_GlobalInvocationID.xy / vec2(img_size.x)) - vec2(0.5, img_size.y / img_size.x * 0.5));
    const vec3 rd = normalize((vec4(norm_coordinates, 1., 1.) * push.cam_transform).xyz);

    const vec3 ro = push.cam_transform[3].xyz;

    for (int i = 0; i < 250; i++) {
        vec3 current_pos = ro + rd * float(i) * 0.1;
        uint value = imageLoad(voxeldata, ivec3(current_pos)).x;
        if(value == 0) {
            return vec3(1. / (i+1));
        }
    }

    return vec3(0.5);
}