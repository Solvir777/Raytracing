#version 460

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


layout(set = 0, binding = 0) writeonly uniform image2D img;
layout(push_constant) readonly uniform PushConstantsPlayerData{
    mat4 transform;
} player;

void main() {
    vec2 norm_coordinates = 2. * (gl_GlobalInvocationID.xy / vec2(imageSize(img)) - vec2(0.5));
    vec3 rd = ( player.transform * vec4(norm_coordinates, 1., 1.)).xyz;
    vec3 ro = player.transform[3].xyz;
    float distance = ro.y / rd.y;

    vec2 c = (ro + rd * distance).xz;
    if ((int(floor(c.x)) + int(floor(c.y))) % 2 == 0) {
        imageStore(img, ivec2(gl_GlobalInvocationID.xy), vec4(1.));
        return;
    }

    imageStore(img, ivec2(gl_GlobalInvocationID.xy), vec4(vec3(0.), 1.));
    return;


    /*
    vec2 z = vec2(0.0, 0.0);
    float i;
    for (i = 0.0; i < 1.0; i += 0.05) {
        z = vec2(
            z.x * z.x - z.y * z.y + c.x,
            z.y * z.x + z.x * z.y + c.y
        );

        if (length(z) > 4.0) {
            break;
        }
    }

    vec4 to_write = vec4(vec3(i), 1.);
    imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);*/
}