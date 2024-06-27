#version 460

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform Data {
    uint data[4096];
};

layout(set = 0, binding = 1, r32ui) writeonly uniform uimage3D myImage;


void main() {
    //imageAtomicMin()
    uint a = data[0];
    uint index = gl_GlobalInvocationID.x * 16 * 16 + gl_GlobalInvocationID.y * 16 + gl_GlobalInvocationID.z;
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    int value = (pos.x + pos.y - 8) / 2;
    //int value = pos.y - 2;
    imageStore(myImage, ivec3(gl_GlobalInvocationID), uvec4(max(value, 0)));
}