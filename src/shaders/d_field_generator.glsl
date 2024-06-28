#version 460

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(set = 0, binding = 0) buffer Data {
    uint data[4096];
};

layout(set = 0, binding = 1, r32ui) uniform uimage3D myImage;


void main() {
    uvec3 upos = gl_GlobalInvocationID;
    ivec3 ipos = ivec3(upos);
    uint index = upos.x * 16 * 16 + upos.y * 16 + upos.z;
    if(data[index] == 1) {
        for (int x = 0; x < 16; x++) {
            for (int y = 0; y < 16; y++) {
                for (int z = 0; z < 16; z++) {
                    ivec3 lol = ivec3(x, y, z);
                    uint cheby_dist = max(abs(ipos.x - lol.x), max(abs(ipos.y - lol.y), abs(ipos.z - lol.z)));
                    imageAtomicMin(myImage, lol, cheby_dist);
                }
            }
        }
    }
}