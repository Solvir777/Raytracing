#version 460
const uint CHUNK_SIZE = 64;

layout(local_size_x = 16, local_size_y = 8, local_size_z = 8) in;


layout(set = 0, binding = 0) buffer Data {
    uint data[CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE];
};

layout(set = 0, binding = 1, r32ui)  uniform uimage3D myImage;

void initial() {
    uvec3 upos = gl_GlobalInvocationID;
    ivec3 ipos = ivec3(upos);
    uint index = upos.x * CHUNK_SIZE * CHUNK_SIZE + upos.y * CHUNK_SIZE + upos.z;
    if(data[index] == 0) {//block is air -> find nearest solid
        uint minimum = CHUNK_SIZE;
        for (int x = 0; x < CHUNK_SIZE; x++) {
            for (int y = 0; y < CHUNK_SIZE; y++) {
                for (int z = 0; z < CHUNK_SIZE; z++) {
                    ivec3 lol = ivec3(x, y, z);
                    uint other_index = x * CHUNK_SIZE * CHUNK_SIZE + y * CHUNK_SIZE + z;
                    if(data[other_index] != 0) {
                        uint cheby_dist = max(abs(ipos.x - lol.x), max(abs(ipos.y - lol.y), abs(ipos.z - lol.z)));
                        minimum = min(minimum, cheby_dist);
                    }
                }
            }
        }
        imageStore(myImage, ipos, uvec4(minimum));
    }
    else{
        imageStore(myImage, ipos, uvec4(data[index] | 2147483648));
    }
}

void main() {
    initial();
}